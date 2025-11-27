#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License. 
# ======================================================================================================================
"""
"""
import os
import pypto
import pytest
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose


# golden
def _apply_rotary_emb_neuron(x, cos, sin):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return torch.cat((o1, o2), dim=-1)

def apply_rotary_pos_emb_v2(q, k, cos, sin):
    x_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    q_embed = _apply_rotary_emb_neuron(q, cos, sin)
    k_embed = _apply_rotary_emb_neuron(k, cos, sin)

    if x_dtype != torch.float32:
        q_embed = q_embed.to(x_dtype)
        k_embed = k_embed.to(x_dtype)
    return q_embed, k_embed

# pypto
def rms_norm_bias(tensor_value, gamma, bias, mean_coff, eps, tile_shape):
    input_dtype = tensor_value.dtype
    # cast
    pypto.set_vec_tile_shapes(*tile_shape)
    tensor_value_fp32 = pypto.cast(tensor_value, pypto.DT_FP32)

    # gamma reshape
    pypto.set_vec_tile_shapes(tile_shape[-1])
    des_shape = [1] * len(tensor_value_fp32.shape)
    des_shape[-1] = gamma.shape[0]
    gamma_3d = pypto.reshape(gamma, des_shape)

    des_shape[-1] = bias.shape[0]
    bias_3d = pypto.reshape(bias, des_shape)

    # gamma cast
    pypto.set_vec_tile_shapes(*tile_shape)
    gamma_fp32 = pypto.cast(gamma_3d, pypto.DT_FP32)
    bias_fp32 = pypto.cast(bias_3d, pypto.DT_FP32)

    # square
    square = pypto.mul(tensor_value_fp32, tensor_value_fp32)

    # mean_res
    mean_res = pypto.mul(square, mean_coff)

    # reduce sum
    reduce_asum = pypto.sum(mean_res, -1, keepdim=True)
    reduce_sum = pypto.add(reduce_asum, eps)

    # sqrt
    reduce_sqrt = pypto.sqrt(reduce_sum)

    # div
    res_div = pypto.div(tensor_value_fp32, reduce_sqrt)

    # gamma mul
    res = pypto.mul(res_div, gamma_fp32)
    res_add = pypto.add(res, bias_fp32)

    # cast
    y_bf16 = pypto.cast(res_add, input_dtype)

    return y_bf16

def rope_data(x1, x2, cos, sin, tile_shape):
    pypto.set_vec_tile_shapes(*tile_shape)
    o1 = pypto.sub(pypto.mul(x1, cos), pypto.mul(x2, sin))
    o2 = pypto.add(pypto.mul(x2, cos), pypto.mul(x1, sin))
    # concat
    res = pypto.concat([o1, o2], 2)

    # cast
    y_bf16 = pypto.cast(res, pypto.DT_BF16)
    return y_bf16

@pypto.jit
def GLM_Quant_attention_pre(in_tensors, out_tensors, enable_residual=True):
    # 1. 添加支持动态的config
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    pypto.set_host_options(only_codegen=True)
    # 2. 从入参拿到输入和输出tensor
    x, residual_input, x_gamma, x_bias, x_scale, x_offset, weight, quant_bias, deq_scale, q_gamma, q_bias, k_gamma, k_bias, cos, sin = in_tensors
    q, k, v, residual = out_tensors

    # 3. 设置axis=0为动态shape
    pypto.mark_dynamic(x, 0)
    if enable_residual:
        pypto.mark_dynamic(residual_input, 0)
    pypto.mark_dynamic(cos, 0)
    pypto.mark_dynamic(sin, 0)
    pypto.mark_dynamic(q, 0)
    pypto.mark_dynamic(k, 0)
    pypto.mark_dynamic(v, 0)
    pypto.mark_dynamic(residual, 0)

    # 4. 得到动态tensor的shape
    bs = x.shape[0]
    max_position_embeddings = x.shape[1]
    hidden_size = weight.shape[1]

    head_size = 128
    x_mean_coff = 1.0 / x.shape[-1]
    qk_mean_coff = 1.0 / head_size
    eps = 1e-05
    half_rotary_dim = cos.shape[-1]
    rotary_dim = cos.shape[-1] * 2
    stay_dim = head_size - rotary_dim

    q_size = q.shape[-1]
    kv_size = k.shape[-1]
    q_num_head = q_size // head_size
    kv_num_head = kv_size // head_size
    kv_index = q_num_head + kv_num_head
    bs_tile = 2
    bs_loop = (bs + bs_tile -1) // bs_tile
    calc_dtype = pypto.DT_FP32
    input_dtype = x.dtype
    tiling_value = 128

    # 5. 定义动态函数
    with pypto.function("ATTENTION_PRE", [x, residual_input, x_gamma, x_bias, x_scale, x_offset, weight, quant_bias, deq_scale, q_gamma, q_bias, k_gamma, k_bias, cos, sin], [q, k, v, residual]):
        def inside_attention_pre_func():
            # 6. 实现kernel逻辑，循环展开BS动态轴
            for bs_idx in pypto.loop(bs_loop, name="LOOP_ATT_PRE_L0", idx_name="bs_idx"):
                def bs_loop_func(bs_idx):
                    # 7. 通过view得到x_tile、cos_tile、sin_tile
                    x_tile = pypto.view(x, [bs_tile, max_position_embeddings], [bs_idx * bs_tile, 0])
                    cos_tile = pypto.view(cos, [bs_tile, 1, half_rotary_dim], [bs_idx * bs_tile, 0, 0])
                    sin_tile = pypto.view(sin, [bs_tile, 1, half_rotary_dim], [bs_idx * bs_tile, 0, 0])

                    # 8. 按照计算图实现运算逻辑
                    # 设置set_vec_tile_shape时 尽可能用满UB，但不要超过UB的大小
                    # init
                    pypto.set_vec_tile_shapes(tiling_value)
                    x_gamma_fp32 = pypto.cast(x_gamma, calc_dtype)
                    x_bias_fp32 = pypto.cast(x_bias, calc_dtype)

                    x_gamma_2d = pypto.unsqueeze(x_gamma_fp32, 0)
                    x_bias_2d = pypto.unsqueeze(x_bias_fp32, 0)

                    pypto.set_vec_tile_shapes(tiling_value, tiling_value)
                    x_tile_fp32 = pypto.cast(x_tile, calc_dtype)
                    # add
                    if enable_residual:
                        residual_input_tile = pypto.view(residual_input, [bs_tile, max_position_embeddings], [bs_idx * bs_tile, 0])
                        residual_input_tile_fp32 = pypto.cast(residual_input_tile, calc_dtype)
                        x_f32 = pypto.add(residual_input_tile_fp32, x_tile_fp32) # tile_x
                    else:
                        x_f32 = x_tile_fp32
                    # rms norm
                    square = pypto.mul(x_f32, x_f32) # square
                    mean_res = pypto.mul(square, x_mean_coff) # mean_res = square * mean_coff
                    reduce_asum = pypto.sum(mean_res, -1, keepdim=True) # reduce_asum = mean_res.sum(dim=-1, keepdim=True)
                    reduce_sum = pypto.add(reduce_asum, eps) # reduce_sum = reduce_asum + eps
                    reduce_sqrt = pypto.sqrt(reduce_sum) # reduce_sqrt = torch.sqrt(reduce_sum)
                    res_div = pypto.div(x_f32, reduce_sqrt) # res_div = x_f32 / reduce_sqrt
                    
                    res = pypto.mul(res_div, x_gamma_2d) # res = res_div * weight
                    res_add = pypto.add(res, x_bias_2d)
                    
                    x_norm = pypto.cast(res_add, input_dtype)
                    residual_bf16 = pypto.cast(x_f32, input_dtype)
                    
                    # init
                    pypto.set_vec_tile_shapes(tiling_value)
                    x_scale_fp32 = pypto.cast(x_scale, calc_dtype) # bf16 -> fp32
                    x_offset_fp32 = pypto.cast(x_offset, calc_dtype) # bf16 -> fp32

                    x_scale_2d = pypto.unsqueeze(x_scale_fp32, 0)
                    x_offset_2d = pypto.unsqueeze(x_offset_fp32, 0)
                    quant_bias_2d = pypto.unsqueeze(quant_bias, 0)
                    deq_scale_2d = pypto.unsqueeze(deq_scale, 0)
                    # x quant
                    pypto.set_vec_tile_shapes(tiling_value, tiling_value)
                    x_norm_fp32 = pypto.cast(x_norm, calc_dtype) # bf16 -> fp32
                    x_mul = pypto.mul(x_norm_fp32, x_scale_2d)
                    x_add = pypto.add(x_mul, x_offset_2d)
                    x_int32 = pypto.cast(x_add, pypto.DT_INT32, pypto.CastMode.CAST_RINT) # Align ascendC
                    x_fp16 = pypto.cast(x_int32, pypto.DT_FP16)
                    x_int8 = pypto.cast(x_fp16, pypto.DT_INT8)
                    # matmul
                    pypto.set_cube_tile_shapes([tiling_value, tiling_value], [tiling_value, tiling_value], [tiling_value, tiling_value])
                    mm_int32 = pypto.matmul(x_int8, weight, pypto.DT_INT32, a_trans=False, b_trans=False)
                    mm_add = pypto.add(mm_int32, quant_bias_2d)

                    mm_fp32 = pypto.cast(mm_add, calc_dtype) # int32 -> fp32
                    mm_deq_scale = pypto.mul(mm_fp32, deq_scale_2d)
                    mm_bf16 = pypto.cast(mm_deq_scale, input_dtype) # fp32 -> bf16

                    pypto.set_vec_tile_shapes(tiling_value, tiling_value, tiling_value)
                    mm_3d = pypto.reshape(mm_bf16, [bs_tile, hidden_size // head_size, head_size])

                    # split
                    q_tile = pypto.view(mm_3d, [bs_tile, q_num_head, head_size], [0, 0, 0])
                    k_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, q_num_head, 0])
                    v_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, kv_index, 0])

                    # rms norm
                    q_norm = rms_norm_bias(q_tile, q_gamma, q_bias, qk_mean_coff, eps, [bs_tile, q_num_head, head_size])
                    k_norm = rms_norm_bias(k_tile, k_gamma, k_bias, qk_mean_coff, eps, [bs_tile, kv_num_head, head_size])

                    q_rot = pypto.view(q_norm, [bs_tile, q_num_head, rotary_dim], [0, 0, 0])
                    q_pass = pypto.view(q_norm, [bs_tile, q_num_head, stay_dim], [0, 0, rotary_dim])

                    k_rot = pypto.view(k_norm, [bs_tile, kv_num_head, rotary_dim], [0, 0, 0])
                    k_pass = pypto.view(k_norm, [bs_tile, kv_num_head, stay_dim], [0, 0, rotary_dim])

                    # apply rope
                    # cast
                    pypto.set_vec_tile_shapes(bs_tile, q_num_head, rotary_dim)
                    q_fp32 = pypto.cast(q_rot, calc_dtype)
                    k_fp32 = pypto.cast(k_rot, calc_dtype)

                    pypto.set_vec_tile_shapes(bs_tile, kv_num_head, half_rotary_dim)
                    cos_fp32 = pypto.cast(cos_tile, calc_dtype)
                    sin_fp32 = pypto.cast(sin_tile, calc_dtype)

                    # q split
                    q1 = pypto.view(q_fp32, [bs_tile, q_num_head, half_rotary_dim], [0, 0, 0])
                    q2 = pypto.view(q_fp32, [bs_tile, q_num_head, half_rotary_dim], [0, 0, half_rotary_dim])
                    
                    # rope data
                    q_rope = rope_data(q1, q2, cos_fp32, sin_fp32, [bs_tile, q_num_head, half_rotary_dim])
                    q_cat = pypto.concat([q_rope, q_pass], 2)

                    # k split
                    k1 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_rotary_dim], [0, 0, 0])
                    k2 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_rotary_dim], [0, 0, half_rotary_dim])
                    
                    # rope data
                    k_rope = rope_data(k1, k2, cos_fp32, sin_fp32, [bs_tile, kv_num_head, half_rotary_dim])
                    k_cat = pypto.concat([k_rope, k_pass], 2)

                    # post process
                    q_res = pypto.reshape(q_cat, [bs_tile, q_size])
                    k_res = pypto.reshape(k_cat, [bs_tile, kv_size])
                    v_res = pypto.reshape(v_tile, [bs_tile, kv_size])

                    k_f = pypto.cast(k_res, calc_dtype)
                    k_b = pypto.cast(k_f, input_dtype)

                    q_f = pypto.cast(q_res, calc_dtype)
                    q_b = pypto.cast(q_f, input_dtype)

                    v_f = pypto.cast(v_res, calc_dtype)
                    v_b = pypto.cast(v_f, input_dtype)

                    # # 9. 将结果搬运到输出tensor上
                    # # update output
                    q[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = q_b
                    k[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = k_b
                    v[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = v_b
                    residual[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = residual_bf16

                bs_loop_func(bs_idx)
        inside_attention_pre_func()


def test_attention_pre_GLM():
    # 1. 设置参数
    bs = 2
    max_position_embeddings = 5120
    hidden_size = 1792
    head_size = 128
    q_size = 1536
    kv_size = 128
    rotary_dim = 64
    half_rotary_dim = rotary_dim // 2
    eps = 1e-05

    device_id = int(os.environ.get('TILE_FWK_STEST_DEVICE_ID', 7))
    torch.npu.set_device(device_id)

    # 2. 构造多种shape，测试动态case
    for i in range(0, 1):
        if (i == 2):
            bs = 5

        # 3. 准备测试数据
        np.random.seed(0)
        # inputs
        # x = torch.rand(bs, max_position_embeddings, dtype=torch.bfloat16, device=f'npu:{device_id}')
        # residual_input = torch.rand(bs, max_position_embeddings, dtype=torch.bfloat16, device=f'npu:{device_id}')
        # x_gamma = torch.rand(max_position_embeddings, dtype=torch.bfloat16, device=f'npu:{device_id}')
        # x_bias = torch.rand(max_position_embeddings, dtype=torch.bfloat16, device=f'npu:{device_id}')
        # x_scale = torch.rand(max_position_embeddings, dtype=torch.bfloat16, device=f'npu:{device_id}')
        # x_offset = torch.rand(max_position_embeddings, dtype=torch.bfloat16, device=f'npu:{device_id}')
        # weight = torch.rand(max_position_embeddings, hidden_size, dtype=torch.int8, device=f'npu:{device_id}')
        # quant_bias = torch.rand(hidden_size, dtype=torch.int32, device=f'npu:{device_id}')
        # deq_scale = torch.rand(hidden_size, dtype=torch.float32, device=f'npu:{device_id}')
        # q_gamma = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        # q_bias = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        # k_gamma = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        # k_bias = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        # cos = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')
        # sin = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')

        pid = 66913
        x = torch.load("/home/c00610849/glm4/save_pt/input_pt/0.pt_{}".format(pid))
        residual_input = torch.load("/home/c00610849/glm4/save_pt/input_pt/1.pt_{}".format(pid))
        x_gamma = torch.load("/home/c00610849/glm4/save_pt/input_pt/2.pt_{}".format(pid))
        x_bias = torch.load("/home/c00610849/glm4/save_pt/input_pt/3.pt_{}".format(pid))

        x_scale = torch.load("/home/c00610849/glm4/save_pt/input_pt/4.pt_{}".format(pid))
        x_offset = torch.load("/home/c00610849/glm4/save_pt/input_pt/5.pt_{}".format(pid))
        weight = torch.load("/home/c00610849/glm4/save_pt/input_pt/6.pt_{}".format(pid))
        quant_bias = torch.load("/home/c00610849/glm4/save_pt/input_pt/7.pt_{}".format(pid))
        deq_scale = torch.load("/home/c00610849/glm4/save_pt/input_pt/8.pt_{}".format(pid))
        q_gamma = torch.load("/home/c00610849/glm4/save_pt/input_pt/9.pt_{}".format(pid))
        q_bias = torch.load("/home/c00610849/glm4/save_pt/input_pt/10.pt_{}".format(pid))
        k_gamma = torch.load("/home/c00610849/glm4/save_pt/input_pt/11.pt_{}".format(pid))
        k_bias = torch.load("/home/c00610849/glm4/save_pt/input_pt/12.pt_{}".format(pid))
        cos = torch.load("/home/c00610849/glm4/save_pt/input_pt/13.pt_{}".format(pid))
        sin = torch.load("/home/c00610849/glm4/save_pt/input_pt/14.pt_{}".format(pid))

        # outputs
        q = torch.zeros((bs, q_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
        k = torch.zeros((bs, kv_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
        v = torch.zeros((bs, kv_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
        residual = torch.zeros((bs, max_position_embeddings), dtype=torch.bfloat16, device=f'npu:{device_id}')
        enable_residual = True
        if torch.all(residual_input == 0):
            residual_input = torch.zeros((bs, x.shape[1]), dtype=x.dtype, device=f'{x.device}')
            enable_residual = False

        # # 4. 执行kernel并获取结果
        inputs = [x, residual_input, x_gamma, x_bias, x_scale, x_offset, weight, quant_bias, deq_scale, q_gamma, q_bias, k_gamma, k_bias, cos, sin]
        outputs = [q, k, v, residual]
        GLM_Quant_attention_pre(inputs, outputs, enable_residual)
        pypto.runtime._device_synchronize()

        # 5. 与PyTorch参考实现对比
        # add rms norm
        x_g, _, residual_g = torch_npu.npu_add_rms_norm(x, residual_input, x_gamma, eps)
        x_g.add_(x_bias)

        # matmul
        x_quant = torch_npu.npu_quantize(x_g, x_scale, x_offset, torch.qint8, -1, False)
        mm_golden = torch_npu.npu_quant_matmul(x_quant, weight, deq_scale, bias=quant_bias, output_dtype=torch.bfloat16)

        # split
        q_g, k_g, v_g = mm_golden.split([q_size, kv_size, kv_size], dim=-1)
        # nms norm
        q_by_head = q_g.view(*q_g.shape[:-1], q_g.shape[-1] // head_size, head_size)
        q_by_head, _ = torch_npu.npu_rms_norm(q_by_head, q_gamma, eps)
        q_by_head.add_(q_bias)
        
        k_by_head = k_g.view(*k_g.shape[:-1], k_g.shape[-1] // head_size, head_size)
        k_by_head, _ = torch_npu.npu_rms_norm(k_by_head, k_gamma, eps)
        k_by_head.add_(q_bias)
        # apply rope
        q_rot = q_by_head[..., :rotary_dim]
        q_pass = q_by_head[..., rotary_dim:]
        k_rot = k_by_head[..., :rotary_dim]
        k_pass = k_by_head[..., rotary_dim:]
        q_r, k_r = apply_rotary_pos_emb_v2(q_rot, k_rot, cos, sin)
        q_cat = torch.cat((q_r, q_pass), dim=-1)
        k_cat = torch.cat((k_r, k_pass), dim=-1)
        # post process
        q_r = q_cat.view(bs, q_size)
        k_r = k_cat.view(bs, kv_size)

        # q_b = torch.load("/home/c00610849/glm4/save_pt/output_pt/g_0.pt_{}".format(pid))
        # k_b = torch.load("/home/c00610849/glm4/save_pt/output_pt/g_1.pt_{}".format(pid))
        # v_b = torch.load("/home/c00610849/glm4/save_pt/output_pt/g_2.pt_{}".format(pid))
        # residual_b = torch.load("/home/c00610849/glm4/save_pt/output_pt/g_3.pt_{}".format(pid)) # golden

        # compare result
        from vllm_ascend.pypto.utils.common_compare import detailed_allclose_manual
        detailed_allclose_manual(np.array(residual_g.cpu().float().numpy()).flatten(), np.array(residual.cpu().float().numpy()).flatten(), "residual")
        detailed_allclose_manual(np.array(q_r.cpu().float().numpy()).flatten(), np.array(q.cpu().float().numpy()).flatten(), "q")
        detailed_allclose_manual(np.array(k_r.cpu().float().numpy()).flatten(), np.array(k.cpu().float().numpy()).flatten(),"k")
        detailed_allclose_manual(np.array(v_g.cpu().float().numpy()).flatten(), np.array(v.cpu().float().numpy()).flatten(),"v")

        # assert_allclose(np.array(q_r.cpu().flatten().tolist()), np.array(q_b.flatten().tolist()),rtol=0.001,atol=0.001)
        # assert_allclose(np.array(k_r.cpu().flatten().tolist()), np.array(k_b.flatten().tolist()),rtol=0.001,atol=0.001)
        # assert_allclose(np.array(v_g.cpu().flatten().tolist()), np.array(v_b.flatten().tolist()),rtol=0.001,atol=0.001)
        # assert_allclose(np.array(residual_g.cpu().flatten().tolist()), np.array(residual_b.flatten().tolist()),rtol=0.001,atol=0.001)

def main():
    test_attention_pre_GLM()

if __name__ == "__main__":
    main()