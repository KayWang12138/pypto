#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
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
def add_rms_norm_npu_golden(residual_input, x, x_gamma, x_bias, eps):
    x_bias_fp32 = x_bias.to(torch.float32)
    x_fp32 = x.to(torch.float32)
    residual_input_fp32 = residual_input.to(torch.float32)
    x_fp32 = x_fp32 + residual_input_fp32
    x_mean_coff = 1.0 / x.shape[-1]
    x_square = x_fp32 * x_fp32
    x_mean = x_square * x_mean_coff
    x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + eps
    x_reduce_sqrt = torch.sqrt(x_reduce_sum)
    x_res_div = x_fp32 / x_reduce_sqrt
    x_mul_res = x_res_div * x_gamma.to(torch.float32)
    x_add_bias = x_mul_res + x_bias_fp32

    return x_add_bias.to(torch.bfloat16), x_fp32.to(torch.bfloat16)


def rms_norm_npu_golden(x, x_gamma, x_bias, eps):
    x_bias_fp32 = x_bias.to(torch.float32)
    x_fp32 = x.to(torch.float32)
    x_mean_coff = 1.0 / x.shape[-1]
    x_square = x_fp32 * x_fp32
    x_mean = x_square * x_mean_coff
    x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + eps
    x_reduce_sqrt = torch.sqrt(x_reduce_sum)
    x_res_div = x_fp32 / x_reduce_sqrt
    x_mul_res = x_res_div * x_gamma.to(torch.float32)
    x_add_bias = x_mul_res + x_bias_fp32

    return x_add_bias.to(torch.bfloat16)


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
    target_shape = [1] * len(tensor_value_fp32.shape)
    target_shape[-1] = gamma.shape[0]
    gamma_3d = pypto.reshape(gamma, target_shape)

    target_shape[-1] = bias.shape[0]
    bias_3d = pypto.reshape(bias, target_shape)

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
def glm_quant_attention_pre(in_tensors, out_tensors, enable_residual=True):
    # 1. 添加支持动态的config
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    pypto.set_host_options(only_codegen=True)
    pypto.set_option('profile_enable', True)
    pypto.set_runtime_options(estimated_stitch_task_max_loop_num=256)
    pypto.set_runtime_options(workspace_recycle_period=256)
    # 2. 从入参拿到输入和输出tensor
    x, residual_input, x_gamma, x_bias, x_scale, x_offset, weight, quant_bias, deq_scale, \
    q_gamma, q_bias, k_gamma, k_bias, cos, sin = in_tensors
    q, k, v, residual = out_tensors
    bs_tile = x.shape[0]

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
    hidden_size = x.shape[1]
    total_head_size = weight.shape[1]

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

    bs_loop = (bs + bs_tile - 1) // bs_tile
    calc_dtype = pypto.DT_FP32
    input_dtype = x.dtype
    tiling_value = 128
    vec_tile_value = 5120
    q_batch_tile = 4
    # 5. 定义动态函数

    for _ in pypto.loop(1, name="LOOP_RESHAPE_INPLACE", idx_name="tmp_idx"):
        pypto.set_vec_tile_shapes(5120)
        x_gamma_2d = pypto.reshape(x_gamma, [1, 5120], inplace=True)
        x_bias_2d = pypto.reshape(x_bias, [1, 5120], inplace=True)
        x_scale_2d = pypto.reshape(x_scale, [1, 5120], inplace=True)
        x_offset_2d = pypto.reshape(x_offset, [1, 5120], inplace=True)
        quant_bias_2d = pypto.reshape(quant_bias, [1, 1792], inplace=True)
        deq_scale_2d = pypto.reshape(deq_scale, [1, 1792], inplace=True)

    # 6. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_ATT_PRE_L0", idx_name="bs_idx"):

        x_tile = pypto.view(x, [bs_tile, hidden_size], [bs_idx * bs_tile, 0])
        # init
        pypto.set_vec_tile_shapes(1, vec_tile_value)
        x_tile_fp32 = pypto.cast(x_tile, calc_dtype)
        # add
        if enable_residual:
            residual_input_tile = pypto.view(residual_input, [bs_tile, hidden_size], [bs_idx * bs_tile, 0])
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

        x_int8 = pypto.tensor([bs_tile, 5120], pypto.DT_INT8, "x_int8")
        residual_bf16 = pypto.cast(x_f32, input_dtype)

        for tmp_idx in range(bs_tile):
            pypto.set_vec_tile_shapes(1, vec_tile_value)
            x_gamma_2d_fp32 = pypto.cast(x_gamma_2d, calc_dtype)
            x_bias_2d_fp32 = pypto.cast(x_bias_2d, calc_dtype)
            x_scale_2d_fp32 = pypto.cast(x_scale_2d, calc_dtype)
            x_offset_2d_fp32 = pypto.cast(x_offset_2d, calc_dtype)

            res_div_single = pypto.view(res_div, [1, hidden_size], [tmp_idx, 0])

            res = pypto.mul(res_div_single, x_gamma_2d_fp32) # res = res_div * weight
            res_add = pypto.add(res, x_bias_2d_fp32)
            x_norm = pypto.cast(res_add, input_dtype)

            # x quant
            pypto.set_vec_tile_shapes(1, vec_tile_value)
            x_norm_fp32 = pypto.cast(x_norm, calc_dtype) # bf16 -> fp32
            x_mul = pypto.mul(x_norm_fp32, x_scale_2d_fp32)
            x_add = pypto.add(x_mul, x_offset_2d_fp32)
            x_int32 = pypto.cast(x_add, pypto.DT_INT32, pypto.CastMode.CAST_RINT) # Align ascendC
            x_fp16 = pypto.cast(x_int32, pypto.DT_FP16)
            x_int8[tmp_idx:tmp_idx + 1, 0:] = pypto.cast(x_fp16, pypto.DT_INT8)

        pypto.set_vec_tile_shapes(32, 256)
        tmp_c = pypto.full([bs_tile, total_head_size], 0, pypto.DT_INT32)
        matmul_result = []
        k_split = 10
        k_split_size = hidden_size // k_split # 5120 // 10 = 512
        pypto.set_cube_tile_shapes([32, 32], [256, 256], [256, 256])
        for ki in range(k_split):
            input_mk = pypto.view(x_int8, [bs_tile, k_split_size], [0, ki * k_split_size])
            input_kn = pypto.view(weight, [k_split_size, total_head_size], [ki * k_split_size, 0])
            tmp = pypto.matmul(input_mk, input_kn, pypto.DT_INT32, mat3=tmp_c)
            matmul_result.append(tmp)
        pypto.set_vec_tile_shapes(32, 256)
        tmp_c = pypto.reduce(matmul_result, pypto.ReduceMode.ATOMIC_ADD)
        mm_add = pypto.add(tmp_c, quant_bias_2d)

        pypto.set_vec_tile_shapes(8, 1792)
        mm_fp32 = pypto.cast(mm_add, calc_dtype) # int32 -> fp32
        mm_deq_scale = pypto.mul(mm_fp32, deq_scale_2d)
        mm_bf16 = pypto.cast(mm_deq_scale, input_dtype) # fp32 -> bf16

        pypto.set_vec_tile_shapes(tiling_value, tiling_value, head_size)
        mm_3d = pypto.reshape(mm_bf16, [bs_tile, total_head_size // head_size, head_size], inplace=True)

        # split
        q_tile = pypto.view(mm_3d, [bs_tile, q_num_head, head_size], [0, 0, 0])
        k_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, q_num_head, 0])
        v_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, kv_index, 0])

        # rms norm
        q_norm = rms_norm_bias(q_tile, q_gamma, q_bias, qk_mean_coff, eps, [q_batch_tile, q_num_head, head_size])
        k_norm = rms_norm_bias(k_tile, k_gamma, k_bias, qk_mean_coff, eps, [q_batch_tile, kv_num_head, head_size])

        q_rot = pypto.view(q_norm, [bs_tile, q_num_head, rotary_dim], [0, 0, 0])
        q_pass = pypto.view(q_norm, [bs_tile, q_num_head, stay_dim], [0, 0, rotary_dim])

        k_rot = pypto.view(k_norm, [bs_tile, kv_num_head, rotary_dim], [0, 0, 0])
        k_pass = pypto.view(k_norm, [bs_tile, kv_num_head, stay_dim], [0, 0, rotary_dim])

        # apply rope
        # cast
        pypto.set_vec_tile_shapes(q_batch_tile, q_num_head, head_size)
        cos_tile = pypto.view(cos, [bs_tile, 1, half_rotary_dim], [bs_idx * bs_tile, 0, 0])
        sin_tile = pypto.view(sin, [bs_tile, 1, half_rotary_dim], [bs_idx * bs_tile, 0, 0])
        q_fp32 = pypto.cast(q_rot, calc_dtype)
        k_fp32 = pypto.cast(k_rot, calc_dtype)
        cos_fp32 = pypto.cast(cos_tile, calc_dtype)
        sin_fp32 = pypto.cast(sin_tile, calc_dtype)

        # q split
        q1 = pypto.view(q_fp32, [bs_tile, q_num_head, half_rotary_dim], [0, 0, 0])
        q2 = pypto.view(q_fp32, [bs_tile, q_num_head, half_rotary_dim], [0, 0, half_rotary_dim])

        # rope data
        q_rope = rope_data(q1, q2, cos_fp32, sin_fp32, [q_batch_tile, q_num_head, half_rotary_dim])
        q_cat = pypto.concat([q_rope, q_pass], 2)

        # k split
        k1 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_rotary_dim], [0, 0, 0])
        k2 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_rotary_dim], [0, 0, half_rotary_dim])

        # rope data
        k_rope = rope_data(k1, k2, cos_fp32, sin_fp32, [q_batch_tile, q_num_head, half_rotary_dim])
        k_cat = pypto.concat([k_rope, k_pass], 2)

        # post process
        q_res = pypto.reshape(q_cat, [bs_tile, q_size])
        k_res = pypto.reshape(k_cat, [bs_tile, kv_size])
        v_res = pypto.reshape(v_tile, [bs_tile, kv_size])

        # # 9. 将结果搬运到输出tensor上
        # # update output
        q[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = q_res
        k[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = k_res
        v[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = v_res
        residual[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = residual_bf16


@pytest.mark.skip(reason="Dependent on the integration of the matmul interface update")
def test_glm_attention_pre():
    # 1. 设置参数
    bs = 8
    hidden_size = 5120
    total_head_size = 1792
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
            bs = 16

        # 3. 准备测试数据
        np.random.seed(0)
        # inputs
        x = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        residual_input = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        x_gamma = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        x_bias = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        x_scale = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        x_offset = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        weight = torch.randint(0, 255, size=(hidden_size, total_head_size), dtype=torch.int8, device=f'npu:{device_id}')
        quant_bias = torch.randint(0, 255, size=(total_head_size,), dtype=torch.int32, device=f'npu:{device_id}')
        deq_scale = torch.rand(total_head_size, dtype=torch.float32, device=f'npu:{device_id}')
        q_gamma = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        q_bias = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        k_gamma = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        k_bias = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        cos = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')
        sin = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')

        # outputs
        q = torch.zeros((bs, q_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
        k = torch.zeros((bs, kv_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
        v = torch.zeros((bs, kv_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
        residual = torch.zeros((bs, hidden_size), dtype=torch.bfloat16, device=f'npu:{device_id}')

        enable_residual = True
        if torch.all(residual_input == 0):
            residual_input = torch.zeros((bs, hidden_size), dtype=x.dtype, device=f'{x.device}')
            enable_residual = False

        # # 4. 执行kernel并获取结果
        inputs = [
            pypto.from_torch(x, name="IN"), 
            pypto.from_torch(residual_input, name="IN"), 
            pypto.from_torch(x_gamma, name="IN"), 
            pypto.from_torch(x_bias, name="IN"), 
            pypto.from_torch(x_scale, name="IN"), 
            pypto.from_torch(x_offset, name="IN"), 
            pypto.from_torch(weight, name="IN"), 
            pypto.from_torch(quant_bias, name="IN"), 
            pypto.from_torch(deq_scale, name="IN"), 
            pypto.from_torch(q_gamma, name="IN"), 
            pypto.from_torch(q_bias, name="IN"), 
            pypto.from_torch(k_gamma, name="IN"), 
            pypto.from_torch(k_bias, name="IN"), 
            pypto.from_torch(cos, name="IN"), 
            pypto.from_torch(sin, name="IN")
        ]
        outputs = [
            pypto.from_torch(q, name="OUT"), 
            pypto.from_torch(k, name="OUT"), 
            pypto.from_torch(v, name="OUT"), 
            pypto.from_torch(residual, name="OUT")
        ]
        glm_quant_attention_pre(inputs, outputs, enable_residual)
        pypto.runtime._device_synchronize()

        # 5. 与PyTorch参考实现对比
        # add rms norm
        x_g, residual_g = add_rms_norm_npu_golden(x, residual_input, x_gamma, x_bias, eps)

        # matmul
        x_quant = torch_npu.npu_quantize(x_g, x_scale, x_offset, torch.qint8, -1, False)
        mm_golden = torch_npu.npu_quant_matmul(x_quant, weight, deq_scale, bias=quant_bias, output_dtype=torch.bfloat16)

        # split
        q_g, k_g, v_g = mm_golden.split([q_size, kv_size, kv_size], dim=-1)

        # nms norm
        q_by_head = q_g.view(*q_g.shape[:-1], q_g.shape[-1] // head_size, head_size)
        q_by_head = rms_norm_npu_golden(q_by_head, q_gamma, q_bias, eps)

        k_by_head = k_g.view(*k_g.shape[:-1], k_g.shape[-1] // head_size, head_size)
        k_by_head = rms_norm_npu_golden(k_by_head, k_gamma, k_bias, eps)

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

        assert_allclose(np.array(q_r.cpu().flatten().tolist()), np.array(q.flatten().tolist()),
            rtol=0.0078125, atol=0.0001)
        assert_allclose(np.array(k_r.cpu().flatten().tolist()), np.array(k.flatten().tolist()),
            rtol=0.0078125, atol=0.0001)
        assert_allclose(np.array(v_g.cpu().flatten().tolist()), np.array(v.flatten().tolist()),
            rtol=0.0078125, atol=0.0001)
        assert_allclose(np.array(residual_g.cpu().flatten().tolist()), np.array(residual.flatten().tolist()),
            rtol=0.0078125, atol=0.0001)


def main():
    test_glm_attention_pre()


if __name__ == "__main__":
    main()