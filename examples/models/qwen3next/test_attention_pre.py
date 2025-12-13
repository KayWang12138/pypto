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
def rms_norm_golden(x, gamma, eps):
    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]
    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff
    
    reduce_sum = torch.sum(mean_res, dim=-1, keepdim=True) + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt
    
    gamma_f32 = gamma.to(torch.float32)
    res = res_div * (gamma_f32 + 1)
    
    if x_dtype != torch.float32:
        res = res.to(x_dtype)
    return res


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
def rms_norm(tensor_value, gamma, eps, tile_shape):
    input_dtype = tensor_value.dtype
    # cast
    pypto.set_vec_tile_shapes(*tile_shape)
    tensor_value_fp32 = pypto.cast(tensor_value, pypto.DT_FP32)

    # gamma reshape
    pypto.set_vec_tile_shapes(tile_shape[-1])
    gamma_shape = [1] * len(tensor_value_fp32.shape)
    gamma_shape[-1] = gamma.shape[0]
    gamma_3d = pypto.reshape(gamma, gamma_shape)
    # gamma cast
    pypto.set_vec_tile_shapes(*tile_shape)
    gamma_fp32 = pypto.cast(gamma_3d, pypto.DT_FP32)

    # square
    square = pypto.mul(tensor_value_fp32, tensor_value_fp32)
    # mean_res
    mean_coff = 1.0 / tensor_value_fp32.shape[-1]
    mean_res = pypto.mul(square, mean_coff)
    # reduce sum
    reduce_asum = pypto.sum(mean_res, dim=-1, keepdim=True)
    reduce_sum = pypto.add(reduce_asum, eps)
    # sqrt
    reduce_sqrt = pypto.sqrt(reduce_sum)
    # div
    res_div = pypto.div(tensor_value_fp32, reduce_sqrt)
    # gamma mul
    res = pypto.mul(res_div, pypto.add(gamma_fp32, 1.0))

    # cast
    y_bf16 = pypto.cast(res, input_dtype)

    return y_bf16


def rope_data(x1, x2, cos, sin, tile_shape):
    pypto.set_vec_tile_shapes(*tile_shape)
    # mul
    o1 = pypto.sub(pypto.mul(x1, cos), pypto.mul(x2, sin))
    o2 = pypto.add(pypto.mul(x2, cos), pypto.mul(x1, sin))
    # concat
    res = pypto.concat([o1, o2], 2)

    # cast
    y_bf16 = pypto.cast(res, pypto.DT_BF16)
    return y_bf16


# 1. 添加支持动态的config
@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_unaligned": True}
)
def attention_pre_pto_inner(in_tensors, out_tensors):
    # 2. 从入参拿到输入和输出tensor
    x = in_tensors[0]
    weight_qkvz = in_tensors[1]
    q_gamma = in_tensors[2]
    k_gamma = in_tensors[3]
    cos = in_tensors[4]
    sin = in_tensors[5]

    q = out_tensors[0]
    k = out_tensors[1]
    v = out_tensors[2]
    z = out_tensors[3]

    # 3. 得到动态tensor的shape
    bs = x.shape[0]
    hidden_size = x.shape[1]
    head_dim = q_gamma.shape[0]
    q_size = q.shape[-1]
    kv_size = k.shape[-1]
    q_num_head = q_size // head_dim
    kv_num_head = kv_size // head_dim
    half_rotary_dim = cos.shape[-1]
    rotary_dim = half_rotary_dim * 2
    bs_tile = 1
    bs_loop = (bs + bs_tile - 1) // bs_tile
    eps = 1e-6

    # 4. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_ATT_PRE_L0", idx_name="bs_idx"):
        # 5. 通过view得到x_tile、cos_tile、sin_tile
        pypto.set_vec_tile_shapes(128, 128, 128)
        x_tile = pypto.view(x, [bs_tile, hidden_size], [bs_idx * bs_tile, 0])
        cos_tile = pypto.view(cos, [bs_tile, 1, half_rotary_dim], [bs_idx * bs_tile, 0, 0])
        sin_tile = pypto.view(sin, [bs_tile, 1, half_rotary_dim], [bs_idx * bs_tile, 0, 0])
        
        # 6. 按照计算图实现运算逻辑
        # Linear projection
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        mm_res = pypto.matmul(x_tile, weight_qkvz, pypto.DT_BF16, a_trans=False, b_trans=True)  

        # Split into Q, K, V, Z
        qz_tile = pypto.view(mm_res, [bs_tile, 2 * q_size], [0, 0])      
        k_tile = pypto.view(mm_res, [bs_tile, kv_size], [0, 2 * q_size])
        v_tile = pypto.view(mm_res, [bs_tile, kv_size], [0, 2 * q_size + kv_size])   

        qz_temp = pypto.reshape(qz_tile, [bs_tile, q_num_head, 2 * head_dim])
        q_temp = pypto.view(qz_temp, [bs_tile, q_num_head, head_dim], [0, 0, 0])      
        z_temp = pypto.view(qz_temp, [bs_tile, q_num_head, head_dim], [0, 0, head_dim])      
        q_tile = pypto.reshape(q_temp, [bs_tile, q_size])      
        z_tile = pypto.reshape(z_temp, [bs_tile, q_size])      
        
        # Reshape to 3D
        pypto.set_vec_tile_shapes(bs_tile, q_num_head, head_dim)
        q_3d = pypto.reshape(q_tile, [bs_tile, q_num_head, head_dim])
        k_3d = pypto.reshape(k_tile, [bs_tile, kv_num_head, head_dim])
        v_3d = pypto.reshape(v_tile, [bs_tile, kv_num_head, head_dim])

        # RMS normalization for Q and K
        q_norm = rms_norm(q_3d, q_gamma, eps, [bs_tile, q_num_head, head_dim])
        k_norm = rms_norm(k_3d, k_gamma, eps, [bs_tile, kv_num_head, head_dim])

        # Apply Rotary Position Embedding (RoPE)
        # cast
        pypto.set_vec_tile_shapes(bs_tile, q_num_head, head_dim)
        q_fp32 = pypto.cast(q_norm, pypto.DT_FP32)
        k_fp32 = pypto.cast(k_norm, pypto.DT_FP32)

        pypto.set_vec_tile_shapes(bs_tile, kv_num_head, half_rotary_dim)
        cos_fp32 = pypto.cast(cos_tile, pypto.DT_FP32)
        sin_fp32 = pypto.cast(sin_tile, pypto.DT_FP32)

        # q split
        q_rot = pypto.view(q_fp32, [bs_tile, q_num_head, rotary_dim], [0, 0, 0])
        q_pass = pypto.view(q_fp32, [bs_tile, q_num_head, head_dim - rotary_dim], [0, 0, rotary_dim])
        q_pass1 = pypto.cast(q_pass, pypto.DT_BF16)
        # rope data
        q1 = pypto.view(q_rot, [bs_tile, q_num_head, half_rotary_dim], [0, 0, 0])
        q2 = pypto.view(q_rot, [bs_tile, q_num_head, half_rotary_dim], [0, 0, half_rotary_dim])
        q_rope = rope_data(q1, q2, cos_fp32, sin_fp32, [bs_tile, q_num_head, half_rotary_dim])
        q_res1 = pypto.concat([q_rope, q_pass1], -1)

        # k split
        k_rot = pypto.view(k_fp32, [bs_tile, kv_num_head, rotary_dim], [0, 0, 0])
        k_pass = pypto.view(k_fp32, [bs_tile, kv_num_head, head_dim - rotary_dim], [0, 0, rotary_dim])
        k_pass1 = pypto.cast(k_pass, pypto.DT_BF16)
        # rope data
        k1 = pypto.view(k_rot, [bs_tile, kv_num_head, half_rotary_dim], [0, 0, 0])
        k2 = pypto.view(k_rot, [bs_tile, kv_num_head, half_rotary_dim], [0, 0, half_rotary_dim])
        k_rope = rope_data(k1, k2, cos_fp32, sin_fp32, [bs_tile, kv_num_head, half_rotary_dim])
        k_res1 = pypto.concat([k_rope, k_pass1], -1)

        #Sigmoid
        z_sig = pypto.sigmoid(z_tile)

        # Reshape outputs
        q_res = pypto.reshape(q_res1, [bs_tile, q_size])
        k_res = pypto.reshape(k_res1, [bs_tile, kv_size])
        v_res = pypto.reshape(v_3d, [bs_tile, kv_size])

        # 7. 将结果搬运到输出tensor上
        # update output
        q[bs_idx * bs_tile:, 0:] = q_res
        k[bs_idx * bs_tile:, 0:] = k_res
        v[bs_idx * bs_tile:, 0:] = v_res
        z[bs_idx * bs_tile:, 0:] = z_sig


# 封装pypto实现代码
def attention_pre_pto(**kwargs):
    x = kwargs.get("x")
    weight_qkvz = kwargs.get("weight_qkvz")
    q_gamma = kwargs.get("q_gamma")
    k_gamma = kwargs.get("k_gamma")
    cos_sin_cache = kwargs.get("cos_sin_cache")
    positions = kwargs.get("positions")
    num_kv_heads = kwargs.get("num_kv_heads")
    num_heads = kwargs.get("num_heads")

    dtype = x.dtype
    device = x.device

    # 从入参中获取维度
    bs = x.shape[0]
    head_dim = q_gamma.shape[0]

    # 从传进的cos_sin_cache获取旋转位置编码所需的cos，sin
    cos_sin = cos_sin_cache.index_select(0, positions)
    half = cos_sin.size(-1) // 2
    cos = cos_sin[:, :half]
    sin = cos_sin[:, half:]
    cos = cos.view(-1, 1, half).contiguous()
    sin = sin.view(-1, 1, half).contiguous()

    # outputs
    q = torch.zeros((bs, num_heads * head_dim), dtype=dtype, device=device)
    k = torch.zeros((bs, num_kv_heads * head_dim), dtype=dtype, device=device)
    v = torch.zeros((bs, num_kv_heads * head_dim), dtype=dtype, device=device)
    gate = torch.zeros((bs, num_heads * head_dim), dtype=dtype, device=device)

    inputs = {
        x: [0],
        weight_qkvz: [],
        q_gamma: [],
        k_gamma: [],
        cos: [0],
        sin: [0]
    }
    outputs = {
        q: [0],
        k: [0],
        v: [0],
        gate: [0]
    }

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    attention_pre_pto_inner(pto_inputs, pto_outputs)

    return q, k, v, gate


# torch实现的对比代码
def attention_pre(**kwargs):
    x = kwargs.get("x")
    weight_qkvz = kwargs.get("weight_qkvz")
    q_gamma = kwargs.get("q_gamma")
    k_gamma = kwargs.get("k_gamma")
    cos_sin_cache = kwargs.get("cos_sin_cache")
    positions = kwargs.get("positions")
    num_kv_heads = kwargs.get("num_kv_heads")
    num_heads = kwargs.get("num_heads")

    # 从传进的cos_sin_cache获取旋转位置编码所需的cos，sin
    cos_sin = cos_sin_cache.index_select(0, positions)
    half = cos_sin.size(-1) // 2
    cos = cos_sin[:, :half]
    sin = cos_sin[:, half:]
    cos = cos.view(-1, 1, half).contiguous()
    sin = sin.view(-1, 1, half).contiguous()

    # 从参数中获取维度
    bs = x.shape[0]
    head_dim = q_gamma.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    half_rotary_dim = cos.shape[-1]
    rotary_dim = half_rotary_dim * 2
    eps = 1e-6

    # linear
    mm_golden = torch.matmul(x, weight_qkvz.T)
    # split
    q_gate, k_g, v_g = mm_golden.split([q_size * 2, kv_size, kv_size], dim=-1)
    orig_shape = q_gate.shape[:-1]
    q_gate = q_gate.view(*orig_shape, num_heads, -1)
    q_g, z_g = torch.chunk(q_gate, 2, dim=-1)
    q_g = q_g.reshape(*orig_shape, -1)
    z_g = z_g.reshape(*orig_shape, -1)

    # reshape
    q_by_head = q_g.view(*q_g.shape[:-1], q_g.shape[-1] // head_dim, head_dim)
    k_by_head = k_g.view(*k_g.shape[:-1], k_g.shape[-1] // head_dim, head_dim)
    # nms norm   
    q_by_head = rms_norm_golden(q_by_head, q_gamma, eps)
    k_by_head = rms_norm_golden(k_by_head, k_gamma, eps)

    # apply rope
    q_rot = q_by_head[:, :, :rotary_dim]
    q_pass = q_by_head[:, :, rotary_dim:]
    k_rot = k_by_head[:, :, :rotary_dim]
    k_pass = k_by_head[:, :, rotary_dim:]
    q_rope, k_rope = apply_rotary_pos_emb_v2(q_rot, k_rot, cos, sin)
    q_r = torch.cat((q_rope, q_pass), dim=2)
    k_r = torch.cat((k_rope, k_pass), dim=2)

    #sigmoid
    z_s = torch.sigmoid(z_g)

    #reshape 
    q_r = q_r.view(bs, q_size)
    k_r = k_r.view(bs, kv_size)

    return q_r, k_r, v_g, z_s


def test_attention_pre():
    device_id = int(os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # 设置参数
    bs = 64
    hidden_size = 2048
    head_dim = 256
    q_size = 1024
    kv_size = 256
    total_hidden_size = 2 * q_size + 2 * kv_size
    max_embbending_size = 262144
    rotary_dim = 64
    loc_shape = [bs, ]

    
    # 准备测试数据
    np.random.seed(0)
    # inputs
    x = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    weight_qkvz = torch.rand(total_hidden_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    q_gamma = torch.rand(head_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')
    k_gamma = torch.rand(head_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')
    cos_sin_cache = torch.rand(max_embbending_size, rotary_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')
    positions = torch.randint(0, max_embbending_size, loc_shape, dtype=torch.int64).to(device=f'npu:{device_id}')

    num_kv_heads = 1
    num_heads = 4

    q, k, v, gate = attention_pre_pto(
                        x=x, 
                        weight_qkvz=weight_qkvz, 
                        q_gamma=q_gamma, 
                        k_gamma=k_gamma, 
                        cos_sin_cache=cos_sin_cache, 
                        positions=positions, 
                        num_kv_heads=num_kv_heads, 
                        num_heads=num_heads
                        )
    q_r, k_r, v_r, gate_r = attention_pre(
                                x=x.clone(), 
                                weight_qkvz=weight_qkvz.clone(), 
                                q_gamma=q_gamma.clone(), 
                                k_gamma=k_gamma.clone(), 
                                cos_sin_cache=cos_sin_cache.clone(), 
                                positions=positions.clone(), 
                                num_kv_heads=num_kv_heads, 
                                num_heads=num_heads
                                )
    # 与PyTorch参考实现对比
    assert_allclose(np.array(q_r.cpu().flatten().tolist()), np.array(q.cpu().flatten().tolist()), 
                    rtol=0.001, atol=0.001)
    assert_allclose(np.array(k_r.cpu().flatten().tolist()), np.array(k.cpu().flatten().tolist()), 
                    rtol=0.001, atol=0.001)
    assert_allclose(np.array(v_r.cpu().flatten().tolist()), np.array(v.cpu().flatten().tolist()), 
                    rtol=0.001, atol=0.001)
    assert_allclose(np.array(gate_r.cpu().flatten().tolist()), np.array(gate.cpu().flatten().tolist()),
                    rtol=0.001, atol=0.001)


if __name__ == "__main__":
    test_attention_pre()