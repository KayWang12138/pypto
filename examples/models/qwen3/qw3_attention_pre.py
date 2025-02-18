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

    res = res_div * gamma

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
    reduce_asum = pypto.sum(mean_res, -1, True)
    reduce_sum = pypto.add(reduce_asum, eps)

    # sqrt
    reduce_sqrt = pypto.sqrt(reduce_sum)

    # div
    res_div = pypto.div(tensor_value_fp32, reduce_sqrt)

    # gamma mul
    res = pypto.mul(res_div, gamma_fp32)

    # cast
    y_bf16 = pypto.cast(res, input_dtype)

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
def attention_pre(x, weight, q_gamma, k_gamma, cos, sin, q, k, v):
    # 1. 添加支持动态的config
    pypto.set_host_options(only_codegen=True)

    # 3. 得到动态tensor的shape
    bs = x.shape[0]
    hidden_size = x.shape[1]
    total_hidden_size = weight.shape[0]

    head_size = 128
    half_head_size = head_size // 2
    q_size = q.shape[-1]
    kv_size = k.shape[-1]
    q_num_head = q_size // head_size
    kv_num_head = kv_size // head_size
    kv_index = q_num_head + kv_num_head
    eps = 1e-6
    bs_tile = 1
    bs_loop = (bs + bs_tile - 1) // bs_tile

    # 4. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_ATT_PRE_L0", idx_name="bs_idx"):
        def bs_loop_func(bs_idx):
            # 5. 通过view得到x_tile、cos_tile、sin_tile
            x_tile = pypto.view(x, [bs_tile, hidden_size], [bs_idx * bs_tile, 0])
            cos_tile = pypto.view(cos, [bs_tile, 1, half_head_size], [bs_idx * bs_tile, 0, 0])
            sin_tile = pypto.view(sin, [bs_tile, 1, half_head_size], [bs_idx * bs_tile, 0, 0])

            # 6. 按照计算图实现运算逻辑
            # matmul
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
            mm_res = pypto.matmul(x_tile, weight, pypto.DT_BF16, a_trans=False, b_trans=True)
            pypto.set_vec_tile_shapes(128, 128, 128)
            mm_3d = pypto.reshape(mm_res, [bs_tile, total_hidden_size // head_size, head_size])

            # split
            q_tile = pypto.view(mm_3d, [bs_tile, q_num_head, head_size], [0, 0, 0])
            k_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, q_num_head, 0])
            v_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, kv_index, 0])

            # rms norm
            q_norm = rms_norm(q_tile, q_gamma, eps, [bs_tile, q_num_head, head_size])
            k_norm = rms_norm(k_tile, k_gamma, eps, [bs_tile, kv_num_head, head_size])

            # apply rope
            # cast
            pypto.set_vec_tile_shapes(bs_tile, q_num_head, head_size)
            q_fp32 = pypto.cast(q_norm, pypto.DT_FP32)
            k_fp32 = pypto.cast(k_norm, pypto.DT_FP32)

            pypto.set_vec_tile_shapes(bs_tile, kv_num_head, half_head_size)
            cos_fp32 = pypto.cast(cos_tile, pypto.DT_FP32)
            sin_fp32 = pypto.cast(sin_tile, pypto.DT_FP32)

            # q split
            q1 = pypto.view(q_fp32, [bs_tile, q_num_head, half_head_size], [0, 0, 0])
            q2 = pypto.view(q_fp32, [bs_tile, q_num_head, half_head_size], [0, 0, half_head_size])
            # rope data
            q_rope = rope_data(q1, q2, cos_fp32, sin_fp32, [bs_tile, q_num_head, half_head_size])

            # k split
            k1 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_head_size], [0, 0, 0])
            k2 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_head_size], [0, 0, half_head_size])
            # rope data
            k_rope = rope_data(k1, k2, cos_fp32, sin_fp32, [bs_tile, kv_num_head, half_head_size])

            # post process
            q_res = pypto.reshape(q_rope, [bs_tile, q_size])
            k_res = pypto.reshape(k_rope, [bs_tile, kv_size])
            v_res = pypto.reshape(v_tile, [bs_tile, kv_size])

            # 7. 将结果搬运到输出tensor上
            # update output
            q[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = q_res
            k[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = k_res
            v[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = v_res

        bs_loop_func(bs_idx)


def test_attention_pre():
    # 1. 设置参数
    bs = 48
    hidden_size = 2048
    total_hidden_size = 768
    head_size = 128
    q_size = 512
    kv_size = 128
    half_head_size = head_size // 2

    device_id = int(os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    eps = 1e-6

    # 2. 构造多种shape，测试动态case
    for i in range(0, 4):
        if (i == 2):
            bs = 5

        # 3. 准备测试数据
        np.random.seed(0)
        # inputs
        x = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        weight = torch.rand(total_hidden_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        q_gamma = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        k_gamma = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        cos = torch.rand(bs, 1, half_head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        sin = torch.rand(bs, 1, half_head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        # outputs
        q = torch.zeros((bs, q_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
        k = torch.zeros((bs, kv_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
        v = torch.zeros((bs, kv_size), dtype=torch.bfloat16, device=f'npu:{device_id}')

        # 4. 执行kernel并获取结果
        inputs = {
            x: [0],
            weight: [],
            q_gamma: [],
            k_gamma: [],
            cos: [0],
            sin: [0]
        }
        outputs = {
            q: [0],
            k: [0],
            v: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        attention_pre(*pto_inputs, *pto_outputs)
        pypto.runtime._device_synchronize()

        # 5. 与PyTorch参考实现对比
        # matmul
        mm_golden = torch.matmul(x, weight.T)
        # split
        q_g, k_g, v_g = mm_golden.split([q_size, kv_size, kv_size], dim=-1)
        # rms norm
        q_by_head = q_g.view(*q_g.shape[:-1], q_g.shape[-1] // head_size, head_size)
        q_by_head = rms_norm_golden(q_by_head, q_gamma, eps)
        k_by_head = k_g.view(*k_g.shape[:-1], k_g.shape[-1] // head_size, head_size)
        k_by_head = rms_norm_golden(k_by_head, k_gamma, eps)
        # apply rope
        q_r, k_r = apply_rotary_pos_emb_v2(q_by_head, k_by_head, cos, sin)
        # post process
        q_r = q_r.view(bs, q_size)
        k_r = k_r.view(bs, kv_size)

        # compare result
        assert_allclose(np.array(q_r.cpu().flatten().tolist()), np.array(q.flatten().tolist()), rtol=0.001, atol=0.001)
        assert_allclose(np.array(k_r.cpu().flatten().tolist()), np.array(k.flatten().tolist()), rtol=0.001, atol=0.001)
        assert_allclose(np.array(v_g.cpu().flatten().tolist()), np.array(v.flatten().tolist()), rtol=0.001, atol=0.001)


def main():
    test_attention_pre()

if __name__ == "__main__":
    main()
