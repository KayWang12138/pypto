#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
GLM-4.5 Attention Module

This module implements the Attention mechanism for GLM-4.5 model, which uses
a paged memory management approach similar to operating systems to efficiently
handle variable-length sequences and dynamic batch sizes in attention computation.

Main Functions:
    - attention: Main attention function with Attention support
    - ifa_func: JIT compiled kernel implementing Flash Attention with paged KV cache
    - gen_block_table: Generate block mapping table for Attention
    - kv_cache_concat_bsnd: Convert paged KV cache to BSND format
"""
import os
import math
from dataclasses import dataclass
import torch
import torch_npu
import pytest
import numpy as np
from numpy.testing import assert_allclose
from typing import Tuple
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
from utils.get_format import get_format

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


def check_args(
    query,
    key_cache,
    value_cache,
    block_tables,
    actual_seqs,
    attn_res
):
    assert query.dim() == 3
    assert get_format(query) == 'ND'
    assert query.dtype == torch.bfloat16
    assert key_cache.dim() == 4
    assert get_format(key_cache) == 'ND'
    assert key_cache.dtype == torch.bfloat16
    assert value_cache.dim() == 4
    assert get_format(value_cache) == 'ND'
    assert value_cache.dtype == torch.bfloat16
    assert block_tables.dim() == 2
    assert get_format(block_tables) == 'ND'
    assert block_tables.dtype == torch.int32
    assert actual_seqs.dim() == 1
    assert get_format(actual_seqs) == 'ND'
    assert actual_seqs.dtype == torch.int32
    assert attn_res.dim() == 3
    assert get_format(attn_res) == 'ND'
    assert attn_res.dtype == torch.bfloat16


@dataclass
class AttentionTileConfig:
    g_tile: int
    s2_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


@dataclass
class AttentionConfig:
    b: int
    s1: int
    s2: int
    n1: int
    n2: int
    q_d: int
    kv_d: int
    block_size: int = 128
    max_num_blocks_per_query: int = 0
    softmax_scale: float = 1.0
    kv_layout: str = "PA_BSND"
    actual_seq: torch.Tensor = None  # 改为 torch.Tensor 类型
    block_table_batch: int = 0
    kv_num_blocks: int = 0


def get_qwen_common_config(device="cpu"):
    b = 16
    s1 = 1
    s2 = 8*1024
    q_d = 128
    nq = 12
    nkv = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_table_batch = b
    block_size = 128
    kv_num_blocks = b * ((s2 + block_size - 1) // block_size)

    # 创建 torch tensor 类型的 actual_seq
    actual_seq_values = [s2] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    atten_cfg = AttentionConfig(b=b, s1=s1, s2=s2, n1=nq, n2=nkv, softmax_scale=softmax_scale, kv_layout=kv_layout,
                                q_d=q_d, kv_d=q_d, block_size=block_size, block_table_batch=block_table_batch,
                                kv_num_blocks=kv_num_blocks, actual_seq=actual_seq_tensor)  # 传入 tensor
    atten_cfg.max_num_blocks_per_query = (s2 + block_size - 1) // block_size
    cube_tile = 128
    m_tile = 128
    s2_tile = 1024
    tile_cfg = AttentionTileConfig(
        nq,
        s2_tile,
        [[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [m_tile, 1024],
        [[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [m_tile, 512])
    return atten_cfg, tile_cfg


def gen_block_table(actual_seq_len, block_size, block_table_shape):
    block_num_per_batch = []
    block_num = 0

    # 处理 torch tensor 类型的 actual_seq_len
    if isinstance(actual_seq_len, torch.Tensor):
        # 如果 tensor 在 GPU/NPU 上，先移动到 CPU
        if actual_seq_len.device.type != 'cpu':
            actual_seq_len_cpu = actual_seq_len.cpu()
        else:
            actual_seq_len_cpu = actual_seq_len

        # 转换为 numpy 数组进行处理，或者直接使用 torch 操作
        for actual_seq in actual_seq_len_cpu:
            block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
            block_num += math.ceil(actual_seq.item() / block_size)
    else:
        # 保持对 list 的兼容
        for actual_seq in actual_seq_len:
            block_num_per_batch.append(math.ceil(actual_seq / block_size))
            block_num += math.ceil(actual_seq / block_size)

    # 使用 torch 替换 numpy
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]  # 随机排列

    # 创建 block_table 张量
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def kv_cache_concat_bsnd(kr_cache_out, kv_cache_out, k_scale, block_table, atten_config):
    b = atten_config.b
    n2 = atten_config.n2
    kv_lora_rank = atten_config.q_d
    rope_dim = atten_config.kv_d
    block_size = atten_config.block_size
    kv_cache_actual_seq = atten_config.actual_seq
    dtype = kr_cache_out.dtype
    scale_dtype = k_scale.dtype

    # 处理 torch tensor 类型的 kv_cache_actual_seq
    if isinstance(kv_cache_actual_seq, torch.Tensor):
        if kv_cache_actual_seq.device.type != 'cpu':
            kv_cache_actual_seq_cpu = kv_cache_actual_seq.cpu()
        else:
            kv_cache_actual_seq_cpu = kv_cache_actual_seq
        kv_max = (torch.max(kv_cache_actual_seq_cpu).item() + block_size - 1) // block_size * block_size
    else:
        kv_max = (max(kv_cache_actual_seq) + block_size - 1) // block_size * block_size

    # 使用 torch 创建张量，保持在同一设备上
    device = kr_cache_out.device
    k_cache = torch.zeros([b, kv_max, n2, kv_lora_rank], dtype=dtype, device=device)
    k_sclae_cache = torch.zeros([b, kv_max, n2, 1], dtype=scale_dtype, device=device)
    v_cache = torch.zeros([b, kv_max, n2, rope_dim], dtype=kv_cache_out.dtype, device=device)

    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, kv_max, n2, kv_lora_rank], dtype=kv_cache_out.dtype, device=device)
        kv_rope_temp_tensor = torch.zeros([1, kv_max, n2, rope_dim], dtype=dtype, device=device)
        k_scale_temp_tensor = torch.zeros([1, kv_max, n2, 1], dtype=scale_dtype, device=device)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            # 使用 torch 的切片操作
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            kv_nope_temp_tensor[:, start_idx:end_idx, :, :] = kv_cache_out[block_idx:block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, start_idx:end_idx, :, :] = kr_cache_out[block_idx:block_idx + 1, :, :, :]
            k_scale_temp_tensor[:, start_idx:end_idx, :, :] = k_scale[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        v_cache[b_idx:b_idx + 1, :, :, :] = kv_nope_temp_tensor
        k_cache[b_idx:b_idx + 1, :, :, :] = kv_rope_temp_tensor
        k_sclae_cache[b_idx:b_idx + 1, :, :, :] = k_scale_temp_tensor

    return k_cache, v_cache, k_sclae_cache


def get_special_array(m, n):
    q_shape = [m, n]

    # 生成递增的行值
    base = np.arange(1, m + 1)  # 生成 [1, 2, ..., m]

    # 将 base 扩展到二维形状 [m, n]
    q = base[:, np.newaxis]  # 增加一个新维度，形状变为 [m, 1]
    q = np.broadcast_to(q, q_shape)  # 广播到目标形状 [m, n]

    # 转换为 float16 类型
    q = q.astype(np.float16)
    return q


def softmax(x, is_fp16=False):
    # 使用 torch 的 softmax 实现
    if is_fp16:
        original_dtype = x.dtype
        x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    if is_fp16:
        ans = ans.to(original_dtype)
        x_max = x_max.to(original_dtype)
        x_sum = x_sum.to(original_dtype)
    return ans, x_max, x_sum


def symmetric_quantization_per_token_fp8_e4m3(input_tensor) -> Tuple:
    """
    Perform symmetric quantization per token (per row).

    Args:
        input_tensor: Input tensor to quantize

    Returns:
        Tuple of (quantized_f8_e4m3_tensor, dequantization_scale)
    """
    fp8_e4m3_max_value = 448.0
    x_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    x_abs = pypto.abs(x_fp32)
    x_max = pypto.amax(x_abs, -1, True)
    shape_0, shape_1 = x_max.shape[:2]
    x_scale = pypto.div(pypto.full([shape_0, shape_1], fp8_e4m3_max_value, pypto.DT_FP32), x_max)
    x_mul = pypto.mul(x_fp32, x_scale)
    x_fp8_e4m3 = pypto.cast(x_mul, pypto.DT_FP8E4M3)
    x_scale_quant = pypto.div(pypto.full([shape_0, shape_1], 1.0, pypto.DT_FP32), x_scale)
    return x_fp8_e4m3, x_scale_quant


def dequant_dynamic(in_tensor, scale_1, scale_2):
    """
    Perform dynamic dequantization using two scale factors.

    Args:
        in_tensor: Quantized input tensor
        scale_1: First scale factor
        scale_2: Second scale factor

    Returns:
        Dequantized tensor
    """
    in_tensor_fp32 = pypto.cast(in_tensor, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_1_fp32 = pypto.cast(scale_1, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_2_fp32 = pypto.cast(scale_2, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    out_scale_2 = pypto.mul(in_tensor_fp32, scale_2_fp32)
    out = pypto.mul(out_scale_2, scale_1_fp32)
    return out


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
    # 当子图大小达到上界不允许与其他子图合并
    pass_options={
        "pg_upper_bound": 1536,
        # Q常驻，0代表第一组mmad，4代表4次matmul合并
        "cube_l1_reuse_setting": {0: 4},
        "vec_nbuffer_setting":{-1:4},
        "cube_nbuffer_setting":{-1:4}
    },
    verify_options={
        "enable_pass_verify":False,
        "pass_verify_save_tensor":False,
    },
    debug_options={"runtime_debug_mode": 1}
)
def ifa_func_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    q_scale: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    k_scale: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    v_scale: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    block_table: pypto.Tensor([], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    atten_out: pypto.Tensor([], pypto.DT_BF16)
):

    # 1. 添加支持动态的config
    pypto.experimental.set_operation_options(combine_axis=True)

    atten_cfg, tile_cfg = get_qwen_common_config()
    softmax_scale = atten_cfg.softmax_scale

    # 2. 从入参拿到输入和输出tensor
    shape_q = q.shape
    shape_k = k.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    b_scalar = kv_act_seqs.shape[0]

    # dtype = q.dtype
    dtype = pypto.DT_BF16
    group = nq // nkv
    n2_sym = nkv

    g_tile = tile_cfg.g_tile
    s2_tile = tile_cfg.s2_tile
    c1_tile = tile_cfg.c1_tile_shape
    v1_tile = tile_cfg.v1_tile_shape
    c2_tile = tile_cfg.c2_tile_shape
    v2_tile = tile_cfg.v2_tile_shape

    # 3. 得到动态tensor的shape
    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)
    q_scale_2d_shape = (b_scalar * 1 * nq, 1)
    k_scale_2d_shape = (block_num_scalar * block_size, n2_sym * 1)
    v_scale_2d_shape = (b_scalar * 1, n2_sym * dn)

    k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
    k_scale_2d = pypto.reshape(k_scale, k_scale_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
    v_scale_2d = pypto.reshape(v_scale, v_scale_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    q_scale_2d = pypto.reshape(q_scale, q_scale_2d_shape, inplace=True)

    # 4. 实现kernel逻辑，循环展开B动态轴
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 4, 2, 1]):
                        block_num = s2_tile // block_size
                        idx = s2_idx * block_num
                        bs_ofs = b_idx * s1_scalar + s1_idx
                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                        # 5. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                        qi_scale = pypto.view(q_scale_2d, [g_tile, 1], [bs_ofs * nq + n1g_ofs, 0])

                        kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                        kj_sclae_assemble = pypto.tensor([s2_tile, 1], k_scale_2d.dtype, "kj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(k_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                            kj_sclae_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(k_scale_2d, [block_size, 1], [block_idx_valid * block_size, 0])
                        kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[s2_tile, dn])
                        kj_sclae_assemble = pypto.view(kj_sclae_assemble, [s2_tile, 1], [0, 0], valid_shape=[s2_tile, 1])

                        # c1
                        # 6. 下面是flash attention的计算逻辑
                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij_quant = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False,
                                            b_trans=True)
                        # dequant
                        kj_sclae_assemble_t = pypto.transpose(kj_sclae_assemble, 0, 1)
                        # print(f"dequant_dynamic line_415 = {sij_quant.dtype}, {qi_scale.dtype}, {kj_sclae_assemble_t.dtype}")
                        sij_fp32 = dequant_dynamic(sij_quant, qi_scale, kj_sclae_assemble_t)
                        sij = pypto.view(sij_fp32, [g_tile, s2_tile], [0, 0],
                                            valid_shape=[g_tile, actual_s2_tile])
                        # v1
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        if pypto.is_loop_begin(s2_idx):
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

                            tsub = pypto.sub(sij_scale, tilda_mij)
                            tilda_pij = pypto.exp(tsub)
                            # tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                            sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            max_update[:] = tilda_mij

                            #  quant
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            tilda_pij_fp8_e4m3, tilda_pij_scale = symmetric_quantization_per_token_fp8_e4m3(tilda_pij)
                            # c2
                            vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")

                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            vj_scale_assemble = pypto.tensor([1, dn], v_scale_2d.dtype, "vj_assemble")
                            vj_scale_assemble = pypto.view(v_scale_2d, [1, dn], [b_idx, 0])

                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                    pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                            vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                        [0, 0], valid_shape=[actual_s2_tile, dn])

                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp_quant = pypto.matmul(tilda_pij_fp8_e4m3, vj_assemble, pypto.DT_FP32)
                            # dequant
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            oi_tmp = dequant_dynamic(oi_tmp_quant, tilda_pij_scale, vj_scale_assemble)
                            oi_update[:] = oi_tmp
                        else:
                            pypto.set_pass_options(sg_set_scope=1)
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                            max_new = pypto.maximum(max_update, tilda_mij)
                            tsub = pypto.sub(sij_scale, max_new)
                            tilda_pij = pypto.exp(tsub)
                            sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            pypto.set_pass_options(sg_set_scope=-1)

                            pypto.set_pass_options(sg_set_scope=2)
                            tsub2 = pypto.sub(max_update, max_new)
                            max_update[:] = max_new
                            update_mul = pypto.exp(tsub2)
                            sum_update[:] = sum_update * update_mul + sum_local
                            pypto.set_pass_options(sg_set_scope=-1)

                            # c2
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                    pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                            vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                        [0, 0], valid_shape=[actual_s2_tile, dn])
                            vj_scale_assemble = pypto.tensor([1, dn], v_scale_2d.dtype, "vj_assemble")
                            vj_scale_assemble = pypto.view(v_scale_2d, [1, dn], [b_idx, 0])
                            tilda_pij_fp8_e4m3, tilda_pij_scale = symmetric_quantization_per_token_fp8_e4m3(tilda_pij)

                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp_quant = pypto.matmul(tilda_pij_fp8_e4m3, vj_assemble, pypto.DT_FP32)
                            # dequant
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            oi_tmp = dequant_dynamic(oi_tmp_quant, tilda_pij_scale, vj_scale_assemble)
                            # v2
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update[:] = oi_update * update_mul + oi_tmp
                        if pypto.is_loop_end(s2_idx):
                            oi_final = pypto.div(oi_update, sum_update)
                            pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                            oi_final_3d = pypto.cast(
                                pypto.reshape(oi_final, [1, g_tile, dn]),
                                dtype)
                            # 7. 将结果搬运到输出tensor上
                            pypto.assemble(oi_final_3d, oi_ofs, atten_out)


def quant_fp8e4m3_per_token(x: torch.Tensor):
    # perblock
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale_quant = 448.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_fp8e4m3 = y_fp32.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    # (b, s, n, d) fp8e4m3, (b, s, n, 1) fp32
    return y_fp8e4m3, scale_dequant


def quant_fp8e4m3_per_token_key(x: torch.Tensor):
    # perblock
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale_quant = 448.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_fp8e4m3 = y_fp32.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    # (b, s, n, d) fp8e4m3, (b, s, n, 1) fp32
    return y_fp8e4m3, scale_dequant


def quant_fp8e4m3_per_channel_value(x: torch.Tensor):
    # perblock
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=1, keepdim=True)
    scale_quant = 448.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_fp8e4m3 = y_fp32.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    # (b, s, n, d) fp8e4m3, (b, 1, n, d) fp32
    return y_fp8e4m3, scale_dequant


def fp8_bsnd_to_pa_format(tensor_bsnd, block_table, actual_seq, block_size):
    """
    转换为最终 PA 格式：shape [total_blocks, 128, n, d]
    每个全局 block 存储 128 个 token 位置，有效 token 填充，无效位置补 0
    """
    # 1. 输入合法性校验
    # assert tensor_bsnd.device == torch.device("cpu"), "仅支持 CPU 张量"
    # assert block_table.device == torch.device("cpu"), "block_table 需为 CPU 张量"
    # assert actual_seq.device == torch.device("cpu"), "actual_seq 需为 CPU 张量"

    b, s, n, d = tensor_bsnd.shape
    num_blocks_per_batch = s // block_size
    total_blocks = num_blocks_per_batch * b
    # print(f"total_blcok = {total_blocks}")
    # assert block_table.shape == (b, num_blocks_per_batch), f"block_table 形状应为 {b, num_blocks_per_batch}"
    # assert actual_seq.shape == (b,), f"actual_seq 形状应为 {b,}"
    # assert total_blocks == b * num_blocks_per_batch, "total_blocks 必须等于 b*s//128"

    # 2. 初始化 PA 张量（核心：shape [total_blocks, 128, n, d]，全 0 填充）
    pa_tensor = torch.zeros((total_blocks, block_size, n, d), dtype=tensor_bsnd.dtype, device="cpu")

    # 3. 逐 Batch + 逐逻辑 Block 填充
    for batch_idx in range(b):
        curr_actual_seq = actual_seq[batch_idx].item()
        if curr_actual_seq <= 0:
            # warnings.warn(f"Batch {batch_idx} 有效序列长度 ≤ 0，跳过填充")
            continue
        # 截断有效序列长度到总长度以内
        curr_actual_seq = min(curr_actual_seq, s)

        # 当前 batch 的原始数据和 Block Table
        curr_tokens = tensor_bsnd[batch_idx]  # [s, n, d]
        curr_global_block_ids = block_table[batch_idx]  # [num_blocks_per_batch]：每个逻辑 block 对应的全局 ID

        # 按 128 切分逻辑 block，逐个处理
        for logical_block_idx in range(num_blocks_per_batch):
            # 步骤 1：获取当前逻辑 block 对应的全局 PA block ID（唯一）
            global_pa_block_id = curr_global_block_ids[logical_block_idx].item()
            # 校验全局 ID 合法性
            if global_pa_block_id < 0 or global_pa_block_id >= total_blocks:
                raise ValueError(f"全局 Block ID {global_pa_block_id} 超出范围 [0, {total_blocks-1}]")

            # 步骤 2：计算当前逻辑 block 的 token 范围
            token_start = logical_block_idx * block_size  # 逻辑 block 起始 token
            token_end = min((logical_block_idx + 1) * block_size, curr_actual_seq)  # 结束 token（不超过有效长度）
            # 无有效 token，跳过
            if token_start >= token_end:
                continue

            # 步骤 3：填充当前逻辑 block 的有效 token 到全局 PA block 中
            # token_offset：token 在 block 内的偏移（0~127）
            for token_in_block_offset in range(token_end - token_start):
                src_token_idx = token_start + token_in_block_offset  # 原始张量的 token 索引
                # 填充：PA[全局block_id, 块内偏移, :, :] = 原始token数据
                pa_tensor[global_pa_block_id, token_in_block_offset] = curr_tokens[src_token_idx]

            # 调试日志（可选）
            # print(f"Batch {batch_idx} 逻辑 block {logical_block_idx} → 全局 PA block {global_pa_block_id}")
            # print(f"  填充 token 范围：[{token_start}, {token_end}) → 块内偏移 [{0}, {token_end-token_start})")

    return pa_tensor


@allow_in_graph
def attention(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key_cache: torch.Tensor,
    key_cache_scale: torch.Tensor,
    value_cache: torch.Tensor,
    value_cache_sclae: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seqs: torch.Tensor,
    attn_res: torch.Tensor
) -> None:
    """
    Main attention function with Attention support.

    This function implements scaled dot-product attention using Attention
    mechanism, which efficiently handles variable-length sequences and dynamic
    batch sizes by managing KV cache in non-contiguous blocks.

    Args:
        query: Query tensor with shape [num_tokens, num_head, head_size]
        key_cache: Key cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        value_cache: Value cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        block_tables: Block mapping table with shape [batch_size, max_num_blocks_per_query]
        actual_seqs: Actual sequence lengths with shape [batch_size]
        attn_res: Output attention tensor with shape [num_tokens, num_head, head_size]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    inputs = [query, query_scale, key_cache, key_cache_scale, value_cache, value_cache_sclae, block_tables, actual_seqs, attn_res]
    ifa_func_kernel(*inputs)


def IFA(atten_cfg):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 1)
    torch_dtype = torch.bfloat16
    torch.npu.set_device(int(device_id))
    b = atten_cfg.b
    s1 = atten_cfg.s1
    d = atten_cfg.q_d
    nq = atten_cfg.n1
    nkv = atten_cfg.n2

    block_size = atten_cfg.block_size
    max_num_blocks_per_query = atten_cfg.max_num_blocks_per_query

    # 获取 torch tensor 类型的 actual_seq
    kv_cache_actual_seq = atten_cfg.actual_seq

    q_shape = [b * s1, nq, d]
    kv_shape = [atten_cfg.kv_num_blocks, block_size, nkv, d]
    block_table_shape = [atten_cfg.block_table_batch, max_num_blocks_per_query]

    # 使用 torch 生成数据
    device = f'npu:{device_id}'
    q = torch.empty(q_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    q_fp8_e4m3, q_scale = quant_fp8e4m3_per_token(q)
    k = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    k_fp8_e4m3, k_scale = quant_fp8e4m3_per_token_key(k)
    v = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    # v_fp8_e4m3, _ = quant_fp8e4m3_per_channel_value(v)
    attention_output = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    # 2. 生成block table - 传入 torch tensor
    block_table = gen_block_table(kv_cache_actual_seq, block_size, block_table_shape)

    # 3. 根据block table 将pa格式的数据转换成
    k_cache_bsnd, v_cache_bsnd, k_sclae_bsnd = kv_cache_concat_bsnd(k_fp8_e4m3, v, k_scale, block_table, atten_cfg)
    v_fp8_e4m3_bsnd, v_scale= quant_fp8e4m3_per_channel_value(v_cache_bsnd)
    # print(f"v_fp8_e4m3_bsnd shape = {v_fp8_e4m3_bsnd.shape}")
    # print(f"block_table shape = {block_table.shape}")
    # print(f"kv_cache_actual_seq shape = {kv_cache_actual_seq.shape}")
    v_fp8_e4m3 = fp8_bsnd_to_pa_format(v_fp8_e4m3_bsnd.cpu(), block_table.cpu(), kv_cache_actual_seq.cpu(), atten_cfg.block_size)
    v_scale = v_scale.reshape(b * 1, nkv, d)
    k_cache_bsnd_cpu = k_cache_bsnd.cpu()
    v_cache_bsnd_cpu = v_fp8_e4m3_bsnd.cpu()
    q_fp8_e4m3 = q_fp8_e4m3.cpu()

    for i in range(b):
        for j in range(s1):
            for n2_idx in range(nkv):
                # 从 torch tensor 获取值
                kv_seq_len = kv_cache_actual_seq[i].item()  # 使用 .item() 获取标量值
                seq_len = kv_seq_len - s1 + 1 + j
                q_bs = q_fp8_e4m3[i * s1 + j]
                q_scale_b = q_scale[i]
                k_bs = k_cache_bsnd_cpu[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                k_value_bs = k_sclae_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, 1)
                v_bs = v_cache_bsnd_cpu[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                v_bs_scale = v_scale[i, n2_idx:n2_idx + 1].reshape(1, d)
                # k_bs = k_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                # v_bs = v_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                # MM1: 矩阵乘法
                qk_bmm_res = torch.matmul(q_bs.to(torch.float32), k_bs.to(torch.float32).transpose(1, 0))  # 1,nq, d  -> n_q,d @ d, s2_actual_len
                qk_bmm_res_npu = qk_bmm_res.npu()
                q_scale_b_npu = q_scale_b.npu()
                k_value_bs_npu = k_value_bs.npu()
                qk_bmm_res_npu = qk_bmm_res_npu * q_scale_b_npu
                qk_bmm_res_npu = qk_bmm_res_npu * k_value_bs_npu.transpose(1, 0)
                qk_ele_res = qk_bmm_res_npu * atten_cfg.softmax_scale
                # Softmax计算
                softmax_res, _, _ = softmax(qk_ele_res, False)
                softmax_res_fp8_e4m3, softmax_res_scale = quant_fp8e4m3_per_token(softmax_res)
                softmax_res_cpu = softmax_res_fp8_e4m3.cpu()
                # MM2: 矩阵乘法
                bmm2_res = torch.matmul(softmax_res_cpu.to(torch.float32), v_bs.to(torch.float32))
                bmm2_res_npu = bmm2_res.npu()
                bmm2_res_npu = bmm2_res_npu * softmax_res_scale
                bmm2_res_npu = bmm2_res_npu * v_bs_scale
                bmm2_res_npu = bmm2_res_npu.to(torch_dtype)
                bmm2_res_cpu = bmm2_res_npu.cpu()
                # 存储结果
                attention_output[i * s1 + j] = bmm2_res_cpu

    # 4. 准备测试数据 - 直接使用 torch 张量
    block_table_torch = block_table.to(dtype=torch.int32, device=device)
    act_seq_torch = kv_cache_actual_seq.to(dtype=torch.int32, device=device)  # 直接使用已有的 tensor
    q_fp8_e4m3 = q_fp8_e4m3.to(device=device)
    q_scale = q_scale.to(device=device)
    k_fp8_e4m3 = k_fp8_e4m3.to(device=device)
    k_scale = k_scale.to(device=device)
    v_fp8_e4m3 = v_fp8_e4m3.to(device=device)
    v_scale = v_scale.to(device=device)
    out_torch = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    inputs = [
        q_fp8_e4m3,
        q_scale,
        k_fp8_e4m3,
        k_scale,
        v_fp8_e4m3,
        v_scale,
        block_table_torch,
        act_seq_torch,
        out_torch
    ]
    # 5. 执行kernel并获取结果
    # verify_check
    # attention_output = attention_output.cpu()
    # pypto.set_verify_golden_data(goldens=[None, None, None, None, None, None, None, None, attention_output])
    attention(*inputs)

    # 6. 与PyTorch参考实现对比
    assert_allclose(np.array(attention_output.cpu().flatten().tolist()),
                    np.array(out_torch.cpu().flatten().tolist()),
                    rtol=0.0078125, atol=0.001)


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_ifa():
    # 1. 设置参数
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 1)
    device = f'npu:{device_id}'
    atten_cfg, _ = get_qwen_common_config(device=device)

    # 检查 B 的大小和 actual_seq 长度是否相等
    assert atten_cfg.b == len(
        atten_cfg.actual_seq), f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'

    # 检查所有值是否都小于 s2
    if atten_cfg.actual_seq.device.type != 'cpu':
        actual_seq_cpu = atten_cfg.actual_seq.cpu()
    else:
        actual_seq_cpu = atten_cfg.actual_seq

    assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
    IFA(atten_cfg)


if __name__ == "__main__":
    test_ifa()