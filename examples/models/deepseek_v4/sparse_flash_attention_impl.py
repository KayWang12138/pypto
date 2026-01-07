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
Sparse Flash Attention Quantization Module

This module implements sparse flash attention with quantization support for DeepSeek V4.
It performs attention computation on top-k selected key-value pairs from cache,
supporting both standard and flash attention algorithms.

Main Functions:
    - sparse_flash_attention_compute: Standard sparse attention computation
    - sparse_flash_attention_compute_flash: Flash attention variant with online softmax
    - sparse_flash_attention_d: JIT-compiled decode version
    - sparse_flash_attention_p: JIT-compiled prefill version

Example:
    See deepseekv4_sparse_flash_attention.py for usage examples.
"""
from dataclasses import dataclass
import math
import os
import pypto
import numpy as np
from pypto.experimental import gather_in_l1, gather_in_ub


@dataclass
class SaTileShapeConfig:
    g_tile: int
    s_kv_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


def sparse_flash_attention_compute(query_nope, query_rope, key_nope_2d, key_rope_2d,
                                   topk_indices, block_table, kv_act_seqs, atten_sink,
                                   attention_out, nq, n_kv, softmax_scale, topk,
                                   block_size, max_blocknum_perbatch, tile_config):
    """Compute sparse flash attention with quantization support.

    Performs attention computation on top-k selected key-value pairs from cache.
    The function processes queries and keys in batches, computing attention scores
    and aggregating values. Supports both quantized (INT8) and non-quantized keys.

    Args:
        query_nope: Query tensor without RoPE, shape (t * n_q, kv_lora_rank), dtype BF16
        query_rope: Query tensor with RoPE, shape (t * n_q, rope_dim), dtype BF16
        key_nope_2d: Key tensor without RoPE, shape (block_num * block_size, kv_lora_rank),
                     dtype BF16 or INT8
        key_rope_2d: Key tensor with RoPE, shape (block_num * block_size, rope_dim), dtype BF16
        topk_indices: Top-k indices for each query token, shape (t, n_kv * topk), dtype INT32
        block_table: Block mapping table for PagedAttention, shape (b, max_blocknum_perbatch),
                     dtype INT32
        kv_act_seqs: Actual sequence lengths for each batch, shape (b,), dtype INT32
        attention_out: Output attention tensor, shape (b, s, n_q, kv_lora_rank), dtype BF16
        nq: Number of query heads
        n_kv: Number of key-value heads
        softmax_scale: Scaling factor for attention scores, typically 1/sqrt(head_dim)
        topk: Number of top-k keys to attend to
        block_size: Size of each block in PagedAttention
        max_blocknum_perbatch: Maximum number of blocks per batch
        tile_config: SaTileShapeConfig object containing tiling parameters:
            - g_tile: Group tile size
            - s_kv_tile: Key-value sequence tile size
            - c1_tile_shape: Cube tile shape for first matmul
            - v1_tile_shape: Vector tile shape for softmax
            - c2_tile_shape: Cube tile shape for second matmul

    Note:
        The function uses nested loops to process batches, sequences, heads, and groups.
        For quantized keys, it performs dequantization before attention computation.
        The attention computation uses standard softmax normalization.
    """
    dtype = query_nope.dtype
    dn = query_nope.shape[1]
    dr = query_rope.shape[1]
    group = nq // n_kv
    group_tile = tile_config.g_tile
    s2_tile = tile_config.s_kv_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    n_kv_sym = n_kv

    batch_size_sym = kv_act_seqs.shape[0]

    s1_n2_gsym = query_nope.shape[0] // batch_size_sym
    s1_sym = s1_n2_gsym // nq

    g_loop_sym = group // group_tile

    atten_sink_2d = pypto.unsqueeze(atten_sink, -1)

    atten_out_2dim = pypto.tensor(
        [batch_size_sym * s1_n2_gsym, dn], dtype, "attenOut2Dim")
    for batch_idx in pypto.loop(0, batch_size_sym, 1, name="LOOP_L0_idx", idx_name="bIdx"):
        cur_act_seq = kv_act_seqs[batch_idx]
        for slc_idx in pypto.loop(0, s1_sym, 1, name="LOOP_L1_s1_SA", idx_name="s1Idx"):
            cur_seq = (cur_act_seq - s1_sym + 1 + slc_idx).max(0).min(topk)
            cur_seq.as_variable()
            bn_per_batch = (cur_seq + s2_tile - 1) // s2_tile

            for n_kv_idx in pypto.loop(0, n_kv_sym, 1, name="LOOP_L2_n_kv_SA", idx_name="n_kvIdx"):
                for group_idx in pypto.loop(0, g_loop_sym, 1, name="LOOP_L3_g_SA", idx_name="gIdx"):
                    cur_group_tile = group_tile
                    cur_offset = batch_idx * s1_n2_gsym + slc_idx * \
                        nq + n_kv_idx * group + group_idx * cur_group_tile
                    for s2_idx, _ in pypto.loop_unroll(0, bn_per_batch, 1,
                                                       name="LOOP_L4_s2_SA", idx_name="s2_idx", unroll_list={1}):
                        cur_s2_tile = s2_tile

                        cur_topk_indices = pypto.view(topk_indices, [1, cur_s2_tile],
                                                      [batch_idx * s1_sym + slc_idx,
                                                          s2_idx * cur_s2_tile],
                                                      valid_shape=[1, (cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile)])
                        cur_block_table = pypto.view(
                            block_table, [1, max_blocknum_perbatch], [batch_idx, 0])

                        pypto.set_semantic_label("Sa_V0")
                        pypto.set_vec_tile_shapes(64, 512)

                        k_nope_2d_view = pypto.view(key_nope_2d, [cur_s2_tile, dn],
                                                    [0, 0], valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])

                        # ---- gather: GM --> UB  [s2_tile, dn]
                        kn_fp16 = gather_in_ub(
                            k_nope_2d_view, cur_topk_indices, cur_block_table, block_size, -2)
                        pypto.set_vec_tile_shapes(64, 512)
                        kn = pypto.view(kn_fp16, [cur_s2_tile, dn], [0, 0],
                                        valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])

                        key_rope_2d_view = pypto.view(
                            key_rope_2d, [cur_s2_tile, dr],
                            [0, 0], valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dr])

                        # ---- gather: GM --> UB  ----  [s2_tile, dr]
                        kr_fp16 = gather_in_ub(
                            key_rope_2d_view, cur_topk_indices, cur_block_table, block_size, -2)
                        pypto.set_vec_tile_shapes(16, 64)
                        kr = pypto.view(kr_fp16, [cur_s2_tile, dr], [0, 0],
                                        valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dr])

                        # （1）kr和kn分开搬出，（2）kr和kb UB内assemble，再连续内存搬出
                        kj = pypto.tensor([cur_s2_tile, dn + dr], dtype, "kj")
                        pypto.assemble(kn, [0, 0], kj)
                        pypto.assemble(pypto.clone(kr), [0, dn], kj)
                        kj_view = pypto.view(kj, [cur_s2_tile, dn + dr], [0, 0],
                                             valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn + dr])

                        # C1
                        pypto.set_semantic_label("Sa_C1")
                        pypto.set_vec_tile_shapes(32, 512)
                        pypto.set_cube_tile_shapes([c1_tile[0],
                                                    c1_tile[1]], [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])

                        qn = pypto.view(query_nope, [cur_group_tile, dn], [cur_offset, 0],
                                        valid_shape=[cur_group_tile, dn])
                        qr = pypto.view(query_rope, [cur_group_tile, dr], [cur_offset, 0],
                                        valid_shape=[cur_group_tile, dr])
                        qi = pypto.tensor(
                            [cur_group_tile, dn + dr], dtype, "qi")
                        pypto.assemble(qn, [0, 0], qi)
                        pypto.assemble(qr, [0, dn], qi)

                        sij = pypto.matmul(
                            qi, kj_view, pypto.DT_FP32, a_trans=False, b_trans=True)

                        pypto.set_semantic_label("Sa_V1")
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij_reduce = pypto.amax(
                            sij_scale, dim=-1, keepdim=True)
                        t_sub = pypto.sub(sij_scale, tilda_mij_reduce)
                        tilda_pij = pypto.exp(t_sub)
                        tilda_lij_reduce = pypto.sum(
                            tilda_pij, dim=-1, keepdim=True)

                        tilda_lij_reduce = pypto.add(tilda_lij_reduce, atten_sink_2d)

                        t_softmax = pypto.div(tilda_pij, tilda_lij_reduce)
                        tilda_pij_f16 = pypto.cast(t_softmax, dtype)

                        pypto.set_semantic_label("Sa_C2")
                        pypto.set_cube_tile_shapes([c2_tile[0],
                                                    c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]])
                        pypto.set_matrix_size([tilda_pij_f16.shape[0],
                                               tilda_pij_f16.shape[1], kn.shape[1]])

                        vj = pypto.view(kn, [cur_s2_tile, dn], [0, 0],
                                        valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])

                        # [cur_group_tile, dn]
                        q1 = pypto.matmul(tilda_pij_f16, vj, dtype)

                        pypto.assemble(q1, [cur_offset, 0], atten_out_2dim)
                        attention_out[:] = pypto.reshape(atten_out_2dim,
                                                         [attention_out.shape[0], attention_out.shape[1],
                                                          attention_out.shape[2], attention_out.shape[3]], inplace=True)


@pypto.jit(
    pass_options={
        "mg_copyin_upper_bound": 2 * 1024 * 1024,
        "pg_upper_bound": 50000,
        "pg_lower_bound": 512,
        "pg_parallel_lower_bound": 20,
        "vec_nbuffer_mode": 2,
        "vec_nbuffer_setting": {-1: 2, 0: 8},
        "cube_l1_reuse_mode": 2
    },
    runtime_options={
        "stitch_function_inner_memory": 128,
        "stitch_function_outcast_memory": 128,
        "device_sched_mode": 3
    },
    host_options={"only_codegen": True}
)
def sparse_flash_attention_d(query_nope, query_rope, key_nope_2d, key_rope_2d,
                             topk_indices, block_table, kv_act_seqs, atten_sink,
                             attention_out, nq, n_kv, softmax_scale, topk,
                             block_size, max_blocknum_perbatch, tile_config):
    """JIT-compiled sparse flash attention for decode phase.

    Optimized version for decode phase with specific pass configurations.
    Uses flash attention algorithm with online softmax for numerical stability.

    Args:
        query_nope: Query tensor without RoPE, shape (t * n_q, kv_lora_rank), dtype BF16
        query_rope: Query tensor with RoPE, shape (t * n_q, rope_dim), dtype BF16
        key_nope_2d: Key tensor without RoPE, shape (block_num * block_size, kv_lora_rank),
                     dtype BF16 or INT8
        key_rope_2d: Key tensor with RoPE, shape (block_num * block_size, rope_dim), dtype BF16
        topk_indices: Top-k indices for each query token, shape (t, n_kv * topk), dtype INT32
        block_table: Block mapping table for PagedAttention, shape (b, max_blocknum_perbatch),
                     dtype INT32
        kv_act_seqs: Actual sequence lengths for each batch, shape (b,), dtype INT32
        atten_sink: Softmax parameter, shape (n1,), dtype FP32
        attention_out: Output attention tensor, shape (b, s, n_q, kv_lora_rank), dtype BF16
        nq: Number of query heads
        n_kv: Number of key-value heads
        softmax_scale: Scaling factor for attention scores
        topk: Number of top-k keys to attend to
        block_size: Size of each block in PagedAttention
        max_blocknum_perbatch: Maximum number of blocks per batch
        tile_config: SaTileShapeConfig object containing tiling parameters

    Note:
        Configured for decode phase with optimized memory and parallelism settings.
        Uses flash attention algorithm for better numerical stability.
    """
    pypto.set_debug_options(runtime_debug_mode=1)

    pypto.experimental.set_operation_config(combine_axis=True)

    sparse_flash_attention_compute(query_nope, query_rope, key_nope_2d, key_rope_2d,
                                   topk_indices, block_table, kv_act_seqs, atten_sink,
                                   attention_out, nq, n_kv, softmax_scale, topk,
                                   block_size, max_blocknum_perbatch, tile_config)


@pypto.jit(
    pass_options={
        "mg_copyin_upper_bound": 2 * 1024 * 1024,
        "pg_upper_bound": 50000,
        "pg_lower_bound": 512,
        "pg_parallel_lower_bound": 20,
        "vec_nbuffer_mode": 2,
        "vec_nbuffer_setting": {-1: 4, 0: 16},
        "cube_l1_reuse_mode": 4
    },
    runtime_options={
        "stitch_function_inner_memory": 32,
        "stitch_function_outcast_memory": 32,
        "stitch_function_num_initial": 128
    },
    host_options={"only_codegen": True}
)
def sparse_flash_attention_p(query_nope, query_rope, key_nope_2d, key_rope_2d,
                             topk_indices, block_table, kv_act_seqs, atten_sink,
                             attention_out, nq, n_kv, softmax_scale, topk,
                             block_size, max_blocknum_perbatch, tile_config):
    """JIT-compiled sparse flash attention for prefill phase.

    Optimized version for prefill phase with specific pass configurations.
    Uses flash attention algorithm with online softmax for numerical stability.

    Args:
        query_nope: Query tensor without RoPE, shape (t * n_q, kv_lora_rank), dtype BF16
        query_rope: Query tensor with RoPE, shape (t * n_q, rope_dim), dtype BF16
        key_nope_2d: Key tensor without RoPE, shape (block_num * block_size, kv_lora_rank),
                     dtype BF16 or INT8
        key_rope_2d: Key tensor with RoPE, shape (block_num * block_size, rope_dim), dtype BF16
        topk_indices: Top-k indices for each query token, shape (t, n_kv * topk), dtype INT32
        block_table: Block mapping table for PagedAttention, shape (b, max_blocknum_perbatch),
                     dtype INT32
        kv_act_seqs: Actual sequence lengths for each batch, shape (b,), dtype INT32
        atten_sink: Softmax parameter, shape (n1,), dtype FP32
        attention_out: Output attention tensor, shape (b, s, n_q, kv_lora_rank), dtype BF16
        nq: Number of query heads
        n_kv: Number of key-value heads
        softmax_scale: Scaling factor for attention scores
        topk: Number of top-k keys to attend to
        block_size: Size of each block in PagedAttention
        max_blocknum_perbatch: Maximum number of blocks per batch
        tile_config: SaTileShapeConfig object containing tiling parameters

    Note:
        Configured for prefill phase with optimized memory and parallelism settings.
        Uses flash attention algorithm for better numerical stability.
    """
    pypto.experimental.set_operation_config(combine_axis=True)

    sparse_flash_attention_compute(query_nope, query_rope, key_nope_2d, key_rope_2d,
                                   topk_indices, block_table, kv_act_seqs, atten_sink,
                                   attention_out, nq, n_kv, softmax_scale, topk,
                                   block_size, max_blocknum_perbatch, tile_config)
