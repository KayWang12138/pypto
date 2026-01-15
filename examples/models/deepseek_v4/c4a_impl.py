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
from pypto.experimental import gather_in_ub
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import FakeTensor


@dataclass
class SaTileShapeConfig:
    g_tile: int
    s_kv_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


def c4a_compute(q, ori_kv, cmp_kv, ori_block_table, cmp_block_table, atten_sinks, seqused_kv, cmp_sparse_indices, attention_out,
                                    softmax_scale, cmp_ratio, win_size,
                                   topk, tile_config):
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
        seqused_kv: Actual sequence lengths for each batch, shape (b,), dtype INT32
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
    dtype = q.dtype
    dn = q.shape[3]
    nq = q.shape[2]
    n_kv = 1
    group = nq // n_kv
    group_tile = tile_config.g_tile
    s2_tile = tile_config.s_kv_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    n_kv_sym = n_kv

    batch_size_sym = seqused_kv.shape[0]
    max_blocknum_perbatch = cmp_block_table.shape[1]

    s1_sym = q.shape[1]
    s1_n2_gsym = s1_sym * nq

    g_loop_sym = group // group_tile

    pypto.set_vec_tile_shapes(atten_sinks.shape[0])
    atten_sinks_2d = pypto.reshape(atten_sinks, [atten_sinks.shape[0], 1], inplace=True)
    topk_tile = topk if s2_tile > topk else s2_tile

    cmp_kv_block_num = cmp_kv.shape[0]
    block_size = cmp_kv.shape[1]
    kv_2d_shape = (cmp_kv_block_num * block_size, n_kv * dn)

    cmp_kv_2d = pypto.reshape(cmp_kv, kv_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, [q.shape[0] * s1_sym * nq, dn], inplace=True)

    atten_out_2dim = pypto.tensor([batch_size_sym * s1_n2_gsym, dn], dtype, "attenOut2Dim")
    for batch_idx in pypto.loop(0, batch_size_sym, 1, name="LOOP_L0_idx", idx_name="bIdx"):
        cur_act_seq = seqused_kv[batch_idx]
        valid_data_len = pypto.min(win_size + s1_sym - 1, cur_act_seq)   
        for slc_idx in pypto.loop(0, s1_sym, 1, name="LOOP_L1_s1_SA", idx_name="s1Idx"):
            cur_seq = (cur_act_seq - s1_sym + 1 + slc_idx).max(0).min(topk)
            cur_seq.as_variable()
            valid_end_pos = valid_data_len - s1_sym + slc_idx
            valid_start_pos = valid_end_pos - pypto.min(win_size - 1, valid_end_pos)
            valid_win_len = valid_end_pos - valid_start_pos + 1
            bn_per_batch = (cur_seq + s2_tile - 1) // s2_tile + 1 # 1 for swf

            for n_kv_idx in pypto.loop(0, n_kv_sym, 1, name="LOOP_L2_n_kv_SA", idx_name="n_kvIdx"):
                for group_idx in pypto.loop(0, g_loop_sym, 1, name="LOOP_L3_g_SA", idx_name="gIdx"):
                    cur_group_tile = group_tile
                    cur_offset = batch_idx * s1_n2_gsym + slc_idx * \
                        nq + n_kv_idx * group + group_idx * cur_group_tile
                    oi_update = pypto.tensor([group_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([group_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([group_tile, 1], pypto.DT_FP32, "max_update")
                    for s2_idx, _ in pypto.loop_unroll(0, bn_per_batch, 1,
                                                       name="LOOP_L4_s2_SA", idx_name="s2_idx", unroll_list={1}):
                        if pypto.cond(pypto.is_loop_begin(s2_idx)):
                            pypto.set_vec_tile_shapes(128, 128, 512, 512)
                            q_tensor_cur = pypto.view(q_2d, [cur_group_tile, dn], [cur_offset, 0], 
                                                valid_shape=[cur_group_tile, dn])
                            start_block = valid_start_pos // block_size
                            end_block = valid_end_pos // block_size

                            physical_block_id = ori_block_table[batch_idx, start_block]
                            pypto.set_vec_tile_shapes(128, 256, 128, 256)
                            kv_block_0 = pypto.view(ori_kv, [1, block_size, 1, dn], [physical_block_id, 0, 0, 0])
                            kv_block_reshape_0 = pypto.reshape(kv_block_0, (block_size, dn))

                            physical_block_id = ori_block_table[batch_idx, end_block]
                            pypto.set_vec_tile_shapes(128, 256, 128, 256)
                            kv_block_1 = pypto.view(ori_kv, [1, block_size, 1, dn], [physical_block_id, 0, 0, 0])
                            kv_block_reshape_1 = pypto.reshape(kv_block_1, (block_size, dn))

                            pypto.set_vec_tile_shapes(128, 256)
                            kv_gather = pypto.concat([kv_block_reshape_0, kv_block_reshape_1], dim=0)

                            pypto.set_vec_tile_shapes(128, 256)
                            kv_cur = pypto.view(kv_gather, [win_size, dn], [valid_start_pos, 0], valid_shape=[valid_win_len, dn])

                            # 以下为block循环部分
                            pypto.set_cube_tile_shapes([64, 64], [256, 256], [128, 128], True, False)
                            sij = pypto.matmul(q_tensor_cur, kv_cur, pypto.DT_FP32, b_trans=True)

                            pypto.set_vec_tile_shapes(64, 128)
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, -1, True)
                            t_sub = pypto.sub(sij_scale, tilda_mij)
                            tilda_pij = pypto.exp(t_sub)
                            tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)

                            sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            max_update[:] = tilda_mij

                            #c2
                            mm2_res = pypto.matmul(tilda_pij_fp16, kv_cur, pypto.DT_FP32)

                            oi_update[:] = mm2_res
                        else:
                            cur_s2_tile = s2_tile
                            cmp_s2_idx = s2_idx - 1

                            cur_topk_indices = pypto.view(cmp_sparse_indices, [1, topk_tile],
                                                        [batch_idx * s1_sym + slc_idx, 0],
                                                        valid_shape=[1, (cur_seq - cmp_s2_idx * cur_s2_tile).min(topk_tile)])
                            cur_block_table = pypto.view(
                                cmp_block_table, [1, max_blocknum_perbatch], [batch_idx, 0])

                            pypto.set_semantic_label("Sa_V0")
                            pypto.set_vec_tile_shapes(128, 512)

                            k_nope_2d_view = pypto.view(cmp_kv_2d, [cur_s2_tile, dn],
                                                        [0, 0], valid_shape=[(cur_seq - cmp_s2_idx * cur_s2_tile).min(cur_s2_tile), dn])

                            # ---- gather: GM --> UB  [s2_tile, dn]
                            kn_fp16 = gather_in_ub(
                                k_nope_2d_view, cur_topk_indices, cur_block_table, block_size, -2)
                            pypto.set_vec_tile_shapes(128, 512)
                            kv = pypto.view(kn_fp16, [topk_tile, dn], [0, 0],
                                            valid_shape=[(cur_seq - cmp_s2_idx * cur_s2_tile).min(topk_tile), dn])

                            # C1
                            pypto.set_semantic_label("Sa_C1")
                            pypto.set_cube_tile_shapes([c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]], 
                                                        [c1_tile[4], c1_tile[5]], enable_multi_data_load=True)

                            qi = pypto.view(q_2d, [cur_group_tile, dn], [cur_offset, 0],
                                            valid_shape=[cur_group_tile, dn])

                            sij = pypto.matmul(qi, kv, pypto.DT_FP32, a_trans=False, b_trans=True)

                            pypto.set_semantic_label("Sa_V1")
                            pypto.set_vec_tile_shapes(32, 512)
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                            max_new = pypto.maximum(max_update, tilda_mij)
                            t_sub = pypto.sub(sij_scale, max_new)
                            tilda_pij = pypto.exp(t_sub)
                            tilda_pij_f16 = pypto.cast(tilda_pij, dtype)
                            sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)

                            t_sub2 = pypto.sub(max_update, max_new)
                            update_mul = pypto.exp(t_sub2)
                            sum_update[:] = sum_update * update_mul + sum_local

                           
                            #C2
                            pypto.set_semantic_label("Sa_C2")
                            pypto.set_cube_tile_shapes([c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], 
                                                        [c2_tile[4], c2_tile[5]], enable_multi_data_load=True)
                            pypto.set_matrix_size([tilda_pij_f16.shape[0],
                                                tilda_pij_f16.shape[1], kv.shape[1]])
                            
                            mm2_res = pypto.matmul(tilda_pij_f16, kv, pypto.DT_FP32)

                            pypto.set_vec_tile_shapes(32, 512)
                            oi_update[:] = oi_update * update_mul + mm2_res

                    pypto.set_vec_tile_shapes(64, 512)
                    sub_res = pypto.sub(atten_sinks_2d, max_update)
                    exp_res = pypto.exp(sub_res)
                    sum_total = pypto.add(sum_update, exp_res)
                    atten_out_part = pypto.cast(pypto.div(oi_update, sum_total), dtype)
                    pypto.assemble(atten_out_part, [cur_offset, 0], atten_out_2dim)
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
        "vec_nbuffer_setting": {-1: 2},
        "cube_l1_reuse_mode": 2
    },
    runtime_options={
        "stitch_function_num_initial": 128,
        "stitch_function_inner_memory": 1024,
        "stitch_function_outcast_memory": 1024,
        "device_sched_mode": 3
    },
    host_options={"only_codegen": True}
)
def c4a_d(q, ori_kv, cmp_kv, ori_block_table, cmp_block_table, atten_sinks, seqused_kv, cmp_sparse_indices, attention_out,
                             softmax_scale, cmp_ratio, win_size, tile_config):
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
        seqused_kv: Actual sequence lengths for each batch, shape (b,), dtype INT32
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
    pypto.set_debug_options(runtime_debug_mode=2)

    pypto.experimental.set_operation_config(combine_axis=True)

    topk = 512
    c4a_compute(q, ori_kv, cmp_kv, ori_block_table, cmp_block_table, atten_sinks, seqused_kv, cmp_sparse_indices, attention_out,
                                    softmax_scale, cmp_ratio, win_size,
                                   topk, tile_config)