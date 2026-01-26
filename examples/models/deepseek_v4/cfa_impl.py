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
deepseekv4 Attention Module

This module implements the Attention mechanism for deepseekv4 model, which uses
a paged memory management approach similar to operating systems to efficiently
handle variable-length sequences and dynamic batch sizes in attention computation.

Main Functions:
    - attention: Main attention function with Attention support
    - ifa_flash: JIT compiled kernel implementing Flash Attention with paged KV cache
    - gen_block_table: Generate block mapping table for Attention
    - kv_cache_concat_bsnd: Convert paged KV cache to BSND format
"""

from dataclasses import dataclass
import torch
import pypto
import pytest
import numpy as np
import math
import os
from torch._subclasses.fake_tensor import FakeTensor


def kernel(q, kv, attn_sink, block_table, seqused_kv, kv_win=None, blk_win=None, 
              atten_out = None, cmp_r=1, unroll_list=[], pg_upper_bound=3072, is_prefill = False):
    pypto.set_pass_options(pg_upper_bound=pg_upper_bound)
    enable_c128 = kv_win is not None and blk_win is not None
    shape_q = q.shape
    shape_k = kv.shape
    shape_k_win = kv_win.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_num_win_scalar = shape_k_win[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    softmax_scale = dn ** -0.5
    b_scalar = seqused_kv.shape[0]

    dtype = q.dtype
    n2_sym = nkv

    m_tile = 128
    cube_tile = 128
    k_cube_tile = 256
    s2_tile = 512

    g_tile = min(32, nq)
    c1_tile = [[m_tile, m_tile], [k_cube_tile, k_cube_tile], [cube_tile, cube_tile]]
    v1_tile = [m_tile, s2_tile]
    c2_tile = [[m_tile, m_tile], [cube_tile, cube_tile], [k_cube_tile, k_cube_tile]]
    v2_tile = [m_tile, k_cube_tile]
    v1_win_tile = [m_tile, block_size]
    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    kv_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    kv_win_2d_shape = (block_num_win_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)
    attn_sink_2d_shape = (nq, 1)

    kv_2d = pypto.reshape(kv, kv_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    if enable_c128:
        kv_win_2d = pypto.reshape(kv_win, kv_win_2d_shape, inplace=True)
        win = 128
    attn_sink_2d = pypto.reshape(attn_sink, attn_sink_2d_shape, inplace=True)
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = (seqused_kv[b_idx] - ((s1_scalar - 1) - s1_idx)) // cmp_r
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                if enable_c128:
                    bs_ofs = b_idx * s1_scalar + s1_idx
                    n1g_ofs = g_idx * g_tile
                    valid_len = seqused_kv[b_idx] - (s1_scalar - 1 - s1_idx)
                    actual_s2_tile = pypto.min(valid_len, block_size)
                    
                    valid_win_len = pypto.min(valid_len, win)
                    valid_start_pos = valid_len - valid_win_len
                    valid_end_pos = valid_len - 1
                    start_offset = valid_start_pos % block_size
                    start_block = valid_start_pos // block_size
                    end_block = valid_end_pos // block_size

                    start_block_id = blk_win[b_idx, start_block].max(0)
                    kv_block_0 = pypto.view(kv_win_2d, [block_size, dn], [start_block_id * block_size, 0], valid_shape=[valid_win_len, dn])
                    end_block_id = blk_win[b_idx, end_block].max(0)
                    kv_block_1 = pypto.view(kv_win_2d, [block_size, dn], [end_block_id * block_size, 0], valid_shape=[valid_win_len, dn])

                    pypto.set_vec_tile_shapes(v1_win_tile[0], v1_win_tile[1])
                    kv_gather = pypto.concat([kv_block_0, kv_block_1], dim=0)

                    pypto.set_vec_tile_shapes(v1_win_tile[0], v1_win_tile[1])
                    kv_win_cur = pypto.view(kv_gather, [win, dn], [start_offset, 0], valid_shape=[valid_win_len, dn])
                    
                    
                    # block_idx = blk_win[b_idx, 0]
                    # block_idx_valid = block_idx.max(0)
                    pypto.set_vec_tile_shapes(v1_win_tile[0], v1_win_tile[1])
                    qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                    # kj = pypto.view(kv_win_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                    pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                    sij = pypto.matmul(qi, kv_win_cur, pypto.DT_FP32, a_trans=False, b_trans=True)
                    sij = pypto.view(sij, [g_tile, block_size], [0, 0], valid_shape=[g_tile, actual_s2_tile])
                    pypto.set_vec_tile_shapes(v1_win_tile[0], v1_win_tile[1])
                    sij_scale = pypto.mul(sij, softmax_scale)
                    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                    tsub = pypto.sub(sij_scale, tilda_mij)
                    tilda_pij = pypto.exp(tsub)
                    tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                    sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                    max_update[:] = tilda_mij

                    # c2
                    # vj = pypto.view(kv_win_2d, [block_size, dn], [block_idx_valid * block_size, 0], valid_shape=[actual_s2_tile, dn])
                    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                    oi_tmp = pypto.matmul(tilda_pij_fp16, kv_win_cur, pypto.DT_FP32)
                    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                    oi_update[:] = oi_tmp
                    
                    if pypto.cond(s2_loop == 0):
                        attn_sink_tile = pypto.view(attn_sink_2d, [g_tile, 1], [g_idx * g_tile, 0])
                        attn_sink_tile = pypto.exp(attn_sink_tile - max_update)
                        sum_local = pypto.add(sum_update, attn_sink_tile)
                        oi_final = pypto.div(oi_update, sum_local)
                        pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                        oi_final_3d = pypto.cast(
                            pypto.reshape(oi_final, [1, g_tile, dn]),
                            dtype)

                        pypto.assemble(oi_final_3d, oi_ofs, atten_out)
                        
                for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=unroll_list):
                    block_num = s2_tile // block_size
                    idx = s2_idx * block_num
                    bs_ofs = b_idx * s1_scalar + s1_idx
                    n1g_ofs = g_idx * g_tile
                    actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                    oi_ofs = [bs_ofs, n1g_ofs, 0]
                    # 5. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                    qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])

                    kj_assemble = pypto.tensor([s2_tile, dn], kv_2d.dtype, "kj_assemble")
                    for i in range(block_num):
                        block_idx = block_table[b_idx, idx + i]
                        block_idx_valid = block_idx.max(0)
                        kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                            pypto.view(kv_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                    kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[s2_tile, dn])

                    # c1
                    # 6. 下面是flash attention的计算逻辑
                    pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                    sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False,
                                       b_trans=True)
                    sij = pypto.view(sij, [g_tile, s2_tile], [0, 0],
                                     valid_shape=[g_tile, actual_s2_tile])
                    # v1
                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                    if not enable_c128 and pypto.cond(pypto.is_loop_begin(s2_idx)):
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

                        tsub = pypto.sub(sij_scale, tilda_mij)
                        tilda_pij = pypto.exp(tsub)
                        tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                        sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                        max_update[:] = tilda_mij

                        # c2
                        vj_assemble = pypto.tensor([s2_tile, dn], kv_2d.dtype, "vj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(kv_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                        vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                 [0, 0], valid_shape=[actual_s2_tile, dn])
                        pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                        oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        oi_update[:] = oi_tmp
                    else:
                        # pypto.set_pass_options(sg_set_scope=1)
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        max_new = pypto.maximum(max_update, tilda_mij)
                        tsub = pypto.sub(sij_scale, max_new)
                        tilda_pij = pypto.exp(tsub)
                        tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                        sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                        # pypto.set_pass_options(sg_set_scope=-1)

                        # pypto.set_pass_options(sg_set_scope=2)
                        tsub2 = pypto.sub(max_update, max_new)
                        max_update[:] = max_new
                        update_mul = pypto.exp(tsub2)
                        sum_update[:] = sum_update * update_mul + sum_local
                        # pypto.set_pass_options(sg_set_scope=-1)

                        # c2
                        vj_assemble = pypto.tensor([s2_tile, dn], kv_2d.dtype, "vj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(kv_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                        vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                    [0, 0], valid_shape=[actual_s2_tile, dn])
                        pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                        oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

                        # v2
                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        oi_update[:] = oi_update * update_mul + oi_tmp
                    if pypto.cond(pypto.is_loop_end(s2_idx)):
                        attn_sink_tile = pypto.view(attn_sink_2d, [g_tile, 1], [g_idx * g_tile, 0])
                        attn_sink_tile = pypto.exp(attn_sink_tile - max_update)
                        sum_local = pypto.add(sum_update, attn_sink_tile)
                        oi_final = pypto.div(oi_update, sum_local)

                        pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                        oi_final_3d = pypto.cast(
                            pypto.reshape(oi_final, [1, g_tile, dn]),
                            dtype)
                        # 7. 将结果搬运到输出tensor上
                        pypto.assemble(oi_final_3d, oi_ofs, atten_out)


@pypto.jit(
    runtime_options={"stitch_function_num_initial": 128,
                     "stitch_function_outcast_memory": 2048,
                     "stitch_function_inner_memory": 2048,
                     "device_sched_mode": 1},
    host_options={"only_codegen": True},
    debug_options={"runtime_debug_mode": 1},
    
    # 当子图大小达到上界不允许与其他子图合并
    pass_options={"cube_l1_reuse_setting": {0: 4, 2:4}}
)
def c128_decode(q, kv, attn_sink, block_table, seqused_kv, kv_win=None, blk_win=None, 
              atten_out = None, cmp_r=1, unroll_list=[], pg_upper_bound=3072):
    kernel(q, kv, attn_sink, block_table, seqused_kv, kv_win=kv_win, blk_win=blk_win, 
              atten_out = atten_out, cmp_r=cmp_r, unroll_list=unroll_list, pg_upper_bound=pg_upper_bound, is_prefill=False)


@pypto.jit(
    runtime_options={"stitch_function_num_initial": 128,
                     "stitch_function_outcast_memory": 2048,
                     "stitch_function_inner_memory": 2048,
                     "device_sched_mode": 1},
    host_options={"only_codegen": True},
    debug_options={"runtime_debug_mode": 1},
    
    # 当子图大小达到上界不允许与其他子图合并
    pass_options={"cube_l1_reuse_setting": {0: 4, 2:4}}
)
def c128_prefill(q, kv, attn_sink, block_table, seqused_kv, kv_win=None, blk_win=None, 
              atten_out = None, cmp_r=1, unroll_list=[], pg_upper_bound=3072):
    kernel(q, kv, attn_sink, block_table, seqused_kv, kv_win=kv_win, blk_win=blk_win, 
              atten_out = atten_out, cmp_r=cmp_r, unroll_list=unroll_list, pg_upper_bound=pg_upper_bound, is_prefill=True)