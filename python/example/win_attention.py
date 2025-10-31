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
from dataclasses import dataclass
from typing import List
import logging
import math
import pto

NUM_2 = 2
NUM_5 = 5
NUM_9 = 9
NUM_16 = 16
NUM_64 = 64
NUM_128 = 128
NUM_256 = 256
NUM_512 = 512
NUM_1024 = 1024
KEY_ONLY_CODEGEN = "ONLY_CODEGEN"


@dataclass
class WinAttenTileShapeConfig:
    g_tile: int
    skv_tile: int
    v_nope_tile_shape: List[int]
    v_rope_tile_shape: List[int]
    c1_tile_shape: List[int]
    v1_tile_shape: List[int]
    c2_tile_shape: List[int]
    v2_tile_shape: List[int]
    out_tile_shape: List[int]


def win_attention_compute(**kwargs):
    q_nope = kwargs.get("q_nope")
    v_nope_cache = kwargs.get("v_nope_cache")
    q_rope = kwargs.get("q_rope")
    k_rope_cache = kwargs.get("k_rope_cache")
    n_q = kwargs.get("n_q")
    n_kv = kwargs.get("n_kv")
    block_table = kwargs.get("block_table")
    act_seqs = kwargs.get("act_seqs")
    window_size = kwargs.get("window_size")
    block_size = kwargs.get("block_size")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")

    dtype = q_nope.dtype

    d_nope_size = q_nope.shape[1]
    d_rope_size = q_rope.shape[1]
    g_group = n_q // n_kv
    g_tile = tile_config.g_tile

    nope_tile = tile_config.v_nope_tile_shape
    rope_tile = tile_config.v_rope_tile_shape
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    b_size = block_table.shape[0]
    b_tile = 1
    b_loop = b_size // b_tile
    s1_size = q_nope.shape[0] // b_size // n_q
    s1_tile = 1
    s1_loop = s1_size // s1_tile
    n2_tile = 1
    n2_loop = n_kv // n2_tile
    g_loop = g_group // g_tile

    block_start_index = 0
    block_start_offset = 0
    block_end_index = 0
    win_actual_size = 0
    table_loop = 0

    for b_idx in pto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="b_idx",
        unroll_list=set(), submit_before_loop=True):
        def inside_b_loop(b_idx):
            cur_actual_seq_size = pto.get_input_data(act_seqs, [b_idx])
            for s1_idx in pto.loop(0, s1_loop, 1, name="LOOP_L1_s1Idx", idx_name="s1_idx"):
                def inside_s1_loop(s1_idx):
                    nonlocal block_start_index, block_start_offset, block_end_index, win_actual_size, table_loop
                    win_actual_size = (cur_actual_seq_size - s1_size + s1_idx + 1).min(window_size)
                    block_end_index = (cur_actual_seq_size + block_size - 1) // block_size - 1
                    block_start_index = ((cur_actual_seq_size - win_actual_size - s1_size + 1 + s1_idx) //
                                        block_size).max(0)
                    block_start_offset = (cur_actual_seq_size - win_actual_size - s1_size + 1 + s1_idx) % block_size
                    table_loop = block_end_index - block_start_index + 1
                    for n2_idx in pto.loop(0, n2_loop, 1, name="LOOP_L2_n2Idx", idx_name="n2_idx"):
                        for g_idx in pto.loop(0, g_loop, 1, name="LOOP_L2_gIdx", idx_name="g_idx"):
                            def inside_g_loop(b_idx, s1_idx, n2_idx, g_idx):
                                oi_offset = [b_idx, s1_idx, n2_idx * g_group + g_idx * g_tile, 0]
                                cur_offset = b_idx * s1_size * n_q + s1_idx * n_q + n2_idx * g_group + g_idx * g_tile

                                k_part = pto.tensor([NUM_9 * block_size, (d_nope_size + d_rope_size)], dtype, "k_part")
                                for t_idx in range(NUM_9):
                                    cur_idx = block_start_index + t_idx
                                    cur_block_idx = pto.get_input_data(block_table, [b_idx, cur_idx])

                                    pto.set_vec_tile_shapes(nope_tile[0], nope_tile[1])
                                    k_nope = pto.view(v_nope_cache, [block_size, d_nope_size],
                                                    [cur_block_idx * block_size, n2_idx * d_nope_size])
                                    tmp_k1 = pto.cast(k_nope, pto.DT_FP32)
                                    tmp_k2 = pto.cast(tmp_k1, dtype)
                                    pto.assemble(tmp_k2, [t_idx * block_size, 0], k_part)

                                    pto.set_vec_tile_shapes(rope_tile[0], rope_tile[1])
                                    k_rope = pto.view(k_rope_cache, [block_size, d_rope_size],
                                                    [cur_block_idx * block_size, n2_idx * d_rope_size])
                                    tmp_kr1 = pto.cast(k_rope, pto.DT_FP32)
                                    tmp_kr2 = pto.cast(tmp_kr1, dtype)
                                    pto.assemble(tmp_kr2, [t_idx * block_size, d_nope_size], k_part)

                                start_offset = block_start_offset
                                k_actual_part = pto.view(k_part, [window_size, d_nope_size + d_rope_size], 
                                                    [start_offset, 0], 
                                                    valid_shape=[win_actual_size, d_nope_size + d_rope_size])
                                v_actual_part = pto.view(k_part, [window_size, d_nope_size], 
                                                    [start_offset, 0], 
                                                    valid_shape=[win_actual_size, d_nope_size])

                                # query
                                q_part = pto.tensor([g_tile, d_nope_size + d_rope_size], dtype, "q_part")
                                q_nope_l = pto.view(q_nope, [g_tile, d_nope_size], 
                                                    [cur_offset, 0], valid_shape=[g_tile, d_nope_size])
                                pto.assemble(q_nope_l, [0, 0], q_part)
                                q_rope_r = pto.view(q_rope, [g_tile, d_rope_size], 
                                                    [cur_offset, 0], valid_shape=[g_tile, d_rope_size])
                                pto.assemble(q_rope_r, [0, d_nope_size], q_part)

                                # matmul_1
                                pto.set_cube_tile_shapes(
                                    [c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]], True)
                                q_kt = pto.matmul(q_part, k_actual_part, pto.DT_FP32, a_trans=False, b_trans=True)
                                pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                q_kt_scale = pto.mul_s(q_kt, pto.element(q_kt.dtype, softmax_scale))

                                # softmax
                                tile_max = pto.row_max_single(q_kt_scale)
                                tile_sub = pto.sub(q_kt_scale, tile_max)
                                tile_exp = pto.exp(tile_sub)
                                tile_exp_f16 = pto.cast(tile_exp, dtype)
                                tile_sum = pto.row_sum_single(tile_exp)

                                # matmul_2
                                pto.set_cube_tile_shapes(
                                    [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]], True)
                                oi_tmp = pto.matmul(tile_exp_f16, v_actual_part, pto.DT_FP32)
                                pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])

                                # reshape and copyOut
                                out = pto.div(oi_tmp, tile_sum)
                                pto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                out_final = pto.add_s(pto.reshape(out, [b_tile, s1_tile, g_tile, d_nope_size]),
                                                pto.element(out.dtype, float(0)))
                                pto.assemble(out_final, oi_offset, attention_out)
                            inside_g_loop(b_idx, s1_idx, n2_idx, g_idx)
                inside_s1_loop(s1_idx)
        inside_b_loop(b_idx)


def win_attention_compute_flash(**kwargs):
    q_nope = kwargs.get("q_nope")
    v_nope_cache = kwargs.get("v_nope_cache")
    q_rope = kwargs.get("q_rope")
    k_rope_cache = kwargs.get("k_rope_cache")
    n_q = kwargs.get("n_q")
    n_kv = kwargs.get("n_kv")
    block_table = kwargs.get("block_table")
    act_seqs = kwargs.get("act_seqs")
    window_size = kwargs.get("window_size")
    block_size = kwargs.get("block_size")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")

    dtype = q_nope.dtype

    d_nope_size = q_nope.shape[1]
    d_rope_size = q_rope.shape[1]
    g_group = n_q // n_kv
    g_tile = tile_config.g_tile
    s2_tile = tile_config.skv_tile

    nope_tile = tile_config.v_nope_tile_shape
    rope_tile = tile_config.v_rope_tile_shape
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    b_size = block_table.shape[0]
    b_tile = 1
    b_loop = b_size // b_tile
    s1_size = q_nope.shape[0] // b_size // n_q
    s1_tile = 1
    s1_loop = s1_size // s1_tile
    n2_tile = 1
    n2_loop = n_kv // n2_tile
    g_loop = g_group // g_tile

    block_start_index = 0
    block_start_offset = 0
    block_end_index = 0
    win_actual_size = 0
    table_loop = 0

    for b_idx in pto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="b_idx",
        unroll_list=set(), submit_before_loop=True):
        def inside_b_loop(b_idx):
            cur_actual_seq_size = pto.get_input_data(act_seqs, [b_idx])
            k_part = pto.tensor([b_size * s1_size * NUM_9 * block_size, (d_nope_size + d_rope_size)], dtype, "k_part")
            for s1_idx in pto.loop(0, s1_loop, 1, name="LOOP_L1_s1Idx", idx_name="s1_idx"):
                def inside_s1_loop(s1_idx):
                    nonlocal block_start_index, block_start_offset, block_end_index, win_actual_size, table_loop
                    win_actual_size = (cur_actual_seq_size - s1_size + s1_idx + 1).min(window_size)
                    block_end_index = (cur_actual_seq_size + block_size - 1) // block_size - 1
                    block_start_index = ((cur_actual_seq_size - win_actual_size - s1_size + 1 + s1_idx) //
                                        block_size).max(0)
                    block_start_offset = (cur_actual_seq_size - win_actual_size - s1_size + 1 + s1_idx) % block_size
                    table_loop = block_end_index - block_start_index + 1
                    for n2_idx in pto.loop(0, n2_loop, 1, name="LOOP_L2_n2Idx", idx_name="n2_idx"):
                        for g_idx in pto.loop(0, g_loop, 1, name="LOOP_L2_gIdx", idx_name="g_idx"):
                            def inside_g_loop(b_idx, s1_idx, n2_idx, g_idx):
                                oi_offset = [b_idx, s1_idx, n2_idx * g_group + g_idx * g_tile, 0]
                                cur_offset = b_idx * s1_size * n_q + s1_idx * n_q + n2_idx * g_group + g_idx * g_tile
                                s2_loop = (win_actual_size + s2_tile - 1) // s2_tile

                                oi_update = pto.tensor([g_tile, d_nope_size], pto.DT_FP32, "oi_update")
                                li_update = pto.tensor([g_tile, 1], pto.DT_FP32, "li_update")
                                mi_update = pto.tensor([g_tile, 1], pto.DT_FP32, "mi_update")

                                kv_tensor_idx = (b_idx * s1_size + s1_idx) * NUM_9 * block_size
                                for t_idx in pto.loop(0, table_loop, 1, name="LOOP_L2_tIdx", idx_name="t_idx"):
                                    def inside_t_loop(t_idx):
                                        cur_idx = block_start_index + t_idx
                                        cur_block_idx = pto.get_input_data(block_table, [b_idx, cur_idx])

                                        pto.set_vec_tile_shapes(nope_tile[0], nope_tile[1])
                                        k_nope = pto.view(v_nope_cache, [block_size, d_nope_size],
                                                        [cur_block_idx * block_size, n2_idx * d_nope_size])
                                        tmp_k1 = pto.cast(k_nope, pto.DT_FP32)
                                        tmp_k2 = pto.cast(tmp_k1, dtype)
                                        pto.assemble(tmp_k2, [kv_tensor_idx + t_idx * block_size, 0], k_part)

                                        pto.set_vec_tile_shapes(rope_tile[0], rope_tile[1])
                                        k_rope = pto.view(k_rope_cache, [block_size, d_rope_size],
                                                        [cur_block_idx * block_size, n2_idx * d_rope_size])
                                        tmp_kr1 = pto.cast(k_rope, pto.DT_FP32)
                                        tmp_kr2 = pto.cast(tmp_kr1, dtype)
                                        pto.assemble(tmp_kr2, [kv_tensor_idx + t_idx * block_size, d_nope_size], k_part)
                                    inside_t_loop(t_idx)

                                for s2_idx in pto.loop(0, s2_loop, 1, name="LOOP_L2_s2Idx", idx_name="s2_idx"):
                                    def inside_s2_loop(s2_idx):
                                        start_offset = block_start_offset + s2_idx * s2_tile + kv_tensor_idx
                                        k_actual_part = pto.view(k_part, [s2_tile, d_nope_size + d_rope_size], 
                                                            [start_offset, 0], valid_shape=[(win_actual_size - s2_idx * s2_tile).min(s2_tile), 
                                                            d_nope_size + d_rope_size])
                                        v_actual_part = pto.view(k_part, [s2_tile, d_nope_size], 
                                                            [start_offset, 0], valid_shape=[(win_actual_size - s2_idx * s2_tile).min(s2_tile), 
                                                            d_nope_size])

                                        # query
                                        q_part = pto.tensor([g_tile, d_nope_size + d_rope_size], dtype, "q_part")
                                        q_nope_l = pto.view(q_nope, [g_tile, d_nope_size], 
                                                            [cur_offset, 0], valid_shape=[g_tile, d_nope_size])
                                        pto.assemble(q_nope_l, [0, 0], q_part)
                                        q_rope_r = pto.view(q_rope, [g_tile, d_rope_size], 
                                                            [cur_offset, 0], valid_shape=[g_tile, d_rope_size])
                                        pto.assemble(q_rope_r, [0, d_nope_size], q_part)

                                        # matmul_1
                                        pto.set_cube_tile_shapes([c1_tile[0], c1_tile[1]],
                                            [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]], True)
                                        q_kt = pto.matmul(q_part, k_actual_part, pto.DT_FP32, a_trans=False,
                                                        b_trans=True)
                                        pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                        q_kt_scale = pto.mul_s(q_kt, pto.element(q_kt.dtype, softmax_scale))

                                        # softmax
                                        tile_max = pto.row_max_single(q_kt_scale)
                                        tile_sub = pto.sub(q_kt_scale, tile_max)
                                        tile_exp = pto.exp(tile_sub)
                                        tile_exp_f16 = pto.cast(tile_exp, dtype)
                                        tile_sum = pto.row_sum_single(tile_exp)

                                        if pto.cond(pto.is_loop_begin(s2_idx, 0)):
                                            def inside_if_loop_begin():
                                                nonlocal oi_update, li_update, mi_update
                                                # matmul_2
                                                pto.set_cube_tile_shapes([c2_tile[0], c2_tile[1]],
                                                    [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]], True)
                                                oi_tmp = pto.matmul(tile_exp_f16, v_actual_part, pto.DT_FP32)
                                                pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                if pto.cond(pto.is_loop_end(s2_idx, s2_loop)):
                                                    def inside_if_loop_end():
                                                        nonlocal oi_update
                                                        # reshape and copyOut
                                                        oi_update[:] = pto.div(oi_tmp, tile_sum)
                                                        pto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                                        out_final = pto.add_s(pto.reshape(oi_update,
                                                            [b_tile, s1_tile, g_tile, d_nope_size]),
                                                            pto.element(oi_update.dtype, float(0)))
                                                        pto.assemble(out_final, oi_offset, attention_out)
                                                    inside_if_loop_end()
                                                else:
                                                    def inside_else_loop_end():
                                                        nonlocal oi_update
                                                        oi_update[:] = oi_tmp
                                                    inside_else_loop_end()
                                                li_update[:] = tile_sum
                                                mi_update[:] = tile_max
                                            inside_if_loop_begin()
                                        else:
                                            def inside_else_loop_begin():
                                                nonlocal oi_update, li_update, mi_update
                                                oi = oi_update
                                                li = li_update
                                                mi = mi_update

                                                mi_new = pto.maximum(mi, tile_max)
                                                t1 = pto.sub(mi, mi_new)
                                                t2 = pto.exp(t1)
                                                t3 = pto.sub(tile_max, mi_new)
                                                t4 = pto.exp(t3)
                                                t5 = pto.mul(t4, tile_sum)
                                                t6 = pto.mul(t2, li)
                                                li_new = pto.add(t6, t5)

                                                q3 = pto.mul(oi, t2)
                                                pto.set_cube_tile_shapes([c2_tile[0], c2_tile[1]],
                                                    [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]], True)
                                                q1 = pto.matmul(tile_exp_f16, v_actual_part, pto.DT_FP32)
                                                pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                q2 = pto.mul(q1, t4)
                                                oi_tmp = pto.add(q3, q2)
                                                if pto.cond(pto.is_loop_end(s2_idx, s2_loop)):
                                                    def inside_if_loop_end():
                                                        nonlocal oi_update
                                                        oi_update[:] = pto.div(oi_tmp, li_new)
                                                        pto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                                        out_final = pto.add_s(pto.reshape(oi_update,
                                                            [b_tile, s1_tile, g_tile, d_nope_size]),
                                                            pto.element(oi_update.dtype, float(0)))
                                                        pto.assemble(out_final, oi_offset, attention_out)
                                                    inside_if_loop_end()
                                                else:
                                                    def inside_else_loop_end():
                                                        nonlocal oi_update
                                                        oi_update[:] = oi_tmp
                                                    inside_else_loop_end()
                                                li_update[:] = li_new
                                                mi_update[:] = mi_new
                                            inside_else_loop_begin()
                                    inside_s2_loop(s2_idx)
                            inside_g_loop(b_idx, s1_idx, n2_idx, g_idx)
                inside_s1_loop(s1_idx)
        inside_b_loop(b_idx)


def win_attention_debug_compute(**kwargs):
    q_nope = kwargs.get("q_nope")
    v_nope_cache = kwargs.get("v_nope_cache")
    q_rope = kwargs.get("q_rope")
    k_rope_cache = kwargs.get("k_rope_cache")
    n_q = kwargs.get("n_q")
    n_kv = kwargs.get("n_kv")
    block_table = kwargs.get("block_table")
    act_seqs = kwargs.get("act_seqs")
    window_size = kwargs.get("window_size")
    block_size = kwargs.get("block_size")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")

    dtype = q_nope.dtype

    d_nope_size = q_nope.shape[1]
    d_rope_size = q_rope.shape[1]
    g_group = n_q // n_kv
    g_tile = tile_config.g_tile

    nope_tile = tile_config.v_nope_tile_shape
    rope_tile = tile_config.v_rope_tile_shape
    out_tile = tile_config.out_tile_shape
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    b_size = block_table.shape[0]
    b_tile = 1
    b_loop = b_size // b_tile
    s1_size = q_nope.shape[0] // b_size // n_q
    s1_tile = 1
    s1_loop = s1_size // s1_tile
    n2_tile = 1
    n2_loop = n_kv // n2_tile
    g_loop = g_group // g_tile

    block_start_index = 0
    block_start_offset = 0
    block_end_index = 0
    win_actual_size = 0
    table_loop = 0

    for b_idx in pto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="b_idx",
        unroll_list=set(), submit_before_loop=True):
        def inside_b_loop(b_idx):
            cur_actual_seq_size = pto.get_input_data(act_seqs, [b_idx])
            for s1_idx in pto.loop(0, s1_loop, 1, name="LOOP_L1_s1Idx", idx_name="s1_idx"):
                def inside_s1_loop(s1_idx):
                    nonlocal block_start_index, block_start_offset, block_end_index, win_actual_size, table_loop
                    win_actual_size = (cur_actual_seq_size - s1_size + s1_idx + 1).min(window_size)
                    block_end_index = (cur_actual_seq_size + block_size - 1) // block_size - 1
                    block_start_index = ((cur_actual_seq_size - win_actual_size - s1_size + 1 + s1_idx) //
                                        block_size).max(0)
                    block_start_offset = (cur_actual_seq_size - win_actual_size - s1_size + 1 + s1_idx) % block_size
                    table_loop = block_end_index - block_start_index + 1
                    for n2_idx in pto.loop(0, n2_loop, 1, name="LOOP_L2_n2Idx", idx_name="n2_idx"):
                        for g_idx in pto.loop(0, g_loop, 1, name="LOOP_L2_gIdx", idx_name="g_idx"):
                            def inside_g_loop(b_idx, s1_idx, n2_idx, g_idx):
                                out_offset = [b_idx, s1_idx, n2_idx * g_group + g_idx * g_tile, 0]
                                k_part = pto.tensor([NUM_5 * block_size, (d_nope_size + d_rope_size)], dtype, "k_part")
                                v_part = pto.tensor([NUM_5 * block_size, d_nope_size], dtype, "v_part")
                                for t_idx in pto.loop(0, table_loop, 1, name="LOOP_L2_tIdx", idx_name="t_idx",
                                    unroll_list=set(), submit_before_loop=True):
                                    def inside_t_loop(t_idx):
                                        cur_idx = block_start_index + t_idx
                                        cur_block_idx = pto.get_input_data(block_table, [b_idx, cur_idx])

                                        k_nope = pto.view(v_nope_cache, [block_size, d_nope_size],
                                                        [cur_block_idx * block_size, n2_idx * d_nope_size])
                                        pto.set_vec_tile_shapes(nope_tile[0], nope_tile[1])
                                        tmp_k1 = pto.cast(k_nope, pto.DT_FP32)
                                        tmp_k2 = pto.cast(tmp_k1, dtype)
                                        pto.assemble(tmp_k2, [t_idx * block_size, 0], k_part)

                                        k_rope = pto.view(k_rope_cache, [block_size, d_rope_size],
                                                        [cur_block_idx * block_size, n2_idx * d_rope_size])
                                        pto.set_vec_tile_shapes(rope_tile[0], rope_tile[1])
                                        tmp_kr1 = pto.cast(k_rope, pto.DT_FP32)
                                        tmp_kr2 = pto.cast(tmp_kr1, dtype)
                                        pto.assemble(tmp_kr2, [t_idx * block_size, d_nope_size], k_part)

                                        v_nope = pto.view(v_nope_cache, [block_size, d_nope_size],
                                                        [cur_block_idx * block_size, n2_idx * d_nope_size])
                                        pto.set_vec_tile_shapes(nope_tile[0], nope_tile[1])
                                        tmp_v1 = pto.cast(v_nope, pto.DT_FP32)
                                        tmp_v2 = pto.cast(tmp_v1, dtype)
                                        pto.assemble(tmp_v2, [t_idx * block_size, 0], v_part)
                                    inside_t_loop(t_idx)

                                for o_idx in pto.loop(0, 1, 1, name="LOOP_L2_Idx", idx_name="o_idx",
                                    unroll_list=set(), submit_before_loop=True):
                                    def inside_o_loop(o_idx):
                                        cur_offset = (b_idx * s1_size * n_q + s1_idx * n_q +
                                            n2_idx * g_group + g_idx * g_tile)
                                        k_actual_part = pto.view(k_part, [window_size, d_nope_size + d_rope_size], 
                                            [block_start_offset, 0], 
                                            valid_shape=[win_actual_size, d_nope_size + d_rope_size])
                                        v_actual_part = pto.view(v_part, [window_size, d_nope_size], 
                                            [block_start_offset, 0], 
                                            valid_shape=[win_actual_size, d_nope_size])
                                        q_part = pto.tensor([g_tile, d_nope_size + d_rope_size], dtype, "q_part")
                                        # query
                                        q_nope_l = pto.view(q_nope, [g_tile, d_nope_size], 
                                                            [cur_offset, 0], valid_shape=[g_tile, d_nope_size])
                                        pto.assemble(q_nope_l, [0, 0], q_part)
                                        q_rope_r = pto.view(q_rope, [g_tile, d_nope_size], 
                                                            [cur_offset, 0], valid_shape=[g_tile, d_rope_size])
                                        pto.assemble(q_rope_r, [0, d_nope_size], q_part)

                                        # matmul_1
                                        pto.set_cube_tile_shapes([c1_tile[0], c1_tile[1]],
                                            [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]], True)
                                        q_kt = pto.matmul(q_part, k_actual_part, pto.DT_FP32, a_trans=False,
                                                    b_trans=True)
                                        pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                        q_kt_scale = pto.mul_s(q_kt, pto.element(q_kt.dtype, softmax_scale))

                                        # softmax
                                        tile_max = pto.row_max_single(q_kt_scale)
                                        tile_sub = pto.sub(q_kt_scale, tile_max)
                                        tile_exp = pto.exp(tile_sub)
                                        tile_sum = pto.row_sum_single(tile_exp)
                                        tile_softmx = pto.div(tile_exp, tile_sum)
                                        value_type16 = pto.cast(tile_softmx, dtype)

                                        # matmul_2
                                        pto.set_cube_tile_shapes([c2_tile[0], c2_tile[1]],
                                            [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]], True)
                                        out = pto.matmul(value_type16, v_actual_part, pto.DT_FP32)

                                        # reshape and copyOut
                                        pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                        out_new = pto.reshape(out, [b_tile, s1_tile, g_tile, d_nope_size])
                                        pto.set_vec_tile_shapes(1, 1, out_tile[0], out_tile[1])
                                        out_final = pto.add_s(out_new, pto.element(out_new.dtype, float(0)))
                                        pto.assemble(out_final, out_offset, attention_out)
                                    inside_o_loop(o_idx)
                            inside_g_loop(b_idx, s1_idx, n2_idx, g_idx)
                inside_s1_loop(s1_idx)
        inside_b_loop(b_idx)


def win_attention_debug(**kwargs):
    q_nope = kwargs.get("q_nope")
    v_nope_cache = kwargs.get("v_nope_cache")
    q_rope = kwargs.get("q_rope")
    k_rope_cache = kwargs.get("k_rope_cache")
    n_q = kwargs.get("n_q")
    n_kv = kwargs.get("n_kv")
    block_table = kwargs.get("block_table")
    act_seqs = kwargs.get("act_seqs")
    window_size = kwargs.get("window_size")
    block_size = kwargs.get("block_size")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")

    with pto.function("main", [q_nope, v_nope_cache, q_rope, k_rope_cache, block_table, act_seqs], [attention_out]):
        def inside_main_function():
            win_attention_debug_compute(
                q_nope=q_nope,
                v_nope_cache=v_nope_cache,
                q_rope=q_rope,
                k_rope_cache=k_rope_cache,
                n_q=n_q,
                n_kv=n_kv,
                block_table=block_table,
                act_seqs=act_seqs,
                window_size=window_size,
                block_size=block_size,
                softmax_scale=softmax_scale,
                attention_out=attention_out,
                tile_config=tile_config
            )
        inside_main_function()


def win_attention(**kwargs):
    q_nope = kwargs.get("q_nope")
    v_nope_cache = kwargs.get("v_nope_cache")
    q_rope = kwargs.get("q_rope")
    k_rope_cache = kwargs.get("k_rope_cache")
    n_q = kwargs.get("n_q")
    n_kv = kwargs.get("n_kv")
    block_table = kwargs.get("block_table")
    act_seqs = kwargs.get("act_seqs")
    window_size = kwargs.get("window_size")
    block_size = kwargs.get("block_size")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")

    with pto.function("main", [q_nope, v_nope_cache, q_rope, k_rope_cache, block_table, act_seqs], [attention_out]):
        def inside_main_function():
            win_attention_compute(
                q_nope=q_nope,
                v_nope_cache=v_nope_cache,
                q_rope=q_rope,
                k_rope_cache=k_rope_cache,
                n_q=n_q,
                n_kv=n_kv,
                block_table=block_table,
                act_seqs=act_seqs,
                window_size=window_size,
                block_size=block_size,
                softmax_scale=softmax_scale,
                attention_out=attention_out,
                tile_config=tile_config
            )
        inside_main_function()


def win_attention_flash(**kwargs):
    q_nope = kwargs.get("q_nope")
    v_nope_cache = kwargs.get("v_nope_cache")
    q_rope = kwargs.get("q_rope")
    k_rope_cache = kwargs.get("k_rope_cache")
    n_q = kwargs.get("n_q")
    n_kv = kwargs.get("n_kv")
    block_table = kwargs.get("block_table")
    act_seqs = kwargs.get("act_seqs")
    window_size = kwargs.get("window_size")
    block_size = kwargs.get("block_size")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")

    with pto.function("main", [q_nope, v_nope_cache, q_rope, k_rope_cache, block_table, act_seqs], [attention_out]):
        def inside_main_function():
            win_attention_compute_flash(
                q_nope=q_nope,
                v_nope_cache=v_nope_cache,
                q_rope=q_rope,
                k_rope_cache=k_rope_cache,
                n_q=n_q,
                n_kv=n_kv,
                block_table=block_table,
                act_seqs=act_seqs,
                window_size=window_size,
                block_size=block_size,
                softmax_scale=softmax_scale,
                attention_out=attention_out,
                tile_config=tile_config
            )
        inside_main_function()


def test_win_atten_ut(tile_config, flash, debug):
    pto.set_host_option("KEY_ONLY_CODEGEN", True)

    d_type = pto.DT_FP16

    b = 2
    s_q = 1
    n_q = 128
    n_kv = 1
    s_max = 1024
    d_n = 512
    d_r = 64
    block_size = 128
    window_size = 1024
    softmax_scale = float(1.0 / math.sqrt((d_n + d_r)))

    max_block = (s_max + block_size - 1) // block_size
    q_nope_shape = [b * s_q * n_q, d_n]
    q_rope_shape = [b * s_q * n_q, d_r]
    v_nope_cache_shape = [b * max_block * block_size, n_kv * d_n]
    k_rope_cache_shape = [b * max_block * block_size, n_kv * d_r]
    attention_out_shape = [b, s_q, n_q, d_n]
    block_table_shape = [b, max_block]

    act_seqs = pto.tensor([b], pto.DT_INT32, "act_seqs")
    q_nope = pto.tensor(q_nope_shape, d_type, "q_nope")
    q_rope = pto.tensor(q_rope_shape, d_type, "q_rope")
    v_nope_cache = pto.tensor(v_nope_cache_shape, d_type, "v_nope_cache")
    k_rope_cache = pto.tensor(k_rope_cache_shape, d_type, "k_rope_cache")
    block_table = pto.tensor(block_table_shape, pto.DT_INT32, "block_table")
    attention_out = pto.tensor(attention_out_shape, pto.DT_FP32, "attention_out")

    if flash:
        win_attention_flash(
            q_nope=q_nope,
            v_nope_cache=v_nope_cache,
            q_rope=q_rope,
            k_rope_cache=k_rope_cache,
            n_q=n_q,
            n_kv=n_kv,
            block_table=block_table,
            act_seqs=act_seqs,
            window_size=window_size,
            block_size=block_size,
            softmax_scale=softmax_scale,
            attention_out=attention_out,
            tile_config=tile_config
        )
    elif debug:
        window_size = 512
        win_attention_debug(
            q_nope=q_nope,
            v_nope_cache=v_nope_cache,
            q_rope=q_rope,
            k_rope_cache=k_rope_cache,
            n_q=n_q,
            n_kv=n_kv,
            block_table=block_table,
            act_seqs=act_seqs,
            window_size=window_size,
            block_size=block_size,
            softmax_scale=softmax_scale,
            attention_out=attention_out,
            tile_config=tile_config
        )
    else:
        win_attention(
            q_nope=q_nope,
            v_nope_cache=v_nope_cache,
            q_rope=q_rope,
            k_rope_cache=k_rope_cache,
            n_q=n_q,
            n_kv=n_kv,
            block_table=block_table,
            act_seqs=act_seqs,
            window_size=window_size,
            block_size=block_size,
            softmax_scale=softmax_scale,
            attention_out=attention_out,
            tile_config=tile_config
        )


def main():
    tile_config = WinAttenTileShapeConfig(
        g_tile=128,
        skv_tile=512,
        v_nope_tile_shape=[16, 256],
        v_rope_tile_shape=[128, 64],
        out_tile_shape=[16, 256],
        c1_tile_shape=[128, 128, 64, 64, 128, 128],
        v1_tile_shape=[16, 256],
        c2_tile_shape=[128, 128, 128, 128, 128, 128],
        v2_tile_shape=[16, 256]
    )

    test_win_atten_ut(tile_config, False, False)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    main()
