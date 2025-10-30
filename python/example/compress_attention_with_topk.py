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


KEY_ONLY_CODEGEN = "only_codegen"

SHAPE_DIM0 = 0
SHAPE_DIM1 = 1
SHAPE_DIM2 = 2
SHAPE_DIM3 = 3
SHAPE_DIM4 = 4
SHAPE_DIM5 = 5

NUM_NEG_2 = -2
NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_8 = 8
NUM_16 = 16
NUM_20 = 20
NUM_32 = 32
NUM_64 = 64
NUM_128 = 128
NUM_256 = 256
NUM_512 = 512
NUM_1024 = 1024

FP32 = pto.DT_FP32
BF16 = pto.DT_BF16
INT32 = pto.DT_INT32
tensor = pto.tensor


@dataclass
class CmpTile:
    c1_tile: list
    v1_tile: list
    c2_tile: list
    v2_tile: list


@dataclass
class CmpAttnTopkTile:
    topk_tile: list
    cmp_tile: CmpTile


def compress_attention_with_topk(**kwargs):
    q_nope: tensor = kwargs.get("q_nope")
    q_rope: tensor = kwargs.get("q_rope")
    cmp_kv_cache: tensor = kwargs.get("cmp_kv_cache")
    cmp_kr_cache: tensor = kwargs.get("cmp_kr_cache")
    cmp_block_table: tensor = kwargs.get("cmp_block_table")
    act_seq: tensor = kwargs.get("act_seq")
    aux_tensor: tensor = kwargs.get("aux_tensor")
    cmp_attn_out: tensor = kwargs.get("cmp_attn_out")
    topk_res: tensor = kwargs.get("topk_res")
    block_size: int = kwargs.get("block_size")
    cmp_block_size: int = kwargs.get("cmp_block_size")
    cmp_stride: int = kwargs.get("cmp_stride")
    slc_block_size: int = kwargs.get("slc_block_size")
    softmax_scale: float = kwargs.get("softmax_scale")
    n1: int = kwargs.get("n1")
    topk: int = kwargs.get("topk")
    front: int = kwargs.get("front")
    near: int = kwargs.get("near")
    tile_config: CmpAttnTopkTile = kwargs.get("tile_config")

    c1_tile = tile_config.cmp_tile.c1_tile
    v1_tile = tile_config.cmp_tile.v1_tile
    c2_tile = tile_config.cmp_tile.c2_tile
    v2_tile = tile_config.cmp_tile.v2_tile

    q_dtype = q_nope.dtype
    k_dtype = cmp_kv_cache.dtype

    b = cmp_block_table.shape[SHAPE_DIM0]
    if n1 == 0:
        raise ValueError(f"n1 can't be zero!")
    s1 = pto.symbolic_scalar(q_nope.shape[SHAPE_DIM0] // b // n1)

    d_n = q_nope.shape[SHAPE_DIM1]
    d_r = q_rope.shape[SHAPE_DIM1]
    d_q_k = d_n + d_r
    max_cmp_block = cmp_block_table.shape[SHAPE_DIM1]
    if cmp_stride == 0:
        raise ValueError(f"cmp_stride can't be zero!")
    slc_size = slc_block_size // cmp_stride
    if slc_size == 0:
        raise ValueError(f"slc_size can't be zero!")
    block_slc_num = pto.symbolic_scalar(block_size // slc_size)
    cmp_size = pto.symbolic_scalar(cmp_block_size // cmp_stride)
    slc_window = slc_size + cmp_size - 1

    # 接入Topk的时候让后面子图接收FP32的index结果，另外此处的LOOP必须存在，不能和下面的LOOP合并，否则会存在UB上的View
    topk_num_idx = tensor(dtype=FP32, shape=[1, 1, max_cmp_block * block_slc_num], name="topkNumIdx")
    pto.set_vec_tile_shapes(*tile_config.topk_tile)
    for _ in pto.loop(0, 1, 1, name="GEN_TOPK_RANGE", idx_name="ubReshapeIdx"):
        def inside_ub_reshape_idx_loop():
            nonlocal topk_num_idx
            dump_tensor = pto.full(
                [1, 1, max_cmp_block * block_slc_num], pto.element(FP32, .0), FP32)
            first_k_top = pto.topk(dump_tensor, max_cmp_block * block_slc_num, -1, True)[1] # ! tuple visit
            topk_num_idx[:] = pto.topk(
                pto.cast(first_k_top, FP32), max_cmp_block * block_slc_num, -1, False)[0] # ! tuple visit
        inside_ub_reshape_idx_loop()
    for b_idx in pto.loop(0, b, 1, name="CMP_ATTN_LOOP_BATCH", idx_name="bIdx"):
        def inside_b_idx_loop(b_idx):
            cur_seq = pto.get_tensor_data(act_seq, [b_idx])
            for s1_idx in pto.loop(0, s1, 1, name="CMP_ATTN_LOOP_S1", idx_name="s1Idx"):
                def inside_s1_idx_loop(b_idx, s1_idx):
                    if cmp_stride == 0:
                        raise ValueError(f"slc_size can't be zero!")
                    cas_cmp_seq = (cur_seq - (s1 - s1_idx - 1) - cmp_block_size) // cmp_stride + 1
                    if block_size == 0:
                        raise ValueError("block_size can't be zero!")
                    cur_cmp_block = (cas_cmp_seq + block_size - 1) // block_size
                    slc_loop = (cas_cmp_seq + slc_size - 1) // slc_size
                    q_offset = b_idx * s1 * n1 + s1_idx * n1

                    oi_update = tensor(dtype=FP32, shape=[n1, d_n], name="oiUpdate")
                    li_update = tensor(dtype=FP32, shape=[1, n1], name="liUpdate")
                    mi_update = tensor(dtype=FP32, shape=[1, n1], name="miUpdate")
                    slc_before_g_reduce = tensor(
                        dtype=FP32, shape=[max_cmp_block * block_slc_num, n1], name="slcBeforeGReduce")
                    local_max_gather = tensor(dtype=FP32, shape=[max_cmp_block, n1], name="localMaxGather")
                    slc_pre = tensor(dtype=FP32, shape=[block_slc_num, n1], name="slcPre")
                    for block_idx in pto.loop(0, cur_cmp_block, 1, name="CMP_ATTN_LOOP_BLOCK", idx_name="blockIdx"):
                        def inside_block_idx_loop(b_idx, block_idx):
                            cur_block_idx = pto.to_syms(
                                pto.get_tensor_data(cmp_block_table, [b_idx, block_idx]))
                            cur_block_idx.as_intermediate_variable()

                            cur_valid_seq = (cas_cmp_seq - block_idx * block_size).min(block_size)
                            cur_slc_loop = pto.to_syms(
                                (cur_valid_seq + slc_size - 1) // slc_size)
                            cur_slc_loop.as_intermediate_variable()

                            slc_cur = tensor(dtype=FP32, shape=[block_slc_num, n1], name="slcCur")
                            tilda_pij_pad = tensor(
                                dtype=FP32, shape=[block_size + slc_window - 1, n1], name="tildaPijPad")
                            mi_modify = tensor(dtype=FP32, shape=[1, n1], name="miModify")
                            for _ in pto.loop(0, 1, 1, name="AVOID_LOOP_1", idx_name="avoidLoop1Idx"):
                                def inside_avoid_loop_1_idx_loop(b_idx, s1_idx, block_idx):
                                    vec_tile = NUM_128
                                    pto.set_vec_tile_shapes(vec_tile, vec_tile)
                                    cur_qn = pto.view(q_nope, [n1, d_n], [q_offset, 0])
                                    cur_qr = pto.view(q_rope, [n1, d_r], [q_offset, 0])
                                    cur_q_attn = pto.concat([cur_qn, cur_qr], 1)
                                    cur_q_attn.set_name("curQAttn") # ! check func exist

                                    cmp_kv_cache_2d = pto.reshape(cmp_kv_cache,
                                        [cmp_kv_cache.shape[0] * \
                                        cmp_kv_cache.shape[1] * cmp_kv_cache.shape[2], d_n])
                                    cur_cmp_kv = pto.view(cmp_kv_cache_2d,
                                        [block_size, d_n], [cur_block_idx * block_size, 0],
                                        [cur_valid_seq, d_n])
                                    cmp_kr_cache_2d = pto.reshape(cmp_kr_cache,
                                        [cmp_kr_cache.shape[0] * \
                                        cmp_kr_cache.shape[1] * cmp_kr_cache.shape[2], d_r])
                                    cur_cmp_kr = pto.view(
                                        cmp_kr_cache_2d, [block_size, d_r], [cur_block_idx * block_size, 0],
                                        [cur_valid_seq, d_r])
                                    cur_k_attn = pto.assemble(
                                        [[cur_cmp_kv, [0, 0]], [cur_cmp_kr, [0, d_n]]])
                                    cur_k_attn.set_name("curKAttn") # ! check func exist

                                    cur_v_attn = cur_cmp_kv
                                    cur_v_attn.set_name("curVAttn") # ! check func exist
                                    pto.set_semantic_label("Cmp-Attn-C1")
                                    pto.set_cube_tile_shapes(
                                        [c1_tile[0], c1_tile[1]],
                                        [c1_tile[2], c1_tile[3]],
                                        [c1_tile[4], c1_tile[5]]
                                    )
                                    sij = pto.matmul(cur_k_attn, cur_q_attn, FP32, a_trans=False, b_trans=True)
                                    sij.set_name("sij")

                                    pto.set_vec_tile_shapes(vec_tile, vec_tile)
                                    sij[:] = pto.view(sij, [block_size, n1], [0, 0], [cur_valid_seq, n1])
                                    pto.set_semantic_label("Cmp-Attn-V1")
                                    sij_scale = pto.mul_s(
                                        sij, pto.element(sij.dtype, softmax_scale))
                                    tilda_mij = pto.row_max_single(sij_scale, 0)
                                    tsub = pto.sub(sij_scale, tilda_mij)
                                    tilda_pij = pto.exp(tsub)
                                    tilda_pij_b16 = pto.cast(tilda_pij, k_dtype)
                                    tilda_lij = pto.row_sum_single(tilda_pij, 0)
                                    if pto.cond(pto.is_loop_begin(block_idx, 0)):
                                        def inside_if_loop_begin(b_idx, s1_idx, block_idx):
                                            pto.set_semantic_label("Cmp-Attn-First-Block-C2")
                                            pto.set_cube_tile_shapes(
                                                [c2_tile[0], c2_tile[1]],
                                                [c2_tile[2], c2_tile[3]],
                                                [c2_tile[4], c2_tile[5]],
                                            )
                                            oi_tmp = pto.matmul(
                                                tilda_pij_b16, cur_v_attn, FP32, a_trans=True, b_trans=False)
                                            oi_tmp.set_name("oiTmp")
                                            pto.set_semantic_label("Cmp-Attn-First-Block-V2")
                                            pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                            if pto.cond(pto.is_loop_end(block_idx, cur_cmp_block)):
                                                def inside_if_is_loop_end():
                                                    oi_update[:] = pto.div(oi_tmp, pto.reshape(tilda_lij, [n1, 1]))
                                                    oi_update_reshape = pto.reshape(oi_update, [1, 1, n1, d_n])
                                                    pto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                                    oi_update_cast = pto.assign(pto.cast(
                                                        oi_update_reshape, cmp_attn_out.dtype))
                                                    pto.assemble(oi_update_cast,
                                                        [b_idx, s1_idx, 0, 0], cmp_attn_out)
                                                    pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                inside_if_is_loop_end()
                                            else:
                                                oi_update[:] = oi_tmp
                                            li_update[:] = tilda_lij
                                            mi_update[:] = tilda_mij
                                        inside_if_loop_begin(b_idx, s1_idx, block_idx)
                                    else:
                                        def inside_else_loop_begin(b_idx, s1_idx, block_idx):
                                            pto.set_semantic_label("Cmp-Attn-Other-Update-V1")
                                            oi = oi_update
                                            li = li_update
                                            mi = mi_update
                                            mi_new = pto.maximum(mi, tilda_mij)
                                            t1 = pto.sub(mi, mi_new)
                                            t2 = pto.exp(t1)
                                            t3 = pto.sub(tilda_mij, mi_new)
                                            t4 = pto.exp(t3)
                                            t5 = pto.mul(t4, tilda_lij)
                                            t6 = pto.mul(t2, li)
                                            li_new = pto.add(t6, t5)
                                            q3 = pto.mul(oi, pto.reshape(t2, [n1, 1]))
                                            pto.set_semantic_label("Cmp-Attn-Other-Update-C2")
                                            pto.set_cube_tile_shapes(
                                                [c2_tile[0], c2_tile[1]],
                                                [c2_tile[2], c2_tile[3]],
                                                [c2_tile[4], c2_tile[5]],
                                            )
                                            tilda_pij_b16_t = pto.transpose(tilda_pij_b16, [0, 1])
                                            q1 = pto.matmul(
                                                tilda_pij_b16_t, cur_v_attn, FP32, a_trans=False, b_trans=False)
                                            q1.set_name("q1")
                                            pto.set_semantic_label("Cmp-Attn-Other-Update-V2")
                                            pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                            q2 = pto.mul(q1, pto.reshape(t4, [n1, 1]))
                                            oi_tmp = pto.add(q3, q2)
                                            if pto.cond(pto.is_loop_end(block_idx, cur_cmp_block)):
                                                def inside_if_is_loop_end():
                                                    oi_update[:] = pto.div(oi_tmp, pto.reshape(li_new, [n1, 1]))
                                                    oi_update_reshape = pto.reshape(oi_update, [1, 1, n1, d_n])
                                                    pto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                                    oi_update_cast = pto.assign(
                                                        pto.cast(oi_update_reshape, cmp_attn_out.dtype))
                                                    pto.assemble(oi_update_cast, [b_idx, s1_idx, 0, 0], cmp_attn_out)
                                                    pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                inside_if_is_loop_end()
                                            else:
                                                def inside_else_is_loop_end():
                                                    pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                    oi_update[:] = oi_tmp
                                                inside_else_is_loop_end()
                                            pto.assemble(mi_new, [block_idx - 1, 0], local_max_gather)
                                            mi_modify[:] = pto.sub(mi, mi_new)
                                            mi_modify[:] = pto.exp(mi_modify)

                                            mi_update[:] = mi_new
                                            li_update[:] = li_new

                                            sub_cur = pto.sub(tilda_mij, mi_update)
                                            exp_cur = pto.exp(sub_cur)
                                            tilda_pij[:] = pto.mul(tilda_pij, exp_cur)
                                        inside_else_loop_begin(b_idx, s1_idx, block_idx)
                                    src = pto.element(FP32, .0)
                                    zeros = pto.full([slc_window - 1, n1], src, FP32)
                                    tilda_pij_pad[:] = pto.concat([tilda_pij, zeros], 0)
                                inside_avoid_loop_1_idx_loop(b_idx, s1_idx, block_idx)
                            for slc_idx in pto.loop(0, cur_slc_loop, 1, name="CMP_ATTN_P_SLC_FIRST", idx_name="slcIdx"):
                                def inside_slc_idx_loop(slc_idx):
                                    slc_valid = (cur_valid_seq - slc_idx * slc_size).min(slc_window)
                                    last_view = pto.view(tilda_pij_pad, [slc_window, n1],
                                        [slc_idx * slc_size, 0], [slc_valid, n1])
                                    aux_tmp_tensor = pto.view(
                                        aux_tensor, [slc_window, n1], [0, 0], [slc_valid, n1])
                                    slc_last_no_reduce = pto.mul(last_view, aux_tmp_tensor)
                                    slc_last_reduce = pto.row_sum_single(slc_last_no_reduce, 0)
                                    pto.assemble(slc_last_reduce, [slc_idx, 0], slc_cur)
                                inside_slc_idx_loop(slc_idx)

                            for _ in pto.loop(0, 1, 1, name="GEN_SLC_BEFORE_G_REDUCE", idx_name="unusedIdx"):
                                def inside_unused_idx_loop():
                                    if pto.cond(block_idx != 0):
                                        def inside_if_block_idx_zero():
                                            slc_pre[:] = pto.mul(slc_pre, mi_modify)
                                            modify_tensor = pto.view(
                                                tilda_pij_pad,
                                                [cmp_size - 1, n1],
                                                [0, 0],
                                                [(cmp_size - 1).min(cur_valid_seq), n1])
                                            last_aux_tensor = pto.view(
                                                aux_tensor,
                                                [cmp_size - 1, n1],
                                                [aux_tensor.shape[0] -
                                                    (cmp_size - 1).min(cur_valid_seq), 0],
                                                [(cmp_size - 1).min(cur_valid_seq), n1])
                                            modify_tensor[:] = pto.mul(modify_tensor, last_aux_tensor)
                                            modify_tensor_reduce = pto.row_sum_single(modify_tensor, 0)
                                            pre_view_tensor = pto.view(
                                                slc_pre, [1, n1], [block_slc_num - 1, 0])
                                            pre_view_tensor[:] = pto.add(pre_view_tensor, modify_tensor)
                                            pto.assemble(
                                                pto.assign(pto.view(
                                                    slc_pre, [block_slc_num - 1, n1], [0, 0])),
                                                [(block_idx - 1) * block_slc_num, 0],
                                                slc_before_g_reduce)
                                            slc_pre[:] = pto.assign(slc_cur)
                                        inside_if_block_idx_zero()
                                    else:
                                        slc_pre[:] = pto.assign(slc_cur)
                                    if pto.cond(block_idx == cur_cmp_block - 1):
                                        pto.assemble(
                                            pto.assign(slc_cur),
                                            [block_idx * block_slc_num, 0], slc_before_g_reduce)
                                inside_unused_idx_loop()
                        inside_block_idx_loop(b_idx, block_idx)
                    slc_before_g_reduce_2 = tensor(
                        dtype=FP32, shape=[max_cmp_block * block_slc_num, n1], name="slcBeforeGReduce2")
                    for block_idx in pto.loop(0, cur_cmp_block, 1, name="SOFTMAX_BLOCK", idx_name="blockIdx"):
                        def inside_block_idx_loop(block_idx):
                            slc_before_g_reduce_block = pto.view(
                                slc_before_g_reduce, [block_slc_num, n1], [block_idx * block_slc_num, 0])
                            if pto.cond(pto.is_loop_end(block_idx, cur_cmp_block)):
                                slc_before_g_reduce_block[:] = pto.div(slc_before_g_reduce_block, li_update)
                            else:
                                def inside_else_is_loop_end():
                                    row_max_block = pto.view(local_max_gather, [1, n1], [block_idx, 0])
                                    sub_tmp = pto.sub(row_max_block, mi_update)
                                    exp_tmp = pto.exp(sub_tmp)
                                    slc_before_g_reduce_block[:] = pto.mul(slc_before_g_reduce_block, exp_tmp)
                                    slc_before_g_reduce_block[:] = pto.div(slc_before_g_reduce_block, li_update)
                                    slc_before_g_reduce_block[:] = pto.add_s(
                                        slc_before_g_reduce_block, pto.element(FP32, .0))
                                inside_else_is_loop_end()
                            pto.assemble(
                                slc_before_g_reduce_block,
                                [block_idx * block_slc_num, 0],
                                slc_before_g_reduce_2)
                        inside_block_idx_loop(block_idx)

                    for _ in pto.loop(0, 1, 1, name="TOPK_INDICES", idx_name="unusedIdx"):
                        def inside_unused_idx_loop():
                            slc_reshape = tensor()
                            for _ in pto.loop(0, 1, 1, name="AVOID_LOOP_5", idx_name="ubReshapeIdx"):
                                def inside_ub_reshape_idx_loop():
                                    slc_before_g_reduce_actual = pto.view(
                                        slc_before_g_reduce_2,
                                        [max_cmp_block * block_slc_num, n1],
                                        [0, 0], [slc_loop, n1]
                                    )
                                    slc_reduce = pto.row_sum_single(slc_before_g_reduce_actual)
                                    pto.set_vec_tile_shapes(*tile_config.topk_tile)
                                    slc_reshape[:] = pto.reshape(slc_reduce,
                                        [1, 1, max_cmp_block * block_slc_num], [1, 1, slc_loop])
                                    slc_reshape[:] = pto.add_s(slc_reshape, pto.element(FP32, .0))
                                inside_ub_reshape_idx_loop()
                            for _ in pto.loop(0, 1, 1, name="AVOID_LOOP_6", idx_name="ubReshapeIdx"):
                                def inside_ub_reshape_idx_loop():
                                    if pto.cond(slc_loop < topk):
                                        def inside_if_slc_loop_lt_topk():
                                            cur_idx = pto.view(
                                                topk_num_idx,
                                                [1, 1, max_cmp_block * block_slc_num],
                                                [0, 0, 0],
                                                [1, 1, slc_loop])
                                            cur_idx[:] = pto.cast(cur_idx, INT32)
                                            pto.assemble(cur_idx, [b_idx, s1_idx, 0], topk_res)
                                        inside_if_slc_loop_lt_topk()
                                    else:
                                        def inside_else_slc_loop_lt_topk():
                                            slc_front = pto.view(topk_num_idx, [1, 1, front], [0, 0, 0])
                                            slc_near = pto.view(
                                                topk_num_idx, [1, 1, near], [0, 0, slc_loop - near])
                                            slc_re_view = pto.view(
                                                slc_reshape,
                                                [1, 1, max_cmp_block * block_slc_num - front - near],
                                                [0, 0, front],
                                                [1, 1, slc_loop - front - near])
                                            inner_topk = pto.topk(
                                                slc_re_view, topk - front - near, -1, True)[1]
                                            inner_topk[:] = pto.add_s(inner_topk, pto.element(INT32, 1))
                                            slc_front[:] = pto.cast(slc_front, INT32)
                                            slc_near[:] = pto.cast(slc_near, INT32)
                                            pto.assemble(slc_front, [b_idx, s1_idx, 0], topk_res)
                                            pto.assemble(inner_topk, [b_idx, s1_idx, front], topk_res)
                                            pto.assemble(
                                                slc_near, [b_idx, s1_idx, topk - near], topk_res)
                                        inside_else_slc_loop_lt_topk()
                                inside_ub_reshape_idx_loop()
                        inside_unused_idx_loop()
                inside_s1_idx_loop(b_idx, s1_idx)
        inside_b_idx_loop(b_idx)


def test_cmp_attn_topk(data_type, tile_config: CmpAttnTopkTile, input_param: list, act_seq_len: list):
    pto.set_host_option(KEY_ONLY_CODEGEN, True)
    d_type = data_type

    b = input_param[0]
    s1 = input_param[1]
    n1 = input_param[2]
    dn = input_param[3]
    dr = input_param[4]
    n2 = input_param[5]
    block_size = input_param[6]
    cmp_block_size = input_param[7]
    cmp_stride = input_param[8]
    slc_block_size = input_param[9]
    topk = input_param[10]
    front = input_param[11]
    near = input_param[12]
    softmax_scale = 1.0 / math.sqrt(dn + dr)

    q_type = d_type
    k_type = d_type

    act_cmp_seq = list()

    for cur_seq in act_seq_len:
        cur_cmp_seq = (cur_seq - cmp_block_size) // cmp_stride + 1
        act_cmp_seq.append(cur_cmp_seq)

    cmp_block_num = 0
    for s in act_cmp_seq:
        cmp_block_num += (s + block_size - 1) // block_size
    max_cmp_seq = max(act_cmp_seq)
    max_cmp_block_num = (max_cmp_seq + block_size - 1) // block_size

    slc_size = slc_block_size // cmp_stride
    block_slc_num = block_size // slc_size

    q_nope = tensor(dtype=q_type, shape=[b * s1 * n1, dn], name="qNope")
    q_rope = tensor(dtype=q_type, shape=[b * s1 * n1, dr], name="qRope")
    cmp_kv_cache = tensor(dtype=k_type, shape=[cmp_block_num, block_size, n2, dn], name="cmpKvCache")
    cmp_kr_cache = tensor(dtype=k_type, shape=[cmp_block_num, block_size, n2, dr], name="cmpKrCache")
    cmp_block_table = tensor(dtype=INT32, shape=[b, max_cmp_block_num], name="cmpBlockTable")
    act_seq = tensor(dtype=INT32, shape=[b], name="actSeq")
    aux_tensor = tensor(
        dtype=FP32, shape=[slc_block_size // cmp_stride + cmp_block_size // cmp_stride - 1, n1], name="auxTensor")

    cmp_attn = tensor(dtype=FP32, shape=[b, s1, n1, dn], name="cmpAttnOut")
    topk_res = tensor(dtype=INT32, shape=[b, s1, topk], name="topkRes")

    attn_golden = [.0] * (b * s1 * n1 * dn)
    topk_golden = [0] * (b * s1 * topk)

    input_tensors = [q_nope, q_rope, cmp_kv_cache, cmp_block_table, act_seq, aux_tensor]
    output_tensors = [cmp_attn, topk_res]
    with pto.function("CompressAttentionWithTopk", input_tensors, output_tensors):
        compress_attention_with_topk(
            q_nope=q_nope,
            q_rope=q_rope,
            cmp_kv_cache=cmp_kv_cache,
            cmp_kr_cache=cmp_kr_cache,
            cmp_block_table=cmp_block_table,
            act_seq=act_seq,
            aux_tensor=aux_tensor,
            cmp_attn_out=cmp_attn,
            topk_res=topk_res,
            block_size=block_size,
            cmp_block_size=cmp_block_size,
            cmp_stride=cmp_stride,
            slc_block_size=slc_block_size,
            softmax_scale=softmax_scale,
            n1=n1,
            topk=topk,
            front=front,
            near=near,
            tile_config=tile_config
        )

if __name__ == '__main__':
    topk_tile = [1, 1, 128]
    c1_tile = [128, 128, 128, 128, 128, 128]
    v1_tile = [128, 128]
    c2_tile = [128, 128, 128, 128, 128, 128]
    v2_tile = [128, 64]
    cmp_tile = CmpTile(c1_tile, v1_tile, c2_tile, v2_tile)
    tile_config = CmpAttnTopkTile(topk_tile, cmp_tile)

    input_param = [2, 1, 128, 512, 64, 1, 128, 32, 16, 64, 16, 1, 2]
    act_seq_len = [8192, 8192]
    data_type = BF16

    test_cmp_attn_topk(data_type, tile_config, input_param, act_seq_len)

    graph_dump = pto.dump()
    with open("dump_cmp_attn_topk_py.txt", "w", encoding="utf-8") as f:
        f.write(graph_dump)