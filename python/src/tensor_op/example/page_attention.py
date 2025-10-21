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
import logging
import math
from dataclasses import dataclass
from typing import List
from enum import Enum
import numpy as np
import pto


TILE_VEC_DIMS = 2
TILE_CUBE_DIMS = 6


@dataclass
class PaTileShapeConfig:
    head_num_q_tile: int = 0  # // 由于没有处理尾块，当前仅支持因子切分

    v0_tile_shape = [0] * TILE_VEC_DIMS
    c1_tile_shape = [0] * TILE_CUBE_DIMS   # // (m, M), (k, K), (n, N)
    v1_tile_shape = [0] * TILE_VEC_DIMS
    c2_tile_shape = [0] * TILE_CUBE_DIMS   # // (m, M), (k, K), (n, N)
    v2_tile_shape = [0] * TILE_VEC_DIMS


def page_attention(**kwargs):
    q_nope = kwargs.get("q_nope")
    k_nope_cache = kwargs.get("k_nope_cache")
    v_nope_cache = kwargs.get("v_nope_cache")
    q_rope = kwargs.get("q_rope")
    k_rope_cache = kwargs.get("k_rope_cache")
    block_table = kwargs.get("block_table")
    act_seqs = kwargs.get("act_seqs")
    block_size = kwargs.get("block_size")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")
    max_unroll_times = kwargs.get("max_unroll_times")
    is_nz_format = kwargs.get("is_nz_format")
    is_nz_format = False
   
    dtype = q_nope.get_dtype()
    #// 入参B*S*N合轴
    d_n = q_nope.shape[1]
    d_r = q_rope.shape[1]

    n_tile = tile_config.head_num_q_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    input_tensors = [q_nope, k_nope_cache, v_nope_cache, q_rope, k_rope_cache, block_table, act_seqs]
    output_tensors = [attention_out]
    with pto.dyn_function("main", input_tensors, output_tensors):
        def inside_main_function():
            batch_size = block_table.shape[0]
            n_q = q_nope.shape[0] // batch_size
            n_loop = n_q // n_tile
            with pto.loop_function("LOOP_L0_bIdx", "b_idx", pto.loop_range(0, batch_size, 1)) as b_idx_loop:
                for b_idx in b_idx_loop:
                    def inside_b_idx_loop(b_idx):
                        cur_seq = pto.get_tensor_data(act_seqs, [b_idx])
                        bn_per_batch = (cur_seq + block_size - 1) // block_size
                        bn_per_batch.as_intermediate_variable()
                        with pto.loop_function("LOOP_L1_nIdx", "n_idx", pto.loop_range(0, n_loop, 1)) as n_idx_loop:
                            for n_idx in n_idx_loop:
                                def inside_n_idx_loop(b_idx, n_idx, bn_per_batch):
                                    nonlocal n_tile
                                    cur_n_tile = n_tile
                                    oi_update = pto.tensor([n_tile, d_n], pto.DataType.DT_FP32, "oiUpdate")
                                    li_update = pto.tensor([n_tile, 1], pto.DataType.DT_FP32, "liUpdate")
                                    mi_update = pto.tensor([n_tile, 1], pto.DataType.DT_FP32, "miUpdate")
                                    # 当前curOffset没放到更内层循环，避免重复bnPerBatch次的DAssemble操作
                                    cur_offset = b_idx * n_q + n_idx * n_tile
                                    oi_offset = [cur_offset, 0]  # (B*N*S, d)

                                    # LoopRange(0, bnPerBatch, 1), PowersOf2(1)) {
                                    with pto.loop_function("LOOP_L2_bn", "bn", pto.loop_range(0, bn_per_batch, 1),
                                                            pto.powers_of_2(max_unroll_times)) as bn_loop:
                                        for bn in bn_loop:
                                            def inside_bn_loop(**kwargs):
                                                b_idx = kwargs.get("b_idx")
                                                block_table = kwargs.get("block_table")
                                                cur_seq = kwargs.get("cur_seq")
                                                bn = kwargs.get("bn")
                                                block_size = kwargs.get("block_size")
                                                bn_per_batch = kwargs.get("bn_per_batch")
                                                nonlocal oi_update, li_update, mi_update
                                                # 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                                                cur_s2_tile = block_size
                                                qn = pto.view(q_nope, [cur_n_tile, d_n], [cur_offset, 0])
                                                qr = pto.view(q_rope, [cur_n_tile, d_r], [cur_offset, 0])
                                                qi = pto.tensor([cur_n_tile, d_n + d_r], dtype, "qi")
                                                pto.assemble(qn, [0, 0], qi)
                                                pto.assemble(qr, [0, d_n], qi)

                                                cur_block_idx = pto.get_tensor_data(block_table, [b_idx, bn])
                                                cur_block_idx.as_intermediate_variable()
                                                kn = pto.view(k_nope_cache, [cur_s2_tile, d_n], 
                                                              [(cur_seq - bn * block_size).min(block_size), d_n],
                                                              [cur_block_idx * block_size, 0])
                                                kr = pto.view(k_rope_cache, [cur_s2_tile, d_r], 
                                                              [(cur_seq - bn * block_size).min(block_size), d_r],
                                                              [cur_block_idx * block_size, 0])
                                                kj_format = pto.tile_op_format.TILEOP_NZ if is_nz_format else (
                                                    pto.tile_op_format.TILEOP_ND
                                                )
                                                kj = pto.tensor([cur_s2_tile, d_n + d_r], dtype, "kj", kj_format)
                                                pto.assemble(kn, [0, 0], kj)
                                                pto.assemble(kr, [0, d_n], kj)
                                                kj = pto.view(kj, [cur_s2_tile, d_n + d_r], 
                                                [(cur_seq - bn * block_size).min(block_size), d_r + d_n], [0, 0])
                                                vj = pto.view(v_nope_cache, [cur_s2_tile, d_n], 
                                                              [(cur_seq - bn * block_size).min(block_size), d_n],
                                                              [cur_block_idx * block_size, 0])

                                                pto.set_semantic_label("MatMul")
                                                pto.set_cube_tile_shapes(
                                                    [c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]], 
                                                    [c1_tile[4], c1_tile[5]])
                                                pto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
                                                sij = pto.matmul(pto.DataType.DT_FP32, qi, kj, False, True)
                                                
                                                pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])

                                                pto.set_semantic_label("SoftMax")
                                                sij_scale = pto.mul_s(
                                                    sij, pto.element(sij.get_dtype(), softmax_scale))

                                                pto.set_semantic_label("SoftMax")
                                                tilda_mij = pto.row_max_single(sij_scale)  
                                                tsub = pto.sub(sij_scale, tilda_mij)
                                                tilda_pij = pto.exp(tsub)
                                                tilda_pij_f16 = pto.cast(tilda_pij, dtype)
                                                tilda_lij = pto.row_sum_single(tilda_pij)  
                                                # (nTileCur, s2TileCur) -> (nTileCur, 1)

                                                if pto.cond(pto.is_loop_begin(bn, 0)):
                                                    def inside_if_loop_begin():
                                                        nonlocal oi_update, li_update, mi_update
                                                        pto.set_cube_tile_shapes(
                                                            [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]],
                                                            [c2_tile[4], c2_tile[5]])
                                                        pto.set_semantic_label("b1-matmul2")
                                                        pto.set_matrix_size(
                                                            [tilda_pij_f16.shape[0], tilda_pij_f16.shape[1], 
                                                             vj.shape[1]])
                                                        oi_tmp = pto.matmul(pto.DataType.DT_FP32, tilda_pij_f16, 
                                                                            vj, False, False)
                                                        pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                        pto.set_semantic_label("b1-after-matmul2")
                                                        if pto.cond(pto.is_loop_end(bn, bn_per_batch)):
                                                            pto.set_semantic_label("b1-after-matmul2")
                                                            oi_update[:] = (pto.div(oi_tmp, tilda_lij))  
                                                            # (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                                                            pto.assemble(oi_update, oi_offset, attention_out)
                                                        else:
                                                            oi_update[:] = (oi_tmp)
                                                        li_update[:] = (tilda_lij)
                                                        mi_update[:] = (tilda_mij)
                                                    inside_if_loop_begin()
                                                else:
                                                    def inside_else_loop_begin():
                                                        nonlocal oi_update, li_update, mi_update
                                                        oi = oi_update
                                                        li = li_update
                                                        mi = mi_update
                                                        pto.set_semantic_label("Softmax-acc")
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        mi_new = pto.maximum(mi, tilda_mij)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)  
                                                        t1 = pto.sub(mi, mi_new)           
                                                        t2 = pto.exp(t1)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        t3 = pto.sub(tilda_mij, mi_new)  
                                                        t4 = pto.exp(t3)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        t5 = pto.mul(t4, tilda_lij)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)  
                                                        t6 = pto.mul(t2, li)       
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        li_new = pto.add(t6, t5)    
                                                        # (curNTile, dN), (curNTile, 1) -> (curNTile, dN)
                                                        q3 = pto.mul(oi, t2)
                                                        pto.set_semantic_label("bn-matmul2")
                                                        pto.set_cube_tile_shapes(
                                                        [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], 
                                                        c2_tile[5]])
                                                        pto.set_matrix_size(
                                                            [tilda_pij_f16.shape[0], 
                                                             tilda_pij_f16.shape[1], vj.shape[1]])
                                                        q1 = pto.matmul(pto.DataType.DT_FP32, tilda_pij_f16, vj, 
                                                                        False, False)
                                                        pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                        pto.set_semantic_label("bn-after-matmul2")
                                                        # (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                                                        q2 = pto.mul(q1, t4)    
                                                        # (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                                                        oi_tmp = pto.add(q3, q2)  
                                                        if pto.cond(pto.is_loop_end(bn, bn_per_batch)):
                                                            # (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                                                            oi_update[:] = (pto.div(oi_tmp, li_new))  
                                                            pto.assemble(oi_update, oi_offset, attention_out)
                                                        else:
                                                            oi_update[:] = (oi_tmp)
                                                        li_update[:] = (li_new)
                                                        mi_update[:] = (mi_new)
                                                    inside_else_loop_begin()
                                            inside_bn_loop(
                                                b_idx=b_idx,
                                                block_table=block_table,
                                                cur_seq=cur_seq,
                                                bn=bn,
                                                block_size=block_size,
                                                bn_per_batch=bn_per_batch)
                                    # } # LOOP("LOOP_L2_bn") ends
                                inside_n_idx_loop(b_idx, n_idx, bn_per_batch)
                        # } # LOOP("LOOP_L1_nIdx") ends
                    inside_b_idx_loop(b_idx)
        inside_main_function()


def page_attention_with_imm_scalar(**kwargs):
    q_nope = kwargs.get("q_nope")
    k_nope_cache = kwargs.get("k_nope_cache")
    v_nope_cache = kwargs.get("v_nope_cache")
    q_rope = kwargs.get("q_rope")
    k_rope_cache = kwargs.get("k_rope_cache")
    block_table = kwargs.get("block_table")
    act_seqs = kwargs.get("act_seqs")
    block_size = kwargs.get("block_size")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")
    max_unroll_times = kwargs.get("max_unroll_times")
    is_nz_format = kwargs.get("is_nz_format")
    is_nz_format = False
   
    dtype = q_nope.get_dtype()
    #// 入参B*S*N合轴
    d_n = q_nope.shape[1]
    d_r = q_rope.shape[1]

    n_tile = tile_config.head_num_q_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    input_tensors = [q_nope, k_nope_cache, v_nope_cache, q_rope, k_rope_cache]
    output_tensors = [attention_out]
    with pto.dyn_function("main", input_tensors, output_tensors):
        def inside_main_function():
            batch_size = int(len(block_table))
            n_q = q_nope.shape[0] // pto.symbolic_scalar(batch_size)
            n_loop = n_q // n_tile

            with pto.loop_function("LOOP_L0_bIdx", "b_idx", pto.loop_range(0, batch_size, 1)) as b_idx_loop:
                for b_idx in b_idx_loop:
                    def inside_b_idx_loop(b_idx):
                        cur_seq = pto.symbolic_scalar(int(act_seqs[0]))
                        bn_per_batch = (cur_seq + block_size - 1) // block_size
                        bn_per_batch.as_intermediate_variable()
                        with pto.loop_function("LOOP_L1_nIdx", "n_idx", pto.loop_range(0, n_loop, 1)) as n_idx_loop:
                            for n_idx in n_idx_loop:
                                def inside_n_idx_loop(b_idx, n_idx, bn_per_batch):
                                    nonlocal n_tile
                                    cur_n_tile = n_tile
                                    oi_update = pto.tensor([n_tile, d_n], pto.DataType.DT_FP32, "oiUpdate")
                                    li_update = pto.tensor([n_tile, 1], pto.DataType.DT_FP32, "liUpdate")
                                    mi_update = pto.tensor([n_tile, 1], pto.DataType.DT_FP32, "miUpdate")
                                    # 当前curOffset没放到更内层循环，避免重复bnPerBatch次的DAssemble操作
                                    cur_offset = b_idx * n_q + n_idx * n_tile
                                    oi_offset = [cur_offset, 0]  # (B*N*S, d)

                                    # LoopRange(0, bnPerBatch, 1), PowersOf2(1)) {
                                    with pto.loop_function("LOOP_L2_bn", "bn", pto.loop_range(0, bn_per_batch, 1),
                                                            pto.powers_of_2(max_unroll_times)) as bn_loop:
                                        for bn in bn_loop:
                                            def inside_bn_loop(**kwargs):
                                                b_idx = kwargs.get("b_idx")
                                                block_table = kwargs.get("block_table")
                                                cur_seq = kwargs.get("cur_seq")
                                                bn = kwargs.get("bn")
                                                block_size = kwargs.get("block_size")
                                                bn_per_batch = kwargs.get("bn_per_batch")
                                                nonlocal oi_update, li_update, mi_update
                                                # 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                                                cur_s2_tile = block_size
                                                qn = pto.view(q_nope, [cur_n_tile, d_n], [cur_offset, 0])
                                                qr = pto.view(q_rope, [cur_n_tile, d_r], [cur_offset, 0])
                                                qi = pto.tensor([cur_n_tile, d_n + d_r], dtype, "qi")
                                                pto.assemble(qn, [0, 0], qi)
                                                pto.assemble(qr, [0, d_n], qi)

                                                cur_block_idx = pto.symbolic_scalar(0)
                                                cur_block_idx.as_intermediate_variable()
                                                kn = pto.view(k_nope_cache, [cur_s2_tile, d_n], 
                                                              [(cur_seq - bn * block_size).min(block_size), d_n],
                                                              [cur_block_idx * block_size, 0])
                                                kr = pto.view(k_rope_cache, [cur_s2_tile, d_r], 
                                                              [(cur_seq - bn * block_size).min(block_size), d_r],
                                                              [cur_block_idx * block_size, 0])

                                                kj_format = pto.tile_op_format.TILEOP_NZ if is_nz_format else (
                                                    pto.tile_op_format.TILEOP_ND
                                                )
                                                kj = pto.tensor([cur_s2_tile, d_n + d_r], dtype, "kj", kj_format)
                                                pto.assemble(kn, [0, 0], kj)
                                                pto.assemble(kr, [0, d_n], kj)
                                                kj = pto.view(kj, [cur_s2_tile, d_n + d_r], 
                                                [(cur_seq - bn * block_size).min(block_size), d_r + d_n], [0, 0])
                                                vj = pto.view(v_nope_cache, [cur_s2_tile, d_n], 
                                                              [(cur_seq - bn * block_size).min(block_size), d_n],
                                                              [cur_block_idx * block_size, 0])

                                                pto.set_cube_tile_shapes(
                                                    [c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]], 
                                                    [c1_tile[4], c1_tile[5]])
                                                pto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
                                                sij = pto.matmul(pto.DataType.DT_FP32, qi, kj, False, True)
                                                
                                                pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])

                                                sij_scale = pto.mul_s(
                                                    sij, pto.element(sij.get_dtype(), softmax_scale))

                                                tilda_mij = pto.row_max_single(sij_scale)  
                                                tsub = pto.sub(sij_scale, tilda_mij)
                                                tilda_pij = pto.exp(tsub)
                                                tilda_pij_f16 = pto.cast(tilda_pij, dtype)
                                                tilda_lij = pto.row_sum_single(tilda_pij)  
                                                # (nTileCur, s2TileCur) -> (nTileCur, 1)

                                                if pto.cond(bn == 0):
                                                    def inside_if_loop_begin():
                                                        nonlocal oi_update, li_update, mi_update
                                                        pto.set_cube_tile_shapes(
                                                            [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]],
                                                            [c2_tile[4], c2_tile[5]])
                                                        pto.set_matrix_size(
                                                            [tilda_pij_f16.shape[0], tilda_pij_f16.shape[1], 
                                                             vj.shape[1]])
                                                        oi_tmp = pto.matmul(pto.DataType.DT_FP32, tilda_pij_f16, 
                                                                            vj, False, False)
                                                        pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                        if pto.cond(bn == bn_per_batch - 1):
                                                            oi_update[:] = (pto.div(oi_tmp, tilda_lij))  
                                                            # (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                                                            pto.assemble(oi_update, oi_offset, attention_out)
                                                        else:
                                                            oi_update[:] = (oi_tmp)
                                                        li_update[:] = (tilda_lij)
                                                        mi_update[:] = (tilda_mij)
                                                    inside_if_loop_begin()
                                                else:
                                                    def inside_else_loop_begin():
                                                        nonlocal oi_update, li_update, mi_update
                                                        oi = oi_update
                                                        li = li_update
                                                        mi = mi_update
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        mi_new = pto.maximum(mi, tilda_mij)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)  
                                                        t1 = pto.sub(mi, mi_new)           
                                                        t2 = pto.exp(t1)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        t3 = pto.sub(tilda_mij, mi_new)  
                                                        t4 = pto.exp(t3)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        t5 = pto.mul(t4, tilda_lij)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)  
                                                        t6 = pto.mul(t2, li)       
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        li_new = pto.add(t6, t5)   

                                                        # (curNTile, dN), (curNTile, 1) -> (curNTile, dN)
                                                        q3 = pto.mul(oi, t2)  
                                                        pto.set_cube_tile_shapes(
                                                            [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], 
                                                            [c2_tile[4], c2_tile[5]])
                                                        pto.set_matrix_size(
                                                            [tilda_pij_f16.shape[0], 
                                                             tilda_pij_f16.shape[1], vj.shape[1]])
                                                        q1 = pto.matmul(pto.DataType.DT_FP32, tilda_pij_f16, vj, 
                                                                        False, False)
                                                        pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                        # (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                                                        q2 = pto.mul(q1, t4)    
                                                        # (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                                                        oi_tmp = pto.add(q3, q2)  
                                                        if pto.cond(bn == bn_per_batch - 1):
                                                            # (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                                                            oi_update[:] = (pto.div(oi_tmp, li_new))  
                                                            pto.assemble(oi_update, oi_offset, attention_out)
                                                        else:
                                                            oi_update[:] = (oi_tmp)
                                                        li_update[:] = (li_new)
                                                        mi_update[:] = (mi_new)
                                                    inside_else_loop_begin()
                                            inside_bn_loop(
                                                b_idx=b_idx,
                                                block_table=block_table,
                                                cur_seq=cur_seq,
                                                bn=bn,
                                                block_size=block_size,
                                                bn_per_batch=bn_per_batch)
                                    # } # LOOP("LOOP_L2_bn") ends
                                inside_n_idx_loop(b_idx, n_idx, bn_per_batch)
                        # } # LOOP("LOOP_L1_nIdx") ends
                    inside_b_idx_loop(b_idx)
        inside_main_function()


def page_attention_with_manual_unroll(**kwargs):
    q_nope = kwargs.get("q_nope")
    k_nope_cache = kwargs.get("k_nope_cache")
    v_nope_cache = kwargs.get("v_nope_cache")
    q_rope = kwargs.get("q_rope")
    k_rope_cache = kwargs.get("k_rope_cache")
    block_table = kwargs.get("block_table")
    act_seqs = kwargs.get("act_seqs")
    block_size = kwargs.get("block_size")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")
    max_unroll_times = kwargs.get("max_unroll_times")
   
    dtype = q_nope.get_dtype()
    #// 入参B*S*N合轴
    d_n = q_nope.shape[1]
    d_r = q_rope.shape[1]

    n_tile = tile_config.head_num_q_tile
    v0_tile = tile_config.v0_tile_shape
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    div2 = 2
    input_tensors = [q_nope, k_nope_cache, v_nope_cache, q_rope, k_rope_cache, block_table, act_seqs]
    output_tensors = [attention_out]
    with pto.dyn_function("main", input_tensors, output_tensors):
        def inside_main_function():
            batch_size = block_table.shape[0]
            n_q = q_nope.shape[0] // batch_size
            n_loop = n_q // n_tile
            with pto.loop_function("LOOP_L0_bIdx", "b_idx", pto.loop_range(0, batch_size, 1)) as b_idx_loop:
                for b_idx in b_idx_loop:
                    def inside_b_idx_loop(b_idx):
                        cur_seq = pto.get_tensor_data(act_seqs, [b_idx])
                        bn_per_batch = cur_seq // block_size
                        bn_per_batch.as_intermediate_variable()
                        with pto.loop_function("LOOP_L1_nIdx", "n_idx", pto.loop_range(0, n_loop, 1)) as n_idx_loop:
                            for n_idx in n_idx_loop:
                                def inside_n_idx_loop(b_idx, n_idx, bn_per_batch):
                                    nonlocal n_tile
                                    oi_update = pto.tensor([n_tile, d_n], pto.DataType.DT_FP32, "oiUpdate")
                                    li_update = pto.tensor([n_tile, 1], pto.DataType.DT_FP32, "liUpdate")
                                    mi_update = pto.tensor([n_tile, 1], pto.DataType.DT_FP32, "miUpdate")
                                    # 当前curOffset没放到更内层循环，避免重复bnPerBatch次的DAssemble操作
                                    cur_offset = b_idx * n_q + n_idx * n_tile
                                    oi_offset = [cur_offset, 0]  # (B*N*S, d)

                                    # LoopRange(0, bnPerBatch, 1), PowersOf2(1)) {
                                    with pto.loop_function("LOOP_L2_bn", "bn", pto.loop_range(bn_per_batch),
                                                            pto.powers_of_2(max_unroll_times)) as bn_loop:
                                        for bn in bn_loop:
                                            def inside_bn_loop(**kwargs):
                                                b_idx = kwargs.get("b_idx")
                                                block_table = kwargs.get("block_table")
                                                cur_seq = kwargs.get("cur_seq")
                                                bn = kwargs.get("bn")
                                                block_size = kwargs.get("block_size")
                                                bn_per_batch = kwargs.get("bn_per_batch")
                                                nonlocal oi_update, li_update, mi_update

                                                unroll_times = max_unroll_times
                                                while unroll_times != 0:
                                                    if pto.record_loop_func.MatchUnrollTimes(unroll_times):
                                                        # 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                                                        pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                                        qn = pto.view(q_nope, [n_tile, d_n], [cur_offset, 0])
                                                        qr = pto.view(q_rope, [n_tile, d_r], [cur_offset, 0])
                                                        qi = pto.concat([qn, qr], 1)
                                                        sub_kns = []
                                                        sub_krs = []
                                                        sub_vjs = []
                                                        for idx_offset in range(unroll_times):
                                                            cur_block_idx = pto.get_tensor_data(block_table, 
                                                            [b_idx, bn + idx_offset])
                                                            sub_kns.append(
                                                                pto.view(k_nope_cache, [block_size, d_n], 
                                                                [cur_block_idx * block_size, 0]))
                                                            sub_krs.append(
                                                                pto.view(k_rope_cache, [block_size, d_r], 
                                                                [cur_block_idx * block_size, 0]))
                                                            sub_vjs.append(
                                                                pto.view(v_nope_cache, [block_size, d_n], 
                                                                [cur_block_idx * block_size, 0]))
                                                        

                                                        kn = pto.concat(sub_kns, 0)
                                                        kr = pto.concat(sub_krs, 0)
                                                        kj = pto.concat([kn, kr], 1)
                                                        vj = pto.concat(sub_vjs, 0)

                                                        pto.set_cube_tile_shapes(
                                                            [c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]], 
                                                            [c1_tile[4], c1_tile[5]])

                                                        sij = pto.matmul(pto.DataType.DT_FP32, qi, kj, False, True)
                                                        pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                                        sij_scale = pto.mul_s(
                                                            sij, pto.element(sij.get_dtype(), softmax_scale))

                                                        tilda_mij = pto.row_max_single(sij_scale)  
                                                        tsub = pto.sub(sij_scale, tilda_mij)
                                                        tilda_pij = pto.exp(tsub)
                                                        tilda_pij_f16 = pto.cast(tilda_pij, dtype)
                                                        tilda_lij = pto.row_sum_single(tilda_pij)  
                                                        # (nTileCur, s2TileCur) -> (nTileCur, 1)

                                                        if pto.cond(pto.is_loop_begin(bn, 0)):
                                                            def inside_if_loop_begin():
                                                                nonlocal oi_update, li_update, mi_update
                                                                pto.set_cube_tile_shapes(
                                                                    [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]],
                                                                    [c2_tile[4], c2_tile[5]])
                                                                oi_tmp = pto.matmul(pto.DataType.DT_FP32, tilda_pij_f16, 
                                                                                    vj, False, False)
                                                                pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                                if pto.cond(pto.is_loop_end(bn, bn_per_batch)):
                                                                    oi_update[:] = (pto.div(oi_tmp, tilda_lij))  
                                                                    # (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                                                                    pto.assemble(oi_update, oi_offset, attention_out)
                                                                else:
                                                                    oi_update[:] = (oi_tmp)
                                                                li_update[:] = (tilda_lij)
                                                                mi_update[:] = (tilda_mij)
                                                            inside_if_loop_begin()
                                                        else:
                                                            def inside_else_loop_begin():
                                                                nonlocal oi_update, li_update, mi_update
                                                                oi = oi_update
                                                                li = li_update
                                                                mi = mi_update
                                                                
                                                                # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                                mi_new = pto.maximum(mi, tilda_mij)
                                                                # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)  
                                                                t1 = pto.sub(mi, mi_new)           
                                                                t2 = pto.exp(t1)
                                                                # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                                t3 = pto.sub(tilda_mij, mi_new)  
                                                                t4 = pto.exp(t3)
                                                                # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                                t5 = pto.mul(t4, tilda_lij)
                                                                # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)  
                                                                t6 = pto.mul(t2, li)       
                                                                # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                                li_new = pto.add(t6, t5)    

                                                                # (curNTile, dN), (curNTile, 1) -> (curNTile, dN)
                                                                q3 = pto.mul(oi, t2)
                                                                pto.set_cube_tile_shapes(
                                                                [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], 
                                                                [c2_tile[4], c2_tile[5]])
                                                                
                                                                q1 = pto.matmul(pto.DataType.DT_FP32, tilda_pij_f16, vj, 
                                                                                False, False)
                                                                pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                                # (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                                                                q2 = pto.mul(q1, t4)    
                                                                # (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                                                                oi_tmp = pto.add(q3, q2)  
                                                                if pto.cond(pto.is_loop_end(bn, bn_per_batch)):
                                                                    # (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                                                                    oi_update[:] = (pto.div(oi_tmp, li_new))  
                                                                    pto.assemble(oi_update, oi_offset, attention_out)
                                                                else:
                                                                    oi_update[:] = (oi_tmp)
                                                                li_update[:] = (li_new)
                                                                mi_update[:] = (mi_new)
                                                            inside_else_loop_begin()
                                                    unroll_times = unroll_times // div2
                                            inside_bn_loop(
                                                b_idx=b_idx,
                                                block_table=block_table,
                                                cur_seq=cur_seq,
                                                bn=bn,
                                                block_size=block_size,
                                                bn_per_batch=bn_per_batch)
                                    # } # LOOP("LOOP_L2_bn") ends
                                inside_n_idx_loop(b_idx, n_idx, bn_per_batch)
                        # } # LOOP("LOOP_L1_nIdx") ends
                    inside_b_idx_loop(b_idx)
        inside_main_function()


def page_attention_high_throughput(**kwargs):
    q_nope = kwargs.get("q_nope")
    k_nope_cache = kwargs.get("k_nope_cache")
    v_nope_cache = kwargs.get("v_nope_cache")
    q_rope = kwargs.get("q_rope")
    k_rope_cache = kwargs.get("k_rope_cache")
    block_table = kwargs.get("block_table")
    act_seqs = kwargs.get("act_seqs")
    block_size = kwargs.get("block_size")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")
    max_unroll_times = kwargs.get("max_unroll_times")
    max_unroll_times = 1
   
    dtype = q_nope.get_dtype()
    #// 入参B*S*N合轴
    d_n = q_nope.shape[1]
    d_r = q_rope.shape[1]

    n_tile = tile_config.head_num_q_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    input_tensors = [q_nope, k_nope_cache, v_nope_cache, q_rope, k_rope_cache, block_table, act_seqs]
    output_tensors = [attention_out]
    with pto.dyn_function("main", input_tensors, output_tensors):
        def inside_main_function():
            batch_size = block_table.shape[0]
            n_q = q_nope.shape[0] // batch_size
            
            with pto.loop_function("LOOP_L0_bIdx", "b_idx", pto.loop_range(0, batch_size, 1), 
            pto.powers_of_2(max_unroll_times)) as b_idx_loop:
                for b_idx in b_idx_loop:
                    def inside_b_idx_loop(b_idx):
                        cur_seq = pto.get_tensor_data(act_seqs, [b_idx])
                        bn_per_batch = (cur_seq + block_size - 1) // block_size
                        bn_per_batch.as_intermediate_variable()
                        
                        cur_n_tile = n_tile
                        oi_update = pto.tensor([n_tile, d_n], pto.DataType.DT_FP32, "oiUpdate")
                        
                        # 当前curOffset没放到更内层循环，避免重复bnPerBatch次的Assemble操作
                        cur_offset = b_idx * n_q
                        oi_offset = [cur_offset, 0]  # (B*N*S, d)

                        # 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                        cur_s2_tile = block_size
                        qn = pto.view(q_nope, [cur_n_tile, d_n], [cur_offset, 0])
                        qr = pto.view(q_rope, [cur_n_tile, d_r], [cur_offset, 0])
                        qi = pto.tensor([cur_n_tile, d_n + d_r], dtype, "qi")
                        pto.assemble(qn, [0, 0], qi)
                        pto.assemble(qr, [0, d_n], qi)

                        cur_block_idx = pto.get_tensor_data(block_table, [b_idx, 0])
                        cur_block_idx.as_intermediate_variable()
                        kn = pto.view(k_nope_cache, [cur_s2_tile, d_n], 
                                        [min(cur_seq, block_size), d_n],
                                        [cur_block_idx * block_size, 0])
                        kr = pto.view(k_rope_cache, [cur_s2_tile, d_r], 
                                        [min(cur_seq, block_size), d_r],
                                        [cur_block_idx * block_size, 0])
                        kj = pto.tensor([cur_s2_tile, d_n + d_r], dtype, "kj")
                        pto.assemble(kn, [0, 0], kj)
                        pto.assemble(kr, [0, d_n], kj)
                        vj = pto.view(v_nope_cache, [cur_s2_tile, d_n], 
                                        [min(cur_seq, block_size), d_n],
                                        [cur_block_idx * block_size, 0])

                        pto.set_cube_tile_shapes(
                            [c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]], 
                            [c1_tile[4], c1_tile[5]])
                        sij = pto.matmul(pto.DataType.DT_FP32, qi, kj, False, True)
                        pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pto.mul_s(
                            sij, pto.element(sij.get_dtype(), softmax_scale))

                        tilda_mij = pto.row_max_single(sij_scale)  
                        tsub = pto.sub(sij_scale, tilda_mij)
                        tilda_pij = pto.exp(tsub)
                        tilda_pij_f16 = pto.cast(tilda_pij, dtype)
                        tilda_lij = pto.row_sum_single(tilda_pij)  # (nTileCur, s2TileCur) -> (nTileCur, 1)

                        pto.set_cube_tile_shapes(
                            [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]],
                            [c2_tile[4], c2_tile[5]])
                        oi_tmp = pto.matmul(pto.DataType.DT_FP32, tilda_pij_f16, 
                                            vj, False, False)
                        pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        oi_update[:] = (pto.div(oi_tmp, tilda_lij))  # (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                        pto.assemble(oi_update, oi_offset, attention_out)
                    inside_b_idx_loop(b_idx)
        inside_main_function()

if __name__ == '__main__':
    input_param = [4, 1, 32, 1, 512, 64, 128, 32]
    b = input_param[0]
    sq = input_param[1]
    nq = input_param[2]
    nk = input_param[3]
    dn = input_param[4]
    dr = input_param[5]
    block_size = input_param[6]
    n_tile = input_param[7]
    softmax_scale = 1.0 / math.sqrt(dn + dr)

    tile_config = PaTileShapeConfig()
    tile_config.head_num_q_tile = n_tile
    tile_config.v0_tile_shape = [n_tile, 64]
    tile_config.c1_tile_shape = [n_tile, n_tile, 64, 64, block_size, block_size]
    tile_config.v1_tile_shape = [n_tile, 64]
    tile_config.c2_tile_shape = [n_tile, n_tile, 64, 64, block_size, block_size]
    tile_config.v2_tile_shape = [n_tile, 64]

    seq = [256] * b

    block_num = 0
    for s in seq:
        block_num += ((s + (block_size - 1)) // block_size)
    # // blockTable: (b, maxBlockNumPerBatch)
    max_seq_all_batch = max(seq)
    max_block_num_per_batch = ((max_seq_all_batch + (block_size - 1)) // block_size)

    q_nope = pto.tensor([b * nq * sq, dn], pto.DataType.DT_BF16, "qNope")
    k_nope_cache = pto.tensor([int(block_num * block_size), nk * dn], pto.DataType.DT_BF16, "kNopeCache")
    v_nope_cache = pto.tensor([int(block_num * block_size), nk * dn], pto.DataType.DT_BF16, "vNopeCache")
    q_rope = pto.tensor([b * nq * sq, nk * dr], pto.DataType.DT_BF16, "qRope")
    k_rope_cache = pto.tensor([int(block_num * block_size), nk * dr], pto.DataType.DT_BF16, "kRope")
    block_table = pto.tensor([b, max_block_num_per_batch], pto.DataType.DT_INT32, "blockTable")
    act_seqs = pto.tensor([b], pto.DataType.DT_INT32, "actSeqs")
    pa_out = pto.tensor([b * nq * sq, dn], pto.DataType.DT_FP32, "paOut")

    max_unroll_times = 4
    page_attention(
        q_nope=q_nope, 
        k_nope_cache=k_nope_cache, 
        v_nope_cache=v_nope_cache, 
        q_rope=q_rope, 
        k_rope_cache=k_rope_cache, 
        block_table=block_table, 
        act_seqs=act_seqs, 
        block_size=block_size, 
        softmax_scale=softmax_scale, 
        attention_out=pa_out, 
        tile_config=tile_config, 
        max_unroll_times=max_unroll_times)
