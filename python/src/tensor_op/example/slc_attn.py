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

KEY_SUPPORT_DYNAMIC_UNALIGNED = "SUPPORT_DYNAMIC_UNALIGNED"


@dataclass
class SaConfig:
    manual_unroll: bool = False
    max_unroll_times: int = 1
    only_batch_loop: bool = False
    is_nz_format: bool = False


@dataclass
class SaTileShapeConfig:
    g_tile: int
    s_kv_tile: int
    c1_tile_shape: List[int]
    v1_tile_shape: List[int]
    c2_tile_shape: List[int]
    v2_tile_shape: List[int]


def slc_attn_compute(**kwargs):
    q_nope = kwargs.get("q_nope")
    q_rope = kwargs.get("q_rope")
    k_slc = kwargs.get("k_slc")
    v_slc = kwargs.get("v_slc")
    kv_slc_act_seqs = kwargs.get("kv_slc_act_seqs")
    n_q = kwargs.get("n_q")
    n_kv = kwargs.get("n_kv")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")

    dtype = q_nope.dtype
    d_n = q_nope.shape[1]
    d_r = q_rope.shape[1]
    group = n_q // n_kv

    g_tile = tile_config.g_tile
    s2_tile = tile_config.s_kv_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    batch_size_sym = kv_slc_act_seqs.shape[0]
    s1_n2_g_sym = q_nope.shape[0] // batch_size_sym
    s1_sym = s1_n2_g_sym // n_q
    g_loop_sym = group // g_tile

    s1_n2_s2_sym = k_slc.shape[0] // batch_size_sym
    n2_s2_sym = s1_n2_s2_sym // s1_sym
    n2_sym = n_kv

    for b_idx in pto.loop(0, batch_size_sym, 1, name="LOOP_L0_b_SA", idx_name="b_idx", 
        unroll_list=set(), submit_before_loop=True):
        for s1_idx in pto.loop(0, s1_sym, 1, name="LOOP_L1_s1_SA", idx_name="s1_idx"):
            def inside_s1_loop_sa(b_idx, s1_idx):
                cur_kv_slc_seq = pto.get_input_data(kv_slc_act_seqs, [b_idx, s1_idx])
                cur_seq = (cur_kv_slc_seq - s1_sym + 1 + s1_idx).max(0)
                cur_seq.as_intermediate_variable()
                bn_per_batch = (cur_seq + s2_tile - 1) / s2_tile

                for n2_idx in pto.loop(0, n2_sym, 1, name="LOOP_L2_n2_SA", idx_name="n2_idx"):
                    for g_idx in pto.loop(0, g_loop_sym, 1, name="LOOP_L3_g_SA", idx_name="g_idx"):
                        def inside_g_loop_sa(b_idx, s1_idx, n2_idx, g_idx):
                            cur_g_tile = g_tile
                            oi_update = pto.tensor([cur_g_tile, d_n], pto.DT_FP32, "oi_update")
                            li_update = pto.tensor([cur_g_tile, 1], pto.DT_FP32, "li_update")  
                            mi_update = pto.tensor([cur_g_tile, 1], pto.DT_FP32, "mi_update")

                            cur_offset = b_idx * s1_n2_g_sym + s1_idx * n_q + n2_idx * group + g_idx * cur_g_tile
                            oi_offset = [b_idx, s1_idx, n2_idx * group + g_idx * cur_g_tile, 0]

                            for s2_idx in pto.loop(0, bn_per_batch, 1, name="LOOP_L4_s2_SA", idx_name="s2_idx", 
                                unroll_list=pto.powers_of_2(1)):
                                def inside_s2_loop_sa(b_idx, s2_idx):
                                    cur_s2_tile = s2_tile
                                    cur_kv_offset = b_idx * s1_n2_s2_sym + s1_idx * n2_s2_sym + s2_idx * cur_s2_tile

                                    pto.set_semantic_label("Sa")
                                    qn = pto.view(q_nope, [cur_g_tile, d_n], [cur_g_tile, d_n], [cur_offset, 0])
                                    qr = pto.view(q_rope, [cur_g_tile, d_r], [cur_g_tile, d_r], [cur_offset, 0])
                                    qi = pto.tensor([cur_g_tile, d_n + d_r], dtype, "qi")
                                    pto.assemble(qn, [0, 0], qi)
                                    pto.assemble(qr, [0, d_n], qi)

                                    kj = pto.view(k_slc, [cur_s2_tile, d_n + d_r], 
                                        [(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), d_n + d_r],
                                        [cur_kv_offset, 0]) 
                                    vj = pto.view(v_slc, [cur_s2_tile, d_n], 
                                        [(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), d_n], [cur_kv_offset, 0])

                                    # C1
                                    pto.set_cube_tile_shapes([c1_tile[0], c1_tile[1]], 
                                        [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]], True)
                                    pto.set_semantic_label("Sa_QkMM")
                                    pto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
                                    sij = pto.matmul(pto.DT_FP32, qi, kj, False, True)

                                    # V1
                                    pto.set_semantic_label("Sa_Qkvec1")
                                    pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                    sij_scale = pto.mul_s(sij, pto.element(sij.dtype, softmax_scale))
                                    tilda_mij = pto.row_max_single(sij_scale) 
                                    tsub = pto.sub(sij_scale, tilda_mij) 
                                    tilda_pij = pto.exp(tsub)
                                    tilda_pij_f16 = pto.cast(tilda_pij, dtype)
                                    tilda_lij = pto.row_sum_single(tilda_pij) 

                                    if pto.cond(pto.is_loop_begin(s2_idx, 0)):
                                        def inside_if_loop_begin():
                                            nonlocal oi_update, li_update, mi_update
                                            pto.set_cube_tile_shapes(
                                                [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]],
                                                [c2_tile[4], c2_tile[5]], True)
                                            pto.set_semantic_label("Sa_KvMm")
                                            pto.set_matrix_size(
                                                [tilda_pij_f16.shape[0], tilda_pij_f16.shape[1], vj.shape[1]])
                                            oi_tmp = pto.matmul(pto.DT_FP32, tilda_pij_f16, vj, False, False)
                                            pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                            if pto.cond(pto.is_loop_end(s2_idx, bn_per_batch)):
                                                def inside_if_loop_end():
                                                    nonlocal oi_update
                                                    pto.set_semantic_label("Sa_KvVec2")
                                                    oi_update[:] = pto.div(oi_tmp, tilda_lij)
                                                    pto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                                    oi_update_4_dim = pto.add_s(
                                                        pto.reshape(oi_update, [1, 1, cur_g_tile, d_n]), 
                                                        pto.element(oi_update.dtype, float(0)))
                                                    pto.assemble(oi_update_4_dim, oi_offset, attention_out)
                                                inside_if_loop_end()
                                            else:
                                                def inside_else_loop_end():
                                                    nonlocal oi_update
                                                    oi_update[:] = oi_tmp
                                                inside_else_loop_end()
                                            li_update[:] = tilda_lij
                                            mi_update[:] = tilda_mij
                                        inside_if_loop_begin()
                                    else:
                                        def inside_else_loop_begin():
                                            nonlocal oi_update, li_update, mi_update
                                            pto.set_semantic_label("Sa_UpdateVec2")
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

                                            q3 = pto.mul(oi, t2)
                                            pto.set_cube_tile_shapes(
                                                [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]],
                                                [c2_tile[4], c2_tile[5]], True)
                                            pto.set_semantic_label("Sa_UpdateMM2")
                                            pto.set_matrix_size([tilda_pij_f16.shape[0], 
                                                tilda_pij_f16.shape[1], vj.shape[1]])
                                            q1 = pto.matmul(pto.DT_FP32, tilda_pij_f16, vj, False, False)
                                            pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                            q2 = pto.mul(q1, t4)
                                            oi_tmp = pto.add(q3, q2)
                                            if pto.cond(pto.is_loop_end(s2_idx, bn_per_batch)):
                                                def inside_if_loop_end():
                                                    nonlocal oi_update
                                                    oi_update[:] = pto.div(oi_tmp, li_new)
                                                    pto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                                    oi_update_4_dim = pto.add_s(
                                                        pto.reshape(oi_update, [1, 1, cur_g_tile, d_n]), 
                                                        pto.element(oi_update.dtype, float(0)))
                                                    pto.assemble(oi_update_4_dim, oi_offset, attention_out)
                                                inside_if_loop_end()
                                            else:
                                                def inside_else_loop_end():
                                                    nonlocal oi_update
                                                    oi_update[:] = oi_tmp
                                                inside_else_loop_end()
                                            li_update[:] = li_new
                                            mi_update[:] = mi_new
                                        inside_else_loop_begin()
                                inside_s2_loop_sa(b_idx, s2_idx)
                        inside_g_loop_sa(b_idx, s1_idx, n2_idx, g_idx)
            inside_s1_loop_sa(b_idx, s1_idx)


def slc_attn(**kwargs):
    q_nope = kwargs.get("q_nope")
    q_rope = kwargs.get("q_rope")
    k_slc = kwargs.get("k_slc")
    v_slc = kwargs.get("v_slc")
    kv_slc_act_seqs = kwargs.get("kv_slc_act_seqs")
    n_q = kwargs.get("n_q")
    n_kv = kwargs.get("n_kv")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")

    with pto.function("SA_MAIN", [q_nope, q_rope, k_slc, v_slc, kv_slc_act_seqs], [attention_out]):
        slc_attn_compute(
            q_nope=q_nope,
            q_rope=q_rope,
            k_slc=k_slc,
            v_slc=v_slc,
            kv_slc_act_seqs=kv_slc_act_seqs,
            n_q=n_q,
            n_kv=n_kv,
            softmax_scale=softmax_scale,
            attention_out=attention_out,
            tile_config=tile_config
        )


def test_sa_ut(input_param, tile_config, config):
    d_type = pto.DT_FP16

    b = input_param[0]
    sq = input_param[1]
    nq = input_param[2]
    nkv = input_param[3]
    dn = input_param[4]
    dr = input_param[5]
    smax = input_param[6]
    softmax_scale = float(1.0 / math.sqrt((dn + dr)))

    kv_format = pto.TILEOP_NZ if config.is_nz_format else pto.TILEOP_ND

    q_nope_shape = [b * sq * nq, dn]
    q_rope_shape = [b * sq * nq, dr]
    k_slc_shape = [b * sq * nkv * smax, dn + dr]
    v_slc_shape = [b * sq * nkv * smax, dn]
    sa_out_shape = [b, sq, nq, dn]

    act_seqs = pto.tensor([b, sq], pto.DT_INT32, "act_seqs")
    q_nope = pto.tensor(q_nope_shape, d_type, "q_nope")
    q_rope = pto.tensor(q_rope_shape, d_type, "q_rope")
    k_slc = pto.tensor(k_slc_shape, d_type, "k_slc", kv_format)
    v_slc = pto.tensor(v_slc_shape, d_type, "v_slc", kv_format)
    sa_out = pto.tensor(sa_out_shape, pto.DT_FP32, "sa_out")

    slc_attn(
        q_nope=q_nope,
        q_rope=q_rope,
        k_slc=k_slc,
        v_slc=v_slc,
        kv_slc_act_seqs=act_seqs,
        n_q=nq,
        n_kv=nkv,
        softmax_scale=softmax_scale,
        attention_out=sa_out,
        tile_config=tile_config
    )


def main():
    pto.set_codegen_config(KEY_SUPPORT_DYNAMIC_UNALIGNED, True)

    tile_config = SaTileShapeConfig(
        g_tile=128,
        s_kv_tile=1024,
        c1_tile_shape=[128, 1024, 64, 64, 128, 128],
        v1_tile_shape=[16, 256],
        c2_tile_shape=[128, 1024, 64, 64, 128, 128],
        v2_tile_shape=[16, 256]
    )

    input_param = [32, 1, 128, 1, 512, 64, 1024]
    config = SaConfig()
    test_sa_ut(input_param, tile_config, config)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    main()
