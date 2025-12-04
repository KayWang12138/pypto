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

from dataclasses import dataclass, field
from typing import List
import math
import logging
import pypto
from pypto import pypto_impl
from pypto.operation import op_wrapper


@op_wrapper
def gather_in_l1(src, offsets, size, is_b_matrix, is_trans):
    return pypto_impl.gather_in_l1(src, offsets, size, is_b_matrix, is_trans)


@op_wrapper
def gather_in_ub(
    param,
    indices,
    axis
):
    """gather_in_ub."""

    return pypto_impl.gather_in_ub(param, indices, axis)


SHAPE_DIM_0 = 0
SHAPE_DIM_1 = 1

NUM_1 = 1
NUM_F1 = 1.0
NUM_4 = 4
NUM_16 = 16
NUM_32 = 32
NUM_64 = 64
NUM_128 = 128
NUM_256 = 256
NUM_512 = 512
NUM_2048 = 2048
NUM_4096 = 4096


def set_config():
    pypto.set_host_options(only_codegen=True)
    pypto.set_codegen_options(support_dynamic_unaligned=True)


@dataclass
class SelectedAttentionTileConfigV2:
    g_tile: int
    s_kv_tile: int
    c1_tile: List[List[int]]
    v1_tile: List[int]
    c2_tile: List[List[int]]
    v2_tile: List[int]


@dataclass
class SASimpleParamsV2:
    n_q: int
    n_kv: int
    softmax_scale: float
    topk: int
    tile: SelectedAttentionTileConfigV2


@dataclass
class SAInputsV2:
    q_nope: pypto.Tensor
    q_rope: pypto.Tensor
    k_nope_2d: pypto.Tensor
    k_rope_2d: pypto.Tensor
    k_nope_scales: pypto.Tensor
    offsets: pypto.Tensor
    kv_slc_act_seqs: pypto.Tensor
    attention_out: pypto.Tensor
    params: SASimpleParamsV2


def selected_attention_compute_v2(args: SAInputsV2):
    q_nope = args.q_nope
    q_rope = args.q_rope
    k_nope_2d = args.k_nope_2d
    k_rope_2d = args.k_rope_2d
    k_nope_scales = args.k_nope_scales
    offsets = args.offsets
    kv_slc_act_seqs = args.kv_slc_act_seqs
    attention_out = args.attention_out
    params = args.params

    dtype = q_nope.dtype
    kn_dtype = k_nope_2d.dtype

    d_n = q_nope.shape[SHAPE_DIM_1]
    d_r = q_rope.shape[SHAPE_DIM_1]

    n_q = params.n_q
    n_kv = params.n_kv
    group = n_q // n_kv

    g_tile = params.tile.g_tile
    s2_tile = params.tile.s_kv_tile
    c1_tile = params.tile.c1_tile
    v1_tile = params.tile.v1_tile
    c2_tile = params.tile.c2_tile
    v2_tile = params.tile.v2_tile

    n2_sym = n_kv
    batch_size_sym = kv_slc_act_seqs.shape[SHAPE_DIM_0]
    s1n1g_sym = q_nope.shape[SHAPE_DIM_0] // batch_size_sym
    s1_sym = s1n1g_sym // n_q
    s1s2_sym = s1_sym * params.topk
    g_loop_sym = group // g_tile
    s2_sym = s1s2_sym // s1_sym

    input_tensors = [
        q_nope,
        q_rope,
        k_nope_2d,
        k_rope_2d,
        k_nope_scales,
        offsets,
        kv_slc_act_seqs,
    ]
    output_tensors = [attention_out]
    
    with pypto.function("R2_SA_MAIN_V2", input_tensors, output_tensors):

        def inside_main_function():
            for b_idx in pypto.loop(
                0,
                batch_size_sym,
                1,
                name="LOOP_L0_b_SA",
                idx_name="bIdx",
                submit_before_loop=True,
            ):

                def inside_b_idx_loop(b_idx):
                    cur_kv_slc_seq = kv_slc_act_seqs[b_idx]

                    for s1_idx in pypto.loop(
                        0, s1_sym, 1, name="LOOP_L1_s1_SA", idx_name="s1Idx"
                    ):

                        def inside_s1_idx_loop(b_idx, s1_idx):
                            cur_seq = (
                                (cur_kv_slc_seq - s1_sym + 1 + s1_idx)
                                .max(0)
                                .min(params.topk)
                            )
                            cur_seq.as_variable()
                            bn_per_batch = (cur_seq + s2_tile - 1) // s2_tile

                            for n2_idx in pypto.loop(
                                0, n2_sym, 1, name="LOOP_L2_n2_SA", idx_name="n2Idx"
                            ):

                                def inside_n2_idx_loop(
                                    b_idx, s1_idx, n2_idx, bn_per_batch
                                ):
                                    for g_idx in pypto.loop(
                                        0,
                                        g_loop_sym,
                                        1,
                                        name="LOOP_L3_g_SA",
                                        idx_name="gIdx",
                                    ):

                                        def inside_g_idx_loop(
                                            b_idx,
                                            s1_idx,
                                            n2_idx,
                                            g_idx,
                                            bn_per_batch,
                                        ):
                                            cur_g_tile = g_tile

                                            oi_update = pypto.tensor(
                                                [cur_g_tile, d_n],
                                                pypto.DT_FP32,
                                                "oiUpdate",
                                            )
                                            li_update = pypto.tensor(
                                                [1, cur_g_tile],
                                                pypto.DT_FP32,
                                                "liUpdate",
                                            )
                                            mi_update = pypto.tensor(
                                                [1, cur_g_tile],
                                                pypto.DT_FP32,
                                                "miUpdate",
                                            )

                                            cur_offset = (
                                                b_idx * s1n1g_sym
                                                + s1_idx * n_q
                                                + n2_idx * group
                                                + g_idx * cur_g_tile
                                            )

                                            oi_offset = [
                                                b_idx,
                                                s1_idx,
                                                n2_idx * group + g_idx * cur_g_tile,
                                                0,
                                            ]

                                            for s2_idx, _ in pypto.loop_unroll(
                                                0,
                                                bn_per_batch,
                                                1,
                                                name="LOOP_L4_s2_SA",
                                                idx_name="s2Idx",
                                                unroll_list=[1],
                                            ):

                                                def inside_s2_idx_loop(
                                                    b_idx,
                                                    s1_idx,
                                                    s2_idx,
                                                    bn_per_batch,
                                                    cur_g_tile,
                                                    cur_offset,
                                                    oi_offset,
                                                ):
                                                    nonlocal oi_update, li_update, mi_update
                                                    cur_s2_tile = s2_tile
                                                    cur_kv_offset = (
                                                        b_idx * s1s2_sym
                                                        + s1_idx * s2_sym
                                                        + s2_idx * cur_s2_tile
                                                    )
                                                    pypto.set_semantic_label("Sa_QkMM")

                                                    qn = pypto.view(
                                                        q_nope,
                                                        [cur_g_tile, d_n],
                                                        [cur_offset, 0],
                                                        valid_shape=[cur_g_tile, d_n],
                                                    )
                                                    qr = pypto.view(
                                                        q_rope,
                                                        [cur_g_tile, d_r],
                                                        [cur_offset, 0],
                                                        valid_shape=[cur_g_tile, d_r],
                                                    )
                                                    qi = pypto.tensor(
                                                        [cur_g_tile, d_n + d_r],
                                                        dtype,
                                                        "qi",
                                                    )
                                                    pypto.assemble(qn, [0, 0], qi)
                                                    pypto.assemble(qr, [0, d_n], qi)

                                                    offset_view = pypto.view(
                                                        offsets,
                                                        [1, cur_s2_tile],
                                                        [
                                                            b_idx * s1_sym + s1_idx,
                                                            s2_idx * cur_s2_tile,
                                                        ],
                                                        valid_shape=[
                                                            1,
                                                            (
                                                                cur_seq
                                                                - s2_idx * cur_s2_tile
                                                            ).min(cur_s2_tile),
                                                        ],
                                                    )

                                                    k_nope_2d_view = pypto.view(
                                                        k_nope_2d,
                                                        [cur_s2_tile, d_n],
                                                        [0, 0],
                                                        valid_shape=[
                                                            (
                                                                cur_seq
                                                                - s2_idx * cur_s2_tile
                                                            ).min(cur_s2_tile),
                                                            d_n,
                                                        ],
                                                    )

                                                    k_nope_scales_view = pypto.view(
                                                        k_nope_scales,
                                                        [cur_s2_tile, 4],
                                                        [0, 0],
                                                        valid_shape=[
                                                            (
                                                                cur_seq
                                                                - s2_idx * cur_s2_tile
                                                            ).min(cur_s2_tile),
                                                            4,
                                                        ],
                                                    )

                                                    kn = pypto.tensor(
                                                        [s2_tile, d_n],
                                                        dtype,
                                                        "kn",
                                                    )
                                                    vj = pypto.tensor(
                                                        [s2_tile, d_n],
                                                        dtype,
                                                        "vj",
                                                    )

                                                    if kn_dtype == pypto.DT_INT8:
                                                        pypto.set_vec_tile_shapes(
                                                            NUM_32, NUM_512
                                                        )

                                                        kn_scale = gather_in_ub(
                                                            k_nope_scales_view,
                                                            offset_view,
                                                            axis=-2,
                                                        )

                                                        kn_quant = gather_in_ub(
                                                            k_nope_2d_view,
                                                            offset_view,
                                                            axis=-2,
                                                        )

                                                        kn_quant_f16 = pypto.cast(
                                                            kn_quant, pypto.DT_FP16
                                                        )
                                                        kn_quant_f32 = pypto.cast(
                                                            kn_quant_f16, pypto.DT_FP32
                                                        )

                                                        kn_quant_f32_tmp = (
                                                            pypto.reshape(
                                                                kn_quant_f32,
                                                                [
                                                                    s2_tile * NUM_4,
                                                                    NUM_128,
                                                                ],
                                                            )
                                                        )
                                                        kn_scale_tmp = pypto.reshape(
                                                            kn_scale,
                                                            [s2_tile * NUM_4, NUM_1],
                                                        )

                                                        pypto.set_vec_tile_shapes(
                                                            NUM_128, NUM_128
                                                        )
                                                        kn_f32 = (
                                                            kn_quant_f32_tmp
                                                            * kn_scale_tmp
                                                        )

                                                        kn_f32_reshape = pypto.reshape(
                                                            kn_f32,
                                                            [s2_tile, d_n],
                                                        )

                                                        pypto.set_vec_tile_shapes(
                                                            NUM_32, NUM_512
                                                        )
                                                        cur_kn_fp32 = pypto.view(
                                                            kn_f32_reshape,
                                                            [cur_s2_tile, d_n],
                                                            [0, 0],
                                                            valid_shape=[
                                                                (
                                                                    cur_seq
                                                                    - s2_idx
                                                                    * cur_s2_tile
                                                                ).min(cur_s2_tile),
                                                                d_n,
                                                            ],
                                                        )

                                                        kn[:] = pypto.cast(
                                                            cur_kn_fp32, dtype
                                                        )
                                                        vj[:] = pypto.cast(
                                                            cur_kn_fp32, dtype
                                                        )
                                                    else:
                                                        pypto.set_cube_tile_shapes(
                                                            c1_tile[0],
                                                            c1_tile[1],
                                                            c1_tile[2],
                                                            False,
                                                        )
                                                        kn[:] = gather_in_l1(
                                                            k_nope_2d,
                                                            offset_view,
                                                            d_n,
                                                            True,
                                                            True,
                                                        )

                                                    pypto.set_cube_tile_shapes(
                                                        c1_tile[0],
                                                        c1_tile[1],
                                                        c1_tile[2],
                                                        False,
                                                    )
                                                    kr = gather_in_l1(
                                                        k_rope_2d,
                                                        offset_view,
                                                        d_r,
                                                        True,
                                                        True,
                                                    )

                                                    kj = pypto.tensor(
                                                        [cur_s2_tile, d_n + d_r],
                                                        dtype,
                                                        "kj",
                                                    )
                                                    pypto.assemble(kn, [0, 0], kj)
                                                    pypto.assemble(kr, [0, d_n], kj)

                                                    kj_view = pypto.view(
                                                        kj,
                                                        [cur_s2_tile, d_n + d_r],
                                                        [0, 0],
                                                        valid_shape=[
                                                            (
                                                                cur_seq
                                                                - s2_idx * cur_s2_tile
                                                            ).min(cur_s2_tile),
                                                            d_n + d_r,
                                                        ],
                                                    )
                                                    sij = pypto.matmul(
                                                        qi,
                                                        kj_view,
                                                        pypto.DT_FP32,
                                                        b_trans=True,
                                                    )

                                                    pypto.set_semantic_label(
                                                        "Sa_Qkvec1"
                                                    )
                                                    pypto.set_vec_tile_shapes(
                                                        v1_tile[0],
                                                        v1_tile[1],
                                                    )

                                                    sij_scale = (
                                                        sij * params.softmax_scale
                                                    )

                                                    tilda_mij_reduce = pypto.amax(
                                                        sij_scale, -1, True
                                                    )

                                                    tilda_mij = pypto.reshape(
                                                        tilda_mij_reduce,
                                                        [1, cur_g_tile],
                                                    )
                                                    tilda_mij.name = "tildaMij"

                                                    tsub = sij_scale - tilda_mij_reduce

                                                    tilda_pij = pypto.exp(tsub)
                                                    tilda_pij_f16 = pypto.cast(
                                                        tilda_pij, dtype
                                                    )

                                                    tilda_lij_reduce = pypto.sum(
                                                        tilda_pij, -1, True
                                                    )
                                                    tilda_lij = pypto.reshape(
                                                        tilda_lij_reduce,
                                                        [1, cur_g_tile],
                                                    )
                                                    tilda_lij.name = "tildaLij"

                                                    pypto.set_semantic_label("Sa_KvMm")
                                                    pypto.set_cube_tile_shapes(
                                                        c2_tile[0],
                                                        c2_tile[1],
                                                        c2_tile[2],
                                                        False,
                                                    )
                                                    pypto.set_matrix_size(
                                                        [
                                                            tilda_pij_f16.shape[0],
                                                            tilda_pij_f16.shape[1],
                                                            kn.shape[1],
                                                        ]
                                                    )

                                                    q1 = pypto.tensor(
                                                        [1, d_n],
                                                        dtype,
                                                        "q1",
                                                    )
                                                    if kn_dtype == pypto.DT_INT8:
                                                        q1[:] = pypto.matmul(
                                                            tilda_pij_f16,
                                                            vj,
                                                            pypto.DT_FP32,
                                                        )
                                                    else:
                                                        vj[:] = gather_in_l1(
                                                            k_nope_2d,
                                                            offset_view,
                                                            d_n,
                                                            True,
                                                            False,
                                                        )
                                                        q1[:] = pypto.matmul(
                                                            tilda_pij_f16,
                                                            vj,
                                                            pypto.DT_FP32,
                                                        )

                                                    if pypto.cond(
                                                        pypto.is_loop_begin(s2_idx)
                                                    ):

                                                        def inside_if_loop_begin():
                                                            nonlocal oi_update, li_update, mi_update

                                                            oi_tmp = q1
                                                            pypto.set_vec_tile_shapes(
                                                                v2_tile[0],
                                                                v2_tile[1],
                                                            )

                                                            if pypto.cond(
                                                                pypto.is_loop_end(
                                                                    s2_idx
                                                                )
                                                            ):
                                                                pypto.set_semantic_label(
                                                                    "Sa_KvVec2"
                                                                )
                                                                oi_update[:] = (
                                                                    oi_tmp
                                                                    / tilda_lij_reduce
                                                                )
                                                                pypto.set_vec_tile_shapes(
                                                                    1,
                                                                    1,
                                                                    v2_tile[0],
                                                                    v2_tile[1],
                                                                )

                                                                oi_update_4dim = pypto.cast(
                                                                    pypto.reshape(
                                                                        oi_update,
                                                                        [
                                                                            1,
                                                                            1,
                                                                            cur_g_tile,
                                                                            d_n,
                                                                        ],
                                                                    ),
                                                                    dtype,
                                                                )
                                                                pypto.assemble(
                                                                    oi_update_4dim,
                                                                    oi_offset,
                                                                    attention_out,
                                                                )
                                                            else:
                                                                oi_update[:] = oi_tmp

                                                            pypto.set_vec_tile_shapes(
                                                                v2_tile[0],
                                                                v2_tile[1],
                                                            )
                                                            li_update[:] = pypto.clone(
                                                                tilda_lij
                                                            )
                                                            mi_update[:] = pypto.clone(
                                                                tilda_mij
                                                            )

                                                        inside_if_loop_begin()
                                                    else:

                                                        def inside_else_loop_begin():
                                                            nonlocal oi_update, li_update, mi_update
                                                            pypto.set_semantic_label(
                                                                "Sa_UpdateVec2"
                                                            )

                                                            oi = oi_update
                                                            li = li_update
                                                            mi = mi_update

                                                            pypto.set_vec_tile_shapes(
                                                                v2_tile[0],
                                                                v2_tile[1],
                                                            )

                                                            mi_new = pypto.maximum(
                                                                mi, tilda_mij
                                                            )
                                                            t1 = mi - mi_new
                                                            t2 = pypto.exp(t1)
                                                            t3 = tilda_mij - mi_new
                                                            t4 = pypto.exp(t3)
                                                            t5 = t4 * tilda_lij
                                                            t6 = t2 * li
                                                            li_new = t6 + t5

                                                            q3 = oi * pypto.reshape(
                                                                t2,
                                                                [cur_g_tile, 1],
                                                            )

                                                            pypto.set_vec_tile_shapes(
                                                                v2_tile[0],
                                                                v2_tile[1],
                                                            )
                                                            q2 = q1 * pypto.reshape(
                                                                t4,
                                                                [cur_g_tile, 1],
                                                            )

                                                            oi_tmp = q3 + q2

                                                            if pypto.cond(
                                                                pypto.is_loop_end(
                                                                    s2_idx
                                                                )
                                                            ):

                                                                oi_update[:] = (
                                                                    oi_tmp
                                                                    / pypto.reshape(
                                                                        li_new,
                                                                        [
                                                                            cur_g_tile,
                                                                            1,
                                                                        ],
                                                                    )
                                                                )
                                                                pypto.set_vec_tile_shapes(
                                                                    1,
                                                                    1,
                                                                    v2_tile[0],
                                                                    v2_tile[1],
                                                                )

                                                                oi_update_4dim = pypto.cast(
                                                                    pypto.reshape(
                                                                        oi_update,
                                                                        [
                                                                            1,
                                                                            1,
                                                                            cur_g_tile,
                                                                            d_n,
                                                                        ],
                                                                    ),
                                                                    dtype,
                                                                )
                                                                pypto.assemble(
                                                                    oi_update_4dim,
                                                                    oi_offset,
                                                                    attention_out,
                                                                )
                                                            else:
                                                                oi_update[:] = oi_tmp

                                                            li_update[:] = li_new
                                                            mi_update[:] = mi_new

                                                        inside_else_loop_begin()

                                                inside_s2_idx_loop(
                                                    b_idx,
                                                    s1_idx,
                                                    s2_idx,
                                                    bn_per_batch,
                                                    cur_g_tile,
                                                    cur_offset,
                                                    oi_offset,
                                                )

                                        inside_g_idx_loop(
                                            b_idx,
                                            s1_idx,
                                            n2_idx,
                                            g_idx,
                                            bn_per_batch,
                                        )

                                inside_n2_idx_loop(b_idx, s1_idx, n2_idx, bn_per_batch)

                        inside_s1_idx_loop(b_idx, s1_idx)

                inside_b_idx_loop(b_idx)

        inside_main_function()


@dataclass
class SABuildConfigV2:
    b: int = NUM_32
    s1: int = NUM_4
    n_q: int = NUM_128
    n_kv: int = NUM_1
    qk_nope_head_dim: int = NUM_512
    qk_rope_head_dim: int = NUM_64
    kv_head_dim: int = NUM_512

    block_num: int = NUM_4096
    block_size: int = NUM_128

    topk: int = NUM_2048
    softmax_scale: float = NUM_F1 / float(math.sqrt(NUM_512 + NUM_64))

    g_tile: int = NUM_128
    s_kv_tile: int = NUM_2048
    c1_tile: List[List[int]] = field(
        default_factory=lambda: [
            [NUM_128, NUM_128],
            [NUM_64, NUM_64],
            [NUM_256, NUM_256],
        ]
    )
    v1_tile: List[int] = field(default_factory=lambda: [NUM_16, NUM_256])
    c2_tile: List[List[int]] = field(
        default_factory=lambda: [
            [NUM_128, NUM_128],
            [NUM_128, NUM_128],
            [NUM_128, NUM_128],
        ]
    )
    v2_tile: List[int] = field(default_factory=lambda: [NUM_64, NUM_128])

    is_kn_quant: bool = False
    q_dtype: int = pypto.DT_BF16
    kv_dtype: int = pypto.DT_BF16
    k_nope_dtype: int = pypto.DT_BF16
    k_nope_scales_dtype: int = pypto.DT_FP32

    def __post_init__(self):
        if self.is_kn_quant:
            self.k_nope_dtype = pypto.DT_INT8
        else:
            self.k_nope_dtype = self.kv_dtype


def cfg_from_input_param(input_param: list) -> SABuildConfigV2:
    b, sq, nq, nkv, dn, dr, block_num, block_size, topk, is_kn_quant = input_param
    softmax_scale = NUM_F1 / float(math.sqrt(dn + dr))
    cfg = SABuildConfigV2(
        b=b,
        s1=sq,
        n_q=nq,
        n_kv=nkv,
        qk_nope_head_dim=dn,
        qk_rope_head_dim=dr,
        kv_head_dim=dn,
        block_num=block_num,
        block_size=block_size,
        topk=topk,
        softmax_scale=softmax_scale,
        g_tile=NUM_128,
        s_kv_tile=NUM_2048,
        c1_tile=[
            [NUM_128, NUM_128],
            [NUM_64, NUM_64],
            [NUM_256, NUM_256],
        ],
        v1_tile=[NUM_16, NUM_256],
        c2_tile=[
            [NUM_128, NUM_128],
            [NUM_128, NUM_128],
            [NUM_128, NUM_128],
        ],
        v2_tile=[NUM_64, NUM_128],
        is_kn_quant=bool(is_kn_quant),
    )
    return cfg


def build_selected_args_v2(cfg: SABuildConfigV2 = SABuildConfigV2()):
    d_type_q = cfg.q_dtype
    d_type_kv = cfg.kv_dtype
    d_type_kn = cfg.k_nope_dtype
    d_type_kn_scales = cfg.k_nope_scales_dtype
    i32 = pypto.DT_INT32

    q_nope_shape = [cfg.b * cfg.s1 * cfg.n_q, cfg.qk_nope_head_dim]
    q_rope_shape = [cfg.b * cfg.s1 * cfg.n_q, cfg.qk_rope_head_dim]

    kv_cache_len = cfg.block_num * cfg.block_size
    k_nope_2d_shape = [kv_cache_len, cfg.qk_nope_head_dim]
    k_rope_2d_shape = [kv_cache_len, cfg.qk_rope_head_dim]
    k_nope_scales_shape = [kv_cache_len, NUM_4]

    offsets_shape = [cfg.b * cfg.s1, cfg.n_kv * cfg.topk]
    kv_slc_act_seqs_shape = [cfg.b]

    attention_out_shape = [cfg.b, cfg.s1, cfg.n_q, cfg.qk_nope_head_dim]

    q_nope = pypto.tensor(q_nope_shape, d_type_q, "qNope")
    q_rope = pypto.tensor(q_rope_shape, d_type_q, "qRope")

    k_nope_2d = pypto.tensor(k_nope_2d_shape, d_type_kn, "kNope2D")
    k_rope_2d = pypto.tensor(k_rope_2d_shape, d_type_kv, "kRope2D")
    k_nope_scales = pypto.tensor(k_nope_scales_shape, d_type_kn_scales, "kNopeScales")

    offsets = pypto.tensor(offsets_shape, i32, "offsets")
    kv_slc_act_seqs = (
        pypto.tensor(k_slc_act_seqs_shape, i32, "kvSlcActSeqs")
        if (k_slc_act_seqs_shape := kv_slc_act_seqs_shape)
        else None
    )

    attention_out = pypto.tensor(attention_out_shape, d_type_q, "attentionOut")

    tile = SelectedAttentionTileConfigV2(
        g_tile=cfg.g_tile,
        s_kv_tile=cfg.s_kv_tile,
        c1_tile=cfg.c1_tile,
        v1_tile=cfg.v1_tile,
        c2_tile=cfg.c2_tile,
        v2_tile=cfg.v2_tile,
    )

    params = SASimpleParamsV2(
        n_q=cfg.n_q,
        n_kv=cfg.n_kv,
        softmax_scale=cfg.softmax_scale,
        topk=cfg.topk,
        tile=tile,
    )

    args = SAInputsV2(
        q_nope=q_nope,
        q_rope=q_rope,
        k_nope_2d=k_nope_2d,
        k_rope_2d=k_rope_2d,
        k_nope_scales=k_nope_scales,
        offsets=offsets,
        kv_slc_act_seqs=kv_slc_act_seqs,
        attention_out=attention_out,
        params=params,
    )

    meta = {
        "b": cfg.b,
        "s1": cfg.s1,
        "nQ": cfg.n_q,
        "nKv": cfg.n_kv,
        "blockNum": cfg.block_num,
        "blockSize": cfg.block_size,
        "dims": {
            "qNope": q_nope_shape,
            "qRope": q_rope_shape,
            "kNope2D": k_nope_2d_shape,
            "kRope2D": k_rope_2d_shape,
            "kNopeScales": k_nope_scales_shape,
            "offsets": offsets_shape,
            "kvSlcActSeqs": kv_slc_act_seqs_shape,
            "attentionOut": attention_out_shape,
        },
        "topk": cfg.topk,
        "softmaxScale": cfg.softmax_scale,
        "tiles": {
            "gTile": cfg.g_tile,
            "sKvTile": cfg.s_kv_tile,
            "c1Tile": cfg.c1_tile,
            "v1Tile": cfg.v1_tile,
            "c2Tile": cfg.c2_tile,
            "v2Tile": cfg.v2_tile,
        },
        "dtypes": {
            "q": cfg.q_dtype,
            "kv": cfg.kv_dtype,
            "kNope": cfg.k_nope_dtype,
            "kNopeScales": cfg.k_nope_scales_dtype,
        },
        "isKnQuant": cfg.is_kn_quant,
    }

    return args, meta


def test_selected_attention_v2_from_input_param():
    logging.basicConfig(level=logging.INFO)
    set_config()
    input_param = [32, 1, 128, 1, 512, 64, 128, 128, 2048, 0]
    cfg = cfg_from_input_param(input_param)
    args, meta = build_selected_args_v2(cfg)
    logging.info({"Sanity": meta})
    selected_attention_compute_v2(args)
    assert True


def test_selected_attention_v2_from_input_param_quant():
    logging.basicConfig(level=logging.INFO)
    set_config()
    input_param = [32, 1, 128, 1, 512, 64, 128, 128, 2048, 1]
    cfg = cfg_from_input_param(input_param)
    args, meta = build_selected_args_v2(cfg)
    logging.info({"SanityQuant": meta})
    selected_attention_compute_v2(args)
    assert True