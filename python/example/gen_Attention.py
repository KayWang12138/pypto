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
import pto
from typing import List


ONLY_CODEGEN = "only_codegen"
MACHINE_SCHED_MODE = "machine_sched_mode"


NUM_3 = 3
NUM_8 = 8
NUM_16 = 16
NUM_128 = 128
NUM_512 = 512

TILE_VEC_FOUR_DIMS = 4


@dataclass
class GenAttenTileShapeConfig:
    tile_b_size: int
    tile_s1_size: int
    vec1_tile_shape: List[int]
    vec2_tile_shape: List[int]


def gen_attention_compute(**kwargs):
    cmp_atten = kwargs.get("cmp_atten")
    sel_atten = kwargs.get("sel_atten")
    win_atten = kwargs.get("win_atten")
    gating_score = kwargs.get("gating_score")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")

    n_dim_size = cmp_atten.shape[2]
    d_dim_size = cmp_atten.shape[3]
    d_gate_dim_size = gating_score.shape[3]
    tile_b = tile_config.tile_b_size
    tile_s = tile_config.tile_s1_size
    v1_tile = tile_config.vec1_tile_shape
    v2_tile = tile_config.vec2_tile_shape

    b_dim_size = cmp_atten.shape[0]
    s_dim_size = cmp_atten.shape[1]
    b_loop = b_dim_size // tile_b
    s_loop = s_dim_size // tile_s
    d_type = cmp_atten.dtype
    for b_idx in pto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="b_idx"):
        def loop_l0_b_idx(b_idx):
            b_offset = b_idx * tile_b
            actual_b_size = (b_dim_size - b_idx * tile_b).min(tile_b)
            for s_idx in pto.loop(0, s_loop, 1, name="LOOP_L1_sIdx", idx_name="s_idx"):
                def loop_l1_s_idx(s_idx):
                    s_offset = s_idx * tile_s
                    out_offset = [b_offset, s_offset, 0, 0]
                    actuals_s_size = (s_dim_size - s_idx * tile_s).min(tile_s)
                    pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1], v1_tile[2], v1_tile[3])
                    cmp_atten_tile = pto.view(cmp_atten, [tile_b, tile_s, n_dim_size, d_dim_size],
                        [b_offset, s_offset, 0, 0], 
                        valid_shape=[actual_b_size, actuals_s_size, n_dim_size, d_dim_size])
                    sel_atten_tile = pto.view(sel_atten, [tile_b, tile_s, n_dim_size, d_dim_size],
                        [b_offset, s_offset, 0, 0], 
                        valid_shape=[actual_b_size, actuals_s_size, n_dim_size, d_dim_size])
                    win_atten_tile = pto.view(win_atten, [tile_b, tile_s, n_dim_size, d_dim_size],
                        [b_offset, s_offset, 0, 0], 
                        valid_shape=[actual_b_size, actuals_s_size, n_dim_size, d_dim_size])
                    cmp_atten_fp32_tile = pto.cast(cmp_atten_tile, pto.DataType.DT_FP32)
                    sel_atten_fp32_tile = pto.cast(sel_atten_tile, pto.DataType.DT_FP32)
                    win_atten_fp32_tile = pto.cast(win_atten_tile, pto.DataType.DT_FP32)
                    pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1], v2_tile[2], v2_tile[3])
                    gating_score_tile = pto.view(gating_score, [tile_b, tile_s, n_dim_size, d_gate_dim_size],
                        [b_offset, s_offset, 0, 0], 
                        valid_shape=[actual_b_size, actuals_s_size, n_dim_size, d_gate_dim_size])
                    gating_score_fp32 = pto.cast(gating_score_tile, pto.DataType.DT_FP32)
                    cmp_weight = pto.view(gating_score_fp32, [tile_b, tile_s, n_dim_size, 1], [0, 0, 0, 0])
                    sel_weight = pto.view(gating_score_fp32, [tile_b, tile_s, n_dim_size, 1], [0, 0, 0, 1])
                    win_weight = pto.view(gating_score_fp32, [tile_b, tile_s, n_dim_size, 1], [0, 0, 0, 2])
                    pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1], v1_tile[2], v1_tile[3])
                    mul_cmp = pto.mul(cmp_atten_fp32_tile, cmp_weight)
                    mul_sel = pto.mul(sel_atten_fp32_tile, sel_weight)
                    mul_win = pto.mul(win_atten_fp32_tile, win_weight)
                    add_cmp_sel = pto.add(mul_cmp, mul_sel)
                    out_fp32 = pto.add(add_cmp_sel, mul_win)
                    pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1], v1_tile[2], v1_tile[3])
                    attention_out_tile = pto.cast(out_fp32, d_type)
                    pto.assemble(attention_out_tile, out_offset, attention_out)
                loop_l1_s_idx(s_idx)
        loop_l0_b_idx(b_idx)


def gen_attention(**kwargs):
    cmp_atten = kwargs.get("cmp_atten")
    sel_atten = kwargs.get("sel_atten")
    win_atten = kwargs.get("win_atten")
    gating_score = kwargs.get("gating_score")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")

    input_tensors = [cmp_atten, sel_atten, win_atten, gating_score]
    output_tensor = [attention_out]
    with pto.function("main", input_tensors, output_tensor):
        gen_attention_compute(
            cmp_atten=cmp_atten,
            sel_atten=sel_atten,
            win_atten=win_atten,
            gating_score=gating_score,
            attention_out=attention_out,
            tile_config=tile_config
        )


if __name__ == '__main__':
    b = NUM_16
    n = NUM_128
    s1 = 1
    d = NUM_512
    d_type = pto.DataType.DT_BF16
    shape_cmp_atten = [b, s1, n, d]
    shape_sel_atten = [b, s1, n, d]
    shape_win_atten = [b, s1, n, d]
    shape_gating_score = [b, s1, n, NUM_3]
    shape_attention_out = [b, s1, n, d]
    cmp_atten = pto.tensor(shape_cmp_atten, d_type, "cmp_atten")
    sel_atten = pto.tensor(shape_sel_atten, d_type, "sel_atten")
    win_atten = pto.tensor(shape_win_atten, d_type, "win_atten")
    gating_score = pto.tensor(shape_gating_score, d_type, "gating_score")
    attention_out = pto.tensor(shape_attention_out, d_type, "attention_out")
    
    d_tile_size = NUM_512
    n_tile_size = NUM_128
    tile_config = GenAttenTileShapeConfig(
        tile_b_size=NUM_8,
        tile_s1_size=1,
        vec1_tile_shape=[1, 1, NUM_16, d_tile_size],
        vec2_tile_shape=[1, 1, n_tile_size, NUM_3]
    )
    gen_attention(
        cmp_atten=cmp_atten,
        sel_atten=sel_atten,
        win_atten=win_atten,
        gating_score=gating_score,
        attention_out=attention_out,
        tile_config=tile_config
    )