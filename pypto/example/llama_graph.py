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
from utils import pto_function

T_SHAPE = 128
NUM_64 = 64
NUM_128 = 128
F_1 = 1.0
F_NEGA_1 = -1.0


@dataclass
class AttentionDims:
    b: int
    n: int
    s: int
    d: int
    single_m: int
    single_n: int


@dataclass
class AttentionVecTileConfig:
    default_vec_tile_x: int
    default_vec_tile_y: int
    softmax_tile_x: int
    softmax_tile_y: int
    update_tile_x: int
    update_tile_y: int
    cast_tile_x: int
    cast_tile_y: int


@dataclass
class AttentionCubeTileConfig:
    c1_l1_m: int
    c1_l1_k: int
    c1_l1_n: int
    c2_l1_m: int
    c2_l1_k: int
    c2_l1_n: int
    c1_l0: int = 128
    c2_l0: int = 128


@dataclass
class KeyConfig:
    max: int
    min: int
    db_type: int
    n_buffer: int
    is_partition_cv: int
    c_tile_x: int
    c_tile_y: int


DFS_VEC_CFG = AttentionVecTileConfig(128, 128, 16, 128, 16, 128, 32, 128)
SMALL_DFS_VEC_CFG = AttentionVecTileConfig(64, 128, 16, 128, 16, 128, 32, 128)
DFS_CUBE_CFG = AttentionCubeTileConfig(128, 128, 128, 128, 128, 128)

OOO_VEC_CFG = AttentionVecTileConfig(64, 128, 16, 512, 32, 128, 32, 128)
OOO_CUBE_CFG = AttentionCubeTileConfig(128, 128, 512, 64, 256, 64, 128, 64)

DFT_BASIC_CFG = KeyConfig(8192, 1024, 1, 1, 0, 128, 128)
DFT_SINGLE_M = 128
DFT_SINGLE_N = 128


def set_c1_cube_config(cube_cfg: AttentionCubeTileConfig):
    pto.set_cube_tile_shapes(
        [cube_cfg.c1_l0, cube_cfg.c1_l1_m],
        [cube_cfg.c1_l0, cube_cfg.c1_l1_k],
        [cube_cfg.c1_l0, cube_cfg.c1_l1_n],
    )


def set_c2_cube_config(cube_cfg: AttentionCubeTileConfig):
    pto.set_cube_tile_shapes(
        [cube_cfg.c2_l0, cube_cfg.c2_l1_m],
        [cube_cfg.c2_l0, cube_cfg.c2_l1_k],
        [cube_cfg.c2_l0, cube_cfg.c2_l1_n],
    )


def set_default_l0_cube_config():
    pto.set_cube_tile_shapes([T_SHAPE, T_SHAPE], [T_SHAPE, T_SHAPE], [T_SHAPE, T_SHAPE])


def flash_attention(
    q: pto.tensor,
    k: pto.tensor,
    v: pto.tensor,
    m: pto.tensor,
    l: pto.tensor,
    at_dims: AttentionDims,
    vec_cfg: AttentionVecTileConfig,
    cube_cfg: AttentionCubeTileConfig,
) -> None:
    # m, l are not used
    dim0 = q.shape[0]
    dim1 = q.shape[1]
    b = at_dims.b
    n = at_dims.n
    s = dim0 // b
    d = dim1 // n
    single_m = at_dims.single_m
    assert single_m == 128
    single_n = at_dims.single_n
    s1_loop = s // single_m
    s2_loop = s // single_n

    print(f"FlashAttention, B, N, S, D --------{b},{n},{s},{d}")
    print(f"s1Loop, s2Loop -------{s1_loop},{s2_loop},")

    bns = at_dims.b * at_dims.n * at_dims.s
    shape_reduce = [bns, 1]
    # max and sum are not used in this function
    last_oi = {}
    last_mi = {}
    last_li = {}
    result = None

    for b_idx in range(b):
        for n_idx in range(n):
            for s2_idx in range(s2_loop):
                kj = pto.view(
                    k, [single_n, d], [b_idx * s + s2_idx * single_n, n_idx * d]
                )
                vj = pto.view(
                    v, [single_n, d], [b_idx * s + s2_idx * single_n, n_idx * d]
                )

                for s1_idx in range(s1_loop):
                    print(f"inner fa {s2_idx} {s1_idx} {s2_loop} {s1_loop}")
                    qi = pto.view(
                        q, [single_m, d], [b_idx * s + s1_idx * single_m, n_idx * d]
                    )
                    oi_offset = (b_idx * s + s1_idx * single_m, n_idx * d)
                    li_offset = ((b_idx * n + n_idx) * s + s1_idx * single_m, 0)
                    mi_offset = ((b_idx * n + n_idx) * s + s1_idx * single_m, 0)
                    set_c1_cube_config(cube_cfg)
                    sij = pto.matmul(pto.DT_FP32, qi, kj, b_trans=True)

                    pto.set_vec_tile_shapes(
                        vec_cfg.softmax_tile_x, vec_cfg.softmax_tile_y
                    )

                    tilda_mij = pto.row_max_single(sij)
                    tsub = pto.sub(sij, tilda_mij)
                    tilda_pij = pto.exp(tsub)
                    tilda_pij_f16 = pto.cast(tilda_pij, pto.DT_FP16)
                    tilda_lij = pto.row_sum_single(tilda_pij)

                    set_c2_cube_config(cube_cfg)

                    if s2_idx == 0:
                        oi_tmp = pto.matmul(pto.DT_FP32, tilda_pij_f16, vj)
                        if s2_loop == 1:
                            li_expand = pto.reciprocal(tilda_lij)
                            last_oi[oi_offset] = pto.mul(oi_tmp, li_expand)
                        else:
                            last_oi[oi_offset] = oi_tmp
                        last_li[li_offset] = tilda_lij
                        last_mi[mi_offset] = tilda_mij
                        continue

                    assert oi_offset in last_oi
                    assert li_offset in last_li
                    assert mi_offset in last_mi
                    oi = last_oi[oi_offset]
                    li = last_li[li_offset]
                    mi = last_mi[mi_offset]

                    mi_new = pto.maximum(mi, tilda_mij)
                    t1 = pto.sub(mi, mi_new)
                    t2 = pto.exp(t1)
                    t3 = pto.sub(tilda_mij, mi_new)
                    t4 = pto.exp(t3)
                    t5 = pto.mul(t4, tilda_lij)
                    t6 = pto.mul(t2, li)
                    li_new = pto.add(t6, t5)

                    q3 = pto.mul(oi, t2)
                    q1 = pto.matmul(pto.DT_FP32, tilda_pij_f16, vj)
                    q2 = pto.mul(q1, t4)
                    oi_tmp = pto.add(q3, q2)
                    if s2_idx == s2_loop - 1:
                        last_oi[oi_offset] = pto.mul(oi_tmp, pto.reciprocal(li_new))
                    else:
                        last_oi[oi_offset] = oi_tmp
                    last_li[li_offset] = li_new
                    last_mi[mi_offset] = mi_new

        aggregation = []
        for offset, tensor in last_oi.items():
            aggregation.append((tensor, list(offset)))
        result = pto.assemble(aggregation)
        assert result.shape[0] == b * s
        assert result.shape[1] == n * d
    return result


def multi_attention(
    hidden_states,
    weight,
    m,
    l,
    at_dims: AttentionDims,
    vec_cfg: AttentionVecTileConfig,
    cube_cfg: AttentionCubeTileConfig,
):
    x = pto.cast(hidden_states, pto.DT_FP16)
    qkv = pto.matmul(pto.DT_FP16, x, weight)
    q = pto.view(qkv, hidden_states.shape, [0, 0])
    k = pto.view(qkv, hidden_states.shape, [0, hidden_states.shape[1]])
    v = pto.view(qkv, hidden_states.shape, [0, hidden_states.shape[1] * 2])
    result = flash_attention(q, k, v, m, l, at_dims, vec_cfg, cube_cfg)
    return result


def llama_layer(
    hidden_states,
    attn_weight,
    dense_weight,
    ffn_weight,
    at_dims: AttentionDims,
    vec_cfg: AttentionVecTileConfig,
    cube_cfg: AttentionCubeTileConfig,
):
    pto.set_vec_tile_shapes(vec_cfg.default_vec_tile_x, vec_cfg.default_vec_tile_y)
    set_default_l0_cube_config()
    shape = hidden_states.shape
    residual = hidden_states
    hidden_states = pto.rms_norm(hidden_states)

    bns = at_dims.b * at_dims.n * at_dims.s
    shape_reduce = [bns, 1]
    # max and sum are not used

    m = pto.tensor(pto.DT_FP32, shape_reduce, "m_temp")
    l = pto.tensor(pto.DT_FP32, shape_reduce, "l_temp")
    attention_out = multi_attention(
        hidden_states, attn_weight, m, l, at_dims, vec_cfg, cube_cfg
    )

    attention_out_fp16 = pto.cast(attention_out, pto.DT_FP16)
    # Dense
    set_default_l0_cube_config()
    dense_out = pto.matmul(pto.DT_FP32, attention_out_fp16, dense_weight)
    pto.set_vec_tile_shapes(vec_cfg.default_vec_tile_x, vec_cfg.default_vec_tile_y)
    hidden_states = pto.add(residual, dense_out)

    # Fully Connected
    residual = hidden_states
    hidden_states = pto.rms_norm(hidden_states)

    mlp_res = pto.tensor(pto.DT_FP32, shape, "tmp")

    a = pto.cast(hidden_states, pto.DT_FP16)
    gate = pto.matmul(pto.DT_FP32, a, ffn_weight)

    # swish: x / (1 + e^(-x))
    swish = pto.mul_s(gate, pto.element(pto.DT_FP32, F_NEGA_1))
    swish = pto.exp(swish)
    swish = pto.add_s(swish, pto.element(pto.DT_FP32, F_1))
    swish = pto.div(gate, swish)

    # up_proj
    up = pto.matmul(pto.DT_FP32, a, ffn_weight)
    swish = pto.mul(swish, up)
    swish_fp16 = pto.cast(swish, pto.DT_FP16)

    # down_proj
    mlp_res = pto.matmul(pto.DT_FP32, swish_fp16, ffn_weight, b_trans=True)
    hidden_states = pto.add(residual, mlp_res)
    return hidden_states


if __name__ == "__main__":
    # pto.reset_config()
    dims_cfg = AttentionDims(1, 1, 128, 128, DFT_SINGLE_M, DFT_SINGLE_N)
    b = dims_cfg.b
    n = dims_cfg.n
    s = dims_cfg.s
    d = dims_cfg.d
    H = pto.tensor(pto.DT_FP32, [b * s, n * d], "H")
    AW = pto.tensor(pto.DT_FP16, [n * d, n * d * 3], "AW")
    DW = pto.tensor(pto.DT_FP16, [n * d, n * d], "DW")
    FW = pto.tensor(pto.DT_FP16, [n * d, n * d * 3], "FW")
    res = pto.tensor(pto.DT_FP32, [b * s, n * d], "Res")

    graph_t = pto.graph_type.TENSOR_GRAPH
    func_t = pto.function_type.STATIC
    with pto_function("LLAMA", graph_t, func_t, H, AW, DW, FW, res):
        res = llama_layer(H, AW, DW, FW, dims_cfg, SMALL_DFS_VEC_CFG, DFS_CUBE_CFG)

    # NOTE: or set `GLOBAL_LOG_LEVEL=1` to dump
    print(pto.dump())
