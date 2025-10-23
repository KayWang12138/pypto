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
import math
from dataclasses import dataclass
from typing import Tuple, List
import pto

TILE_VEC_DIMS = 2
TILE_CUBE_DIMS = 6


@dataclass
class IfaTileShapeConfig:
    block_size: int
    head_num_q_tile: int
    v0_tile_shape: list[int]
    c1_tile_shape: list[int]
    v1_tile_shape: list[int]
    c2_tile_shape: list[int]
    v2_tile_shape: list[int]


def incre_flash_attention(**kwargs):
    q_nope = kwargs.get("q_nope")
    k_nope_cache = kwargs.get("k_nope_cache")
    v_nope_cache = kwargs.get("v_nope_cache")
    q_rope = kwargs.get("q_rope")
    k_rope_cache = kwargs.get("k_rope_cache")
    block_table = kwargs.get("block_table")
    act_seqs = kwargs.get("act_seqs")
    softmax_scale = kwargs.get("softmax_scale")
    attention_out = kwargs.get("attention_out")
    tile_config = kwargs.get("tile_config")

    batch_size = len(block_table) # assert batch_size == len(act_seqs)
    d_n = q_nope.shape[1]
    d_r = q_rope.shape[1]
    n_q = q_nope.shape[0] // batch_size
    n_tile = tile_config.head_num_q_tile
    block_size = tile_config.block_size
    n_loop = math.ceil(n_q / n_tile)
    v0_tile = tile_config.v0_tile_shape
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    aggregation: List[Tuple[pto.tensor, List[int]]] = []
    tiled_out: List[pto.tensor] = []

    for b_idx in range(batch_size):
        cur_seq = act_seqs[b_idx]
        bn_per_batch = math.ceil(cur_seq / block_size)
        for n_idx in range(n_loop):
            oi_update = pto.tensor()
            li_update = pto.tensor()
            mi_update = pto.tensor()

            n_tile_cur = min(n_tile, n_q - n_idx * n_tile)
            cur_offset = b_idx * n_q + n_idx * n_tile

            pto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
            qn = pto.view(q_nope, [n_tile_cur, d_n], [int(cur_offset), 0])
            qr = pto.view(q_rope, [n_tile_cur, d_r], [int(cur_offset), 0])
            qi = pto.assemble([[qn, [0, 0]], [qr, [0, d_n]]])
            for bn in range(bn_per_batch):
                cur_block_idx = block_table[b_idx][bn]
                s2_tile_cur = min(block_size, cur_seq - bn * block_size)
                pto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
                kn = pto.view(k_nope_cache, [s2_tile_cur, d_n], [cur_block_idx * block_size, 0])
                kr = pto.view(k_rope_cache, [s2_tile_cur, d_r], [cur_block_idx * block_size, 0])
                kj = pto.assemble([[kn, [0, 0]], [kr, [0, d_n]]])
                vj = pto.view(v_nope_cache, [s2_tile_cur, d_n], [cur_block_idx * block_size, 0])

                pto.set_cube_tile_shapes(
                    [c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]], True)

                # (nTileCur, dN+dR), (s2TileCur, dN+dR) -> (nTileCur, s2TileCur)
                pto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
                sij = pto.matmul(pto.DT_FP32, qi, kj, False, True)

                pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                sij_scale = pto.mul_s(sij, pto.element(pto.DT_FP32, softmax_scale)) # (nTileCur, s2TileCur)
                tilda_mij = pto.row_max_single(sij_scale);   # (nTileCur, s2TileCur) -> (nTileCur, 1)
                tsub = pto.sub(sij_scale, tilda_mij); # (nTileCur, s2TileCur) - (nTileCur, 1) -> (nTileCur, s2TileCur)
                tilda_pij = pto.exp(tsub)
                tilda_pij_f16 = pto.cast(tilda_pij, pto.DT_BF16)
                tilda_lij = pto.row_sum_single(tilda_pij); # (nTileCur, s2TileCur) -> (nTileCur, 1)

                if bn == 0:
                    pto.set_cube_tile_shapes(
                        [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]], True)
                    pto.set_matrix_size(
                        [tilda_pij_f16.shape[0], tilda_pij_f16.shape[1], vj.shape[1]])
                    oi_tmp = pto.matmul(pto.DT_FP32, tilda_pij_f16, vj, False, False)

                    pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                    oi_update = pto.div(oi_tmp, tilda_lij) if bn_per_batch == 1 else oi_tmp
                    li_update = tilda_lij
                    mi_update = tilda_mij
                    continue

                oi = oi_update
                li = li_update
                mi = mi_update

                mi_new = pto.maximum(mi, tilda_mij) # (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                t1 = pto.sub(mi, mi_new)           # (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                t2 = pto.exp(t1)
                t3 = pto.sub(tilda_mij, mi_new)  # (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                t4 = pto.exp(t3)
                t5 = pto.mul(t4, tilda_lij) # (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                t6 = pto.mul(t2, li)       # (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                li_new = pto.add(t6, t5)    # (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                q3 = pto.mul(oi, t2) # (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                pto.set_cube_tile_shapes(
                    [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]], True)
                # (nTileCur, s2TileCur), (s2TileCur, dN) -> (nTileCur, dN)
                pto.set_matrix_size(
                    [tilda_pij_f16.shape[0], tilda_pij_f16.shape[1], vj.shape[1]])
                q1 = pto.matmul(pto.DT_FP32, tilda_pij_f16, vj, False, False)
                pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                q2 = pto.mul(q1, t4);    # (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                oi_tmp = pto.add(q3, q2); # (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                oi_update = pto.div(oi_tmp, li_new) if bn == bn_per_batch - 1 else oi_tmp
                li_update = li_new
                mi_update = mi_new
            tiled_out.append(oi_update)
    attention_out[:] = pto.concat(tiled_out, 0)

if __name__ == "__main__":
    s2 = 256
    sq = 1
    nq = 32
    dn = 512
    nkv = 1
    dr = 64
    block_size = 256
    b = 4
    tile_config = IfaTileShapeConfig(
        block_size=256,
        head_num_q_tile=32,
        v0_tile_shape=[256, 128],
        c1_tile_shape=[32, 32, 256, 256, 128, 128],
        v1_tile_shape=[32, 256],
        c2_tile_shape=[32, 32, 256, 256, 128, 128],
        v2_tile_shape=[32, 256]
    )
    act_seqs = [s2] * b
    block_num = 0
    for s in act_seqs:
        block_num += math.ceil(s / block_size)

    q_nope = pto.tensor([b * sq * nq, dn], pto.DT_BF16, "q_nope")
    q_rope = pto.tensor([b * sq * nq, dr], pto.DT_BF16, "q_rope")
    kv_nope_cache = pto.tensor([block_num * block_size * nkv, dn], pto.DT_BF16,
                            "k_nope_cache", pto.TileOpFormat.TILEOP_NZ)
    k_rope_cache = pto.tensor([block_num * block_size * nkv, dr], pto.DT_BF16,
                            "k_rope", pto.TileOpFormat.TILEOP_NZ)

    max_seq_all_batch = max(act_seqs)
    max_block_num_per_batch = math.ceil(max_seq_all_batch / block_size)
    block_table = [[0] * max_block_num_per_batch for _ in range(b)]

    for i in range(b):
        for j in range(max_block_num_per_batch):
            block_table[i][j] = i * max_block_num_per_batch + j

    softmax_scale = float(1.0 / math.sqrt(512 + 64))
    attention_out = pto.tensor([b * sq * nq, dn], pto.DT_BF16, "attention_out")

    graph_t = pto.GraphType.TENSOR_GRAPH
    func_t = pto.FunctionType.STATIC
    with pto.pto_function("incre_flash_attention", graph_t, func_t):
        incre_flash_attention(q_nope=q_nope, k_nope_cache=kv_nope_cache, v_nope_cache=kv_nope_cache,
            q_rope=q_rope, k_rope_cache=k_rope_cache, block_table=block_table, act_seqs=act_seqs,
            softmax_scale=softmax_scale, attention_out=attention_out, tile_config=tile_config)