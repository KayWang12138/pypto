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

# RuntimeConfig KEYS
SG_CYCLE_UPPER_BOUND = "cycle_upper_bound"
L1_REUSE = "l1_reuse"
CUBE_NBUFFER_MAP = "cube_nbuffer_map"
COPYIN_THRESHOLD = "copyin_threshold"
NBUFFER_MERGE_MODE = "nbuffer_merge_mode"
KEY_SUPPORT_DYNAMIC_UNALIGNED = "SUPPORT_DYNAMIC_UNALIGNED"
TILE_VEC_DIMS = 2
TILE_CUBE_DIMS = 6
SHAPE_DIM2 = 2
SHAPE_DIM3 = 3
SHAPE_DIM4 = 4
SHAPE_DIM5 = 5


NUM_65536 = 65536
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_5 = 5
NUM_8 = 8
NUM_16 = 16
NUM_20 = 20
NUM_32 = 32
NUM_64 = 64
NUM_128 = 128
NUM_256 = 256
NUM_512 = 512
NUM_1024 = 1024
NUM_1536 = 1536
NUM_7168 = 7168
NUM_4096 = 4096
DF_1E_5 = 1e-5


class GateMode(Enum):
    standard = 0
    simple = 1


class SimpleParams:
    b: int
    s: int
    s2: int
    d: int
    m: int
    k: int
    n: int
    n2: int
    right: int
    h: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    q_head_dim: int
    cache_mode: str
    block_size: int
    vec_tile = []
    cube_m_tile = []
    cube_k_tile = []
    cube_n_tile = []
    tile_b: int


    @staticmethod
    def get_common_garams():
        params: SimpleParams = SimpleParams()
        params.n2 = NUM_1
        params.s = NUM_1
        params.h = NUM_7168 # 7168
        params.q_lora_rank = NUM_1536 # 1536
        params.kv_lora_rank = NUM_512 # 512
        params.qk_rope_head_dim = NUM_64 # 64
        params.qk_nope_head_dim = NUM_128 # 128
        params.q_head_dim = params.qk_rope_head_dim + params.qk_nope_head_dim
        params.cache_mode = "BNSD"
        params.block_size = NUM_128 # 128
        return params


    @staticmethod
    def get_low_params():
        params: SimpleParams = SimpleParams.get_common_garams()
        params.b = NUM_4
        params.n = NUM_32
        params.s2 = NUM_256
        return params


    @staticmethod
    def get_high_params():
        params: SimpleParams = SimpleParams.get_common_garams()
        params.b = NUM_32
        params.n = NUM_128
        params.s2 = NUM_4096
        return params


def gen_gated_score(**kwargs):
    x = kwargs.get("x")
    gate_w1 = kwargs.get("gate_w1")
    gate_w2 = kwargs.get("gate_w2")
    gate_sim_w1 = kwargs.get("gate_sim_w1")
    gating_score = kwargs.get("gating_score")
    gate_mode = kwargs.get("gate_mode")

    d_type = x.dtype

    b = x.shape[0]
    s = x.shape[1]
    h = x.shape[2]
    n1 = gate_w2.shape[1] // 3
    tile_b = b
    tile_s = s
    tile_bs = tile_b * tile_s

    b_loop = b // tile_b
    s_loop = s // tile_s
    for b_idx in pto.loop(0, b_loop, 1, name="LOOP_L0_bIdx_gated_score", idx_name="b_idx"):
        def insideb_idx_loop(b_idx):
            for s_idx in pto.loop(0, s_loop, 1, name="LOOP_L0_sIdx_gated_score", idx_name="s_idx"):
                def insides_idx_loop(s_idx):
                    pto.set_vec_tile_shapes(tile_b, tile_s, h)
                    pto.set_cube_tile_shapes([tile_bs, tile_bs], [NUM_128, NUM_128], [NUM_128, NUM_128])
                    b_ofs = b_idx * tile_b
                    s_ofs = s_idx * tile_s
                    bs_ofs = b_ofs * s_ofs

                    x_reshape = pto.reshape(x, [b * s, h])
                    x_view = pto.view(x_reshape, [tile_bs, h], [bs_ofs, 0])
                    mm1_res = pto.matmul(pto.DataType.DT_FP32, x_reshape, gate_w1)

                    pto.set_vec_tile_shapes(1, h)
                    sigmoid_res = pto.sigmoid(mm1_res)
                    sigmoid_res = pto.cast(sigmoid_res, d_type)
                    pto.set_cube_tile_shapes([tile_bs, tile_bs], [NUM_128, NUM_128], [NUM_16, NUM_16])
                    mm2_res = pto.matmul(pto.DataType.DT_FP32, sigmoid_res, gate_w2)
                    pto.set_vec_tile_shapes(tile_bs, n1)

                    res = pto.reshape(mm2_res, [tile_b, tile_s, 3, n1])
                    pto.set_vec_tile_shapes(1, tile_s, 3, n1)

                    res = pto.transpose(res, [2, 3])
                    if gating_score.dtype != pto.DataType.DT_FP32:
                        res = pto.cast(res, d_type)
                    pto.assemble(res, [b_ofs, s_idx, 0, 0], gating_score)
                insides_idx_loop(s_idx)
        insideb_idx_loop(b_idx)


def gen_gated_score_compute(**kwargs):
    x = kwargs.get("x")
    gate_w1 = kwargs.get("gate_w1")
    gate_w2 = kwargs.get("gate_w2")
    gate_sim_w1 = kwargs.get("gate_sim_w1")
    gating_score = kwargs.get("gating_score")
    gate_mode = kwargs.get("gate_mode")
    input_tensors = [x, gate_w1, gate_w2, gate_sim_w1]
    output_tensors = [gating_score]
    with pto.function("fused_compress_kv_select", input_tensors, output_tensors):
        gen_gated_score(
            x=x,
            gate_w1=gate_w1,
            gate_w2=gate_w2,
            gate_sim_w1=gate_sim_w1,
            gating_score=gating_score,
            gate_mode=gate_mode
        )


def test_nsa(params):
    b = params.b
    s = params.s
    n = params.n
    h = params.h

    d_type = pto.DataType.DT_FP16

    x_shape = [b, s, h]
    gate_w1_shape = [h, 4 * h]
    gate_w2_shape = [4 * h, 3 * n]
    gate_sim_w1_shape = [h, 3 * n]
    gating_score_shape = [b, s, n, 3]
    temp_shape = [b * s, n * 3]
    mm1_shape = [b * s, 4 * h]

    x = pto.tensor(x_shape, d_type, "x")
    gate_w1 = pto.tensor(gate_w1_shape, d_type, "gateW1")
    gate_w2 = pto.tensor(gate_w2_shape, d_type, "gateW2")
    gate_sim_w1 = pto.tensor(gate_sim_w1_shape, d_type, "gateSimW1")
    gating_score = pto.tensor(gating_score_shape, d_type, "gatingScore")

    gen_gated_score_compute(
        x=x,
        gate_w1=gate_w1,
        gate_w2=gate_w2,
        gate_sim_w1=gate_sim_w1,
        gating_score=gating_score,
        gate_mode=GateMode.standard,
        )


def main():
    params = SimpleParams.get_high_params()

    params.h = 128
    params.s = NUM_2

    test_nsa(params)


if __name__ == "__main__":
    main()

