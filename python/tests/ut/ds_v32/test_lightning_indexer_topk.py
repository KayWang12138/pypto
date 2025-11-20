# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import sys
from dataclasses import dataclass, field
from typing import List, Set, Optional
import logging
import pytest
import pto

SHAPE_DIM0 = 0
SHAPE_DIM1 = 1
SHAPE_DIM2 = 2
SHAPE_DIM3 = 3

NUM_NEG1 = -1
NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_8 = 8
NUM_16 = 16
NUM_32 = 32
NUM_64 = 64
NUM_100 = 100
NUM_128 = 128
NUM_1024 = 1024
NUM_1127 = 1127
NUM_2048 = 2048
NUM_4096 = 4096
NUM_8192 = 8192
AVOID_FP32_TO_FP16_OVERFLOW_SCALE = 1.0 / 2048.0


@dataclass
class LightningIndexerTileConfig:
    weight_tile: List[int]
    c1_tile: List[List[int]]
    v1_tile: List[int]
    topk_tile: List[int]
    adds_tile: List[int]


@dataclass
class LightningIndexerParams:
    b: int
    s1: int
    index_n1: int
    qk_nope: int
    qk_rope: int
    n2: int
    block_size: int
    block_num: int
    selected_count: int
    is_quant: bool = False


@dataclass
class LightningIndexerInputs:
    query: pto.Tensor
    key: pto.Tensor
    weights: pto.Tensor
    act_seq_key: pto.Tensor
    block_table: pto.Tensor
    topk_res: pto.Tensor
    q_scale: Optional[pto.Tensor]
    k_scale: Optional[pto.Tensor]
    tmp_out: Optional[pto.Tensor]
    topk_value: Optional[pto.Tensor]
    tile_config: LightningIndexerTileConfig
    unroll_list: Set[int]
    params: LightningIndexerParams


def lightning_indexer_topk_impl(args: LightningIndexerInputs):
    query = args.query
    key = args.key
    weights = args.weights
    act_seq_key = args.act_seq_key
    block_table = args.block_table
    topk_res = args.topk_res
    q_scale = args.q_scale
    k_scale = args.k_scale
    tmp_out = args.tmp_out
    topk_value = args.topk_value
    tile_config = args.tile_config
    unroll_list = args.unroll_list
    params = args.params
    is_quant = params.is_quant
    selected_count = params.selected_count

    b = query.shape[SHAPE_DIM0]
    s1 = query.shape[SHAPE_DIM1]
    block_num = key.shape[SHAPE_DIM0]

    index_n1 = query.shape[SHAPE_DIM2]
    index_d = query.shape[SHAPE_DIM3]
    block_size = key.shape[SHAPE_DIM1]
    n2 = key.shape[SHAPE_DIM2]

    qk_dtype = query.dtype
    scale_dtype = q_scale.dtype if is_quant else pto.DataType.DT_FP16
    w_dtype = weights.dtype

    group = index_n1 // n2

    c1_tile = tile_config.c1_tile
    max_batch = NUM_128
    max_s1 = NUM_4
    max_n2 = NUM_1
    max_s2 = NUM_128 * NUM_1024

    query_2d = pto.tensor([b * s1 * index_n1, index_d], qk_dtype, "query2D")
    key_2d = pto.tensor([block_num * block_size, n2 * index_d], qk_dtype, "key2D")
    q_scale_2d = pto.tensor([b * s1 * index_n1, 1], scale_dtype, "qScale2D")
    k_scale_2d = pto.tensor([block_num * block_size, n2], scale_dtype, "kScale2D")
    weight_2d = pto.tensor([b * s1 * index_n1, 1], w_dtype, "weight2D")
    local_sum = pto.tensor(
        [max_batch * max_s1 * max_n2, max_s2],
        pto.DataType.DT_FP32,
        "localSum",
    )

    for _ in pto.loop(0, 1, 1, name="INPUT_4D_2_2D", idx_name="unUsedIdx"):
        query_2d[:] = pto.reshape(query, [b * s1 * index_n1, index_d], inplace=True)
        key_2d[:] = pto.reshape(
            key, [block_num * block_size, n2 * index_d], inplace=True
        )
        weight_2d[:] = pto.reshape(weights, [b * s1 * index_n1, 1], inplace=True)
        if is_quant:
            q_scale_2d[:] = pto.reshape(q_scale, [b * s1 * index_n1, 1], inplace=True)
            k_scale_2d[:] = pto.reshape(
                k_scale, [block_num * block_size, n2], inplace=True
            )

    for b_idx in pto.loop(0, b, 1, name="INDEX_LOOP_BATCH", idx_name="bIdx"):

        def _inside_b(b_idx):
            cur_seq = act_seq_key[b_idx]
            for s1_idx in pto.loop(0, s1, 1, name="INDEX_LOOP_S1", idx_name="s1Idx"):

                def _inside_s1(s1_idx):
                    causal_offset = s1 - s1_idx - 1
                    eff_seq = cur_seq - causal_offset
                    act_block = (eff_seq + block_size - 1) // block_size
                    for n2_idx in pto.loop(
                        0, n2, 1, name="INDEX_LOOP_N2", idx_name="n2Idx"
                    ):

                        def _inside_n2(n2_idx):
                            bs1n2_offset = b_idx * s1 * n2 + s1_idx * n2 + n2_idx
                            q_offset = (
                                b_idx * s1 * index_n1
                                + s1_idx * index_n1
                                + n2_idx * group
                            )

                            def unrolling_process(
                                unroll_length: int,
                                first_block_idx: pto.symbolic_scalar,
                                b_idx,
                                s1_idx,
                                n2_idx,
                                eff_seq,
                                bs1n2_offset,
                                q_offset,
                            ):
                                cur_q = pto.view(
                                    query_2d, [group, index_d], [q_offset, 0]
                                )

                                concat_srcs = []

                                for sub_block_idx in range(unroll_length):
                                    block_idx = first_block_idx + sub_block_idx
                                    cur_block_idx = block_table[b_idx, block_idx]

                                    cur_k = pto.view(
                                        key_2d,
                                        [block_size, index_d],
                                        [cur_block_idx * block_size, n2_idx * index_d],
                                        valid_shape=[
                                            pto.min(
                                                block_size,
                                                eff_seq - (block_idx * block_size),
                                            ),
                                            index_d,
                                        ],
                                    )

                                    pto.set_cube_tile_shapes(
                                        c1_tile[0], c1_tile[1], c1_tile[2], False
                                    )

                                    mm_res = pto.matmul(
                                        cur_q,
                                        cur_k,
                                        pto.DT_FP32,
                                        a_trans=False,
                                        b_trans=True,
                                    )
                                    concat_srcs.append(mm_res)

                                pto.set_vec_tile_shapes(*tile_config.weight_tile)

                                cur_w = pto.view(weight_2d, [group, 1], [q_offset, 0])
                                w_b32 = pto.cast(cur_w, pto.DT_FP32)

                                mm_res_cat = pto.concat(concat_srcs, -1)

                                pto.set_vec_tile_shapes(*tile_config.v1_tile)

                                relu_res = pto.maximum(mm_res_cat, 0.0)
                                mul_res = relu_res * w_b32
                                sum_res = pto.sum(mul_res, 0)

                                pto.assemble(
                                    sum_res,
                                    [bs1n2_offset, first_block_idx * block_size],
                                    local_sum,
                                )
                                if tmp_out is not None:
                                    pto.assemble(
                                        sum_res,
                                        [bs1n2_offset, first_block_idx * block_size],
                                        tmp_out,
                                    )

                            def unrolling_process_quant(
                                unroll_length: int,
                                first_block_idx: pto.symbolic_scalar,
                                b_idx,
                                s1_idx,
                                n2_idx,
                                eff_seq,
                                bs1n2_offset,
                                q_offset,
                            ):
                                cur_q = pto.view(
                                    query_2d, [group, index_d], [q_offset, 0]
                                )
                                cur_q_scale = pto.view(
                                    q_scale_2d, [group, 1], [q_offset, 0]
                                )

                                mm_res_quant_concat_srcs = []
                                k_scale_concat_srcs = []

                                for sub_block_idx in range(unroll_length):
                                    block_idx = first_block_idx + sub_block_idx
                                    cur_block_idx = block_table[b_idx, block_idx]

                                    cur_k = pto.view(
                                        key_2d,
                                        [block_size, index_d],
                                        [cur_block_idx * block_size, n2_idx * index_d],
                                        valid_shape=[
                                            pto.min(
                                                block_size,
                                                eff_seq - (block_idx * block_size),
                                            ),
                                            index_d,
                                        ],
                                    )

                                    pto.set_cube_tile_shapes(
                                        c1_tile[0], c1_tile[1], c1_tile[2], False
                                    )

                                    mm_res = pto.matmul(
                                        cur_q,
                                        cur_k,
                                        pto.DataType.DT_INT32,
                                        a_trans=False,
                                        b_trans=True,
                                    )
                                    mm_res_quant_concat_srcs.append(mm_res)

                                    cur_k_scale = pto.view(
                                        k_scale_2d,
                                        [block_size, 1],
                                        [cur_block_idx * block_size, n2_idx],
                                        valid_shape=[
                                            pto.min(
                                                block_size,
                                                eff_seq - (block_idx * block_size),
                                            ),
                                            1,
                                        ],
                                    )
                                    k_scale_concat_srcs.append(cur_k_scale)

                                pto.set_vec_tile_shapes(*tile_config.weight_tile)

                                cur_w = pto.view(weight_2d, [group, 1], [q_offset, 0])
                                w_f16 = pto.cast(cur_w, pto.DataType.DT_FP16)

                                pto.set_vec_tile_shapes(*tile_config.v1_tile)

                                cur_k_scale = pto.concat(k_scale_concat_srcs, 0)
                                mm_res_i32 = pto.concat(mm_res_quant_concat_srcs, -1)
                                mm_res_fp32 = (
                                    pto.cast(mm_res_i32, pto.DataType.DT_FP32)
                                    * AVOID_FP32_TO_FP16_OVERFLOW_SCALE
                                )
                                mm_res_fp16 = pto.cast(
                                    mm_res_fp32, pto.DataType.DT_FP16
                                )
                                mm_res_dequant = (
                                    mm_res_fp16
                                    * cur_q_scale
                                    * pto.transpose(cur_k_scale, 0, 1)
                                )
                                relu_res = pto.maximum(mm_res_dequant, 0.0)
                                mul_res = relu_res * w_f16

                                sum_res = pto.sum(
                                    pto.cast(mul_res, pto.DataType.DT_FP32),
                                    0,
                                    True
                                )

                                pto.assemble(
                                    sum_res,
                                    [bs1n2_offset, first_block_idx * block_size],
                                    local_sum,
                                )
                                if tmp_out is not None:
                                    pto.assemble(
                                        sum_res,
                                        [bs1n2_offset, first_block_idx * block_size],
                                        tmp_out,
                                    )

                            for loop_block_idx, unroll_length in pto.loop_unroll(
                                0,
                                act_block,
                                1,
                                name="INDEX_LOOP_MATMUL",
                                idx_name="loopBlockIdx",
                                unroll_list=unroll_list,
                            ):

                                def _inside_block(loop_block_idx, unroll_length):
                                    if is_quant:
                                        unrolling_process_quant(
                                            unroll_length,
                                            loop_block_idx,
                                            b_idx,
                                            s1_idx,
                                            n2_idx,
                                            eff_seq,
                                            bs1n2_offset,
                                            q_offset,
                                        )
                                    else:
                                        unrolling_process(
                                            unroll_length,
                                            loop_block_idx,
                                            b_idx,
                                            s1_idx,
                                            n2_idx,
                                            eff_seq,
                                            bs1n2_offset,
                                            q_offset,
                                        )

                                _inside_block(loop_block_idx, unroll_length)

                        _inside_n2(n2_idx)

                _inside_s1(s1_idx)

        _inside_b(b_idx)

    assert selected_count == NUM_2048

    x_dtype = local_sum.dtype
    idx_dtype = topk_res.dtype
    pad_idx_value = NUM_NEG1
    tile_size = NUM_8192
    descending = True
    pad_value = -sys.float_info.max if descending else sys.float_info.max

    length_2k = selected_count
    length_8k = NUM_1024 * NUM_8
    length_64k = NUM_1024 * NUM_64
    length_128k = max_s2

    pto.set_vec_tile_shapes(1, tile_size)

    for bs1n2_offset in pto.loop(
        0,
        b * s1 * n2,
        1,
        name="INDEX_LOOP_TOPK_bs1n2Offset",
        idx_name="bs1n2Offset",
    ):

        def _inside_bs1n2(bs1n2_offset):
            b_idx = bs1n2_offset // (s1 * n2)
            s1_idx = (bs1n2_offset % (s1 * n2)) // n2
            n2_idx = bs1n2_offset % n2

            cur_seq = act_seq_key[b_idx]
            causal_offset = s1 - s1_idx - 1
            eff_seq = cur_seq - causal_offset

            length_is_le2k = eff_seq <= length_2k
            length_is_gt2k = eff_seq > length_2k

            pad_x_2k = pto.tensor(
                [max_batch * max_s1 * max_n2, length_2k],
                x_dtype,
                "padX2K",
            )
            pto.set_vec_tile_shapes(1, tile_size)

            for unused in pto.loop(
                0, length_is_le2k, 1, name="2K_LOOP", idx_name="unused"
            ):

                def _inside_2k(unused):
                    pto.set_pass_options(sg_skip_partition=True)

                    for unused1 in pto.loop(0, 1, 1, name="2K_PAD", idx_name="unused1"):

                        def _inside_2k_pad(unused1):
                            pto.set_vec_tile_shapes(1, length_2k)

                            eff_sum_res = pto.view(
                                local_sum,
                                [1, length_2k],
                                [bs1n2_offset, 0],
                                valid_shape=[1, eff_seq],
                            )
                            ax = pto.view(
                                eff_sum_res,
                                [1, length_2k],
                                [0, 0],
                                valid_shape=[1, eff_seq],
                            )
                            bx = pto.full(
                                [1, length_2k],
                                pad_value,
                                x_dtype,
                                valid_shape=[
                                    1,
                                    length_2k - eff_seq,
                                ],
                            )

                            pto.assemble(
                                pto.clone(ax),
                                [bs1n2_offset, 0],
                                pad_x_2k,
                            )
                            pto.assemble(
                                bx,
                                [bs1n2_offset, eff_seq],
                                pad_x_2k,
                            )

                        _inside_2k_pad(unused1)

                    pto.set_pass_options(sg_skip_partition=False)

                    for unused2 in pto.loop(
                        0, 1, 1, name="2K_TOPK", idx_name="unused2"
                    ):

                        def _inside_2k_topk(unused2):
                            res, res_idx = pto.topk(
                                pto.view(
                                    pad_x_2k,
                                    [1, length_2k],
                                    [bs1n2_offset, 0],
                                ),
                                selected_count,
                                1,
                            )
                            pto.set_vec_tile_shapes(*tile_config.adds_tile)

                            topk_4d = pto.reshape(
                                pto.view(
                                    res_idx,
                                    [1, selected_count],
                                    [0, 0],
                                    valid_shape=[1, eff_seq],
                                ),
                                [1, 1, 1, selected_count],
                                valid_shape=[1, 1, 1, eff_seq],
                            )
                            pto.assemble(
                                pto.clone(topk_4d),
                                [b_idx, s1_idx, n2_idx, 0],
                                topk_res,
                            )

                            topk_indices_pad = pto.full(
                                [1, 1, 1, selected_count],
                                pad_idx_value,
                                idx_dtype,
                                valid_shape=[
                                    1,
                                    1,
                                    1,
                                    selected_count - eff_seq,
                                ],
                            )
                            pto.assemble(
                                topk_indices_pad,
                                [b_idx, s1_idx, n2_idx, eff_seq],
                                topk_res,
                            )

                            if topk_value is not None:
                                topk_4d_value = pto.reshape(
                                    pto.view(
                                        res,
                                        [1, selected_count],
                                        [0, 0],
                                        valid_shape=[1, eff_seq],
                                    ),
                                    [1, 1, 1, selected_count],
                                    valid_shape=[
                                        1,
                                        1,
                                        1,
                                        eff_seq,
                                    ],
                                )
                                pto.assemble(
                                    pto.clone(topk_4d_value),
                                    [b_idx, s1_idx, n2_idx, 0],
                                    topk_value,
                                )
                                topk_value_pad = pto.full(
                                    [1, 1, 1, selected_count],
                                    pad_value,
                                    pto.DataType.DT_FP32,
                                    valid_shape=[
                                        1,
                                        1,
                                        1,
                                        selected_count - eff_seq,
                                    ],
                                )
                                pto.assemble(
                                    topk_value_pad,
                                    [
                                        b_idx,
                                        s1_idx,
                                        n2_idx,
                                        eff_seq,
                                    ],
                                    topk_value,
                                )

                            pto.set_vec_tile_shapes(1, tile_size)

                        _inside_2k_topk(unused2)

                _inside_2k(unused)

            length_is_le8k = eff_seq <= length_8k
            length_is_gt8k = eff_seq > length_8k

            pad_x_8k = pto.tensor(
                [max_batch * max_s1 * max_n2, length_8k], x_dtype, "padX8K"
            )

            for unused in pto.loop(
                0,
                length_is_gt2k * length_is_le8k,
                1,
                name="8K_LOOP",
                idx_name="unused",
            ):

                def _inside_8k(unused):
                    for unused0 in pto.loop(0, 1, 1, name="8K_PAD", idx_name="unused0"):

                        def _inside_8k_pad(unused0):
                            pto.set_vec_tile_shapes(1, tile_size)

                            eff_sum_res = pto.view(
                                local_sum,
                                [1, length_8k],
                                [bs1n2_offset, 0],
                                valid_shape=[1, eff_seq],
                            )
                            ax = pto.view(
                                eff_sum_res,
                                [1, length_8k],
                                [0, 0],
                                valid_shape=[1, eff_seq],
                            )
                            bx = pto.full(
                                [1, length_8k],
                                pad_value,
                                x_dtype,
                                valid_shape=[
                                    1,
                                    length_8k - eff_seq,
                                ],
                            )
                            pto.assemble(pto.clone(ax), [bs1n2_offset, 0], pad_x_8k)
                            pto.assemble(bx, [bs1n2_offset, eff_seq], pad_x_8k)

                        _inside_8k_pad(unused0)

                    pto.set_vec_tile_shapes(1, tile_size)

                    for unused1 in pto.loop(
                        0, 1, 1, name="8K_TOPK", idx_name="unused1"
                    ):

                        def _inside_8k_topk(unused1):
                            res, res_idx = pto.topk(
                                pto.view(pad_x_8k, [1, length_8k], [bs1n2_offset, 0]),
                                selected_count,
                                1,
                            )
                            pto.set_vec_tile_shapes(*tile_config.adds_tile)

                            topk_4d = pto.reshape(
                                res_idx,
                                [1, 1, 1, selected_count],
                            )
                            pto.assemble(
                                pto.clone(topk_4d),
                                [b_idx, s1_idx, n2_idx, 0],
                                topk_res,
                            )

                            if topk_value is not None:
                                pto.set_vec_tile_shapes(*tile_config.adds_tile)
                                topk_4d_value = pto.reshape(
                                    res, [1, 1, 1, selected_count]
                                )
                                pto.assemble(
                                    pto.clone(topk_4d_value),
                                    [b_idx, s1_idx, n2_idx, 0],
                                    topk_value,
                                )

                            pto.set_vec_tile_shapes(1, tile_size)

                        _inside_8k_topk(unused1)

                _inside_8k(unused)

            length_is_le64k = eff_seq <= length_64k
            length_is_gt64k = eff_seq > length_64k

            pad_x_64k = pto.tensor(
                [max_batch * max_s1 * max_n2, length_64k], x_dtype, "padX64K"
            )

            for unused in pto.loop(
                0,
                length_is_gt8k * length_is_le64k,
                1,
                name="64K_LOOP",
                idx_name="unused",
            ):

                def _inside_64k(unused):
                    for unused0 in pto.loop(
                        0, 1, 1, name="64K_PAD", idx_name="unused0"
                    ):

                        def _inside_64k_pad(unused0):
                            pto.set_vec_tile_shapes(1, tile_size)

                            eff_sum_res = pto.view(
                                local_sum,
                                [1, length_64k],
                                [bs1n2_offset, 0],
                                valid_shape=[1, eff_seq],
                            )
                            ax = pto.view(
                                eff_sum_res,
                                [1, length_64k],
                                [0, 0],
                                valid_shape=[1, eff_seq],
                            )
                            bx = pto.full(
                                [1, length_64k],
                                pad_value,
                                x_dtype,
                                valid_shape=[
                                    1,
                                    length_64k - eff_seq,
                                ],
                            )
                            pto.assemble(pto.clone(ax), [bs1n2_offset, 0], pad_x_64k)
                            pto.assemble(bx, [bs1n2_offset, eff_seq], pad_x_64k)

                        _inside_64k_pad(unused0)

                    pto.set_vec_tile_shapes(1, tile_size)

                    for unused1 in pto.loop(
                        0, 1, 1, name="64K_TOPK", idx_name="unused1"
                    ):

                        def _inside_64k_topk(unused1):
                            res, res_idx = pto.topk(
                                pto.view(pad_x_64k, [1, length_64k], [bs1n2_offset, 0]),
                                selected_count,
                                1,
                            )
                            pto.set_vec_tile_shapes(*tile_config.adds_tile)
                            topk_4d = pto.reshape(res_idx, [1, 1, 1, selected_count])
                            pto.assemble(
                                pto.clone(topk_4d),
                                [b_idx, s1_idx, n2_idx, 0],
                                topk_res,
                            )

                            if topk_value is not None:
                                pto.set_vec_tile_shapes(*tile_config.adds_tile)
                                topk_4d_value = pto.reshape(
                                    res, [1, 1, 1, selected_count]
                                )
                                pto.assemble(
                                    pto.clone(topk_4d_value),
                                    [b_idx, s1_idx, n2_idx, 0],
                                    topk_value,
                                )

                            pto.set_vec_tile_shapes(1, tile_size)

                        _inside_64k_topk(unused1)

                _inside_64k(unused)

            pad_x_128k = pto.tensor(
                [max_batch * max_s1 * max_n2, length_128k], x_dtype, "padX128K"
            )

            for unused in pto.loop(
                0, length_is_gt64k, 1, name="128K_LOOP", idx_name="unused"
            ):

                def _inside_128k(unused):
                    for unused0 in pto.loop(
                        0, 1, 1, name="128K_PAD", idx_name="unused0"
                    ):

                        def _inside_128k_pad(unused0):
                            pto.set_vec_tile_shapes(1, tile_size)

                            eff_sum_res = pto.view(
                                local_sum,
                                [1, length_128k],
                                [bs1n2_offset, 0],
                                valid_shape=[1, eff_seq],
                            )
                            ax = pto.view(
                                eff_sum_res,
                                [1, length_128k],
                                [0, 0],
                                valid_shape=[1, eff_seq],
                            )
                            bx = pto.full(
                                [1, length_128k],
                                pad_value,
                                x_dtype,
                                valid_shape=[
                                    1,
                                    length_128k - eff_seq,
                                ],
                            )
                            pto.assemble(pto.clone(ax), [bs1n2_offset, 0], pad_x_128k)
                            pto.assemble(bx, [bs1n2_offset, eff_seq], pad_x_128k)

                        _inside_128k_pad(unused0)

                    pto.set_vec_tile_shapes(1, tile_size)

                    for unused1 in pto.loop(
                        0, 1, 1, name="128K_TOPK", idx_name="unused1"
                    ):

                        def _inside_128k_topk(unused1):
                            res, res_idx = pto.topk(
                                pto.view(
                                    pad_x_128k, [1, length_128k], [bs1n2_offset, 0]
                                ),
                                selected_count,
                                1,
                            )
                            pto.set_vec_tile_shapes(*tile_config.adds_tile)
                            topk_4d = pto.reshape(res_idx, [1, 1, 1, selected_count])
                            pto.assemble(
                                pto.clone(topk_4d),
                                [b_idx, s1_idx, n2_idx, 0],
                                topk_res,
                            )

                            if topk_value is not None:
                                pto.set_vec_tile_shapes(*tile_config.adds_tile)
                                topk_4d_value = pto.reshape(
                                    res, [1, 1, 1, selected_count]
                                )
                                pto.assemble(
                                    pto.clone(topk_4d_value),
                                    [b_idx, s1_idx, n2_idx, 0],
                                    topk_value,
                                )

                            pto.set_vec_tile_shapes(1, tile_size)

                        _inside_128k_topk(unused1)

                _inside_128k(unused)

        _inside_bs1n2(bs1n2_offset)


def lightning_indexer_topk_inner(args: LightningIndexerInputs):
    input_tensors = [
        args.query,
        args.key,
        args.weights,
        args.act_seq_key,
        args.block_table,
    ]
    if args.params.is_quant:
        input_tensors += [args.q_scale, args.k_scale]

    output_tensors = [args.topk_res]
    if args.tmp_out is not None:
        output_tensors.append(args.tmp_out)
    if args.topk_value is not None:
        output_tensors.append(args.topk_value)

    with pto.function("LightningIndexerTopkInner", input_tensors, output_tensors):

        def inside_main_function():
            lightning_indexer_topk_impl(args)

        inside_main_function()


@dataclass
class LightningIndexerBuildConfig:
    b: int = NUM_4
    s1: int = NUM_2
    index_n1: int = NUM_64
    qk_nope: int = NUM_128
    qk_rope: int = NUM_0
    n2: int = NUM_1
    block_size: int = NUM_128
    block_num: int = NUM_1127
    selected_count: int = NUM_2048
    is_quant: bool = True
    c1_tile: List[List[int]] = field(
        default_factory=lambda: [
            [NUM_64, NUM_64],
            [NUM_128, NUM_128],
            [NUM_128, NUM_128],
        ]
    )
    v1_tile: List[int] = field(default_factory=lambda: [NUM_64, NUM_128])
    topk_tile: List[int] = field(default_factory=lambda: [NUM_1, NUM_4096])
    adds_tile: List[int] = field(
        default_factory=lambda: [NUM_1, NUM_1, NUM_1, NUM_4096]
    )


def setup_lightning_indexer_topk_config():
    pto.set_codegen_options(support_dynamic_unaligned=True,
                            codegen_expression_fusion=True)

    pto.set_pass_options(copyin_threshold=NUM_100 * NUM_1024 * NUM_1024,
                         cycle_lower_bound=NUM_1024,
                         cycle_upper_bound=NUM_1024 * NUM_1024,
                         l1_reuse=NUM_32,
                         sg_skip_partition=NUM_2,
                         nbuffer_merge_mode=NUM_2,
                         vec_nbuffer_map={NUM_NEG1: NUM_16})
    pto.set_runtime_options(machine_sched_mode=NUM_3,
                            workspace_recycle_period=NUM_128,
                            estimated_stitch_task_max_loop_num=NUM_128)


def build_lightning_indexer_topk_args(
    cfg: LightningIndexerBuildConfig = LightningIndexerBuildConfig(),
):
    d_bf16 = pto.DT_FP16
    d_i32 = pto.DT_INT32
    d_int8 = pto.DT_INT8
    d_f16 = pto.DT_FP16

    index_d = cfg.qk_nope + cfg.qk_rope
    max_block_num = NUM_1024

    if cfg.is_quant:
        qk_dtype = d_int8
        scale_dtype = d_f16
    else:
        qk_dtype = d_bf16
        scale_dtype = d_f16

    query = pto.tensor(
        [cfg.b, cfg.s1, cfg.index_n1, index_d],
        qk_dtype,
        "query",
    )

    key = pto.tensor(
        [cfg.block_num, cfg.block_size, cfg.n2, index_d],
        qk_dtype,
        "key",
    )

    weights = pto.tensor(
        [cfg.b, cfg.s1, cfg.index_n1],
        d_bf16,
        "weights",
    )

    act_seq_key = pto.tensor(
        [cfg.b],
        d_i32,
        "actSeqKey",
    )

    block_table = pto.tensor(
        [cfg.b, max_block_num],
        d_i32,
        "blockTable",
    )

    topk_res = pto.tensor(
        [cfg.b, cfg.s1, cfg.n2, cfg.selected_count],
        d_i32,
        "topkRes",
    )

    q_scale = (
        pto.tensor(
            [cfg.b, cfg.s1, cfg.index_n1, 1],
            scale_dtype,
            "qScale",
        )
        if cfg.is_quant
        else None
    )
    k_scale = (
        pto.tensor(
            [cfg.block_num, cfg.block_size, cfg.n2, 1],
            scale_dtype,
            "kScale",
        )
        if cfg.is_quant
        else None
    )

    tmp_out = None
    topk_value = None

    tile_cfg = LightningIndexerTileConfig(
        weight_tile=[NUM_64, NUM_128],
        c1_tile=cfg.c1_tile,
        v1_tile=cfg.v1_tile,
        topk_tile=cfg.topk_tile,
        adds_tile=cfg.adds_tile,
    )

    unroll_list: List[int] = [1, 2, 4, 8, 16, 32, 64]

    params = LightningIndexerParams(
        b=cfg.b,
        s1=cfg.s1,
        index_n1=cfg.index_n1,
        qk_nope=cfg.qk_nope,
        qk_rope=cfg.qk_rope,
        n2=cfg.n2,
        block_size=cfg.block_size,
        block_num=cfg.block_num,
        selected_count=cfg.selected_count,
        is_quant=cfg.is_quant,
    )

    args = LightningIndexerInputs(
        query=query,
        key=key,
        weights=weights,
        act_seq_key=act_seq_key,
        block_table=block_table,
        topk_res=topk_res,
        q_scale=q_scale,
        k_scale=k_scale,
        tmp_out=tmp_out,
        topk_value=topk_value,
        tile_config=tile_cfg,
        unroll_list=unroll_list,
        params=params,
    )

    meta = {
        "B": cfg.b,
        "S1": cfg.s1,
        "indexN1": cfg.index_n1,
        "indexD": index_d,
        "N2": cfg.n2,
        "blockSize": cfg.block_size,
        "blockNum": cfg.block_num,
        "maxBlockNum": max_block_num,
        "selectedCount": cfg.selected_count,
        "isQuant": cfg.is_quant,
        "dims": {
            "query": [cfg.b, cfg.s1, cfg.index_n1, index_d],
            "key": [cfg.block_num, cfg.block_size, cfg.n2, index_d],
            "weights": [cfg.b, cfg.s1, cfg.index_n1],
            "actSeqKey": [cfg.b],
            "blockTable": [cfg.b, max_block_num],
            "topkRes": [cfg.b, cfg.s1, cfg.n2, cfg.selected_count],
            "qScale": ([cfg.b, cfg.s1, cfg.index_n1, 1] if cfg.is_quant else None),
            "kScale": (
                [cfg.block_num, cfg.block_size, cfg.n2, 1] if cfg.is_quant else None
            ),
        },
        "tiles": {
            "weightTile": tile_cfg.weight_tile,
            "c1Tile": tile_cfg.c1_tile,
            "v1Tile": tile_cfg.v1_tile,
            "topkTile": tile_cfg.topk_tile,
            "addsTile": tile_cfg.adds_tile,
        },
        "unrollList": sorted(list(unroll_list)),
    }

    return args, meta


def test_lightning_indexer_topk():
    logging.basicConfig(level=logging.INFO)
    setup_lightning_indexer_topk_config()
    args, meta = build_lightning_indexer_topk_args()
    logging.info({"Sanity": meta})
    lightning_indexer_topk_inner(args)
    assert True
