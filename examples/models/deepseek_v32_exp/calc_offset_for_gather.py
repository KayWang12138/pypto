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
"""
"""
from dataclasses import dataclass
import math
import os
import pypto
from pypto import pypto_impl
from pypto.operation import op_wrapper
import torch
import numpy as np
from numpy.testing import assert_allclose

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


@op_wrapper
def load(src, dst):
    return pypto_impl.Load(src, dst)

@pypto.jit
def calc_offsets_for_gather_prefill_compute(inputs, outputs, block_size, topk, batch_value, seq_value):
    topk_indecies, block_table, kv_act_seqs = inputs
    offsets,  = outputs
    n_kv = 1
    max_block_num_per_batch = block_table.shape[1]
    b = batch_value
    s1 = seq_value

    pypto.set_host_options(only_codegen=True)
    pypto.set_pass_options(copyin_threshold=NUM_100 * NUM_1024 * NUM_1024,
                         cycle_lower_bound=NUM_1024,
                         cycle_upper_bound=NUM_1024 * NUM_1024,
                         l1_reuse=NUM_32,
                         parallel_threshold=NUM_2,
                         nbuffer_merge_mode=1)
    pypto.set_runtime_options(machine_sched_mode=NUM_3,
                            workspace_recycle_period=NUM_128,
                            estimated_stitch_task_max_loop_num=NUM_128)
    pypto.set_codegen_options(support_dynamic_unaligned=True)

    for idx in pypto.loop(0, b * s1, 1, name="LOOP_L0_idx", idx_name="idx"):
        batch_idx = idx // s1
        slc_idx = idx % s1
        pypto.set_semantic_label("calc_offset")
        topk_loop = (kv_act_seqs[batch_idx, ] - s1 + 1 + slc_idx).max(0).min(topk)
        tile_0 = 1
        tile_1 = 256
        pypto.set_vec_tile_shapes(tile_0, tile_1)
        topk_indcies_reshape = pypto.view(topk_indecies, [1, n_kv * topk],
                                                    [idx, 0], valid_shape=[1, topk_loop])

        topk_indcies_reshape_fp32 = pypto.cast(topk_indcies_reshape, pypto.DataType.DT_FP32)
        topk_indcies_reshape_fp32 = pypto.add(topk_indcies_reshape_fp32, 0.5)
        block_idx_in_batchs_fp32 = pypto.div(topk_indcies_reshape_fp32, float(block_size))
        block_idx_in_batchs = pypto.cast(block_idx_in_batchs_fp32,
            pypto.DataType.DT_INT32, pypto.CastMode.CAST_FLOOR)

        tails = pypto.sub(topk_indcies_reshape, pypto.mul(block_idx_in_batchs, block_size))

        block_table_raw_offsets = pypto.full([1, n_kv * topk], batch_idx * max_block_num_per_batch,
            pypto.DataType.DT_INT32, valid_shape=[1, topk_loop])
        add_res = pypto.add(block_table_raw_offsets, block_idx_in_batchs)
        slc_block_idxs = load(block_table, add_res)
        block_offsets = pypto.mul(slc_block_idxs, block_size)
        offset = pypto.add(block_offsets, tails)
        pypto.assemble(offset, [idx, 0], offsets)
