#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import logging
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


S1 = 64
S2 = 1024
DQK = 128
DV = 128

SQ = 64
MAX_UNROLL_TIMES = 4


def softmax(x):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = (x - x_max)
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    res = y / x_sum
    return res


@pypto.frontend.jit()
def test_flash_attention(
    q: pypto.Tensor((S1, DQK), pypto.DT_FP16),
    k: pypto.Tensor((S2, DQK), pypto.DT_FP16),
    v: pypto.Tensor((S2, DV), pypto.DT_FP16),
) -> (
    pypto.Tensor((S1, DV), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 64)
    pypto.set_cube_tile_shapes([32, 64], [32, 64], [32, 64])

    softmax_scale = np.reciprocal(np.sqrt(DQK))

    qlen = S1
    seqlen = S2
    dk = DQK
    dv = DV

    sq = SQ

    max_unroll_times = MAX_UNROLL_TIMES

    oi_update = pypto.tensor((qlen, dv), pypto.DT_FP32)
    li_update = pypto.tensor((qlen, 1), pypto.DT_FP32)
    mi_update = pypto.tensor((qlen, 1), pypto.DT_FP32)

    seq_block_num = (seqlen + sq - 1) // sq

    for seq_block in pypto.loop(seq_block_num, unroll_List=[max_unroll_times]):
        cur_offset = seq_block * sq
        kj = pypto.view(k, (sq, dk), [cur_offset, 0], valid_shape=[(seqlen - seq_block * sq).min(sq), dk])
        vj = pypto.view(v, (sq, dv), [cur_offset, 0], valid_shape=[(seqlen - seq_block * sq).min(sq), dv])

        sij = pypto.matmul(q, kj, out_dtype=pypto.DT_FP32, b_trans=True)
        sij = sij * softmax_scale
        tilda_mij = pypto.amax(sij, -1, True)
        tsub = sij - tilda_mij
        tilda_pij = pypto.exp(tsub)
        tilda_lij = pypto.sum(tilda_pij, -1, True)
        tilda_pij = pypto.cast(tilda_pij, pypto.DT_FP16)

        if seq_block == 0:
            oi_update[:] = pypto.matmul(tilda_pij, vj, out_dtype=pypto.DT_FP32)
            if (seq_block == seq_block_num - 1):
                attention = pypto.div(oi_update, tilda_lij)
            li_update[:] = tilda_lij
            mi_update[:] = tilda_mij
        else:
            mi = mi_update
            mi_update = pypto.maximum(mi, tilda_mij)
            t1 = mi - mi_update
            t2 = pypto.exp(t1)
            t3 = tilda_mij - mi_update
            t4 = pypto.exp(t3)
            t5 = t4 * tilda_lij
            t6 = t2 * li_update
            li_update[:] = t6 + t5

            q3 = oi_update * t2
            q1 = pypto.matmul(tilda_pij, vj, out_dtype=pypto.DT_FP32)
            q2 = q1 * t4
            oi_update[:] = q3 + q2
            if (seq_block == seq_block_num - 1):
                attention = pypto.div(oi_update, li_update)

    return attention


def test_flash_attention_run():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 8))
    torch.npu.set_device(device_id)

    s1 = S1
    s2 = S2
    dqk = DQK
    dv = DV

    q = np.random.uniform(-1, 1, (s1, dqk)).astype(np.float16)
    k = np.random.uniform(-1, 1, (s2, dqk)).astype(np.float16)
    v = np.random.uniform(-1, 1, (s2, dv)).astype(np.float16)

    attention_golden = (softmax(q @ k.T / np.sqrt(dqk)) @ v)

    q = torch.from_numpy(q).to(device=f"npu:{device_id}")
    k = torch.from_numpy(k).to(device=f"npu:{device_id}")
    v = torch.from_numpy(v).to(device=f"npu:{device_id}")

    attention = test_flash_attention(q, k, v).cpu()

    logging.info(f"Attention shape: {attention.shape}")
    logging.info(f"Attention golden shape: {attention_golden.shape}")
    assert_allclose(
        attention,
        attention_golden,
        rtol=1e-3,
        atol=1e-3
    )


if __name__ == "__main__":
    test_flash_attention_run()
