#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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


S1 = 32
S2 = 128
DQK = 64
DV = 128


def attention_data_gen(s1, s2, dqk, dv, device_id):
    q = np.random.uniform(-1, 1, (s1, dqk)).astype(np.float16)
    k = np.random.uniform(-1, 1, (s2, dqk)).astype(np.float16)
    v = np.random.uniform(-1, 1, (s2, dv)).astype(np.float16)

    attention_golden = (softmax(q @ k.T / np.sqrt(dqk)) @ v)

    q = torch.from_numpy(q).to(device=f"npu:{device_id}")
    k = torch.from_numpy(k).to(device=f"npu:{device_id}")
    v = torch.from_numpy(v).to(device=f"npu:{device_id}")

    return q, k, v, attention_golden


def softmax(x):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = (x - x_max)
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    res = y / x_sum
    return res


@pypto.frontend.jit()
def test_attention(
    q: pypto.tensor((S1, DQK), pypto.DT_FP16),
    k: pypto.tensor((S2, DQK), pypto.DT_FP16),
    v: pypto.tensor((S2, DV), pypto.DT_FP16),
) -> (
    pypto.tensor((S1, DV), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])

    qk = pypto.cast(pypto.matmul(q, k, out_dtype=pypto.DT_FP32, b_trans=True), pypto.DT_FP16)
    qk_scale = np.reciprocal(np.sqrt(DQK))
    s = pypto.softmax(qk * qk_scale, -1)
    out = pypto.matmul(s, v, out_dtype=pypto.DT_FP32)

    attention = out
    return attention


def test_attention_run():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    q, k, v, attention_golden = attention_data_gen(S1, S2, DQK, DV, device_id)

    attention = test_attention(q, k, v).cpu()

    assert_allclose(
        attention,
        attention_golden,
        rtol=1e-3,
        atol=1e-3
    )

    logging.info("\n" + "=" * 80)
    logging.info("TEST PASSED: Attention works correctly!")
    logging.info("=" * 80)


if __name__ == "__main__":
    test_attention_run()
