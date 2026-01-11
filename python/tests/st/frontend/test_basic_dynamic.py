#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
"""
import os

import numpy as np
import pypto
import torch
from numpy.testing import assert_allclose

N = pypto.frontend.dynamic("N")
M = 1024
VIEW_SHAPE = (32, 32)

FLAG = False

@pypto.frontend.jit()
def basic_dynamic(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32),
    c: pypto.Tensor((N, M), pypto.DT_FP32),
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
    pypto.Tensor((N, M), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    d = pypto.tensor((N, M), pypto.DT_FP32)
    e = pypto.tensor((N, M), pypto.DT_FP32)

    for bs_idx in pypto.loop(32, unroll_List={16}):
        tile_a = pypto.view(a, (32, 1024), [bs_idx * 32, 0])
        tile_b = pypto.view(b, (32, 1024), [bs_idx * 32, 0])
        tile_c = pypto.view(c, (32, 1024), [bs_idx * 32, 0])
        if FLAG:
            tile_a[:] = pypto.add(tile_a, tile_c)
        else:
            tile_a[:] = pypto.sub(tile_a, tile_c)
        tile_b[:] = pypto.sub(tile_b, tile_c)
        d[bs_idx * 32: (bs_idx + 1) * 32, :] = pypto.add(tile_a, tile_a)
        e[bs_idx * 32: (bs_idx + 1) * 32, :] = pypto.add(tile_b, tile_b)
    return d, e


def test_basic_dynamic_run():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    n, m = 1024, 1024
    np.random.seed(0)
    a = torch.zeros((n, m), dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.zeros((n, m), dtype=torch.float32, device=f"npu:{device_id}")
    c = torch.rand((n, m), dtype=torch.float32, device=f"npu:{device_id}")

    d, e = basic_dynamic(a, b, c)

    pypto.runtime._device_synchronize()

    for _ in range(1):
        if FLAG:
            a = a + c
        else:
            a = a - c
        b = b - c
    d_golden = (a + a).cpu()
    e_golden = (b + b).cpu()

    assert_allclose(d.cpu().flatten(), d_golden.cpu().flatten(), rtol=1e-5, atol=1e-5)
    assert_allclose(e.cpu().flatten(), e_golden.cpu().flatten(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_basic_dynamic_run()
