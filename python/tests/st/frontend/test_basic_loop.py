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
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose

N = 1024
M = 1024
LOOP_NUM = 10


@pypto.frontend.jit()
def basic_loop(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32),
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
):

    c = pypto.tensor((N, M), pypto.DT_FP32)
    pypto.set_vec_tile_shapes(32, 32)

    for _ in pypto.loop(LOOP_NUM):
        a[:] = pypto.add(a, b)
        c = a

    return c


def test_basic_loop_run():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    n, m = 1024, 1024
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.rand((n, m), dtype=torch.float32, device=f"npu:{device_id}")

    c = basic_loop(a, b)

    pypto.runtime._device_synchronize()

    for _ in range(LOOP_NUM):
        a = a + b
    c_golden = a.cpu()

    assert_allclose(c.cpu().flatten(), c_golden.flatten(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_basic_loop_run()
