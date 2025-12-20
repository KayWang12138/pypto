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
import os
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose

N = 1024
M = 1024
IS_ADD = True

@pypto.frontend.jit()
def basic_if_else(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32),
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
):

    c = pypto.tensor((N, M), pypto.DT_FP32)

    pypto.set_vec_tile_shapes(32, 32)

    if IS_ADD:
        c[:] = pypto.add(a, b)
    else:
        c[:] = pypto.sub(a, b)

    return c


def test_basic_if_else_run():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    n, m = 1024, 1024
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.rand((n, m), dtype=torch.float32, device=f"npu:{device_id}")

    c = basic_if_else(a, b)

    pypto.runtime._device_synchronize()

    if IS_ADD:
        c_golden = (a + b).cpu()
    else:
        c_golden = (a - b).cpu()

    assert_allclose(c.cpu().flatten(), c_golden.flatten(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_basic_if_else_run()
