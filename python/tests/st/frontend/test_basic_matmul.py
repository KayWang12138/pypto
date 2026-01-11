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

N = 128
M = 128


@pypto.frontend.jit()
def basic_matmul(
    a: pypto.Tensor((N, M), pypto.DT_FP16),
    b: pypto.Tensor((N, M), pypto.DT_FP16),
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
    pypto.Tensor((N, M), pypto.DT_FP32),
):

    c = pypto.tensor((N, M), pypto.DT_FP32)
    d = pypto.tensor((N, M), pypto.DT_FP32)

    pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])

    c[:] = pypto.matmul(a, b, out_dtype=pypto.DT_FP32)
    d[:] = pypto.matmul(a, b, out_dtype=pypto.DT_FP32, b_trans=True)
    return c, d


def test_basic_matmul_run():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    n, m = 128, 128
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float16, device=f"npu:{device_id}")
    b = torch.rand((n, m), dtype=torch.float16, device=f"npu:{device_id}")

    c, d = basic_matmul(a, b)

    pypto.runtime._device_synchronize()

    c_golden = (a @ b).cpu()
    d_golden = (a @ b.T).cpu()
    assert_allclose(c.cpu().flatten(), c_golden.flatten(), rtol=1e-3, atol=1e-3)
    assert_allclose(d.cpu().flatten(), d_golden.flatten(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_basic_matmul_run()
