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

SHAPE = (1, 2, 64, 128)

@pypto.frontend.jit
def element_wise_op(
    a: pypto.Tensor((1, 2, 64, 128), pypto.DT_FP32),
    b: pypto.Tensor((1, 2, 64, 128), pypto.DT_FP32),
) -> pypto.Tensor((1, 2, 64, 128), pypto.DT_FP32):
    pypto.set_vec_tile_shapes(1, 1, 16, 32)
    c = pypto.sin(a) * pypto.sin(b)
    return c


def test_element_wise_op():
    # Setup device
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    # Prepare test data
    np.random.seed(0)
    a = torch.rand(SHAPE, dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.rand(SHAPE, dtype=torch.float32, device=f"npu:{device_id}")

    # Execute kernel
    t3 = element_wise_op(a, b)
    pypto.runtime._device_synchronize()

    print("End to execute kernel")

    # PyTorch reference implementation
    ref = torch.sin(a) * torch.sin(b)

    assert_allclose(
        np.array(t3.cpu().flatten().tolist()),
        np.array(ref.cpu().flatten().tolist()),
        rtol=5e-3,
        atol=5e-3,
    )


if __name__ == "__main__":
    test_element_wise_op()
