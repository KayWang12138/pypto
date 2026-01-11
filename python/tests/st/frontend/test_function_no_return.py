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
import torch
import pypto
from numpy.testing import assert_allclose


@pypto.frontend.function
def inner_add_inplace(
    x: pypto.Tensor((8,), pypto.DT_FP32),
    bias: pypto.Tensor((8,), pypto.DT_FP32),
    out: pypto.Tensor((8,), pypto.DT_FP32),
):
    out[:] = pypto.add(x, bias)


@pypto.frontend.jit
def outer_kernel_inplace(
    a: pypto.Tensor((8,), pypto.DT_FP32),
    b: pypto.Tensor((8,), pypto.DT_FP32),
    out: pypto.Tensor((8,), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 16, 32)
    inner_add_inplace(a, b, out)


def _reference_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.add(a, b)


def test_function_no_return():
    print("=" * 80)
    print("Testing Function with no return")
    print("=" * 80)

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    print(f"\nDevice ID: {device_id}")
    shape = (8,)
    print(f"Input shape: {shape}")

    print("\nGenerating test data...")
    a = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")
    out = torch.empty(shape, dtype=torch.float32, device=f"npu:{device_id}")

    print("\nExecuting function kernel...")
    result = outer_kernel_inplace(a, b, out)
    pypto.runtime._device_synchronize()

    assert result is None

    print("\nComputing reference result...")
    reference = _reference_impl(a, b)

    print("\nValidating results...")
    assert_allclose(
        np.array(out.cpu().flatten().tolist()),
        np.array(reference.cpu().flatten().tolist()),
        rtol=5e-3,
        atol=5e-3,
    )

    print("\n" + "=" * 80)
    print("TEST PASSED: Function with no return works correctly!")
    print("=" * 80)


if __name__ == "__main__":
    test_function_no_return()