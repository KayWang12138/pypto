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

import numpy as np
import torch
import pypto
from numpy.testing import assert_allclose


@pypto.frontend.function
def inner_add(
    x: pypto.Tensor((8,), pypto.DT_FP32),
    bias: pypto.Tensor((8,), pypto.DT_FP32),
) -> (
    pypto.Tensor((8,), pypto.DT_FP32),
):
    tmp = pypto.add(x, bias)
    return tmp


@pypto.frontend.jit
def outer_kernel(
    a: pypto.Tensor((8,), pypto.DT_FP32),
    b: pypto.Tensor((8,), pypto.DT_FP32),
) -> (
    pypto.Tensor((8,), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 16, 32)
    res = inner_add(a, b)
    return res


def _reference_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.add(a, b)


def test_nested_function_kernel_inline_expansion():
    print("=" * 80)
    print("Testing Nested Function Kernel Inline Expansion")
    print("=" * 80)

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    print(f"\nDevice ID: {device_id}")
    shape = (8,)
    print(f"Input shape: {shape}")

    # Generate test data
    print("\nGenerating test data...")
    a = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")

    print(f"Input a: {a}")
    print(f"Input b: {b}")

    # Execute nested function kernel
    print("\nExecuting nested function kernel (outer_kernel calling inner_add)...")
    result = outer_kernel(a, b)
    pypto.runtime._device_synchronize()

    print(f"Kernel output: {result}")

    # Compute reference
    print("\nComputing reference result...")
    reference = _reference_impl(a, b)
    print(f"Reference output: {reference}")

    # Validate result
    print("\nValidating results...")
    assert_allclose(
        np.array(result.cpu().flatten().tolist()),
        np.array(reference.cpu().flatten().tolist()),
        rtol=5e-3,
        atol=5e-3,
    )

    print("\n" + "=" * 80)
    print("TEST PASSED: Nested function inline expansion works correctly!")
    print("=" * 80)

if __name__ == "__main__":
    test_nested_function_kernel_inline_expansion()