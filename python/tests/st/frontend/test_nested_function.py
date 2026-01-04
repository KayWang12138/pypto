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

"""
"""
import os
import logging
import numpy as np
import torch
import pypto
from numpy.testing import assert_allclose
from test_function_no_return import outer_kernel_inplace

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


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
    logging.info("=" * 80)
    logging.info("Testing Nested Function Kernel Inline Expansion")
    logging.info("=" * 80)

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    logging.info(f"\nDevice ID: {device_id}")
    shape = (8,)
    logging.info(f"Input shape: {shape}")

    # Generate test data
    logging.info("\nGenerating test data...")
    a = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")

    logging.info(f"Input a: {a}")
    logging.info(f"Input b: {b}")

    # Execute nested function kernel
    logging.info("\nExecuting nested function kernel (outer_kernel calling inner_add)...")
    result = outer_kernel(a, b)

    logging.info(f"Kernel output: {result}")

    # Compute reference
    logging.info("\nComputing reference result...")
    reference = _reference_impl(a, b)
    logging.info(f"Reference output: {reference}")

    # Validate result
    logging.info("\nValidating results...")
    assert_allclose(
        np.array(result.cpu().flatten().tolist()),
        np.array(reference.cpu().flatten().tolist()),
        rtol=5e-3,
        atol=5e-3,
    )

    logging.info("\n" + "=" * 80)
    logging.info("TEST PASSED: Nested function inline expansion works correctly!")
    logging.info("=" * 80)


def test_nested_function_no_return():
    logging.info("=" * 80)
    logging.info("Testing Nested Function Inline Expansion (No Return)")
    logging.info("=" * 80)

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    logging.info(f"\nDevice ID: {device_id}")
    shape = (8,)
    logging.info(f"Input shape: {shape}")

    logging.info("\nGenerating test data...")
    a = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")
    out = torch.empty(shape, dtype=torch.float32, device=f"npu:{device_id}")

    logging.info("\nExecuting nested function kernel (no return value)...")
    result = outer_kernel_inplace(a, b, out)

    assert result is None

    logging.info("\nComputing reference result...")
    reference = _reference_impl(a, b)

    logging.info("\nValidating results...")
    assert_allclose(
        np.array(out.cpu().flatten().tolist()),
        np.array(reference.cpu().flatten().tolist()),
        rtol=5e-3,
        atol=5e-3,
    )

    logging.info("\n" + "=" * 80)
    logging.info("TEST PASSED: Nested function with no return works correctly!")
    logging.info("=" * 80)

if __name__ == "__main__":
    test_nested_function_kernel_inline_expansion()
    test_nested_function_no_return()
