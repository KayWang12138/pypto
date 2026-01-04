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
import pypto
import torch
from numpy.testing import assert_allclose

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


@pypto.frontend.function
def inner_compute(
    x: pypto.Tensor([16, 32], pypto.DT_FP32),
    y: pypto.Tensor([16, 32], pypto.DT_FP32),
) -> pypto.Tensor([16, 32], pypto.DT_FP32):
    mul_res = pypto.mul(x, y)
    return mul_res


@pypto.frontend.function
def middle_layer(
    a: pypto.Tensor([16, 32], pypto.DT_FP32),
    b: pypto.Tensor([16, 32], pypto.DT_FP32),
) -> pypto.Tensor([16, 32], pypto.DT_FP32):
    mul_res = inner_compute(a, b)
    add_res = pypto.add(mul_res, 2.0)
    return add_res


@pypto.frontend.jit
def top_level_kernel(
    in1: pypto.Tensor([16, 32], pypto.DT_FP32),
    in2: pypto.Tensor([16, 32], pypto.DT_FP32),
) -> pypto.Tensor([16, 32], pypto.DT_FP32):
    pypto.set_vec_tile_shapes(16, 32)
    res = middle_layer(in1, in2)
    return res


@pypto.frontend.function
def inner_compute_inplace(
    x: pypto.Tensor([16, 32], pypto.DT_FP32),
    y: pypto.Tensor([16, 32], pypto.DT_FP32),
    out: pypto.Tensor([16, 32], pypto.DT_FP32),
):
    out[:] = pypto.mul(x, y)
    return None


@pypto.frontend.function
def middle_layer_inplace(
    a: pypto.Tensor([16, 32], pypto.DT_FP32),
    b: pypto.Tensor([16, 32], pypto.DT_FP32),
    out: pypto.Tensor([16, 32], pypto.DT_FP32),
):
    inner_compute_inplace(a, b, out)
    out[:] = pypto.add(out, 2.0)


@pypto.frontend.jit
def top_level_kernel_inplace(
    in1: pypto.Tensor([16, 32], pypto.DT_FP32),
    in2: pypto.Tensor([16, 32], pypto.DT_FP32),
    out: pypto.Tensor([16, 32], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(16, 32)
    middle_layer_inplace(in1, in2, out)


def test_multi_level_nested():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    logging.info(f"Running on NPU device {device_id}")
    shape = (16, 32)
    logging.info(f"Input shape: {shape}")

    # Generate test data
    logging.info("\nGenerating test data...")
    np.random.seed(42)
    input1 = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")
    input2 = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")

    logging.info(f"Input1 stats: mean={input1.mean().item():.4f}, std={input1.std().item():.4f}")
    logging.info(f"Input2 stats: mean={input2.mean().item():.4f}, std={input2.std().item():.4f}")
    logging.info(f"Input1 sample (first 5 elements): {input1.flatten()[:5].tolist()}")
    logging.info(f"Input2 sample (first 5 elements): {input2.flatten()[:5].tolist()}")

    # Execute multi-level nested kernel
    output_tensor = top_level_kernel(input1, input2)

    logging.info(f"\nKernel output shape: {output_tensor.shape}")
    logging.info(f"Kernel output stats: mean={output_tensor.mean().item():.4f}, std={output_tensor.std().item():.4f}")
    logging.info(f"Kernel output sample (first 5 elements): {output_tensor.flatten()[:5].tolist()}")

    # Compute reference
    logging.info("\nComputing reference result...")
    logging.info("  Formula: (input1 * input2) + 2.0")
    output_ref = (input1 * input2) + 2.0

    logging.info(f"Reference output stats: mean={output_ref.mean().item():.4f}, std={output_ref.std().item():.4f}")
    logging.info(f"Reference output sample (first 5 elements): {output_ref.flatten()[:5].tolist()}")

    # Validate results
    logging.info("\nValidating results...")
    assert_allclose(
        np.array(output_tensor.cpu().flatten().tolist()),
        np.array(output_ref.cpu().flatten().tolist()),
        rtol=5e-3,
        atol=5e-3,
    )

    logging.info("\n" + "=" * 80)
    logging.info("TEST PASSED: Multi-level nested function calls work correctly!")
    logging.info("=" * 80)


def test_multi_level_nested_no_return():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    logging.info(f"Running on NPU device {device_id} (no return)")
    shape = (16, 32)
    logging.info(f"Input shape: {shape}")

    logging.info("\nGenerating test data...")
    np.random.seed(123)
    input1 = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")
    input2 = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")
    output_tensor = torch.empty(shape, dtype=torch.float32, device=f"npu:{device_id}")

    result = top_level_kernel_inplace(input1, input2, output_tensor)

    assert result is None

    logging.info("\nComputing reference result...")
    output_ref = (input1 * input2) + 2.0

    logging.info("\nValidating results...")
    assert_allclose(
        np.array(output_tensor.cpu().flatten().tolist()),
        np.array(output_ref.cpu().flatten().tolist()),
        rtol=5e-3,
        atol=5e-3,
    )

    logging.info("\n" + "=" * 80)
    logging.info("TEST PASSED: Multi-level nested function with no return calls work correctly!")
    logging.info("=" * 80)


if __name__ == "__main__":
    test_multi_level_nested()
    test_multi_level_nested_no_return()
