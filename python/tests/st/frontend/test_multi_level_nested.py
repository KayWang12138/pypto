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
import pypto
import torch
from numpy.testing import assert_allclose

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


def test_multi_level_nested():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    print(f"Running on NPU device {device_id}")
    shape = (16, 32)
    print(f"Input shape: {shape}")

    # Generate test data
    print("\nGenerating test data...")
    np.random.seed(42)
    input1 = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")
    input2 = torch.rand(shape, dtype=torch.float32, device=f"npu:{device_id}")

    print(f"Input1 stats: mean={input1.mean().item():.4f}, std={input1.std().item():.4f}")
    print(f"Input2 stats: mean={input2.mean().item():.4f}, std={input2.std().item():.4f}")
    print(f"Input1 sample (first 5 elements): {input1.flatten()[:5].tolist()}")
    print(f"Input2 sample (first 5 elements): {input2.flatten()[:5].tolist()}")

    # Execute multi-level nested kernel
    output_tensor = top_level_kernel(input1, input2)
    pypto.runtime._device_synchronize()

    print(f"\nKernel output shape: {output_tensor.shape}")
    print(f"Kernel output stats: mean={output_tensor.mean().item():.4f}, std={output_tensor.std().item():.4f}")
    print(f"Kernel output sample (first 5 elements): {output_tensor.flatten()[:5].tolist()}")

    # Compute reference
    print("\nComputing reference result...")
    print("  Formula: (input1 * input2) + 2.0")
    output_ref = (input1 * input2) + 2.0

    print(f"Reference output stats: mean={output_ref.mean().item():.4f}, std={output_ref.std().item():.4f}")
    print(f"Reference output sample (first 5 elements): {output_ref.flatten()[:5].tolist()}")

    # Validate results
    print("\nValidating results...")
    assert_allclose(
        np.array(output_tensor.cpu().flatten().tolist()),
        np.array(output_ref.cpu().flatten().tolist()),
        rtol=5e-3,
        atol=5e-3,
    )

    print("\n" + "=" * 80)
    print("TEST PASSED: Multi-level nested function calls work correctly!")
    print("=" * 80)

if __name__ == "__main__":
    test_multi_level_nested()