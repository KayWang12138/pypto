#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Simple end-to-end test for Quantize operation
"""
import pypto
import numpy as np


def test_quantize_symmetric_int8_basic():
    """Test basic symmetric quantization: FP32 -> INT8"""
    print("=" * 60)
    print("Test: Symmetric Quantization (FP32 -> INT8)")
    print("=" * 60)

    # Define shapes
    shape = (3, 4)
    tiles = (4, 16)

    # Create tensors
    x = pypto.tensor(shape, pypto.DT_FP32, "x")
    scale = pypto.tensor((3,), pypto.DT_FP32, "scale")
    y = pypto.tensor(shape, pypto.DT_INT8, "y")

    # Build computation graph
    with pypto.function("quantize_sym_int8", x, scale, y):
        for _ in pypto.loop(1, name="quantizeLoop"):
            pypto.set_vec_tile_shapes(*tiles)
            y[:] = pypto.quantize(x, scale, pypto.DT_INT8, -1)

    # Compile
    print("Compiling quantize operation...")
    success = pypto.compile()
    if success:
        print("✓ Compilation successful")
    else:
        print("✗ Compilation failed")
        return False

    return True


def test_quantize_asymmetric_uint8_basic():
    """Test basic asymmetric quantization: FP32 -> UINT8"""
    print("\n" + "=" * 60)
    print("Test: Asymmetric Quantization (FP32 -> UINT8)")
    print("=" * 60)

    # Define shapes
    shape = (3, 4)
    tiles = (4, 16)

    # Create tensors
    x = pypto.tensor(shape, pypto.DT_FP32, "x")
    scale = pypto.tensor((3, 1), pypto.DT_FP32, "scale")
    zero_points = pypto.tensor((3, 1), pypto.DT_FP32, "zero_points")
    y = pypto.tensor(shape, pypto.DT_UINT8, "y")

    # Build computation graph
    with pypto.function("quantize_asym_uint8", x, scale, zero_points, y):
        for _ in pypto.loop(1, name="quantizeLoop"):
            pypto.set_vec_tile_shapes(*tiles)
            y[:] = pypto.quantize(x, scale, pypto.DT_UINT8, -1, zero_points)

    # Compile
    print("Compiling quantize operation...")
    success = pypto.compile()
    if success:
        print("✓ Compilation successful")
    else:
        print("✗ Compilation failed")
        return False

    return True


def test_quantize_axis2():
    """Test quantization with axis=-2"""
    print("\n" + "=" * 60)
    print("Test: Quantization with axis=-2")
    print("=" * 60)

    # Define shapes
    shape = (4, 8)
    tiles = (4, 16)

    # Create tensors
    x = pypto.tensor(shape, pypto.DT_FP32, "x")
    scale = pypto.tensor((1, 8), pypto.DT_FP32, "scale")
    y = pypto.tensor(shape, pypto.DT_INT8, "y")

    # Build computation graph
    with pypto.function("quantize_axis2", x, scale, y):
        for _ in pypto.loop(1, name="quantizeLoop"):
            pypto.set_vec_tile_shapes(*tiles)
            y[:] = pypto.quantize(x, scale, pypto.DT_INT8, -2)

    # Compile
    print("Compiling quantize operation with axis=-2...")
    success = pypto.compile()
    if success:
        print("✓ Compilation successful")
    else:
        print("✗ Compilation failed")
        return False

    return True


def test_quantize_3d():
    """Test quantization with 3D input"""
    print("\n" + "=" * 60)
    print("Test: Quantization with 3D input")
    print("=" * 60)

    # Define shapes
    shape = (2, 4, 8)
    tiles = (4, 16)

    # Create tensors
    x = pypto.tensor(shape, pypto.DT_FP32, "x")
    scale = pypto.tensor((2, 1, 1), pypto.DT_FP32, "scale")
    y = pypto.tensor(shape, pypto.DT_INT8, "y")

    # Build computation graph
    with pypto.function("quantize_3d", x, scale, y):
        for _ in pypto.loop(1, name="quantizeLoop"):
            pypto.set_vec_tile_shapes(*tiles)
            y[:] = pypto.quantize(x, scale, pypto.DT_INT8, -1)

    # Compile
    print("Compiling quantize operation with 3D input...")
    success = pypto.compile()
    if success:
        print("✓ Compilation successful")
    else:
        print("✗ Compilation failed")
        return False

    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Quantize Operation Tests")
    print("=" * 60)

    tests = [
        ("Symmetric INT8 (axis=-1)", test_quantize_symmetric_int8_basic),
        ("Asymmetric UINT8 (axis=-1)", test_quantize_asymmetric_uint8_basic),
        ("Axis=-2", test_quantize_axis2),
        ("3D Input", test_quantize_3d),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test '{name}' failed with error: {e}")
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<40} {status}")

    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
