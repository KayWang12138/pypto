#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file in the compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Simple end-to-end test for Dequantize operation
"""
import pypto
import numpy as np


def test_dequantize_symmetric_int8_basic():
    """Test basic symmetric dequantization: INT8 -> FP32"""
    print("=" * 60)
    print("Test: Symmetric Dequantization (INT8 -> FP32)")
    print("=" * 60)

    # Define shapes
    shape = (3, 4)
    tiles = (4, 16)

    # Create tensors
    x = pypto.tensor(shape, pypto.DT_INT8, "x")
    scale = pypto.tensor((3, 1), pypto.DT_FP32, "scale")
    y = pypto.tensor(shape, pypto.DT_FP32, "y")

    # Build computation graph
    with pypto.function("dequantize_sym_int8", x, scale, y):
        for _ in pypto.loop(1, name="dequantizeLoop"):
            pypto.set_vec_tile_shapes(*tiles)
            y[:] = pypto.dequantize(x, scale, pypto.DT_FP32, -1)

    # Compile
    print("Compiling dequantize operation...")
    success = pypto.compile()
    if success:
        print("✓ Compilation successful")
    else:
        print("✗ Compilation failed")
        return False

    return True


def test_dequantize_symmetric_int16_basic():
    """Test basic symmetric dequantization: INT16 -> FP32"""
    print("\n" + "=" * 60)
    print("Test: Symmetric Dequantization (INT16 -> FP32)")
    print("=" * 60)

    # Define shapes
    shape = (3, 4)
    tiles = (4, 16)

    # Create tensors
    x = pypto.tensor(shape, pypto.DT_INT16, "x")
    scale = pypto.tensor((3, 1), pypto.DT_FP32, "scale")
    y = pypto.tensor(shape, pypto.DT_FP32, "y")

    # Build computation graph
    with pypto.function("dequantize_sym_int16", x, scale, y):
        for _ in pypto.loop(1, name="dequantizeLoop"):
            pypto.set_vec_tile_shapes(*tiles)
            y[:] = pypto.dequantize(x, scale, pypto.DT_FP32, -1)

    # Compile
    print("Compiling dequantize operation...")
    success = pypto.compile()
    if success:
        print("✓ Compilation successful")
    else:
        print("✗ Compilation failed")
        return False

    return True


def test_dequantize_asymmetric_int8_basic():
    """Test basic asymmetric dequantization: INT8 -> FP32"""
    print("\n" + "=" * 60)
    print("Test: Asymmetric Dequantization (INT8 -> FP32)")
    print("=" * 60)

    # Define shapes
    shape = (3, 4)
    tiles = (4, 16)

    # Create tensors
    x = pypto.tensor(shape, pypto.DT_INT8, "x")
    scale = pypto.tensor((3, 1), pypto.DT_FP32, "scale")
    zero_points = pypto.tensor((3, 1), pypto.DT_FP32, "zero_points")
    y = pypto.tensor(shape, pypto.DT_FP32, "y")

    # Build computation graph
    with pypto.function("dequantize_asym_int8", x, scale, zero_points, y):
        for _ in pypto.loop(1, name="dequantizeLoop"):
            pypto.set_vec_tile_shapes(*tiles)
            y[:] = pypto.dequantize(x, scale, pypto.DT_FP32, -1, zero_points)

    # Compile
    print("Compiling dequantize operation...")
    success = pypto.compile()
    if success:
        print("✓ Compilation successful")
    else:
        print("✗ Compilation failed")
        return False

    return True


def test_dequantize_asymmetric_int16_basic():
    """Test basic asymmetric dequantization: INT16 -> FP32"""
    print("\n" + "=" * 60)
    print("Test: Asymmetric Dequantization (INT16 -> FP32)")
    print("=" * 60)

    # Define shapes
    shape = (3, 4)
    tiles = (4, 16)

    # Create tensors
    x = pypto.tensor(shape, pypto.DT_INT16, "x")
    scale = pypto.tensor((3, 1), pypto.DT_FP32, "scale")
    zero_points = pypto.tensor((3, 1), pypto.DT_FP32, "zero_points")
    y = pypto.tensor(shape, pypto.DT_FP32, "y")

    # Build computation graph
    with pypto.function("dequantize_asym_int16", x, scale, zero_points, y):
        for _ in pypto.loop(1, name="dequantizeLoop"):
            pypto.set_vec_tile_shapes(*tiles)
            y[:] = pypto.dequantize(x, scale, pypto.DT_FP32, -1, zero_points)

    # Compile
    print("Compiling dequantize operation...")
    success = pypto.compile()
    if success:
        print("✓ Compilation successful")
    else:
        print("✗ Compilation failed")
        return False

    return True


def test_dequantize_axis2():
    """Test dequantization with axis=-2"""
    print("\n" + "=" * 60)
    print("Test: Dequantization with axis=-2")
    print("=" * 60)

    # Define shapes
    shape = (4, 8)
    tiles = (4, 16)

    # Create tensors
    x = pypto.tensor(shape, pypto.DT_INT8, "x")
    scale = pypto.tensor((1, 8), pypto.DT_FP32, "scale")
    y = pypto.tensor(shape, pypto.DT_FP32, "y")

    # Build computation graph
    with pypto.function("dequantize_axis2", x, scale, y):
        for _ in pypto.loop(1, name="dequantizeLoop"):
            pypto.set_vec_tile_shapes(*tiles)
            y[:] = pypto.dequantize(x, scale, pypto.DT_FP32, -2)

    # Compile
    print("Compiling dequantize operation with axis=-2...")
    success = pypto.compile()
    if success:
        print("✓ Compilation successful")
    else:
        print("✗ Compilation failed")
        return False

    return True


def test_dequantize_3d():
    """Test dequantization with 3D input"""
    print("\n" + "=" * 60)
    print("Test: Dequantization with 3D input")
    print("=" * 60)

    # Define shapes
    shape = (2, 4, 8)
    tiles = (4, 16)

    # Create tensors
    x = pypto.tensor(shape, pypto.DT_INT8, "x")
    scale = pypto.tensor((2, 1, 1), pypto.DT_FP32, "scale")
    y = pypto.tensor(shape, pypto.DT_FP32, "y")

    # Build computation graph
    with pypto.function("dequantize_3d", x, scale, y):
        for _ in pypto.loop(1, name="dequantizeLoop"):
            pypto.set_vec_tile_shapes(*tiles)
            y[:] = pypto.dequantize(x, scale, pypto.DT_FP32, -1)

    # Compile
    print("Compiling dequantize operation with 3D input...")
    success = pypto.compile()
    if success:
        print("✓ Compilation successful")
    else:
        print("✗ Compilation failed")
        return False

    return True


def test_dequantize_4d():
    """Test dequantization with 4D input"""
    print("\n" + "=" * 60)
    print("Test: Dequantization with 4D input")
    print("=" * 60)

    # Define shapes
    shape = (2, 2, 4, 8)
    tiles = (4, 16)

    # Create tensors
    x = pypto.tensor(shape, pypto.DT_INT8, "x")
    scale = pypto.tensor((2, 2, 1, 1), pypto.DT_FP32, "scale")
    y = pypto.tensor(shape, pypto.DT_FP32, "y")

    # Build computation graph
    with pypto.function("dequantize_4d", x, scale, y):
        for _ in pypto.loop(1, name="dequantizeLoop"):
            pypto.set_vec_tile_shapes(*tiles)
            y[:] = pypto.dequantize(x, scale, pypto.DT_FP32, -1)

    # Compile
    print("Compiling dequantize operation with 4D input...")
    success = pypto.compile()
    if success:
        print("✓ Compilation successful")
    else:
        print("✗ Compilation failed")
        return False

    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Dequantize Operation Tests")
    print("=" * 60)

    tests = [
        ("Symmetric INT8 (axis=-1)", test_dequantize_symmetric_int8_basic),
        ("Symmetric INT16 (axis=-1)", test_dequantize_symmetric_int16_basic),
        ("Asymmetric INT8 (axis=-1)", test_dequantize_asymmetric_int8_basic),
        ("Asymmetric INT16 (axis=-1)", test_dequantize_asymmetric_int16_basic),
        ("Axis=-2", test_dequantize_axis2),
        ("3D Input", test_dequantize_3d),
        ("4D Input", test_dequantize_4d),
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
