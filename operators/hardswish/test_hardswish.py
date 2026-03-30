#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""Hardswish 算子测试文件."""

import sys
import os
import argparse
import torch
from numpy.testing import assert_allclose

from hardswish_golden import hardswish_golden
from hardswish_impl import hardswish_wrapper


def get_device_id():
    """Get TILE_FWK_DEVICE_ID from environment."""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    return int(os.environ["TILE_FWK_DEVICE_ID"])


# ============================================================
# 测试用例
# ============================================================

def test_hardswish_1d(device=None):
    """测试 1D 输入."""
    print("\n[Test 1D]")
    shape = (4096,)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    golden_output = hardswish_golden(x)
    impl_output = hardswish_wrapper(x)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_hardswish_2d(device=None):
    """测试 2D 输入."""
    print("\n[Test 2D]")
    shape = (128, 1024)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    golden_output = hardswish_golden(x)
    impl_output = hardswish_wrapper(x)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_hardswish_3d(device=None):
    """测试 3D 输入."""
    print("\n[Test 3D]")
    shape = (4, 128, 1024)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    golden_output = hardswish_golden(x)
    impl_output = hardswish_wrapper(x)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_hardswish_4d(device=None):
    """测试 4D 输入."""
    print("\n[Test 4D]")
    shape = (2, 4, 128, 1024)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    golden_output = hardswish_golden(x)
    impl_output = hardswish_wrapper(x)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_hardswish_performance(device=None):
    """测试性能场景."""
    print("\n[Test Performance]")
    shape = (4096, 4096)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    golden_output = hardswish_golden(x)
    impl_output = hardswish_wrapper(x)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_hardswish_fp16(device=None):
    """测试 FP16 输入."""
    print("\n[Test FP16]")
    shape = (128, 1024)
    x = torch.randn(shape, dtype=torch.float16, device=device)

    golden_output = hardswish_golden(x)
    impl_output = hardswish_wrapper(x)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-3, atol=1e-3)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-3, atol=1e-3)

    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_hardswish_boundary(device=None):
    """测试边界值."""
    print("\n[Test Boundary]")

    # Test with values near the hardswish boundaries (-3, 0, 3)
    x = torch.tensor([-4.0, -3.0, -2.0, 0.0, 1.0, 3.0, 4.0], dtype=torch.float32, device=device)

    golden_output = hardswish_golden(x)
    impl_output = hardswish_wrapper(x)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input: {x.cpu().tolist() if device else x.tolist()}")
    print(f"  Output: {impl_output.cpu().tolist() if device else impl_output.tolist()}")
    print("[PRECISION_PASS]")


# ============================================================
# 运行所有测试
# ============================================================

def run_all_tests(device=None):
    """运行所有测试."""
    print("=" * 60)
    print("Hardswish Operator Tests")
    print("=" * 60)

    try:
        test_hardswish_1d(device)
        test_hardswish_2d(device)
        test_hardswish_3d(device)
        test_hardswish_4d(device)
        test_hardswish_performance(device)
        test_hardswish_fp16(device)
        test_hardswish_boundary(device)

        print("\n" + "=" * 60)
        print("[PRECISION_PASS] All tests passed!")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n[PRECISION_FAIL] {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\nRuntime error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Hardswish operator test")
    parser.add_argument("--run_mode", default="npu", choices=["npu", "cpu"])
    args = parser.parse_args()

    device = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        device = f"npu:{device_id}"
        print(f"Running tests on NPU (device {device_id})...")
    else:
        print("Running tests on CPU...")

    success = run_all_tests(device)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
