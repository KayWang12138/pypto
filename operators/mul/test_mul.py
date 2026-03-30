#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""Mul 算子测试文件."""

import sys
import os
import argparse
import torch
from numpy.testing import assert_allclose

from mul_golden import mul_golden
from mul_impl import mul_wrapper


def get_device_id():
    """Get TILE_FWK_DEVICE_ID from environment."""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    return int(os.environ["TILE_FWK_DEVICE_ID"])


# ============================================================
# 测试用例
# ============================================================

def test_mul_1d(device=None):
    """测试 1D 输入."""
    print("\n[Test 1D]")
    x1 = torch.randn(1024, dtype=torch.float32, device=device)
    x2 = torch.randn(1024, dtype=torch.float32, device=device)

    golden_output = mul_golden(x1, x2)
    impl_output = mul_wrapper(x1, x2)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shapes: {x1.shape}, {x2.shape}")
    print("[PRECISION_PASS]")


def test_mul_2d(device=None):
    """测试 2D 输入."""
    print("\n[Test 2D]")
    x1 = torch.randn(128, 1024, dtype=torch.float32, device=device)
    x2 = torch.randn(128, 1024, dtype=torch.float32, device=device)

    golden_output = mul_golden(x1, x2)
    impl_output = mul_wrapper(x1, x2)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shapes: {x1.shape}, {x2.shape}")
    print("[PRECISION_PASS]")


def test_mul_3d(device=None):
    """测试 3D 输入."""
    print("\n[Test 3D]")
    x1 = torch.randn(4, 128, 1024, dtype=torch.float32, device=device)
    x2 = torch.randn(4, 128, 1024, dtype=torch.float32, device=device)

    golden_output = mul_golden(x1, x2)
    impl_output = mul_wrapper(x1, x2)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shapes: {x1.shape}, {x2.shape}")
    print("[PRECISION_PASS]")


def test_mul_4d(device=None):
    """测试 4D 输入."""
    print("\n[Test 4D]")
    x1 = torch.randn(2, 4, 128, 1024, dtype=torch.float32, device=device)
    x2 = torch.randn(2, 4, 128, 1024, dtype=torch.float32, device=device)

    golden_output = mul_golden(x1, x2)
    impl_output = mul_wrapper(x1, x2)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shapes: {x1.shape}, {x2.shape}")
    print("[PRECISION_PASS]")


def test_mul_broadcast(device=None):
    """测试广播."""
    print("\n[Test Broadcast]")
    x1 = torch.randn(2, 4, 128, 1024, dtype=torch.float32, device=device)
    x2 = torch.randn(1, 4, 1, 1024, dtype=torch.float32, device=device)  # 可广播

    golden_output = mul_golden(x1, x2)
    impl_output = mul_wrapper(x1, x2)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shapes: {x1.shape} * {x2.shape} -> {impl_output.shape}")
    print("[PRECISION_PASS]")


def test_mul_scalar(device=None):
    """测试标量乘法."""
    print("\n[Test Scalar]")
    x1 = torch.randn(128, 1024, dtype=torch.float32, device=device)
    x2 = torch.tensor(2.5, dtype=torch.float32, device=device)  # 标量

    golden_output = mul_golden(x1, x2)
    impl_output = mul_wrapper(x1, x2)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shapes: {x1.shape} * scalar -> {impl_output.shape}")
    print("[PRECISION_PASS]")


def test_mul_fp16(device=None):
    """测试 FP16 输入."""
    print("\n[Test FP16]")
    x1 = torch.randn(128, 1024, dtype=torch.float16, device=device)
    x2 = torch.randn(128, 1024, dtype=torch.float16, device=device)

    golden_output = mul_golden(x1, x2)
    impl_output = mul_wrapper(x1, x2)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-3, atol=1e-3)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-3, atol=1e-3)

    print(f"  Input shapes: {x1.shape}, {x2.shape}, FP16")
    print("[PRECISION_PASS]")


def test_mul_large(device=None):
    """测试大规模输入."""
    print("\n[Test Large]")
    x1 = torch.randn(4096, 4096, dtype=torch.float32, device=device)
    x2 = torch.randn(4096, 4096, dtype=torch.float32, device=device)

    golden_output = mul_golden(x1, x2)
    impl_output = mul_wrapper(x1, x2)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shapes: {x1.shape}, {x2.shape}")
    print("[PRECISION_PASS]")


# ============================================================
# 运行所有测试
# ============================================================

def run_all_tests(device=None):
    """运行所有测试."""
    print("=" * 60)
    print("Mul Operator Tests")
    print("=" * 60)

    try:
        test_mul_1d(device)
        test_mul_2d(device)
        test_mul_3d(device)
        test_mul_4d(device)
        test_mul_broadcast(device)
        test_mul_scalar(device)
        test_mul_fp16(device)
        test_mul_large(device)

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
    parser = argparse.ArgumentParser(description="Mul operator test")
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
