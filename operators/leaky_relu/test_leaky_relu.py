#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""LeakyReLU 算子测试文件."""

import sys
import os
import argparse
import torch
from numpy.testing import assert_allclose

from leaky_relu_golden import leaky_relu_golden
from leaky_relu_impl import leaky_relu_wrapper


def get_device_id():
    """Get TILE_FWK_DEVICE_ID from environment."""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    return int(os.environ["TILE_FWK_DEVICE_ID"])


# ============================================================
# 测试用例
# ============================================================

def test_leaky_relu_1d(device=None, alpha=0.01):
    """测试 1D 输入."""
    print("\n[Test 1D]")
    shape = (4096,)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    golden_output = leaky_relu_golden(x, alpha)
    impl_output = leaky_relu_wrapper(x, alpha)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_leaky_relu_2d(device=None, alpha=0.01):
    """测试 2D 输入."""
    print("\n[Test 2D]")
    shape = (128, 1024)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    golden_output = leaky_relu_golden(x, alpha)
    impl_output = leaky_relu_wrapper(x, alpha)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_leaky_relu_3d(device=None, alpha=0.01):
    """测试 3D 输入."""
    print("\n[Test 3D]")
    shape = (4, 128, 1024)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    golden_output = leaky_relu_golden(x, alpha)
    impl_output = leaky_relu_wrapper(x, alpha)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_leaky_relu_4d(device=None, alpha=0.01):
    """测试 4D 输入."""
    print("\n[Test 4D]")
    shape = (2, 4, 128, 1024)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    golden_output = leaky_relu_golden(x, alpha)
    impl_output = leaky_relu_wrapper(x, alpha)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_leaky_relu_performance(device=None, alpha=0.01):
    """测试性能场景."""
    print("\n[Test Performance]")
    shape = (4096, 4096)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    golden_output = leaky_relu_golden(x, alpha)
    impl_output = leaky_relu_wrapper(x, alpha)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-5, atol=1e-5)

    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_leaky_relu_fp16(device=None, alpha=0.01):
    """测试 FP16 输入."""
    print("\n[Test FP16]")
    shape = (128, 1024)
    x = torch.randn(shape, dtype=torch.float16, device=device)

    golden_output = leaky_relu_golden(x, alpha)
    impl_output = leaky_relu_wrapper(x, alpha)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-3, atol=1e-3)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-3, atol=1e-3)

    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


# ============================================================
# 运行所有测试
# ============================================================

def run_all_tests(device=None, alpha=0.01):
    """运行所有测试."""
    print("=" * 60)
    print("LeakyReLU Operator Tests")
    print("=" * 60)

    try:
        test_leaky_relu_1d(device, alpha)
        test_leaky_relu_2d(device, alpha)
        test_leaky_relu_3d(device, alpha)
        test_leaky_relu_4d(device, alpha)
        test_leaky_relu_performance(device, alpha)
        test_leaky_relu_fp16(device, alpha)

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
    parser = argparse.ArgumentParser(description="LeakyReLU operator test")
    parser.add_argument("--alpha", type=float, default=0.01, help="Negative slope")
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

    success = run_all_tests(device, args.alpha)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
