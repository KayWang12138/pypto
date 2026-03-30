#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""GroupNorm 算子测试文件."""

import sys
import os
import argparse
import torch
from numpy.testing import assert_allclose

from groupnorm_golden import groupnorm_golden
from groupnorm_impl import groupnorm_wrapper


def get_device_id():
    """Get TILE_FWK_DEVICE_ID from environment."""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    return int(os.environ["TILE_FWK_DEVICE_ID"])


# ============================================================
# 测试用例
# ============================================================

def test_groupnorm_basic(device=None):
    """测试基础配置."""
    print("\n[Test Basic]")
    N, C, H, W = 2, 32, 8, 8
    num_groups = 8

    x = torch.randn(N, C, H, W, dtype=torch.float32, device=device)
    weight = torch.randn(C, dtype=torch.float32, device=device)
    bias = torch.randn(C, dtype=torch.float32, device=device)

    golden_output = groupnorm_golden(x, num_groups, weight, bias)
    impl_output = groupnorm_wrapper(x, num_groups, weight, bias)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-4, atol=1e-4)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-4, atol=1e-4)

    print(f"  Input shape: {x.shape}, Groups: {num_groups}")
    print("[PRECISION_PASS]")


def test_groupnorm_resnet_style(device=None):
    """测试 ResNet 风格配置."""
    print("\n[Test ResNet Style]")
    N, C, H, W = 2, 64, 14, 14
    num_groups = 32

    x = torch.randn(N, C, H, W, dtype=torch.float32, device=device)
    weight = torch.randn(C, dtype=torch.float32, device=device)
    bias = torch.randn(C, dtype=torch.float32, device=device)

    golden_output = groupnorm_golden(x, num_groups, weight, bias)
    impl_output = groupnorm_wrapper(x, num_groups, weight, bias)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-4, atol=1e-4)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-4, atol=1e-4)

    print(f"  Input shape: {x.shape}, Groups: {num_groups}")
    print("[PRECISION_PASS]")


def test_groupnorm_small(device=None):
    """测试小规模配置."""
    print("\n[Test Small]")
    N, C, H, W = 1, 16, 4, 4
    num_groups = 4

    x = torch.randn(N, C, H, W, dtype=torch.float32, device=device)
    weight = torch.randn(C, dtype=torch.float32, device=device)
    bias = torch.randn(C, dtype=torch.float32, device=device)

    golden_output = groupnorm_golden(x, num_groups, weight, bias)
    impl_output = groupnorm_wrapper(x, num_groups, weight, bias)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-4, atol=1e-4)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-4, atol=1e-4)

    print(f"  Input shape: {x.shape}, Groups: {num_groups}")
    print("[PRECISION_PASS]")


def test_groupnorm_no_affine(device=None):
    """测试无 affine 变换."""
    print("\n[Test No Affine]")
    N, C, H, W = 2, 32, 8, 8
    num_groups = 8

    x = torch.randn(N, C, H, W, dtype=torch.float32, device=device)

    golden_output = groupnorm_golden(x, num_groups, weight=None, bias=None)
    impl_output = groupnorm_wrapper(x, num_groups, weight=None, bias=None)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-4, atol=1e-4)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-4, atol=1e-4)

    print(f"  Input shape: {x.shape}, Groups: {num_groups}, No affine")
    print("[PRECISION_PASS]")


def test_groupnorm_fp16(device=None):
    """测试 FP16 输入."""
    print("\n[Test FP16]")
    N, C, H, W = 2, 32, 8, 8
    num_groups = 8

    x = torch.randn(N, C, H, W, dtype=torch.float16, device=device)
    weight = torch.randn(C, dtype=torch.float16, device=device)
    bias = torch.randn(C, dtype=torch.float16, device=device)

    golden_output = groupnorm_golden(x, num_groups, weight, bias)
    impl_output = groupnorm_wrapper(x, num_groups, weight, bias)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-3, atol=1e-3)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-3, atol=1e-3)

    print(f"  Input shape: {x.shape}, Groups: {num_groups}, FP16")
    print("[PRECISION_PASS]")


def test_groupnorm_large(device=None):
    """测试大规模配置."""
    print("\n[Test Large]")
    N, C, H, W = 4, 128, 28, 28
    num_groups = 32

    x = torch.randn(N, C, H, W, dtype=torch.float32, device=device)
    weight = torch.randn(C, dtype=torch.float32, device=device)
    bias = torch.randn(C, dtype=torch.float32, device=device)

    golden_output = groupnorm_golden(x, num_groups, weight, bias)
    impl_output = groupnorm_wrapper(x, num_groups, weight, bias)

    if device is not None and str(device).startswith("npu"):
        assert_allclose(impl_output.cpu().numpy(), golden_output.cpu().numpy(), rtol=1e-4, atol=1e-4)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=1e-4, atol=1e-4)

    print(f"  Input shape: {x.shape}, Groups: {num_groups}")
    print("[PRECISION_PASS]")


# ============================================================
# 运行所有测试
# ============================================================

def run_all_tests(device=None):
    """运行所有测试."""
    print("=" * 60)
    print("GroupNorm Operator Tests")
    print("=" * 60)

    try:
        test_groupnorm_basic(device)
        test_groupnorm_resnet_style(device)
        test_groupnorm_small(device)
        test_groupnorm_no_affine(device)
        test_groupnorm_fp16(device)
        test_groupnorm_large(device)

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
    parser = argparse.ArgumentParser(description="GroupNorm operator test")
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
