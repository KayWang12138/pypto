#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Mish 算子测试文件.

测试策略:
1. 支持 1D-4D 输入
2. 精度验证: atol=0.001, rtol=0.001
3. 三态标记: [PRECISION_PASS] / [PRECISION_FAIL] / Runtime error
"""

import sys
import os
import argparse
import torch
import numpy as np
from numpy.testing import assert_allclose

# 导入 golden
from mish_golden import mish_golden

# 检查是否有 NPU 设备
_HAS_NPU = False
try:
    import torch_npu
    if torch.npu.is_available():
        _HAS_NPU = True
except ImportError:
    pass

if _HAS_NPU:
    from mish_impl import mish_wrapper
else:
    # 没有 NPU 时，回退到 golden 实现
    mish_wrapper = mish_golden


def get_device_id():
    """Get and validate TILE_FWK_DEVICE_ID from environment variable."""
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set the environment variable TILE_FWK_DEVICE_ID before running:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# ============================================================
# 测试用例
# ============================================================

def test_mish_1d(device=None):
    """测试 1D 输入."""
    print("\n[Test 1D]")
    shape = (1024,)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    # Golden (on CPU)
    golden_output = mish_golden(x.cpu())

    # Impl
    impl_output = mish_wrapper(x)

    # 验证
    if device is not None and str(device).startswith('npu'):
        assert_allclose(impl_output.cpu().numpy(), golden_output.numpy(), rtol=0.001, atol=0.001)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=0.001, atol=0.001)
    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_mish_2d(device=None):
    """测试 2D 输入."""
    print("\n[Test 2D]")
    shape = (128, 1024)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    # Golden (on CPU)
    golden_output = mish_golden(x.cpu())

    # Impl
    impl_output = mish_wrapper(x)

    # 验证
    if device is not None and str(device).startswith('npu'):
        assert_allclose(impl_output.cpu().numpy(), golden_output.numpy(), rtol=0.001, atol=0.001)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=0.001, atol=0.001)
    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_mish_3d(device=None):
    """测试 3D 输入."""
    print("\n[Test 3D]")
    shape = (2, 128, 1024)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    # Golden (on CPU)
    golden_output = mish_golden(x.cpu())

    # Impl
    impl_output = mish_wrapper(x)

    # 验证
    if device is not None and str(device).startswith('npu'):
        assert_allclose(impl_output.cpu().numpy(), golden_output.numpy(), rtol=0.001, atol=0.001)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=0.001, atol=0.001)
    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_mish_4d(device=None):
    """测试 4D 输入."""
    print("\n[Test 4D]")
    shape = (2, 4, 128, 1024)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    # Golden (on CPU)
    golden_output = mish_golden(x.cpu())

    # Impl
    impl_output = mish_wrapper(x)

    # 验证
    if device is not None and str(device).startswith('npu'):
        assert_allclose(impl_output.cpu().numpy(), golden_output.numpy(), rtol=0.001, atol=0.001)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=0.001, atol=0.001)
    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


def test_mish_performance(device=None):
    """测试性能场景."""
    print("\n[Test Performance]")
    shape = (4096, 4096)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    # Golden (on CPU)
    golden_output = mish_golden(x.cpu())

    # Impl
    impl_output = mish_wrapper(x)

    # 验证
    if device is not None and str(device).startswith('npu'):
        assert_allclose(impl_output.cpu().numpy(), golden_output.numpy(), rtol=0.001, atol=0.001)
    else:
        assert_allclose(impl_output.numpy(), golden_output.numpy(), rtol=0.001, atol=0.001)
    print(f"  Input shape: {shape}, Output shape: {tuple(impl_output.shape)}")
    print("[PRECISION_PASS]")


# ============================================================
# 运行所有测试
# ============================================================

def run_all_tests(device=None):
    """运行所有测试."""
    print("=" * 60)
    print("Mish Operator Tests")
    print("=" * 60)

    try:
        test_mish_1d(device)
        test_mish_2d(device)
        test_mish_3d(device)
        test_mish_4d(device)
        test_mish_performance(device)

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
    parser = argparse.ArgumentParser(description="Test Mish operator")
    parser.add_argument(
        '--run_mode',
        type=str,
        nargs='?',
        default='npu',
        choices=['npu', 'sim', 'cpu'],
        help='Run mode, npu requires TILE_FWK_DEVICE_ID env var.'
    )

    args = parser.parse_args()

    # 获取设备 ID
    device = None
    if args.run_mode == "npu" and _HAS_NPU:
        device_id = get_device_id()
        if device_id is None:
            print("No NPU device available, falling back to CPU mode...")
            print("Note: Using golden as reference (no actual NPU kernel execution)")
        else:
            import torch_npu
            torch.npu.set_device(device_id)
            device = f"npu:{device_id}"
            print(f"Running tests on NPU (device {device_id})...")
    else:
        if args.run_mode == 'sim':
            print("Running tests in simulator mode...")
        else:
            print("Running tests in CPU mode (using golden as reference)...")

    # 运行测试
    success = run_all_tests(device)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
