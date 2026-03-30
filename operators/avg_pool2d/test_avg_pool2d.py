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
avg_pool2d operator test.

测试 PyPTO avg_pool2d 算子的精度和验证。
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from avg_pool2d_golden import avg_pool2d_golden
from avg_pool2d_impl import avg_pool2d_wrapper


# ─────────────────────────────────────────────
# 1. 环境工具
# ─────────────────────────────────────────────

def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID。"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        print("Or use --run_mode sim to run in simulation mode")
        return None
    try:
        device_id = int(os.environ["TILE_FWK_DEVICE_ID"])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# ─────────────────────────────────────────────
# 2. 测试用例
# ─────────────────────────────────────────────

TEST_CASES = [
    # P0 性能测试
    {
        "name": "SAME_P0",
        "type": "performance",
        "priority": "P0",
        "shape": (2, 3, 6, 6),
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding_mode": "SAME",
        "expected_out_shape": (2, 3, 3, 3),
    },
    # P0 功能测试
    {
        "name": "VALID_P0",
        "type": "functional",
        "priority": "P0",
        "shape": (4, 8, 12, 12),
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding_mode": "VALID",
        "expected_out_shape": (4, 8, 5, 5),
    },
    # P1 性能测试
    {
        "name": "SAME_large",
        "type": "performance",
        "priority": "P1",
        "shape": (8, 64, 56, 56),
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding_mode": "SAME",
        "expected_out_shape": (8, 64, 28, 28),
    },
    # P2 功能测试
    {
        "name": "VALID_stride1",
        "type": "functional",
        "priority": "P2",
        "shape": (2, 16, 8, 8),
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding_mode": "VALID",
        "expected_out_shape": (2, 16, 7, 7),
    },
]


# ─────────────────────────────────────────────
# 3. 单个测试
# ─────────────────────────────────────────────

def run_single_test(
    case: dict,
    device: str,
    rtol: float = 0.001,
    atol: float = 0.001,
) -> bool:
    """
    运行单个测试用例。

    Returns:
        True if passed, False otherwise.
    """
    print(f"\n[{case['name']}] Running test...")
    print(f"  Input shape: {case['shape']}")
    print(f"  Kernel: {case['kernel_size']}, Stride: {case['stride']}, Padding: {case['padding_mode']}")

    # 生成输入数据
    np.random.seed(42)
    x_np = np.random.randn(*case["shape"]).astype(np.float32)
    x = torch.from_numpy(x_np).to(device)

    # 计算 golden 输出
    golden_output = avg_pool2d_golden(
        x,
        case["kernel_size"],
        case["stride"],
        case["padding_mode"],
    )

    # 计算 impl 输出
    impl_output = avg_pool2d_wrapper(
        x,
        case["kernel_size"],
        case["stride"],
        case["padding_mode"],
    )

    # 检查 shape
    if impl_output.shape != case["expected_out_shape"]:
        print(f"  FAILED: Shape mismatch. Expected {case['expected_out_shape']}, got {impl_output.shape}")
        return False

    if impl_output.shape != golden_output.shape:
        print(f"  FAILED: Shape mismatch with golden. Golden: {golden_output.shape}, Impl: {impl_output.shape}")
        return False

    # 精度对比
    try:
        assert_allclose(
            impl_output.cpu().numpy(),
            golden_output.cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
        max_diff = np.abs(impl_output.cpu().numpy() - golden_output.cpu().numpy()).max()
        print(f"  PASSED: Max diff = {max_diff:.6e}")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


# ─────────────────────────────────────────────
# 4. 主测试函数
# ─────────────────────────────────────────────

def run_tests(run_mode: str = "npu", test_filter: str = None):
    """
    运行所有测试用例。

    Args:
        run_mode: "npu" 或 "sim"
        test_filter: 可选，指定要运行的测试名称
    """
    print("=" * 60)
    print("avg_pool2d Test Suite")
    print("=" * 60)

    # 设置设备
    if run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            print("ERROR: NPU mode requires TILE_FWK_DEVICE_ID to be set")
            sys.exit(1)
        import torch_npu
        torch.npu.set_device(device_id)
        device = f"npu:{device_id}"
        print(f"Running on NPU device {device_id}")
    else:
        device = "cpu"
        print("Running in simulation mode (CPU)")

    # 运行测试
    passed = 0
    failed = 0
    total = 0

    for case in TEST_CASES:
        # 过滤测试
        if test_filter and test_filter not in case["name"]:
            continue

        total += 1
        try:
            if run_single_test(case, device):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # 汇总
    print("\n" + "=" * 60)
    print(f"Test Summary: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 60)

    # 输出三态标记
    if failed == 0:
        print("[PRECISION_PASS]")
        return 0
    else:
        print("[PRECISION_FAIL]")
        return 1


# ─────────────────────────────────────────────
# 5. 命令行入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="avg_pool2d test suite")
    parser.add_argument(
        "--run_mode",
        type=str,
        default="npu",
        choices=["npu", "sim"],
        help="Run mode: npu or sim",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Run specific test by name (e.g., SAME_P0)",
    )
    args = parser.parse_args()

    exit_code = run_tests(run_mode=args.run_mode, test_filter=args.test)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
