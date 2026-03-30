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
"""PyPTO exp operator test.

测试说明：
  - 本文件测试 exp 算子的精度
  - golden 实现来自 exp_golden.py
  - kernel 实现来自 exp_impl.py
  - 精度对比使用 numpy.testing.assert_allclose
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from exp_golden import exp_golden
from exp_impl import exp_wrapper


# ─────────────────────────────────────────────
# 1. 环境工具
# ─────────────────────────────────────────────

def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID。"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# ─────────────────────────────────────────────
# 2. 测试函数
# ─────────────────────────────────────────────

def test_exp_level0(device_id=None, run_mode="npu"):
    """Level 0: 小数据量基础功能验证（16 元素）。"""
    print("=" * 60)
    print("Test: exp Level 0 (basic)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据
    torch.manual_seed(0)
    shape = (1, 16)  # Level 0: 小 shape
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    # 执行 kernel wrapper
    result = exp_wrapper(x)

    # 执行 golden
    golden = exp_golden(x)

    # 精度对比
    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-5,
        )

    print("  ✓ Passed\n")


def test_exp_level1_2d(device_id=None, run_mode="npu"):
    """Level 1: 2D 典型场景验证。"""
    print("=" * 60)
    print("Test: exp Level 1 (2D typical)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    shape = (1024, 1024)  # 功能_2D_FP32
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    result = exp_wrapper(x)
    golden = exp_golden(x)

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Shape: {shape}, Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-5,
        )

    print("  ✓ Passed\n")


def test_exp_level2_4d(device_id=None, run_mode="npu"):
    """Level 2: 4D 动态轴验证。"""
    print("=" * 60)
    print("Test: exp Level 2 (4D dynamic axis)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(123)
    shape = (2, 4096, 16, 128)  # 功能_4D_FP32
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    result = exp_wrapper(x)
    golden = exp_golden(x)

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Shape: {shape}, Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-5,
        )

    print("  ✓ Passed\n")


def test_exp_level3_fp16(device_id=None, run_mode="npu"):
    """Level 3: FP16 dtype 验证。"""
    print("=" * 60)
    print("Test: exp Level 3 (FP16)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(456)
    shape = (1024, 1024)  # 功能_2D_FP16
    dtype = torch.float16
    x = torch.randn(shape, dtype=dtype, device=device)

    result = exp_wrapper(x)
    golden = exp_golden(x)

    max_diff = np.abs(result.cpu().float().numpy() - golden.cpu().float().numpy()).max()
    print(f"  Shape: {shape}, dtype: {dtype}, Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().float().numpy(),
            golden.cpu().float().numpy(),
            rtol=1e-2, atol=1e-3,
        )

    print("  ✓ Passed\n")


def test_exp_level4_performance(device_id=None, run_mode="npu"):
    """Level 4: 性能场景验证（大规模 4D）。"""
    print("=" * 60)
    print("Test: exp Level 4 (performance 4D)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(789)
    shape = (4, 8192, 32, 128)  # 性能_FP32
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    result = exp_wrapper(x)
    golden = exp_golden(x)

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Shape: {shape}, Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-5,
        )

    print("  ✓ Passed\n")


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

# 用例注册表
EXAMPLES = {
    "exp::test_exp_level0": {
        "name": "exp Level 0",
        "description": "小数据量基础功能验证",
        "function": test_exp_level0,
    },
    "exp::test_exp_level1_2d": {
        "name": "exp Level 1 2D",
        "description": "2D 典型场景验证",
        "function": test_exp_level1_2d,
    },
    "exp::test_exp_level2_4d": {
        "name": "exp Level 2 4D",
        "description": "4D 动态轴验证",
        "function": test_exp_level2_4d,
    },
    "exp::test_exp_level3_fp16": {
        "name": "exp Level 3 FP16",
        "description": "FP16 dtype 验证",
        "function": test_exp_level3_fp16,
    },
    "exp::test_exp_level4_performance": {
        "name": "exp Level 4 Performance",
        "description": "性能场景验证（大规模 4D）",
        "function": test_exp_level4_performance,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO exp operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s exp::test_exp_level0    Run Level 0
  %(prog)s --list                  List all cases
        """,
    )
    parser.add_argument("example_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument(
        "--run_mode", "--run-mode",
        type=str, default="npu", choices=["npu", "sim"],
        help="Run mode (default: npu)",
    )
    args = parser.parse_args()

    # --list
    if args.list:
        print("\nAvailable cases:\n")
        for key, info in sorted(EXAMPLES.items()):
            print(f"  {key}  — {info['description']}")
        return

    # 选择用例
    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{args.example_id}'")
            print(f"Valid: {', '.join(sorted(EXAMPLES))}")
            sys.exit(1)
        to_run = [(args.example_id, EXAMPLES[args.example_id])]
    else:
        to_run = list(sorted(EXAMPLES.items()))

    # NPU 设备初始化
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)

    # 执行
    all_passed = True
    try:
        for key, info in to_run:
            print(f"\n▸ Running {key}: {info['name']}")
            try:
                info["function"](device_id, args.run_mode)
            except AssertionError as e:
                print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
                all_passed = False
            except Exception as e:
                print(f"Runtime error: {e}", file=sys.stderr)
                all_passed = False

        print("\n" + "=" * 60)
        if all_passed:
            print("[PRECISION_PASS]")
            print("All tests passed!")
        else:
            print("Some tests failed!")
        print("=" * 60)

        if not all_passed:
            sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
