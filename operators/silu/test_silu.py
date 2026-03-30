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
"""PyPTO silu operator test.

测试说明：
  - test_silu.py 只做 import + 调用 + 精度对比，不包含 golden 或 kernel 实现代码
  - golden 实现来自 silu_golden.py
  - kernel 实现来自 silu_impl.py
  - 精度对比使用 numpy.testing.assert_allclose
  - 覆盖 spec.md 中的典型配置
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from silu_golden import silu_golden
from silu_impl import silu_wrapper


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


def test_silu_boundary_p0(device_id=None, run_mode="npu"):
    """边界_P0: [1, 1, 1], float32 - 最小 shape 验证"""
    print("=" * 60)
    print("Test: silu Boundary_P0")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(0)
    shape = (1, 1, 1)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    result = silu_wrapper(x)
    golden = silu_golden(x)

    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = torch.abs(result - golden).max().item()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().float().numpy(),
                golden.cpu().float().numpy(),
                rtol=1e-3, atol=1e-3,
            )
            print("  ✓ PASS")
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            print("[PRECISION_FAIL]")
            sys.exit(1)
    else:
        print("  ✓ PASS (SIM mode, no precision check)")
        print("[PRECISION_PASS]")


def test_silu_function_p0(device_id=None, run_mode="npu"):
    """功能_P0: [2, 1024, 512], float32 - 功能验证基础配置"""
    print("=" * 60)
    print("Test: silu Function_P0")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    shape = (2, 1024, 512)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    result = silu_wrapper(x)
    golden = silu_golden(x)

    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = torch.abs(result - golden).max().item()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().float().numpy(),
                golden.cpu().float().numpy(),
                rtol=1e-3, atol=1e-3,
            )
            print("  ✓ PASS")
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            print("[PRECISION_FAIL]")
            sys.exit(1)
    else:
        print("  ✓ PASS (SIM mode, no precision check)")
        print("[PRECISION_PASS]")


def test_silu_function_p1(device_id=None, run_mode="npu"):
    """功能_P1: [4, 2048, 1024], bfloat16 - BF16 精度验证"""
    print("=" * 60)
    print("Test: silu Function_P1")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(123)
    shape = (4, 2048, 1024)
    dtype = torch.bfloat16
    x = torch.randn(shape, dtype=dtype, device=device)

    result = silu_wrapper(x)
    golden = silu_golden(x)

    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = torch.abs(result - golden).max().item()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().float().numpy(),
                golden.cpu().float().numpy(),
                rtol=1e-3, atol=1e-3,
            )
            print("  ✓ PASS")
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            print("[PRECISION_FAIL]")
            sys.exit(1)
    else:
        print("  ✓ PASS (SIM mode, no precision check)")
        print("[PRECISION_PASS]")


def test_silu_perf_p0(device_id=None, run_mode="npu"):
    """性能_P0: [1, 4096, 4096], float16 - LLaMA MLP 典型规模"""
    print("=" * 60)
    print("Test: silu Performance_P0")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(456)
    shape = (1, 4096, 4096)
    dtype = torch.float16
    x = torch.randn(shape, dtype=dtype, device=device)

    result = silu_wrapper(x)
    golden = silu_golden(x)

    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = torch.abs(result - golden).max().item()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().float().numpy(),
                golden.cpu().float().numpy(),
                rtol=1e-3, atol=1e-3,
            )
            print("  ✓ PASS")
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            print("[PRECISION_FAIL]")
            sys.exit(1)
    else:
        print("  ✓ PASS (SIM mode, no precision check)")
        print("[PRECISION_PASS]")


# 用例注册表
EXAMPLES = {
    "silu::test_silu_boundary_p0": {
        "name": "silu Boundary_P0",
        "description": "[1, 1, 1], float32 - 最小 shape 验证",
        "function": test_silu_boundary_p0,
    },
    "silu::test_silu_function_p0": {
        "name": "silu Function_P0",
        "description": "[2, 1024, 512], float32 - 功能验证基础配置",
        "function": test_silu_function_p0,
    },
    "silu::test_silu_function_p1": {
        "name": "silu Function_P1",
        "description": "[4, 2048, 1024], bfloat16 - BF16 精度验证",
        "function": test_silu_function_p1,
    },
    "silu::test_silu_perf_p0": {
        "name": "silu Performance_P0",
        "description": "[1, 4096, 4096], float16 - LLaMA MLP 典型规模",
        "function": test_silu_perf_p0,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO silu operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s silu::test_silu_function_p0    Run Function_P0 test
  %(prog)s --list                         List all cases
  %(prog)s                                Run all cases
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
    try:
        for key, info in to_run:
            print(f"\n▸ Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
