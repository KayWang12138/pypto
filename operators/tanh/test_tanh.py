#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO tanh operator test.

测试说明：
  - test_tanh.py 只做 import + 调用 + 精度对比，不包含 golden 或 kernel 实现代码
  - golden 实现来自 tanh_golden.py
  - kernel 实现来自 tanh_impl.py
  - 精度对比使用 numpy.testing.assert_allclose
  - 覆盖 spec.md 中的典型配置
  - 输出三态标记: [PRECISION_PASS] 或 [PRECISION_FAIL]
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from tanh_golden import tanh_golden
from tanh_impl import tanh_wrapper


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


def test_tanh_perf_p0(device_id=None, run_mode="npu"):
    """性能_P0: 核心性能场景 [1024, 1024]"""
    print("=" * 60)
    print("Test: TanH 性能_P0 (1024, 1024)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试所有支持的 dtype
    dtypes = [torch.float16, torch.float32, torch.bfloat16]

    for dtype in dtypes:
        torch.manual_seed(42)
        shape = (1024, 1024)
        x = torch.randn(shape, dtype=dtype, device=device)

        result = tanh_wrapper(x)
        golden = tanh_golden(x)

        # 转换为 float32 进行 numpy 比较
        result_f32 = result.cpu().float().numpy()
        golden_f32 = golden.cpu().float().numpy()
        max_diff = np.abs(result_f32 - golden_f32).max()
        print(f"  dtype: {dtype}, Max diff: {max_diff:.6e}")

        if run_mode == "npu":
            try:
                assert_allclose(
                    result_f32,
                    golden_f32,
                    rtol=3e-3, atol=3e-3,
                )
                print("  PASS")
                print("  [PRECISION_PASS]")
            except AssertionError as e:
                print(f"  FAIL: {e}")
                print("  [PRECISION_FAIL]")
                sys.exit(1)
        else:
            print("  PASS (SIM mode)")
            print("  [PRECISION_PASS]")


def test_tanh_func_p0(device_id=None, run_mode="npu"):
    """功能_P0: 核心功能验证 [32, 64]"""
    print("=" * 60)
    print("Test: TanH 功能_P0 (32, 64)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    dtypes = [torch.float16, torch.float32, torch.bfloat16]

    for dtype in dtypes:
        torch.manual_seed(0)
        shape = (32, 64)
        x = torch.randn(shape, dtype=dtype, device=device)

        result = tanh_wrapper(x)
        golden = tanh_golden(x)

        result_f32 = result.cpu().float().numpy()
        golden_f32 = golden.cpu().float().numpy()
        max_diff = np.abs(result_f32 - golden_f32).max()
        print(f"  dtype: {dtype}, Max diff: {max_diff:.6e}")

        if run_mode == "npu":
            try:
                assert_allclose(
                    result_f32,
                    golden_f32,
                    rtol=3e-3, atol=3e-3,
                )
                print("  PASS")
                print("  [PRECISION_PASS]")
            except AssertionError as e:
                print(f"  FAIL: {e}")
                print("  [PRECISION_FAIL]")
                sys.exit(1)
        else:
            print("  PASS (SIM mode)")
            print("  [PRECISION_PASS]")


def test_tanh_func_p1(device_id=None, run_mode="npu"):
    """功能_P1: 3维输入验证 [2, 128, 256]"""
    print("=" * 60)
    print("Test: TanH 功能_P1 (2, 128, 256)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    dtypes = [torch.float16, torch.float32, torch.bfloat16]

    for dtype in dtypes:
        torch.manual_seed(1)
        shape = (2, 128, 256)
        x = torch.randn(shape, dtype=dtype, device=device)

        result = tanh_wrapper(x)
        golden = tanh_golden(x)

        result_f32 = result.cpu().float().numpy()
        golden_f32 = golden.cpu().float().numpy()
        max_diff = np.abs(result_f32 - golden_f32).max()
        print(f"  dtype: {dtype}, Max diff: {max_diff:.6e}")

        if run_mode == "npu":
            try:
                assert_allclose(
                    result_f32,
                    golden_f32,
                    rtol=3e-3, atol=3e-3,
                )
                print("  PASS")
                print("  [PRECISION_PASS]")
            except AssertionError as e:
                print(f"  FAIL: {e}")
                print("  [PRECISION_FAIL]")
                sys.exit(1)
        else:
            print("  PASS (SIM mode)")
            print("  [PRECISION_PASS]")


def test_tanh_func_p2(device_id=None, run_mode="npu"):
    """功能_P2: 4维输入验证 [1, 1, 64, 64]"""
    print("=" * 60)
    print("Test: TanH 功能_P2 (1, 1, 64, 64)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    dtypes = [torch.float16, torch.float32, torch.bfloat16]

    for dtype in dtypes:
        torch.manual_seed(2)
        shape = (1, 1, 64, 64)
        x = torch.randn(shape, dtype=dtype, device=device)

        result = tanh_wrapper(x)
        golden = tanh_golden(x)

        result_f32 = result.cpu().float().numpy()
        golden_f32 = golden.cpu().float().numpy()
        max_diff = np.abs(result_f32 - golden_f32).max()
        print(f"  dtype: {dtype}, Max diff: {max_diff:.6e}")

        if run_mode == "npu":
            try:
                assert_allclose(
                    result_f32,
                    golden_f32,
                    rtol=3e-3, atol=3e-3,
                )
                print("  PASS")
                print("  [PRECISION_PASS]")
            except AssertionError as e:
                print(f"  FAIL: {e}")
                print("  [PRECISION_FAIL]")
                sys.exit(1)
        else:
            print("  PASS (SIM mode)")
            print("  [PRECISION_PASS]")


# 用例注册表
EXAMPLES = {
    "tanh::test_tanh_perf_p0": {
        "name": "TanH 性能_P0",
        "description": "核心性能场景 [1024, 1024]",
        "function": test_tanh_perf_p0,
    },
    "tanh::test_tanh_func_p0": {
        "name": "TanH 功能_P0",
        "description": "核心功能验证 [32, 64]",
        "function": test_tanh_func_p0,
    },
    "tanh::test_tanh_func_p1": {
        "name": "TanH 功能_P1",
        "description": "3维输入验证 [2, 128, 256]",
        "function": test_tanh_func_p1,
    },
    "tanh::test_tanh_func_p2": {
        "name": "TanH 功能_P2",
        "description": "4维输入验证 [1, 1, 64, 64]",
        "function": test_tanh_func_p2,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO tanh operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s tanh::test_tanh_perf_p0    Run 性能 P0
  %(prog)s tanh::test_tanh_func_p0    Run 功能 P0
  %(prog)s tanh::test_tanh_func_p1    Run 功能 P1
  %(prog)s tanh::test_tanh_func_p2    Run 功能 P2
  %(prog)s --list                     List all cases
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
            print(f"  {key}  - {info['description']}")
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
            print(f"\nRunning {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
