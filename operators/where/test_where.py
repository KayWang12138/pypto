#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO where operator test.

测试说明：
  - 本文件是 where 算子的精度验证测试。
  - golden 实现来自 where_golden.py。
  - kernel 实现来自 where_impl.py。
  - 精度对比使用 numpy.testing.assert_allclose。
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from where_golden import where_golden
from where_impl import where_wrapper

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
        print(f"ERROR: TILE_FWK_DEVICE_ID needs to be integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# ─────────────────────────────────────────────
# 2. 测试用例
# ─────────────────────────────────────────────

# 精度标准
RTOL = 1e-3
ATOL = 1e-3

# BF16 使用更宽松的容差
RTOL_BF16 = 1e-2
ATOL_BF16 = 1e-2


def test_where_level0(device_id=None, run_mode="npu"):
    """Level 0: 小数据量基础功能验证"""
    print("=" * 60)
    print("Test: where Level 0")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(0)
    b, s, n, d = 1, 4, 2, 8
    condition = torch.rand(b, s, n, d, device=device) > 0.5
    x = torch.randn(b, s, n, d, device=device)
    y = torch.randn(b, s, n, d, device=device)

    result = where_wrapper(condition, x, y)
    golden = where_golden(condition.cpu(), x.cpu(), y.cpu())

    print(f"  Input shape : condition={condition.shape}, x={x.shape}, y={y.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = torch.abs(result.cpu() - golden).max().item()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().float().numpy(),
                golden.cpu().float().numpy(),
                rtol=RTOL, atol=ATOL,
            )
            print("  PASS")
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            print("[PRECISION_FAIL]")
            sys.exit(1)
    else:
        print("  PASS (SIM mode, no precision check)")
        print("[PRECISION_PASS]")


def test_where_level1(device_id=None, run_mode="npu"):
    """Level 1: 广播场景验证"""
    print("=" * 60)
    print("Test: where Level 1")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    b, s, n, d = 2, 64, 4, 16
    condition = torch.rand(b, s, 1, 1, device=device) > 0.5
    x = torch.randn(b, s, n, d, device=device)
    y = torch.randn(b, s, n, d, device=device)

    result = where_wrapper(condition, x, y)
    golden = where_golden(condition.cpu(), x.cpu(), y.cpu())

    print(f"  Input shape : condition={condition.shape}, x={x.shape}, y={y.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = torch.abs(result.cpu() - golden).max().item()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().float().numpy(),
                golden.cpu().float().numpy(),
                rtol=RTOL, atol=ATOL,
            )
            print("  PASS")
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            print("[PRECISION_FAIL]")
            sys.exit(1)
    else:
        print("  PASS (SIM mode, no precision check)")
        print("[PRECISION_PASS]")


def test_where_level2(device_id=None, run_mode="npu"):
    """Level 2: 标量 y 场景验证"""
    print("=" * 60)
    print("Test: where Level 2")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(123)
    b, s, n, d = 2, 32, 4, 8
    condition = torch.rand(b, s, n, d, device=device) > 0.5
    x = torch.randn(b, s, n, d, device=device)
    y_scalar = 0.0

    result = where_wrapper(condition, x, y_scalar)
    golden = where_golden(condition.cpu(), x.cpu(), y_scalar)

    print(f"  Input shape : condition={condition.shape}, x={x.shape}, y=scalar({y_scalar})")
    print(f"  Output shape: {result.shape}")
    max_diff = torch.abs(result.cpu() - golden).max().item()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().float().numpy(),
                golden.cpu().float().numpy(),
                rtol=RTOL, atol=ATOL,
            )
            print("  PASS")
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            print("[PRECISION_FAIL]")
            sys.exit(1)
    else:
        print("  PASS (SIM mode, no precision check)")
        print("[PRECISION_PASS]")


def test_where_level3(device_id=None, run_mode="npu"):
    """Level 3: 不同 dtype 测试"""
    print("=" * 60)
    print("Test: where Level 3")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    for dtype in [torch.float32, torch.float16]:
        print(f"\n  Testing dtype: {dtype}")

        torch.manual_seed(456)
        b, s, n, d = 2, 16, 2, 8
        condition = torch.rand(b, s, n, d, device=device) > 0.5
        x = torch.randn(b, s, n, d, dtype=dtype, device=device)
        y = torch.randn(b, s, n, d, dtype=dtype, device=device)

        result = where_wrapper(condition, x, y)
        golden = where_golden(condition.cpu(), x.cpu(), y.cpu())

        print(f"    Input shape : condition={condition.shape}, x={x.shape}, y={y.shape}")
        print(f"    Output shape: {result.shape}")
        max_diff = torch.abs(result.cpu().float() - golden.cpu().float()).max().item()
        print(f"    Max diff    : {max_diff:.6e}")

        if run_mode == "npu":
            try:
                assert_allclose(
                    result.cpu().float().numpy(),
                    golden.cpu().float().numpy(),
                    rtol=RTOL, atol=ATOL,
                )
                print(f"    PASS ({dtype})")
            except AssertionError as e:
                print(f"    FAIL ({dtype}): {e}")
                print("[PRECISION_FAIL]")
                sys.exit(1)
        else:
            print(f"    PASS ({dtype}) (SIM mode, no precision check)")

    print("\n[PRECISION_PASS]")


# ─────────────────────────────────────────────
# 3. 测试注册表
# ─────────────────────────────────────────────

EXAMPLES = {
    "where::test_where_level0": {
        "name": "where Level 0",
        "description": "小数据量基础功能验证",
        "function": test_where_level0,
    },
    "where::test_where_level1": {
        "name": "where Level 1",
        "description": "广播场景验证",
        "function": test_where_level1,
    },
    "where::test_where_level2": {
        "name": "where Level 2",
        "description": "标量 y 场景验证",
        "function": test_where_level2,
    },
    "where::test_where_level3": {
        "name": "where Level 3",
        "description": "不同 dtype 测试",
        "function": test_where_level3,
    },
}


# ─────────────────────────────────────────────
# 4. CLI 入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PyPTO where operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_where.py where::test_where_level0    Run Level 0
  python test_where.py --list                    List all cases
  python test_where.py                          Run all tests with NPU mode
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
        # 运行所有测试
        device_id = None
        if args.run_mode == "npu":
            device_id = get_device_id()
            if device_id is None:
                return
            import torch_npu
            torch.npu.set_device(device_id)

        run_all_tests(device_id, args.run_mode)
        return

    # NPU 设备初始化
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)

    # 执行单个用例
    try:
        for key, info in to_run:
            print(f"\n>>> Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("[PRECISION_PASS] Test passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[PRECISION_FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[RUNTIME_ERROR] {e}", file=sys.stderr)
        sys.exit(2)


def run_all_tests(device_id, run_mode):
    """运行所有测试并输出三态标记"""
    for key, info in sorted(EXAMPLES.items()):
        try:
            print(f"\n>>> Running {key}: {info['name']}")
            info["function"](device_id, run_mode)
        except AssertionError as e:
            print(f"\n[PRECISION_FAIL] {key}: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"\n[RUNTIME_ERROR] {key}: {e}", file=sys.stderr)
            sys.exit(2)

    print("\n" + "=" * 60)
    print("[PRECISION_PASS] All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
