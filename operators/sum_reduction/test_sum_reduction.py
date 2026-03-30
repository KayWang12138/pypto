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
"""PyPTO sum_reduction operator test.

测试说明:
  - 本文件测试 sum_reduction 算子的精度和功能。
  - golden 实现来自 sum_reduction_golden.py。
  - kernel 实现来自 sum_reduction_impl.py。
  - 精度对比使用 numpy.testing.assert_allclose。
  - 输出三态标记: [PRECISION_PASS] / [PRECISION_FAIL] / 无标记(exit!=0)
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

# 添加当前目录到路径，确保可以导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sum_reduction_golden import sum_reduction_golden
from sum_reduction_impl import sum_reduction_wrapper


# ─────────────────────────────────────────────
# 1. 环境工具
# ─────────────────────────────────────────────

def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID。 """
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

def test_sum_reduction_level0(device_id=None, run_mode="npu"):
    """Level 0: 小数据量基础功能验证（8-16 元素)。 """
    print("=" * 60)
    print("Test: sum_reduction Level 0 (basic)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试 1: dim=-1, keepdim=False
    print("\n[Case 1] dim=-1, keepdim=False")
    torch.manual_seed(0)
    x = torch.randn(2, 4, dtype=torch.float32, device=device)
    print(f"  Input shape : {x.shape}")

    result = sum_reduction_wrapper(x, dim=-1, keepdim=False)
    golden = sum_reduction_golden(x, dim=-1, keepdim=False)

    print(f"  Output shape: {result.shape}")
    print(f"  Expected    : {golden.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().numpy(),
                golden.cpu().numpy(),
                rtol=1e-3, atol=1e-3,
            )
            print("  ✓ Passed")
        except AssertionError as e:
            print(f"  ✗ Failed: {e}")
            raise

    # 测试 2: dim=0, keepdim=True
    print("\n[Case 2] dim=0, keepdim=True")
    x = torch.randn(4, 8, dtype=torch.float32, device=device)
    print(f"  Input shape : {x.shape}")

    result = sum_reduction_wrapper(x, dim=0, keepdim=True)
    golden = sum_reduction_golden(x, dim=0, keepdim=True)

    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        print("  ✓ Passed")


def test_sum_reduction_level1(device_id=None, run_mode="npu"):
    """Level 1: 典型场景验证(1K 元素)。 """
    print("=" * 60)
    print("Test: sum_reduction Level 1 (typical)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    # 性能_P0: dim=1, keepdim=False, [b, s, 4096] -> [b, 4096]
    print("\n[Case 1] dim=1, keepdim=False, [2, 512, 4096]")
    torch.manual_seed(42)
    x = torch.randn(2, 512, 4096, dtype=torch.float32, device=device)
    print(f"  Input shape : {x.shape}")

    result = sum_reduction_wrapper(x, dim=1, keepdim=False)
    golden = sum_reduction_golden(x, dim=1, keepdim=False)

    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        print("  ✓ Passed")

    # 功能_P0: dim=1, keepdim=True
    print("\n[Case 2] dim=1, keepdim=True, [2, 512, 4096]")
    x = torch.randn(2, 512, 4096, dtype=torch.float32, device=device)
    print(f"  Input shape : {x.shape}")

    result = sum_reduction_wrapper(x, dim=1, keepdim=True)
    golden = sum_reduction_golden(x, dim=1, keepdim=True)

    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        print("  ✓ Passed")


def test_sum_reduction_level2(device_id=None, run_mode="npu"):
    """Level 2: 边界和动态 shape 测试。 """
    print("=" * 60)
    print("Test: sum_reduction Level 2 (boundary)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    # 零值输入
    print("\n[Case 1] 零值输入")
    x = torch.zeros(4, 8, 16, dtype=torch.float32, device=device)
    result = sum_reduction_wrapper(x, dim=1, keepdim=False)
    golden = sum_reduction_golden(x, dim=1, keepdim=False)
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-3, atol=1e-3)
        print("  ✓ Passed")

    # 常量输入
    print("\n[Case 2] 常量输入 (全为 5.0)")
    x = torch.full((4, 8, 16), 5.0, dtype=torch.float32, device=device)
    result = sum_reduction_wrapper(x, dim=1, keepdim=False)
    golden = sum_reduction_golden(x, dim=1, keepdim=False)
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-3, atol=1e-3)
        print("  ✓ Passed")

    # 不同维度数
    print("\n[Case 3] 4D tensor [b, s, n, d]")
    x = torch.randn(2, 16, 32, 64, dtype=torch.float32, device=device)
    result = sum_reduction_wrapper(x, dim=2, keepdim=False)
    golden = sum_reduction_golden(x, dim=2, keepdim=False)
    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-3, atol=1e-3)
        print("  ✓ Passed")

    # 负索引
    print("\n[Case 4] 负索引 dim=-1")
    x = torch.randn(4, 8, 16, dtype=torch.float32, device=device)
    result = sum_reduction_wrapper(x, dim=-1, keepdim=False)
    golden = sum_reduction_golden(x, dim=-1, keepdim=False)
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-3, atol=1e-3)
        print("  ✓ Passed")


def test_sum_reduction_dynamic(device_id=None, run_mode="npu"):
    """动态 shape 测试（覆盖不同 batch/seq 组合）。 """
    print("=" * 60)
    print("Test: sum_reduction Dynamic Shape")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    # 多组动态 shape
    test_shapes = [
        (1, 64, 128),
        (8, 256, 512),
        (16, 1024, 256),
        (4, 128, 1024),
    ]

    for b, s, d in test_shapes:
        print(f"\n[Case] shape=[{b}, {s}, {d}], dim=1")
        x = torch.randn(b, s, d, dtype=torch.float32, device=device)
        result = sum_reduction_wrapper(x, dim=1, keepdim=False)
        golden = sum_reduction_golden(x, dim=1, keepdim=False)
        max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
        print(f"  Max diff: {max_diff:.6e}")
        if run_mode == "npu":
            assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-3, atol=1e-3)
            print("  ✓ Passed")


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

# 用例注册表
EXAMPLES = {
    "sum_reduction::test_sum_reduction_level0": {
        "name": "sum_reduction Level 0",
        "description": "小数据量基础功能验证",
        "function": test_sum_reduction_level0,
    },
    "sum_reduction::test_sum_reduction_level1": {
        "name": "sum_reduction Level 1",
        "description": "典型场景验证",
        "function": test_sum_reduction_level1,
    },
    "sum_reduction::test_sum_reduction_level2": {
        "name": "sum_reduction Level 2",
        "description": "边界和动态 shape 测试",
        "function": test_sum_reduction_level2,
    },
    "sum_reduction::test_sum_reduction_dynamic": {
        "name": "sum_reduction Dynamic",
        "description": "动态 shape 测试",
        "function": test_sum_reduction_dynamic,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO sum_reduction operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sum_reduction::test_sum_reduction_level0    Run Level 0
  %(prog)s --list                                        List all cases
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
            print("[ERROR] TILE_FWK_DEVICE_ID not set")
            sys.exit(1)
        import torch_npu
        torch.npu.set_device(device_id)

    # 执行
    try:
        for key, info in to_run:
            print(f"\n>>> Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)

        print("\n" + "=" * 60)
        print("[PRECISION_PASS]")
        print("All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[PRECISION_FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Runtime error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
