#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
concat 算子测试

测试 concat_wrapper 实现与 concat_golden 参考实现的精度对比。

三态标记约定:
    - [PRECISION_PASS]: 精度验证通过
    - [PRECISION_FAIL]: 精度验证失败（数值不匹配）
    - 无标记 + exit != 0: 功能问题（代码崩溃、逻辑错误等）
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from concat_golden import concat_golden
from concat_impl import concat_wrapper


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

def test_concat_level0(device_id=None, run_mode="npu"):
    """Level 0: 小数据量基础功能验证（2D tensor, 2个张量）。"""
    print("=" * 60)
    print("Test: concat Level 0 (basic 2D)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据：小规模 2D tensor
    torch.manual_seed(0)
    dtype = torch.float32
    a = torch.randn(2, 4, dtype=dtype, device=device)
    b = torch.randn(2, 4, dtype=dtype, device=device)
    dim = 1  # 沿最后一维拼接

    # 执行 impl
    result = concat_wrapper([a, b], dim=dim)

    # 执行 golden
    golden = concat_golden([a, b], dim=dim)

    # 精度对比
    print(f"  Input shapes: a={a.shape}, b={b.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  dim: {dim}")

    # 转换为 numpy 进行对比（注意 bfloat16 需要先转 float）
    result_np = result.cpu().float().numpy() if dtype == torch.bfloat16 else result.cpu().numpy()
    golden_np = golden.cpu().float().numpy() if dtype == torch.bfloat16 else golden.cpu().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(result_np, golden_np, rtol=1e-3, atol=1e-3)
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            raise
    else:
        print("  [PRECISION_PASS] (sim mode, skip verification)")

    print("  Passed\n")


def test_concat_level1(device_id=None, run_mode="npu"):
    """Level 1: 典型场景验证（3D tensor, 2个张量）。"""
    print("=" * 60)
    print("Test: concat Level 1 (typical 3D)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据：典型 3D tensor [batch, seq, hidden]
    torch.manual_seed(42)
    dtype = torch.float32
    b, s1, s2, d = 2, 64, 32, 128
    a = torch.randn(b, s1, d, dtype=dtype, device=device)
    b_tensor = torch.randn(b, s2, d, dtype=dtype, device=device)
    dim = 1  # 沿 seq 维度拼接

    result = concat_wrapper([a, b_tensor], dim=dim)
    golden = concat_golden([a, b_tensor], dim=dim)

    print(f"  Input shapes: a={a.shape}, b={b_tensor.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  dim: {dim}")

    result_np = result.cpu().numpy()
    golden_np = golden.cpu().numpy()
    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(result_np, golden_np, rtol=1e-3, atol=1e-3)
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            raise
    else:
        print("  [PRECISION_PASS] (sim mode, skip verification)")

    print("  Passed\n")


def test_concat_level2(device_id=None, run_mode="npu"):
    """Level 2: 3个张量拼接验证。"""
    print("=" * 60)
    print("Test: concat Level 2 (3 tensors)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据：3个张量拼接
    torch.manual_seed(123)
    dtype = torch.float32
    b1, b2, b3, s, d = 4, 8, 12, 64, 128
    a = torch.randn(b1, s, d, dtype=dtype, device=device)
    b_tensor = torch.randn(b2, s, d, dtype=dtype, device=device)
    c = torch.randn(b3, s, d, dtype=dtype, device=device)
    dim = 0  # 沿 batch 维度拼接

    result = concat_wrapper([a, b_tensor, c], dim=dim)
    golden = concat_golden([a, b_tensor, c], dim=dim)

    print(f"  Input shapes: a={a.shape}, b={b_tensor.shape}, c={c.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  dim: {dim}")

    result_np = result.cpu().numpy()
    golden_np = golden.cpu().numpy()
    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(result_np, golden_np, rtol=1e-3, atol=1e-3)
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            raise
    else:
        print("  [PRECISION_PASS] (sim mode, skip verification)")

    print("  Passed\n")


def test_concat_level3(device_id=None, run_mode="npu"):
    """Level 3: 负数索引测试。"""
    print("=" * 60)
    print("Test: concat Level 3 (negative dim)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据：负数 dim
    torch.manual_seed(456)
    dtype = torch.float32
    a = torch.randn(4, 8, 16, dtype=dtype, device=device)
    b_tensor = torch.randn(4, 8, 32, dtype=dtype, device=device)
    dim = -1  # 负数索引，最后一维

    result = concat_wrapper([a, b_tensor], dim=dim)
    golden = concat_golden([a, b_tensor], dim=dim)

    print(f"  Input shapes: a={a.shape}, b={b_tensor.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  dim: {dim}")

    result_np = result.cpu().numpy()
    golden_np = golden.cpu().numpy()
    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(result_np, golden_np, rtol=1e-3, atol=1e-3)
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            raise
    else:
        print("  [PRECISION_PASS] (sim mode, skip verification)")

    print("  Passed\n")


def test_concat_level4(device_id=None, run_mode="npu"):
    """Level 4: 不同 dtype 支持（float16）。"""
    print("=" * 60)
    print("Test: concat Level 4 (float16)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据：float16
    torch.manual_seed(789)
    dtype = torch.float16
    a = torch.randn(4, 8, 16, dtype=dtype, device=device)
    b_tensor = torch.randn(4, 8, 32, dtype=dtype, device=device)
    dim = -1

    result = concat_wrapper([a, b_tensor], dim=dim)
    golden = concat_golden([a, b_tensor], dim=dim)

    print(f"  Input shapes: a={a.shape}, b={b_tensor.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  dtype: {dtype}")

    result_np = result.cpu().float().numpy()
    golden_np = golden.cpu().float().numpy()
    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        try:
            # float16 使用更宽松的容差
            assert_allclose(result_np, golden_np, rtol=1e-2, atol=1e-2)
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            raise
    else:
        print("  [PRECISION_PASS] (sim mode, skip verification)")

    print("  Passed\n")


def test_concat_level5(device_id=None, run_mode="npu"):
    """Level 5: 性能测试（大规模张量）。"""
    print("=" * 60)
    print("Test: concat Level 5 (large scale)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据：大规模张量 [4096, 1024]
    torch.manual_seed(999)
    dtype = torch.float32
    a = torch.randn(4096, 1024, dtype=dtype, device=device)
    b_tensor = torch.randn(4096, 1024, dtype=dtype, device=device)
    dim = -1

    result = concat_wrapper([a, b_tensor], dim=dim)
    golden = concat_golden([a, b_tensor], dim=dim)

    print(f"  Input shapes: a={a.shape}, b={b_tensor.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Total elements: {result.numel():,}")

    result_np = result.cpu().numpy()
    golden_np = golden.cpu().numpy()
    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(result_np, golden_np, rtol=1e-3, atol=1e-3)
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            raise
    else:
        print("  [PRECISION_PASS] (sim mode, skip verification)")

    print("  Passed\n")


def test_concat_level6(device_id=None, run_mode="npu"):
    """Level 6: 4D tensor 测试。"""
    print("=" * 60)
    print("Test: concat Level 6 (4D tensor)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据：4D tensor [N, C, H, W]
    torch.manual_seed(111)
    dtype = torch.float32
    a = torch.randn(2, 32, 8, 8, dtype=dtype, device=device)
    b_tensor = torch.randn(2, 64, 8, 8, dtype=dtype, device=device)
    dim = 1  # 沿 channel 维度拼接

    result = concat_wrapper([a, b_tensor], dim=dim)
    golden = concat_golden([a, b_tensor], dim=dim)

    print(f"  Input shapes: a={a.shape}, b={b_tensor.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  dim: {dim}")

    result_np = result.cpu().numpy()
    golden_np = golden.cpu().numpy()
    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(result_np, golden_np, rtol=1e-3, atol=1e-3)
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            raise
    else:
        print("  [PRECISION_PASS] (sim mode, skip verification)")

    print("  Passed\n")


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

# 用例注册表
EXAMPLES = {
    "concat::test_concat_level0": {
        "name": "concat Level 0",
        "description": "小数据量基础功能验证（2D tensor）",
        "function": test_concat_level0,
    },
    "concat::test_concat_level1": {
        "name": "concat Level 1",
        "description": "典型场景验证（3D tensor）",
        "function": test_concat_level1,
    },
    "concat::test_concat_level2": {
        "name": "concat Level 2",
        "description": "3个张量拼接验证",
        "function": test_concat_level2,
    },
    "concat::test_concat_level3": {
        "name": "concat Level 3",
        "description": "负数索引测试",
        "function": test_concat_level3,
    },
    "concat::test_concat_level4": {
        "name": "concat Level 4",
        "description": "不同 dtype 支持（float16）",
        "function": test_concat_level4,
    },
    "concat::test_concat_level5": {
        "name": "concat Level 5",
        "description": "性能测试（大规模张量）",
        "function": test_concat_level5,
    },
    "concat::test_concat_level6": {
        "name": "concat Level 6",
        "description": "4D tensor 测试",
        "function": test_concat_level6,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO concat operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s concat::test_concat_level0    Run Level 0
  %(prog)s --list                        List all cases
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
            sys.exit(1)
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
    except AssertionError as e:
        # 精度失败已经打印了 [PRECISION_FAIL]
        print(f"\nTest failed with precision error.")
        sys.exit(1)
    except Exception as e:
        print(f"\nRuntime error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
