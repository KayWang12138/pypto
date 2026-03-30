#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""
linear 算子测试代码

测试 PyPTO linear 算子实现与 PyTorch golden 实现的精度对比。

测试用例:
  - Level 0: 小数据量基础功能验证
  - Level 1: 典型场景验证（2D with bias）
  - Level 2: 3D 场景验证
  - Level 3: 4D 场景验证
  - Level 4: 无 bias 场景验证
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from linear_golden import linear_golden
from linear_impl import linear_wrapper


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

def test_linear_level0(device_id=None, run_mode="npu"):
    """Level 0: 小数据量基础功能验证。"""
    print("=" * 60)
    print("Test: linear Level 0 (basic 2D with bias)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据
    torch.manual_seed(0)
    batch, in_features, out_features = 4, 64, 128
    dtype = torch.float32

    input_tensor = torch.randn(batch, in_features, dtype=dtype, device=device)
    weight = torch.randn(out_features, in_features, dtype=dtype, device=device)
    bias = torch.randn(out_features, dtype=dtype, device=device)

    # 执行 kernel wrapper
    result = linear_wrapper(input_tensor, weight, bias)

    # 执行 golden
    golden = linear_golden(input_tensor, weight, bias)

    # 精度对比
    print(f"  Input shape : {input_tensor.shape}")
    print(f"  Weight shape: {weight.shape}")
    print(f"  Bias shape  : {bias.shape}")
    print(f"  Output shape: {result.shape}")

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  [PASS] Level 0 passed\n")


def test_linear_level1(device_id=None, run_mode="npu"):
    """Level 1: 典型场景验证（2D with bias, 较大规模）。"""
    print("=" * 60)
    print("Test: linear Level 1 (typical 2D with bias)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    batch, in_features, out_features = 16, 512, 512
    dtype = torch.float32

    input_tensor = torch.randn(batch, in_features, dtype=dtype, device=device)
    weight = torch.randn(out_features, in_features, dtype=dtype, device=device)
    bias = torch.randn(out_features, dtype=dtype, device=device)

    result = linear_wrapper(input_tensor, weight, bias)
    golden = linear_golden(input_tensor, weight, bias)

    print(f"  Shape: {input_tensor.shape} -> {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  [PASS] Level 1 passed\n")


def test_linear_level2(device_id=None, run_mode="npu"):
    """Level 2: 3D 场景验证。"""
    print("=" * 60)
    print("Test: linear Level 2 (3D with bias)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(100)
    batch, seq_len, in_features, out_features = 4, 64, 256, 512
    dtype = torch.float32

    input_tensor = torch.randn(batch, seq_len, in_features, dtype=dtype, device=device)
    weight = torch.randn(out_features, in_features, dtype=dtype, device=device)
    bias = torch.randn(out_features, dtype=dtype, device=device)

    result = linear_wrapper(input_tensor, weight, bias)
    golden = linear_golden(input_tensor, weight, bias)

    print(f"  Shape: {input_tensor.shape} -> {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  [PASS] Level 2 passed\n")


def test_linear_level3(device_id=None, run_mode="npu"):
    """Level 3: 4D 场景验证。"""
    print("=" * 60)
    print("Test: linear Level 3 (4D with bias)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(200)
    batch, heads, seq_len, in_features, out_features = 2, 4, 32, 128, 256
    dtype = torch.float32

    input_tensor = torch.randn(batch, heads, seq_len, in_features, dtype=dtype, device=device)
    weight = torch.randn(out_features, in_features, dtype=dtype, device=device)
    bias = torch.randn(out_features, dtype=dtype, device=device)

    result = linear_wrapper(input_tensor, weight, bias)
    golden = linear_golden(input_tensor, weight, bias)

    print(f"  Shape: {input_tensor.shape} -> {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  [PASS] Level 3 passed\n")


def test_linear_level4(device_id=None, run_mode="npu"):
    """Level 4: 无 bias 场景验证（2D）。"""
    print("=" * 60)
    print("Test: linear Level 4 (2D without bias)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(300)
    batch, in_features, out_features = 8, 256, 512
    dtype = torch.float32

    input_tensor = torch.randn(batch, in_features, dtype=dtype, device=device)
    weight = torch.randn(out_features, in_features, dtype=dtype, device=device)

    result = linear_wrapper(input_tensor, weight, bias=None)
    golden = linear_golden(input_tensor, weight, bias=None)

    print(f"  Shape: {input_tensor.shape} -> {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  [PASS] Level 4 passed\n")


def test_linear_level5(device_id=None, run_mode="npu"):
    """Level 5: 边界测试 - 最小 batch。"""
    print("=" * 60)
    print("Test: linear Level 5 (min batch)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(400)
    batch, in_features, out_features = 1, 128, 64
    dtype = torch.float32

    input_tensor = torch.randn(batch, in_features, dtype=dtype, device=device)
    weight = torch.randn(out_features, in_features, dtype=dtype, device=device)
    bias = torch.randn(out_features, dtype=dtype, device=device)

    result = linear_wrapper(input_tensor, weight, bias)
    golden = linear_golden(input_tensor, weight, bias)

    print(f"  Shape: {input_tensor.shape} -> {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  [PASS] Level 5 passed\n")


def run_all_tests(device_id=None, run_mode="npu"):
    """运行所有测试并输出三态标记。"""
    tests = [
        ("Level 0", test_linear_level0),
        ("Level 1", test_linear_level1),
        ("Level 2", test_linear_level2),
        ("Level 3", test_linear_level3),
        ("Level 4", test_linear_level4),
        ("Level 5", test_linear_level5),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func(device_id, run_mode)
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name} error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("[PRECISION_FAIL]")
        sys.exit(1)
    else:
        print("[PRECISION_PASS]")


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

# 用例注册表
EXAMPLES = {
    "linear::test_linear_level0": {
        "name": "linear Level 0",
        "description": "小数据量基础功能验证 (2D with bias)",
        "function": test_linear_level0,
    },
    "linear::test_linear_level1": {
        "name": "linear Level 1",
        "description": "典型场景验证 (2D with bias)",
        "function": test_linear_level1,
    },
    "linear::test_linear_level2": {
        "name": "linear Level 2",
        "description": "3D 场景验证",
        "function": test_linear_level2,
    },
    "linear::test_linear_level3": {
        "name": "linear Level 3",
        "description": "4D 场景验证",
        "function": test_linear_level3,
    },
    "linear::test_linear_level4": {
        "name": "linear Level 4",
        "description": "无 bias 场景验证",
        "function": test_linear_level4,
    },
    "linear::test_linear_level5": {
        "name": "linear Level 5",
        "description": "边界测试 - 最小 batch",
        "function": test_linear_level5,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO linear operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s linear::test_linear_level0    Run Level 0
  %(prog)s --list                        List all cases
  %(prog)s                               Run all cases
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

    # NPU 设备初始化
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            sys.exit(2)
        import torch_npu
        torch.npu.set_device(device_id)

    # 选择用例
    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{args.example_id}'")
            print(f"Valid: {', '.join(sorted(EXAMPLES))}")
            sys.exit(1)
        to_run = [(args.example_id, EXAMPLES[args.example_id])]
    else:
        # 运行所有测试
        try:
            run_all_tests(device_id, args.run_mode)
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(2)
        return

    # 执行单个测试
    try:
        for key, info in to_run:
            print(f"\nRunning {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("[PRECISION_PASS]")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[PRECISION_FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
