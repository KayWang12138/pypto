#!/usr/bin/env python3
# coding: utf-8

"""PyPTO batch_matmul operator test.

测试文件说明:
  - 测试 batch_matmul 算子的精度和功能
  - 包含多级测试用例
  - 精度对比使用 numpy.testing.assert_allclose
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from batch_matmul_golden import batch_matmul_golden
from batch_matmul_impl import batch_matmul_wrapper

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

def test_batch_matmul_level0(device_id=None, run_mode="npu"):
    """Level 0: 小数据量基础功能验证. """
    print("=" * 60)
    print("Test: batch_matmul Level 0 (basic)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据 - shape 必须满足 tile shapes 的要求 (>= 128)
    torch.manual_seed(0)
    batch, m, k, n = 2, 128, 128, 128
    x1 = torch.randn((batch, m, k), dtype=torch.float32, device=device)
    x2 = torch.randn((batch, k, n), dtype=torch.float32, device=device)
    expected = torch.bmm(x1, x2)

    # 执行 kernel wrapper
    result = batch_matmul_wrapper(x1, x2)

    # 精度对比
    print(f"  Input shapes: x1={x1.shape}, x2={x2.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            expected.cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        print("[PRECISION_PASS]")
    else:
        print("  [SIM mode] Skipped precision check")

    print("  Passed\n")


def test_batch_matmul_level1(device_id=None, run_mode="npu"):
    """Level 1: 典型场景验证 (1K 元素). """
    print("=" * 60)
    print("Test: batch_matmul Level 1 (typical)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    shape = (4, 1024, 1024)  # Level 1: 典型 shape
    dtype = torch.float32
    x1 = torch.randn(shape, dtype=dtype, device=device)
    x2 = torch.randn(shape, dtype=dtype, device=device)

    result = batch_matmul_wrapper(x1, x2)
    golden = batch_matmul_golden(x1, x2)

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Shape: {shape}, Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        print("[PRECISION_PASS]")

    print("  Passed\n")


def test_batch_matmul_transpose(device_id=None, run_mode="npu"):
    """Level 2: 转置功能验证. """
    print("=" * 60)
    print("Test: batch_matmul Transpose")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(123)

    # x2 转置场景: [b, n, k] @ [b, k, m]
    batch, m, k, n = 2, 128, 128, 128
    x1 = torch.randn((batch, m, k), dtype=torch.float32, device=device)
    x2 = torch.randn((batch, k, n), dtype=torch.float32, device=device)
    # x2 转置后 shape 变为 [b, k, n]
    x2_transposed = x2.transpose(-1, -2)
    expected = torch.bmm(x1, x2_transposed)
    result = batch_matmul_wrapper(x1, x2, transpose_x2=True)
    max_diff = np.abs(result.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"  Transpose x2: x1={x1.shape}, x2={x2.shape} (transposed)")
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            expected.cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        print("[PRECISION_PASS]")

    print("  Passed\n")


def test_batch_matmul_broadcast(device_id=None, run_mode="npu"):
    """Level 3: batch=1 广播到 batch=4 场景验证. """
    print("=" * 60)
    print("Test: batch_matmul Broadcast")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(123)

    # batch=1 广播到 batch=4
    # x1: [1, m, k], x2: [4, k, n]
    m, k, n = 128, 128, 128
    x1 = torch.randn((1, m, k), dtype=torch.float32, device=device)
    x2 = torch.randn((4, k, n), dtype=torch.float32, device=device)
    # 使用 expand 广播 x1 到 [4, m, k]
    x1_expanded = x1.expand(4, -1, -1)
    expected = torch.bmm(x1_expanded, x2)

    result = batch_matmul_wrapper(x1, x2)
    max_diff = np.abs(result.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"  Broadcast: x1={x1.shape}, x2={x2.shape}")
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            expected.cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        print("[PRECISION_PASS]")

    print("  Passed\n")


def test_batch_matmul_large(device_id=None, run_mode="npu"):
    """Level 4: 大规模性能验证 (4096 元素). """
    print("=" * 60)
    print("Test: batch_matmul Large (performance)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(789)

    batch, m, k, n = 2, 1024, 1024, 1024
    x1 = torch.randn((batch, m, k), dtype=torch.float32, device=device)
    x2 = torch.randn((batch, k, n), dtype=torch.float32, device=device)
    expected = torch.bmm(x1, x2)

    result = batch_matmul_wrapper(x1, x2)
    max_diff = np.abs(result.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"  Shape: {x1.shape}, Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            expected.cpu().numpy(),
            rtol=1e-4,  # 大矩阵使用更宽松的容差
            atol=1e-4,
        )
        print("[PRECISION_PASS]")

    print("  Passed\n")


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "batch_matmul::test_batch_matmul_level0": {
        "name": "batch_matmul Level 0",
        "description": "小数据量基础功能验证",
        "function": test_batch_matmul_level0,
    },
    "batch_matmul::test_batch_matmul_level1": {
        "name": "batch_matmul Level 1",
        "description": "典型场景验证",
        "function": test_batch_matmul_level1,
    },
    "batch_matmul::test_batch_matmul_transpose": {
        "name": "batch_matmul Transpose",
        "description": "转置功能验证",
        "function": test_batch_matmul_transpose,
    },
    "batch_matmul::test_batch_matmul_broadcast": {
        "name": "batch_matmul Broadcast",
        "description": "广播场景验证",
        "function": test_batch_matmul_broadcast,
    },
    "batch_matmul::test_batch_matmul_large": {
        "name": "batch_matmul Large",
        "description": "大规模性能验证",
        "function": test_batch_matmul_large,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO batch_matmul operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s batch_matmul::test_batch_matmul_level0    Run Level 0
  %(prog)s --list                                    List all cases
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
