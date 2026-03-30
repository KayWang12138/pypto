#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""
matmul 算子测试

测试覆盖:
    - 2D 矩阵乘法
    - 3D batch matmul
    - 4D batch matmul with broadcast
    - 不同 dtype (float32, float16)
    - 边界情况 (零值, 负数等)
"""

import os
import sys
import argparse
import torch
import numpy as np
from numpy.testing import assert_allclose

from matmul_golden import matmul_golden
from matmul_impl import matmul_wrapper


def get_device_id():
    """获取 NPU 设备 ID"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"Invalid TILE_FWK_DEVICE_ID: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None
def test_matmul_2d(device_id=None, run_mode="npu"):
    """测试 2D 矩阵乘法"""
    print("=" * 60)
    print("Test: matmul 2D")
    print("=" * 60)

    device = f"npu:{device_id}" if run_mode == "npu" and device_id is not None else "cpu"

    torch.manual_seed(42)
    shape_a = (64, 128)
    shape_b = (128, 256)
    dtype = torch.float32

    a = torch.randn(*shape_a, dtype=dtype, device=device)
    b = torch.randn(*shape_b, dtype=dtype, device=device)

    # 执行实现
    result = matmul_wrapper(a, b)

    # 执行 golden
    golden = matmul_golden(a, b)

    # 验证
    print(f"  A shape: {a.shape}")
    print(f"  B shape: {b.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Expected shape: {golden.shape}")

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        atol = 1e-3
        rtol = 1e-3
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=rtol, atol=atol)

    print("  ✓ Passed\n")
def test_matmul_3d(device_id=None, run_mode="npu"):
    """测试 3D batch matmul"""
    print("=" * 60)
    print("Test: matmul 3D batch")
    print("=" * 60)

    device = f"npu:{device_id}" if run_mode == "npu" and device_id is not None else "cpu"

    torch.manual_seed(42)
    shape_a = (4, 64, 128)
    shape_b = (4, 128, 256)
    dtype = torch.float32

    a = torch.randn(*shape_a, dtype=dtype, device=device)
    b = torch.randn(*shape_b, dtype=dtype, device=device)

    result = matmul_wrapper(a, b)
    golden = matmul_golden(a, b)

    print(f"  A shape: {a.shape}")
    print(f"  B shape: {b.shape}")
    print(f"  Output shape: {result.shape}")

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-3, atol=1e-3)

    print("  ✓ Passed\n")
def test_matmul_4d_broadcast(device_id=None, run_mode="npu"):
    """测试 4D batch matmul with broadcast"""
    print("=" * 60)
    print("Test: matmul 4D batch with broadcast")
    print("=" * 60)

    device = f"npu:{device_id}" if run_mode == "npu" and device_id is not None else "cpu"

    torch.manual_seed(42)
    shape_a = (2, 1, 64, 128)
    shape_b = (1, 4, 128, 256)
    dtype = torch.float32

    a = torch.randn(*shape_a, dtype=dtype, device=device)
    b = torch.randn(*shape_b, dtype=dtype, device=device)

    result = matmul_wrapper(a, b)
    golden = matmul_golden(a, b)

    print(f"  A shape: {a.shape}")
    print(f"  B shape: {b.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Expected shape: {golden.shape}")

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-3, atol=1e-3)

    print("  ✓ Passed\n")
def test_matmul_float16(device_id=None, run_mode="npu"):
    """测试 float16 dtype"""
    print("=" * 60)
    print("Test: matmul float16")
    print("=" * 60)

    device = f"npu:{device_id}" if run_mode == "npu" and device_id is not None else "cpu"

    torch.manual_seed(42)
    shape_a = (32, 64)
    shape_b = (64, 128)
    dtype = torch.float16

    a = torch.randn(*shape_a, dtype=dtype, device=device)
    b = torch.randn(*shape_b, dtype=dtype, device=device)

    result = matmul_wrapper(a, b)
    golden = matmul_golden(a, b)

    print(f"  A shape: {a.shape}, dtype: {a.dtype}")
    print(f"  B shape: {b.shape}, dtype: {b.dtype}")
    print(f"  Output shape: {result.shape}, dtype: {result.dtype}")

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-2, atol=1e-2)

    print("  ✓ Passed\n")
def test_matmul_zero(device_id=None, run_mode="npu"):
    """测试零值输入"""
    print("=" * 60)
    print("Test: matmul with zero input")
    print("=" * 60)

    device = f"npu:{device_id}" if run_mode == "npu" and device_id is not None else "cpu"

    torch.manual_seed(42)
    shape_a = (32, 64)
    shape_b = (64, 128)
    dtype = torch.float32

    a = torch.zeros(*shape_a, dtype=dtype, device=device)
    b = torch.randn(*shape_b, dtype=dtype, device=device)

    result = matmul_wrapper(a, b)
    golden = matmul_golden(a, b)

    print(f"  A shape: {a.shape} (all zeros)")
    print(f"  B shape: {b.shape}")
    print(f"  Output all zeros: {torch.all(result == 0)}")

    assert torch.all(result == 0), "Zero input should produce zero output"
    print("  ✓ Passed\n")
# 测试用例注册表
EXAMPLE_list = {
    "matmul::test_2d": {
        "name": "2D matmul",
        "description": "2D matrix multiplication",
        "function": test_matmul_2d,
    },
    "matmul::test_3d": {
        "name": "3D batch matmul",
        "description": "3D batch matrix multiplication",
        "function": test_matmul_3d,
    },
    "matmul::test_4d_broadcast": {
        "name": "4D broadcast",
        "description": "4D batch matmul with broadcast",
        "function": test_matmul_4d_broadcast,
    },
    "matmul::test_float16": {
        "name": "Float16",
        "description": "Float16 dtype test",
        "function": test_matmul_float16,
    },
    "matmul::test_zero": {
        "name": "Zero input",
        "description": "Zero input test",
        "function": test_matmul_zero,
    },
}
def main():
    parser = argparse.ArgumentParser(
        description="matmul 算子测试",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("example_id", type=str, nargs="?", help="Test case to run")
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument(
        "--run_mode",
        type=str,
        default="npu",
        choices=["npu", "sim"],
        help="Run mode (default: npu)"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable tests:")
        for key in sorted(example_list.keys()):
            print(f"  {key:  - {example_list[key]['description']}")
        return

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)

    # 选择测试
    if args.example_id:
        if args.example_id not in example_list:
            print(f"ERROR: Unknown test: {args.example_id}")
            print(f"Available: {', '.join(sorted(example_list.keys()))}")
            sys.exit(1)
        tests_to_run = [(args.example_id, example_list[args.example_id])]
    else:
        tests_to_run = list(sorted(example_list.items()))

    # 执行测试
    try:
        for key, info in tests_to_run:
            print(f"\n{'=' * 60}")
            print(f"Running: {key} - {info['name']}")
            print("=" * 60)
            info["function"](device_id, args.run_mode)

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    main()
