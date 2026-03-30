#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO sqrt operator test."""

import os
import sys
import argparse
import torch
import numpy as np
from numpy.testing import assert_allclose
from sqrt_golden import sqrt_golden
from sqrt_impl import sqrt_wrapper


def get_device_id():
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        return None


def test_sqrt_level0(device_id=None, run_mode="npu"):
    """Level 0: 2D 基础功能验证。"""
    print("=" * 60)
    print("Test: sqrt Level 0 (2D basic)")
    print("=" * 60)
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    torch.manual_seed(0)
    shape = (1024, 1024)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    x = torch.abs(x)  # sqrt 需要非负输入
    result = sqrt_wrapper(x)
    golden = sqrt_golden(x)
    print(f"  Input shape : {x.shape}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-5, atol=1e-5)
    print("  OK Passed\n")


def test_sqrt_level1(device_id=None, run_mode="npu"):
    """Level 1: 3D 典型场景验证。"""
    print("=" * 60)
    print("Test: sqrt Level 1 (3D typical)")
    print("=" * 60)
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    torch.manual_seed(42)
    shape = (4, 64, 128)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    x = torch.abs(x)
    result = sqrt_wrapper(x)
    golden = sqrt_golden(x)
    print(f"  Input shape : {x.shape}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-5, atol=1e-5)
    print("  OK Passed\n")


def test_sqrt_level2(device_id=None, run_mode="npu"):
    """Level 2: 4D 性能场景验证。"""
    print("=" * 60)
    print("Test: sqrt Level 2 (4D performance)")
    print("=" * 60)
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    torch.manual_seed(123)
    shape = (2, 4, 128, 128)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    x = torch.abs(x)
    result = sqrt_wrapper(x)
    golden = sqrt_golden(x)
    print(f"  Input shape : {x.shape}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-5, atol=1e-5)
    print("  OK Passed\n")


def test_sqrt_dtype_fp16(device_id=None, run_mode="npu"):
    """FP16 dtype 支持。"""
    print("=" * 60)
    print("Test: sqrt dtype FP16")
    print("=" * 60)
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    torch.manual_seed(789)
    shape = (32, 64)
    x = torch.randn(shape, dtype=torch.float16, device=device)
    x = torch.abs(x)
    result = sqrt_wrapper(x)
    golden = sqrt_golden(x)
    print(f"  Input shape : {x.shape}, dtype: {x.dtype}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("  OK Passed\n")


def test_sqrt_large(device_id=None, run_mode="npu"):
    """大规模性能验证。"""
    print("=" * 60)
    print("Test: sqrt Large (performance)")
    print("=" * 60)
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    torch.manual_seed(999)
    shape = (4096, 4096)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    x = torch.abs(x)
    result = sqrt_wrapper(x)
    golden = sqrt_golden(x)
    print(f"  Input shape : {x.shape}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-5, atol=1e-5)
    print("  OK Passed\n")


EXAMPLES = {
    "sqrt::test_sqrt_level0": {"name": "sqrt Level 0", "description": "2D 基础功能验证", "function": test_sqrt_level0},
    "sqrt::test_sqrt_level1": {"name": "sqrt Level 1", "description": "3D 典型场景验证", "function": test_sqrt_level1},
    "sqrt::test_sqrt_level2": {"name": "sqrt Level 2", "description": "4D 性能场景验证", "function": test_sqrt_level2},
    "sqrt::test_sqrt_dtype_fp16": {"name": "sqrt dtype FP16", "description": "FP16 dtype 支持", "function": test_sqrt_dtype_fp16},
    "sqrt::test_sqrt_large": {"name": "sqrt Large", "description": "大规模性能验证", "function": test_sqrt_large},
}


def main():
    parser = argparse.ArgumentParser(description="PyPTO sqrt operator test")
    parser.add_argument("example_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument("--run_mode", "--run-mode", type=str, default="npu", choices=["npu", "sim"])
    args = parser.parse_args()

    if args.list:
        print("\nAvailable cases:\n")
        for key, info in sorted(EXAMPLES.items()):
            print(f"  {key}  - {info['description']}")
        return

    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{args.example_id}'")
            sys.exit(1)
        to_run = [(args.example_id, EXAMPLES[args.example_id])]
    else:
        to_run = list(sorted(EXAMPLES.items()))

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)

    try:
        for key, info in to_run:
            print(f"\n> Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("[PRECISION_PASS] All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[PRECISION_FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nRuntime error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
