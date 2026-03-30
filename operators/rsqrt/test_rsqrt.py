#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO rsqrt operator test."""

import os
import sys
import argparse
import torch
from numpy.testing import assert_allclose
from rsqrt_golden import rsqrt_golden
from rsqrt_impl import rsqrt_wrapper


def get_device_id():
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    return int(os.environ["TILE_FWK_DEVICE_ID"]) if os.environ["TILE_FWK_DEVICE_ID"].isdigit() else None


def test_rsqrt_level0(device_id=None, run_mode="npu"):
    """2D 基础功能验证。"""
    print("=" * 60)
    print("Test: rsqrt Level 0 (2D basic)")
    print("=" * 60)
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    x = torch.abs(x) + 0.1  # rsqrt 需要正数输入
    result = rsqrt_wrapper(x)
    golden = rsqrt_golden(x)
    print(f"  Input shape : {x.shape}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-5, atol=1e-5)
    print("  OK Passed\n")


def test_rsqrt_level1(device_id=None, run_mode="npu"):
    """3D 典型场景验证。"""
    print("=" * 60)
    print("Test: rsqrt Level 1 (3D)")
    print("=" * 60)
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    torch.manual_seed(42)
    x = torch.randn(4, 64, 128, dtype=torch.float32, device=device)
    x = torch.abs(x) + 0.1
    result = rsqrt_wrapper(x)
    golden = rsqrt_golden(x)
    print(f"  Input shape : {x.shape}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-5, atol=1e-5)
    print("  OK Passed\n")


def test_rsqrt_level2(device_id=None, run_mode="npu"):
    """4D 性能场景验证。"""
    print("=" * 60)
    print("Test: rsqrt Level 2 (4D)")
    print("=" * 60)
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    torch.manual_seed(123)
    x = torch.randn(2, 4, 128, 128, dtype=torch.float32, device=device)
    x = torch.abs(x) + 0.1
    result = rsqrt_wrapper(x)
    golden = rsqrt_golden(x)
    print(f"  Input shape : {x.shape}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-5, atol=1e-5)
    print("  OK Passed\n")


def test_rsqrt_fp16(device_id=None, run_mode="npu"):
    """FP16 dtype 支持。"""
    print("=" * 60)
    print("Test: rsqrt FP16")
    print("=" * 60)
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    torch.manual_seed(789)
    x = torch.randn(32, 64, dtype=torch.float16, device=device)
    x = torch.abs(x) + 0.1
    result = rsqrt_wrapper(x)
    golden = rsqrt_golden(x)
    print(f"  Input shape : {x.shape}, dtype: {x.dtype}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("  OK Passed\n")


def test_rsqrt_large(device_id=None, run_mode="npu"):
    """大规模性能验证。"""
    print("=" * 60)
    print("Test: rsqrt Large")
    print("=" * 60)
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    torch.manual_seed(999)
    x = torch.randn(4096, 4096, dtype=torch.float32, device=device)
    x = torch.abs(x) + 0.1
    result = rsqrt_wrapper(x)
    golden = rsqrt_golden(x)
    print(f"  Input shape : {x.shape}")
    if run_mode == "npu":
        assert_allclose(result.cpu().numpy(), golden.cpu().numpy(), rtol=1e-5, atol=1e-5)
    print("  OK Passed\n")


EXAMPLES = {
    "rsqrt::test_rsqrt_level0": {"name": "rsqrt Level 0", "function": test_rsqrt_level0},
    "rsqrt::test_rsqrt_level1": {"name": "rsqrt Level 1", "function": test_rsqrt_level1},
    "rsqrt::test_rsqrt_level2": {"name": "rsqrt Level 2", "function": test_rsqrt_level2},
    "rsqrt::test_rsqrt_fp16": {"name": "rsqrt FP16", "function": test_rsqrt_fp16},
    "rsqrt::test_rsqrt_large": {"name": "rsqrt Large", "function": test_rsqrt_large},
}


def main():
    parser = argparse.ArgumentParser(description="PyPTO rsqrt operator test")
    parser.add_argument("example_id", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run_mode", default="npu")
    args = parser.parse_args()

    if args.list:
        for k in sorted(EXAMPLES):
            print(f"  {k}")
        return

    to_run = [(args.example_id, EXAMPLES[args.example_id])] if args.example_id else list(sorted(EXAMPLES.items()))

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)

    try:
        for key, info in to_run:
            print(f"\n> Running {key}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("[PRECISION_PASS] All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[PRECISION_FAIL] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
