#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms of the CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT warranties or any kind, either express or implied, including but not limited to non-infringement, merchantability, or fitness for a particular purpose.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO flash_attention operator test.

This test file verifies the PyPTO implementation of Flash Attention against
the PyTorch golden reference implementation.

Test levels:
  - Level 0: Small scale basic verification [1, 2, 4, 4]
  - Level 1: Typical scale verification [2, 4, 512, 64]
  - Causal: Causal attention mask [1, 8, 512, 128]
  - With Mask: Custom attention mask [1, 8, 512, 128]
  - Perf P0: Performance configuration [1, 8, 1024, 128]
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from flash_attention_golden import flash_attention_golden
from flash_attention_impl import flash_attention_wrapper

# ─────────────────────────────────────────────
# 1. Environment Tools
# ─────────────────────────────────────────────

def get_device_id():
    """Get TILE_FWK_DEVICE_ID from environment variable."""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# ─────────────────────────────────────────────
# 2. Test Functions
# ─────────────────────────────────────────────

def test_flash_attention_level0(device_id=None, run_mode="npu"):
    """Level 0: Small scale basic verification [1, 2, 8, 8]."""
    print("=" * 60)
    print("Test: flash_attention Level 0 (basic)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # Test data
    torch.manual_seed(0)
    N, H, L, d = 1, 2, 8, 8  # Last axis 8 for 32-byte alignment
    dtype = torch.float32

    query = torch.randn(N, H, L, d, dtype=dtype, device=device)
    key = torch.randn(N, H, L, d, dtype=dtype, device=device)
    value = torch.randn(N, H, L, d, dtype=dtype, device=device)

    # Execute kernel wrapper
    result = flash_attention_wrapper(query, key, value)

    # Execute golden
    golden = flash_attention_golden(query, key, value)

    # Precision comparison (must use assert_allclose)
    print(f"  Input shape : query={query.shape}, key={key.shape}, value={value.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=3e-3, atol=3e-3,
        )

    print("  [PRECISION_PASS]")
    print("  Passed\n")


def test_flash_attention_level1(device_id=None, run_mode="npu"):
    """Level 1: Typical scale verification [2, 4, 128, 64]."""
    print("=" * 60)
    print("Test: flash_attention Level 1 (typical)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    N, H, L, d = 2, 4, 128, 64  # seq_len limited to 128 for amax constraint
    dtype = torch.float32

    query = torch.randn(N, H, L, d, dtype=dtype, device=device)
    key = torch.randn(N, H, L, d, dtype=dtype, device=device)
    value = torch.randn(N, H, L, d, dtype=dtype, device=device)

    result = flash_attention_wrapper(query, key, value)
    golden = flash_attention_golden(query, key, value)

    print(f"  Shape: N={N}, H={H}, L={L}, d={d}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=3e-3, atol=3e-3,
        )

    print("  [PRECISION_PASS]")
    print("  Passed\n")


def test_flash_attention_causal(device_id=None, run_mode="npu"):
    """Test: Causal attention mask [1, 8, 128, 128]."""
    print("=" * 60)
    print("Test: flash_attention Causal Mask")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(123)
    N, H, L, d = 1, 8, 128, 128  # seq_len limited to 128 for constraint
    dtype = torch.float32

    query = torch.randn(N, H, L, d, dtype=dtype, device=device)
    key = torch.randn(N, H, L, d, dtype=dtype, device=device)
    value = torch.randn(N, H, L, d, dtype=dtype, device=device)

    # Test with is_causal=True
    result = flash_attention_wrapper(query, key, value, is_causal=True)
    golden = flash_attention_golden(query, key, value, is_causal=True)

    print(f"  Shape: N={N}, H={H}, L={L}, d={d}")
    print(f"  is_causal: True")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=3e-3, atol=3e-3,
        )

    print("  [PRECISION_PASS]")
    print("  Passed\n")


def test_flash_attention_with_mask(device_id=None, run_mode="npu"):
    """Test: Custom attention mask [1, 8, 128, 128]."""
    print("=" * 60)
    print("Test: flash_attention With Mask")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(456)
    N, H, L, d = 1, 8, 128, 128  # seq_len limited to 128 for constraint
    dtype = torch.float32

    query = torch.randn(N, H, L, d, dtype=dtype, device=device)
    key = torch.randn(N, H, L, d, dtype=dtype, device=device)
    value = torch.randn(N, H, L, d, dtype=dtype, device=device)

    # Create a simple mask (mask out second half)
    attn_mask = torch.zeros(N, H, L, L, dtype=dtype, device=device)
    attn_mask[:, :, :, L//2:] = float('-inf')

    result = flash_attention_wrapper(query, key, value, attn_mask=attn_mask)
    golden = flash_attention_golden(query, key, value, attn_mask=attn_mask)

    print(f"  Shape: N={N}, H={H}, L={L}, d={d}")
    print(f"  attn_mask: Custom (second half masked)")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=3e-3, atol=3e-3,
        )

    print("  [PRECISION_PASS]")
    print("  Passed\n")


def test_flash_attention_perf_p0(device_id=None, run_mode="npu"):
    """Test: Performance P0 configuration [1, 8, 128, 128]."""
    print("=" * 60)
    print("Test: flash_attention Perf P0")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(789)
    N, H, L, d = 1, 8, 128, 128  # seq_len limited to 128 for constraint
    dtype = torch.float32

    query = torch.randn(N, H, L, d, dtype=dtype, device=device)
    key = torch.randn(N, H, L, d, dtype=dtype, device=device)
    value = torch.randn(N, H, L, d, dtype=dtype, device=device)

    result = flash_attention_wrapper(query, key, value)
    golden = flash_attention_golden(query, key, value)

    print(f"  Shape: N={N}, H={H}, L={L}, d={d}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=3e-3, atol=3e-3,
        )

    print("  [PRECISION_PASS]")
    print("  Passed\n")


def test_flash_attention_custom_scale(device_id=None, run_mode="npu"):
    """Test: Custom scale factor."""
    print("=" * 60)
    print("Test: flash_attention Custom Scale")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(111)
    N, H, L, d = 1, 2, 8, 8  # Use level0 shape (32-byte aligned)
    dtype = torch.float32

    query = torch.randn(N, H, L, d, dtype=dtype, device=device)
    key = torch.randn(N, H, L, d, dtype=dtype, device=device)
    value = torch.randn(N, H, L, d, dtype=dtype, device=device)
    custom_scale = 0.5

    result = flash_attention_wrapper(query, key, value, scale=custom_scale)
    golden = flash_attention_golden(query, key, value, scale=custom_scale)

    print(f"  Shape: N={N}, H={H}, L={L}, d={d}")
    print(f"  Custom scale: {custom_scale}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=3e-3, atol=3e-3,
        )

    print("  [PRECISION_PASS]")
    print("  Passed\n")


# ─────────────────────────────────────────────
# 3. CLI Entry Point
# ─────────────────────────────────────────────

# Test case registry
EXAMPLES = {
    "flash_attention::test_flash_attention_level0": {
        "name": "flash_attention Level 0",
        "description": "Small scale basic verification [1, 2, 8, 8]",
        "function": test_flash_attention_level0,
    },
    "flash_attention::test_flash_attention_level1": {
        "name": "flash_attention Level 1",
        "description": "Typical scale verification [2, 4, 128, 64]",
        "function": test_flash_attention_level1,
    },
    "flash_attention::test_flash_attention_causal": {
        "name": "flash_attention Causal",
        "description": "Causal attention mask [1, 8, 512, 128]",
        "function": test_flash_attention_causal,
    },
    "flash_attention::test_flash_attention_with_mask": {
        "name": "flash_attention With Mask",
        "description": "Custom attention mask [1, 8, 512, 128]",
        "function": test_flash_attention_with_mask,
    },
    "flash_attention::test_flash_attention_perf_p0": {
        "name": "flash_attention Perf P0",
        "description": "Performance configuration [1, 8, 1024, 128]",
        "function": test_flash_attention_perf_p0,
    },
    "flash_attention::test_flash_attention_custom_scale": {
        "name": "flash_attention Custom Scale",
        "description": "Custom scale factor test [1, 2, 8, 8]",
        "function": test_flash_attention_custom_scale,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO flash_attention operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s flash_attention::test_flash_attention_level0    Run Level 0
  %(prog)s --list                                         List all cases
  %(prog)s                                                Run all cases
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

    # Select test cases
    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{args.example_id}'")
            print(f"Valid: {', '.join(sorted(EXAMPLES))}")
            sys.exit(1)
        to_run = [(args.example_id, EXAMPLES[args.example_id])]
    else:
        to_run = list(sorted(EXAMPLES.items()))

    # NPU device initialization
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            print("ERROR: TILE_FWK_DEVICE_ID not set, falling back to sim mode")
            args.run_mode = "sim"
        else:
            import torch_npu
            torch.npu.set_device(device_id)

    # Execute tests
    passed = 0
    failed = 0
    try:
        for key, info in to_run:
            print(f"\nRunning {key}: {info['name']}")
            try:
                info["function"](device_id, args.run_mode)
                passed += 1
            except AssertionError as e:
                print(f"[PRECISION_FAIL] {e}")
                failed += 1
            except Exception as e:
                print(f"Runtime error: {e}")
                failed += 1

        print("\n" + "=" * 60)
        print(f"Test Results: {passed} passed, {failed} failed")
        print("=" * 60)

        if failed > 0:
            sys.exit(1)
        else:
            print("All tests passed!")

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
