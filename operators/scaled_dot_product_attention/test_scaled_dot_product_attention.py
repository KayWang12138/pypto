#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms of the CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT warranties or any kind, either express or implied, including but not limited to non-infringement, merchantability, or fitness for a particular purpose.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO scaled_dot_product_attention operator test.

Test cases based on spec.md configurations:
- Level 0: Small scale basic verification
- Level 1: Typical scale functional verification
- Level 2: Causal attention test
- Level 3: With attention mask test

Precision verification uses numpy.testing.assert_allclose.
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from scaled_dot_product_attention_golden import scaled_dot_product_attention_golden
from scaled_dot_product_attention_impl import scaled_dot_product_attention_wrapper


# ============================================================================
# Environment utilities
# ============================================================================

def get_device_id():
    """Get and validate TILE_FWK_DEVICE_ID from environment."""
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def get_run_mode() -> str:
    """Get run mode from command line."""
    for idx, arg in enumerate(sys.argv):
        if arg == "--run_mode" and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith("--run_mode="):
            return arg.split("=", 1)[1]
    return "npu"


# ============================================================================
# Test cases
# ============================================================================

def test_level0_basic(device_id=None, run_mode="npu"):
    """Level 0: Small scale basic verification (8 elements)."""
    print("=" * 60)
    print("Test: scaled_dot_product_attention Level 0 (basic)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # Small scale test data
    torch.manual_seed(0)
    batch, num_heads, seq_len, head_dim = 1, 2, 4, 4
    query = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)
    key = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)
    value = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)

    # Execute kernel wrapper
    result = scaled_dot_product_attention_wrapper(query, key, value)

    # Execute golden
    golden = scaled_dot_product_attention_golden(query, key, value)

    # Precision comparison
    print(f"  Input shape : {query.shape}")
    print(f"  Output shape: {result.shape}")

    # Handle BF16 conversion for comparison
    result_np = result.cpu().numpy()
    golden_np = golden.cpu().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result_np,
            golden_np,
            rtol=1e-3,
            atol=1e-3,
        )
        print("[PRECISION_PASS]")
    print("  Passed\n")


def test_level1_typical(device_id=None, run_mode="npu"):
    """Level 1: Typical scale functional verification (P0 config)."""
    print("=" * 60)
    print("Test: scaled_dot_product_attention Level 1 (typical)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # Typical scale from spec.md P0
    torch.manual_seed(42)
    batch, num_heads, seq_len, head_dim = 2, 4, 512, 64
    query = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)
    key = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)
    value = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)

    # Execute kernel wrapper
    result = scaled_dot_product_attention_wrapper(query, key, value)

    # Execute golden
    golden = scaled_dot_product_attention_golden(query, key, value)

    # Precision comparison
    print(f"  Input shape : {query.shape}")
    print(f"  Output shape: {result.shape}")

    result_np = result.cpu().numpy()
    golden_np = golden.cpu().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result_np,
            golden_np,
            rtol=1e-3,
            atol=1e-3,
        )
        print("[PRECISION_PASS]")
    print("  Passed\n")


def test_level2_causal(device_id=None, run_mode="npu"):
    """Level 2: Causal attention test."""
    print("=" * 60)
    print("Test: scaled_dot_product_attention Level 2 (causal)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # Causal attention config from spec.md
    torch.manual_seed(100)
    batch, num_heads, seq_len, head_dim = 1, 8, 512, 128
    query = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)
    key = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)
    value = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)

    # Execute kernel wrapper with is_causal=True
    result = scaled_dot_product_attention_wrapper(
        query, key, value,
        is_causal=True
    )

    # Execute golden with is_causal=True
    golden = scaled_dot_product_attention_golden(
        query, key, value,
        is_causal=True
    )

    # Precision comparison
    print(f"  Input shape : {query.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  is_causal  : True")

    result_np = result.cpu().numpy()
    golden_np = golden.cpu().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result_np,
            golden_np,
            rtol=1e-3,
            atol=1e-3,
        )
        print("[PRECISION_PASS]")
    print("  Passed\n")


def test_level3_with_mask(device_id=None, run_mode="npu"):
    """Level 3: Attention with mask test."""
    print("=" * 60)
    print("Test: scaled_dot_product_attention Level 3 (with mask)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # Masked attention config from spec.md
    torch.manual_seed(200)
    batch, num_heads, seq_len, head_dim = 1, 8, 512, 128
    query = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)
    key = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)
    value = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)

    # Create float attention mask (additive mask)
    attn_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32, device=device)
    # Mask some positions with -inf
    attn_mask[:, :, :, seq_len//2:] = float('-inf')

    # Execute kernel wrapper with mask
    result = scaled_dot_product_attention_wrapper(
        query, key, value,
        attn_mask=attn_mask
    )

    # Execute golden with mask
    golden = scaled_dot_product_attention_golden(
        query, key, value,
        attn_mask=attn_mask
    )

    # Precision comparison
    print(f"  Input shape : {query.shape}")
    print(f"  Mask shape  : {attn_mask.shape}")
    print(f"  Output shape: {result.shape}")

    result_np = result.cpu().numpy()
    golden_np = golden.cpu().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result_np,
            golden_np,
            rtol=1e-3,
            atol=1e-3,
        )
        print("[PRECISION_PASS]")
    print("  Passed\n")


def test_level4_bfloat16(device_id=None, run_mode="npu"):
    """Level 4: BFloat16 input test."""
    print("=" * 60)
    print("Test: scaled_dot_product_attention Level 4 (bfloat16)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # BF16 test
    torch.manual_seed(300)
    batch, num_heads, seq_len, head_dim = 1, 4, 256, 64
    query = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    key = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    value = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    # Execute kernel wrapper
    result = scaled_dot_product_attention_wrapper(query, key, value)

    # Execute golden
    golden = scaled_dot_product_attention_golden(query, key, value)

    # Precision comparison
    print(f"  Input shape : {query.shape}")
    print(f"  Input dtype : {query.dtype}")
    print(f"  Output shape: {result.shape}")

    result_np = result.cpu().float().numpy()
    golden_np = golden.cpu().float().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        # BF16 has relaxed tolerance
        assert_allclose(
            result_np,
            golden_np,
            rtol=0.01,
            atol=0.01,
        )
        print("[PRECISION_PASS]")
    print("  Passed\n")


# ============================================================================
# CLI entry
# ============================================================================

EXAMPLES = {
    "scaled_dot_product_attention::test_level0_basic": {
        "name": "Level 0 Basic",
        "description": "Small scale basic verification (8 elements)",
        "function": test_level0_basic,
    },
    "scaled_dot_product_attention::test_level1_typical": {
        "name": "Level 1 Typical",
        "description": "Typical scale functional verification",
        "function": test_level1_typical,
    },
    "scaled_dot_product_attention::test_level2_causal": {
        "name": "Level 2 Causal",
        "description": "Causal attention test",
        "function": test_level2_causal,
    },
    "scaled_dot_product_attention::test_level3_with_mask": {
        "name": "Level 3 With Mask",
        "description": "Attention with mask test",
        "function": test_level3_with_mask,
    },
    "scaled_dot_product_attention::test_level4_bfloat16": {
        "name": "Level 4 BFloat16",
        "description": "BFloat16 input test",
        "function": test_level4_bfloat16,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO scaled_dot_product_attention operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scaled_dot_product_attention::test_level0_basic
  %(prog)s --list
  %(prog)s --run_mode sim
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

    # List cases
    if args.list:
        print("\nAvailable cases:\n")
        for key, info in sorted(EXAMPLES.items()):
            print(f"  {key}  - {info['description']}")
        return

    # Select cases
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
            return
        import torch_npu
        torch.npu.set_device(device_id)

    # Execute tests
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
