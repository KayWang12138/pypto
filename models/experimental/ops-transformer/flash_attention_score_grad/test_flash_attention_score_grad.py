#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""
Flash Attention Score Grad Test

Tests PyPTO implementation against PyTorch golden reference.

Test cases:
- Causal P0: [2,8,128,64], causal mask (sparse_mode=3)

Precision thresholds:
- FP16: atol=0.005, rtol=0.005
- BF16: atol=0.005, rtol=0.005
"""

import os
import sys
import argparse
import math

import torch
import numpy as np
from numpy.testing import assert_allclose

from flash_attention_score_grad_golden import flash_attention_score_grad_golden
from flash_attention_score_grad_impl import flash_attention_score_grad_wrapper


# ─────────────────────────────────────────────
# Environment utilities
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
# Test functions
# ─────────────────────────────────────────────

def test_causal_p0(device_id=None, run_mode="npu"):
    """Test case: Causal P0 - [2,8,128,64], sparse_mode=3 (causal mask)"""
    print("=" * 60)
    print("Test: Flash Attention Score Grad - Causal P0")
    print("=" * 60)
    
    B, N, S, D = 2, 8, 128, 64
    scale = 1.0 / math.sqrt(D)
    
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    
    torch.manual_seed(42)
    query = torch.randn(B, N, S, D, dtype=torch.float16, device=device)
    key = torch.randn(B, N, S, D, dtype=torch.float16, device=device)
    value = torch.randn(B, N, S, D, dtype=torch.float16, device=device)
    dy = torch.randn(B, N, S, D, dtype=torch.float16, device=device)
    
    print(f"  Input shapes: Q{list(query.shape)}, K{list(key.shape)}, V{list(value.shape)}, dY{list(dy.shape)}")
    print(f"  Sparse mode: 3 (causal mask)")
    
    # Run golden with causal mask
    dq_golden, dk_golden, dv_golden = flash_attention_score_grad_golden(
        query.cpu(), key.cpu(), value.cpu(), dy.cpu(),
        scale_value=scale,
        keep_prob=1.0,
        input_layout="BNSD",
        sparse_mode=3,  # causal
        pse_type=1,
    )
    
    # Run PyPTO implementation
    try:
        dq_impl, dk_impl, dv_impl = flash_attention_score_grad_wrapper(
            query, key, value, dy,
            sparse_mode=3,  # causal
        )
    except Exception as e:
        print(f"  [ERROR] Implementation failed: {e}")
        import traceback
        traceback.print_exc()
        print("[PRECISION_FAIL] Implementation error")
        sys.exit(1)
    
    dq_impl = dq_impl.cpu()
    dk_impl = dk_impl.cpu()
    dv_impl = dv_impl.cpu()
    
    rtol = 0.005
    atol = 0.005
    
    print(f"\n  Precision check (rtol={rtol}, atol={atol}):")
    
    try:
        dq_diff = torch.abs(dq_impl - dq_golden).max().item()
        print(f"    dQ max diff: {dq_diff:.6e}")
        assert_allclose(dq_impl.numpy(), dq_golden.numpy(), rtol=rtol, atol=atol)
        
        dk_diff = torch.abs(dk_impl - dk_golden).max().item()
        print(f"    dK max diff: {dk_diff:.6e}")
        assert_allclose(dk_impl.numpy(), dk_golden.numpy(), rtol=rtol, atol=atol)
        
        dv_diff = torch.abs(dv_impl - dv_golden).max().item()
        print(f"    dV max diff: {dv_diff:.6e}")
        assert_allclose(dv_impl.numpy(), dv_golden.numpy(), rtol=rtol, atol=atol)
        
        print("\n  [PRECISION_PASS] All gradients match within tolerance")
        return True
        
    except AssertionError as e:
        print(f"\n  [PRECISION_FAIL] {e}")
        return False


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

EXAMPLES = {
    "flash_attention_score_grad::test_causal_p0": {
        "name": "Causal P0",
        "description": "[2,8,128,64], causal mask (sparse_mode=3)",
        "function": test_causal_p0,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Flash Attention Score Grad Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s flash_attention_score_grad::test_causal_p0
  %(prog)s --list
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
            print(f"  {key}  — {info['description']}")
        return
    
    # Select case
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
            print("ERROR: TILE_FWK_DEVICE_ID not set")
            sys.exit(1)
        import torch_npu
        torch.npu.set_device(device_id)
    
    # Execute tests
    all_passed = True
    for key, info in to_run:
        print(f"\n▸ Running {key}: {info['name']}")
        try:
            result = info["function"](device_id, args.run_mode)
            if not result:
                all_passed = False
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
        print("=" * 60)
    else:
        print("Some tests failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()