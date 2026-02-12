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
"""
matmul_add 算子测试及验证

y = a @ b^T + c
"""

import os
import sys
import argparse
import torch
import numpy as np
from numpy.testing import assert_allclose
import pypto

from matmul_add_impl import matmul_add_op


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("If no NPU environment is available, set --run_mode sim to run in simulation mode;")
        print("otherwise, set the environment variable TILE_FWK_DEVICE_ID.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def matmul_add_golden(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of matmul_add.

    Args:
        a: Input tensor [m, k].
        b: Input tensor [n, k].
        c: Input tensor [m, n].

    Returns:
        y = a @ b^T + c
    """
    b_t = b.transpose(0, 1)
    return torch.matmul(a, b_t) + c


def test_matmul_add(device_id: int = None, run_mode: str = "npu") -> None:
    """
    Test matmul_add operator.

    Args:
        device_id: NPU device ID.
        run_mode: Execution mode ("npu" or "sim").
    """
    print("=" * 60)
    print("Test: matmul_add Operator")
    print("=" * 60)

    if run_mode == "npu" and device_id is not None:
        device = 'npu:0'  # PyTorch 看到的设备 ID 是 0
    else:
        device = 'cpu'

    test_cases = [
        {
            "name": "Level 0: Small matrices (8x8)",
            "m": 8,
            "k": 16,
            "n": 8,
        },
        {
            "name": "Level 1: Typical matrices (128x128)",
            "m": 128,
            "k": 256,
            "n": 128,
        },
        {
            "name": "Level 2: Non-square matrix (64x32)",
            "m": 64,
            "k": 128,
            "n": 32,
        },
        {
            "name": "Level 2: Wide matrix (32x128)",
            "m": 32,
            "k": 64,
            "n": 128,
        },
        {
            "name": "Level 2: Tall matrix (128x32)",
            "m": 128,
            "k": 64,
            "n": 32,
        },
    ]

    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("-" * 60)

        m = test_case["m"]
        k = test_case["k"]
        n = test_case["n"]

        a_torch = torch.randn((m, k), dtype=torch.bfloat16, device=device)
        b_torch = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        c_torch = torch.randn((m, n), dtype=torch.bfloat16, device=device)

        # Execute
        out_torch = matmul_add_op((m, k, n), run_mode)(a_torch, b_torch, c_torch)

        # Verify
        expected = matmul_add_golden(a_torch, b_torch, c_torch)
        max_diff = (out_torch - expected).abs().max().item()
        mean_diff = (out_torch - expected).abs().mean().item()
        rel_diff = ((out_torch - expected).abs() / (expected.abs() + 1e-6)).max().item()

        print(f"Input a shape: {a_torch.shape}")
        print(f"Input b shape: {b_torch.shape}")
        print(f"Input c shape: {c_torch.shape}")
        print(f"Output shape: {out_torch.shape}")
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Max relative difference: {rel_diff:.6f}")

        if run_mode == "npu":
            # bfloat16 精度较低，容忍度设置为 1e-2
            assert rel_diff < 1e-2, f"Result mismatch! Max relative diff: {rel_diff}"
            print("✓ Test passed")
        else:
            print("✓ Test passed (sim mode)")

    print()
    print("=" * 60)
    print("All matmul_add tests passed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test matmul_add operator")
    parser.add_argument(
        "--run_mode",
        type=str,
        default="npu",
        choices=["npu", "sim"],
        help="Execution mode: 'npu' or 'sim' (default: npu)"
    )
    args = parser.parse_args()

    device_id = get_device_id()
    if device_id is None:
        print("Warning: No NPU environment available. Using CPU for golden reference.")
        device_id = 0

    test_matmul_add(device_id, args.run_mode)


if __name__ == "__main__":
    main()
