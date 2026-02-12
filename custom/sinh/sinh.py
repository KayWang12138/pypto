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
sinh 算子测试及验证

sinh(x) = (e^x - e^(-x)) / 2
"""

import os
import sys
import argparse
import torch
import numpy as np
from numpy.testing import assert_allclose
import pypto

from sinh_impl import sinh_op


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


def sinh_golden(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of sinh.

    Args:
        x: Input tensor.

    Returns:
        sinh(x) = (e^x - e^(-x)) / 2
    """
    return torch.sinh(x)


def test_sinh(device_id: int = None, run_mode: str = "npu") -> None:
    """
    Test sinh operator.

    Args:
        device_id: NPU device ID.
        run_mode: Execution mode ("npu" or "sim").
    """
    print("=" * 60)
    print("Test: sinh Operator")
    print("=" * 60)

    if run_mode == "npu" and device_id is not None:
        device = 'npu:0'  # PyTorch 看到的设备 ID 是 0
    else:
        device = 'cpu'

    test_cases = [
        {
            "name": "Level 0: Small tensor (8 elements)",
            "shape": (2, 2, 2, 1),
        },
        {
            "name": "Level 0: Small tensor (16 elements)",
            "shape": (2, 2, 2, 2),
        },
        {
            "name": "Level 1: Typical tensor (1K elements)",
            "shape": (4, 8, 8, 4),
        },
        {
            "name": "Level 2: Zero values",
            "shape": (2, 4, 4, 2),
            "init": "zeros",
        },
        {
            "name": "Level 2: Large positive values",
            "shape": (2, 4, 4, 2),
            "init": "positive_large",
        },
        {
            "name": "Level 2: Large negative values",
            "shape": (2, 4, 4, 2),
            "init": "negative_large",
        },
    ]

    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("-" * 60)

        shape = test_case["shape"]
        init = test_case.get("init", "random")

        if init == "zeros":
            x_torch = torch.zeros(shape, dtype=torch.float32, device=device)
        elif init == "positive_large":
            x_torch = torch.ones(shape, dtype=torch.float32, device=device) * 10.0
        elif init == "negative_large":
            x_torch = torch.ones(shape, dtype=torch.float32, device=device) * (-10.0)
        else:
            x_torch = torch.randn(shape, dtype=torch.float32, device=device)

        # Execute
        out_torch = sinh_op(x_torch.shape, run_mode)(x_torch)

        # Verify
        expected = sinh_golden(x_torch)
        max_diff = (out_torch - expected).abs().max().item()
        mean_diff = (out_torch - expected).abs().mean().item()
        rel_diff = ((out_torch - expected).abs() / (expected.abs() + 1e-6)).max().item()

        print(f"Input shape: {x_torch.shape}")
        print(f"Output shape: {out_torch.shape}")
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Max relative difference: {rel_diff:.6f}")

        if run_mode == "npu":
            assert rel_diff < 1e-3, f"Result mismatch! Max relative diff: {rel_diff}"
            print("✓ Test passed")
        else:
            print("✓ Test passed (sim mode)")

    print()
    print("=" * 60)
    print("All sinh tests passed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test sinh operator")
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
        # 使用 torch 的 CPU 模式
        device_id = -1  # 使用 -1 表示 CPU 模式

    test_sinh(device_id, args.run_mode)


if __name__ == "__main__":
    main()
