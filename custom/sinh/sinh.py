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
Sinh (Hyperbolic Sine) Activation Function for PyPTO

This example demonstrates how to implement the hyperbolic sine function using PyPTO.
Formula: sinh(x) = (e^x - e^(-x)) / 2
"""

import os
import sys
import argparse
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


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
    """PyTorch reference implementation of sinh."""
    return torch.sinh(x)


def configure_tiling(x):
    """Configure tiling for optimal performance."""
    if len(x.shape) >= 2:
        tile_list = [32 for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_list)
    else:
        pypto.set_vec_tile_shapes(32, 128)


def sinh_activation(shape: tuple, run_mode: str = "npu", dynamic: bool = False):
    """
    Create sinh activation kernel function.

    Formula: sinh(x) = (e^x - e^(-x)) / 2

    Args:
        shape: Input tensor shape
        run_mode: Execution mode ('npu' or 'sim')
        dynamic: Whether to use dynamic shapes

    Returns:
        Kernel function that can be called with input tensor
    """
    if dynamic:
        _, n = shape
        m = pypto.frontend.dynamic("M")
    else:
        m, n = shape

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def sinh_activation_kernel(
        x: pypto.Tensor((m, n), pypto.DT_FP32),
    ) -> pypto.Tensor((m, n), pypto.DT_FP32):
        """
        Hyperbolic sine function: sinh(x) = (e^x - e^(-x)) / 2

        The hyperbolic sine function is an odd function that appears in
        various mathematical and engineering applications.

        Formula breakdown:
        1. Compute e^x
        2. Compute -x
        3. Compute e^(-x)
        4. Subtract: e^x - e^(-x)
        5. Divide by 2
        """
        out = pypto.tensor((m, n), pypto.DT_FP32)
        configure_tiling(x)

        exp_x = pypto.exp(x)
        neg_x = pypto.neg(x)
        exp_neg_x = pypto.exp(neg_x)
        diff = pypto.sub(exp_x, exp_neg_x)
        out[:] = pypto.div(diff, 2.0)
        return out

    return sinh_activation_kernel


def test_sinh(device_id: int = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Test sinh activation with multiple test cases."""
    print("=" * 60)
    print("Test: Sinh Activation")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    test_cases = [
        {"name": "Level 0: Small tensor (8x8)", "shape": (8, 8)},
        {"name": "Level 1: Medium tensor (32x128)", "shape": (32, 128)},
        {"name": "Level 2: Large tensor (64x256)", "shape": (64, 256)},
    ]

    for test_case in test_cases:
        name = test_case["name"]
        shape = test_case["shape"]

        print(f"\n--- {name} ---")
        print(f"Shape: {shape}")

        x_torch = torch.randn(shape, dtype=torch.float32, device=device)

        out_torch = sinh_activation(x_torch.shape, run_mode, dynamic)(x_torch)
        expected = sinh_golden(x_torch)

        max_diff = (out_torch - expected).abs().max().item()
        mean_diff = (out_torch - expected).abs().mean().item()

        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")

        if run_mode == "npu":
            assert max_diff < 1e-3, f"Result mismatch! Max diff: {max_diff}"

        print(f"✓ {name} passed")

    print("\n" + "=" * 60)
    print("All sinh tests passed!")
    print("=" * 60)


def main():
    """Run sinh activation examples.

    Usage:
        python sinh.py              # Run all examples
        python sinh.py --list       # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Sinh Activation Function Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs='?',
        help='Example ID to run. If not specified, all examples will run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available examples and exit'
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        nargs='?',
        default="npu",
        choices=["npu", "sim"],
        help='Run mode, such as npu/sim etc.'
    )

    args = parser.parse_args()

    examples = {
        'sinh::test_sinh': {
            'name': 'Sinh Activation',
            'description': 'Hyperbolic sine function',
            'function': test_sinh,
        }
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  ID: {ex_id}")
            print(f"     name: {ex_info['name']}")
            print(f"     description: {ex_info['description']}\n")
        return

    print("\n" + "=" * 60)
    print("PyPTO Sinh Activation Function Example")
    print("=" * 60 + "\n")

    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        examples_to_run = list(examples.items())

    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running examples that require NPU hardware...")
        print("(Make sure CANN environment is configured and NPU is available)\n")

    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id, args.run_mode)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All sinh tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
