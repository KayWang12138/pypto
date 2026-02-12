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
Sinh Activation Function Example for PyPTO

This example demonstrates how to implement the sinh (hyperbolic sine) activation function.
Formula: sinh(x) = (x - x) / 2
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


def configure_tiling(x):
    """Configure vector tile shapes based on input tensor shape."""
    if len(x.shape) >= 2:
        tile_list = [32 for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_list)
    else:
        pypto.set_vec_tile_shapes(32, 128)


def sinh_golden(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of sinh."""
    return torch.sinh(x)


def test_sinh_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of sinh function (Level 0: 8-16 elements)"""
    print("=" * 60)
    print("Test: Basic Usage of sinh Function (Level 0)")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    dtype = torch.float32
    x = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5], dtype=dtype, device=device)

    from sinh_impl import sinh_op
    out = sinh_op(x, run_mode)

    expected = sinh_golden(x)
    max_diff = (out - expected).abs().max().item()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Input: {x}")
    print(f"Output: {out}")
    print(f"Expected: { {expected}}")
    if run_mode == "npu":
        print(f"Max difference: {max_diff:.6f}")
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=0.005, atol=0.000025)
    print("✓ Basic usage of sinh function completed successfully")
    print()


def test_sinh_1k(device_id: int = None, run_mode: str = "npu"):
    """Test sinh with 1K elements (Level 1: typical scenario)"""
    print("=" * 60)
    print("Test: Sinh with 1K Elements (Level 1)")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    dtype = torch.float32
    x = torch.randn(1024, dtype=dtype, device=device)

    from sinh_impl import sinh_op
    out = sinh_op(x, run_mode)

    expected = sinh_golden(x)
    max_diff = (out - expected).abs().max().item()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    if run_mode == "npu":
        print(f"Max difference: {max_diff:.6f}")
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=0.005, atol=0.000025)
    print("✓ Sin with 1K elements completed successfully")
    print()


def test_sinh_edge_cases(device_id: int = None, run_mode: str = "npu"):
    """Test sinh with extreme values and zeros (Level 2: boundary cases)"""
    print("=" * 60)
    print("Test: Sinh with Edge Cases (Level 2)")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    dtype = torch.float32
    x = torch.tensor([0.0, -0.0, 1e-6, -1e-6, 10.0, -10.0], dtype=dtype, device=device)

    from sinh_impl import sinh_op
    out = sinh_op(x, run_mode)

    expected = sinh_golden(x)
    max_diff = (out - expected).abs().max().item()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Input: {x}")
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    if run_mode == "npu":
        print(f"Max difference: {max_diff:.6f}")
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=0.005, atol=0.000025)
    print("✓ Sinh with edge cases completed successfully")
    print()


def test_sinh_4d(device_id: int = None, run_mode: str = "npu"):
    """Test sinh with 4D tensor [b, s, n, d]"""
    print("=" * 60)
    print("Test: Sinh with 4D Tensor [b, s, n, d]")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    dtype = torch.float32
    x = torch.randn(2, 4, 8, 16, dtype=dtype, device=device)

    from sinh_impl import sinh_op
    out = sinh_op(x, run_mode)

    expected = sinh_golden(x)
    max_diff = (out - expected).abs().max().item()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    if run_mode == "npu":
        print(f"Max difference: {max_diff:.6f}")
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=0.005, atol=0.000025)
    print("✓ Sinh with 4D tensor completed successfully")
    print()


def main():
    """Run sinh examples.

    Usage:
        python sinh.py              # Run all examples
        python sinh.py --list       # List all available examples
        python sinh.py basic::test_sinh_basic    # Run a specific case
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Sinh Activation Function Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s --list       List all available examples
  %(prog)s basic::test_sinh_basic    Run a specific case
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
        'basic::test_sinh_basic': {
            'name': 'Basic Usage of sinh Function',
            'description': 'Basic usage of sinh function (Level 0: 8-16 elements)',
            'function': test_sinh_basic,
        },
        '1k::test_sinh_1k': {
            'name': 'Sinh with 1K Elements',
            'description': 'Test sinh with 1K elements (Level 1: typical scenario)',
            'function': test_sinh_1k,
        },
        'edge::test_sinh_edge_cases': {
            'name': 'Sinh with Edge Cases',
            'description': 'Test sinh with extreme values and zeros (Level 2: boundary cases)',
            'function': test_sinh_edge_cases,
        },
        '4d::test_sinh_4d': {
            'name': 'Sinh with 4D Tensor',
            'description': 'Test sinh with 4D tensor [b, s, n, d]',
            'function': test_sinh_4d,
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

    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO Sinh Activation Function Examples")
    print("=" * 60 + "\n")

    device_id = None
    examples_to_run = []

    if args.example_id is not None:
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
