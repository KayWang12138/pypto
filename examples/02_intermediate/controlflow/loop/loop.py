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
Loop Feature Example for PyPTO

This example demonstrates:
- Basic Loop Usage
- Loop Compile Phase Print Feature
"""

import os
import sys
import argparse
import torch
import pypto
import numpy as np
from numpy.testing import assert_allclose


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("ERROR: Environment variable TILE_FWK_DEVICE_ID is not set.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ["TILE_FWK_DEVICE_ID"])
        return device_id
    except ValueError:
        print(
            f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}"
        )
        return None


DTYPE = pypto.DT_FP16


def loop_basic(t0, t1, dynamic: bool = True):
    if dynamic:
        N = pypto.frontend.dynamic("N")
    else:
        N = 8

    S = 64
    SHAPE = (N * S, S)

    @pypto.frontend.jit()
    def loop_basic_kernel(
        t0: pypto.Tensor(SHAPE, DTYPE),
        t1: pypto.Tensor(SHAPE, DTYPE),
    ) -> (
        pypto.Tensor(SHAPE, DTYPE),
        pypto.Tensor(SHAPE, DTYPE),
    ):
        out0 = pypto.tensor(SHAPE, DTYPE)
        out1 = pypto.tensor(SHAPE, DTYPE)
        pypto.set_vec_tile_shapes(64, 64)
        for bs_idx in pypto.loop(0, N, 1):  # start, stop, step
            t0s = t0[bs_idx * S : (bs_idx + 1) * S, :]
            t1s = t1[bs_idx * S : (bs_idx + 1) * S, :]
            out0[bs_idx * S : (bs_idx + 1) * S, :] = pypto.add(t0s, t1s)
        new_step = 2
        for bs_idx in pypto.loop(0, N, new_step):  # start, stop, step
            t0s = t0[bs_idx * S : (bs_idx + new_step) * S, :]
            t1s = t1[bs_idx * S : (bs_idx + new_step) * S, :]
            out1[bs_idx * S : (bs_idx + new_step) * S, :] = pypto.add(t0s, t1s)
        return out0, out1

    # launch the kernel
    y1, y2 = loop_basic_kernel(t0, t1)
    return y1, y2


def test_loop_basic(device_id=None, dynamic: bool = False) -> None:
    """Test basic loop usage."""
    print("=" * 60)
    print("Test: Basic Loop Usage")
    print("=" * 60)

    device_id = torch.npu.current_device()

    s, n = 64, 8
    shape = (n * s, s)
    input_t1 = torch.randn(shape, dtype=torch.float16, device=f"npu:{device_id}")
    input_t2 = torch.randn(shape, dtype=torch.float16, device=f"npu:{device_id}")
    output1, output2 = loop_basic(input_t1, input_t2, dynamic)
    pypto.runtime._device_synchronize()

    # Verify
    expected = input_t1 + input_t2
    max_diff1 = (output1 - expected).abs().max().item()
    max_diff2 = (output2 - expected).abs().max().item()
    equal_output_1_2 = (output1 - output2).abs().max().item() < 1e-6
    print(f"Max difference from PyTorch: {max_diff1:.6f}")
    assert max_diff1 < 1e-2, "Result mismatch!"
    assert max_diff2 < 1e-2, "Result mismatch!"
    print(f"Whether output1 equals output2: {equal_output_1_2}")
    print("✓ Basic loop usage completed successfully")
    print()


def main():
    """Run loop_feature examples.

    Usage:
        python loop.py              # Run all examples
        python loop.py loop_basic::test_loop_basic
            Run example loop_basic::test_loop_basic
        python loop.py --list       # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Loop Feature Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s loop_basic::test_loop_basic
            Run example loop_basic::test_loop_basic
  %(prog)s --list       List all available examples
        """,
    )
    parser.add_argument(
        "example_id",
        type=str,
        nargs="?",
        help="Example ID to run. If not specified, all examples will run.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available examples and exit"
    )

    args = parser.parse_args()

    # Define available examples
    examples = {
        "loop_basic::test_loop_basic": {
            "name": "Test basic loop usage",
            "description": "Basic loop usages example",
            "function": test_loop_basic,
            "requires_npu": True,
        }
    }

    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  ID: {ex_id}")
            print(f"     name: {ex_info['name']}")
            print(f"     description: {ex_info['description']}\n")
        return

    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(
                f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}"
            )
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO Loop Feature Examples")
    print("=" * 60 + "\n")

    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        # Run single example
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        # Run all examples
        examples_to_run = list(examples.items())

    # Check if any example requires NPU
    requires_npu = any(ex_info["requires_npu"] for _, ex_info in examples_to_run)

    if requires_npu:
        device_id = get_device_id()
        if device_id is None:
            return
        # Set the device once for all examples
        torch.npu.set_device(device_id)

    try:
        for ex_id, ex_info in examples_to_run:
            if ex_info["requires_npu"] and device_id is None:
                print(
                    f"Skipping example {ex_id} ({ex_info['name']}): NPU device not configured"
                )
                continue

            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info["function"]()

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All loop tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
