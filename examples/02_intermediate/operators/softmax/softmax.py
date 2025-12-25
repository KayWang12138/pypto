#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Softmax Example for PyPTO

This example demonstrates how to implement a softmax operation using PyPTO, including:
- Manual softmax computation from basic operations
- Dynamic axis marking for variable batch sizes
- Tiling configuration for efficient execution
- Loop-based processing for large tensors

Softmax is a fundamental operation in neural networks, especially for attention mechanisms.
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


def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
    """
    Core softmax computation: exp(x - max(x)) / sum(exp(x - max(x))).

    Parameters
    ----------
    input_tensor : pypto.tensor
        Input tensor to apply softmax to

    Returns
    -------
    pypto.tensor
        Softmax normalized tensor
    """
    row_max = pypto.amax(x, dim=-1, keepdim=True)
    sub = x - row_max
    exp = pypto.exp(sub)
    esum = pypto.sum(exp, dim=-1, keepdim=True)
    return exp / esum


B = pypto.frontend.dynamic("B")
N1, N2, DIM = 32, 1, 256


@pypto.frontend.jit()
def softmax_kernel_npu(
    input_tensor: pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32),
) -> pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32):
    output_tensor = pypto.tensor((B, N1, N2, DIM), pypto.DT_FP32)
    tile_b = 1  # Process one batch at a time
    b_loop = B // tile_b

    # Tiling shape setting for efficient execution
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
        b_offset = idx * tile_b
        b_offset_end = pypto.min((idx + 1) * tile_b, B)
        input_view = pypto.view(input_tensor, [tile_b, N1, N2, DIM], [b_offset, 0, 0, 0], valid_shape=[b_offset_end - b_offset, N1, N2, DIM])
        softmax_out = softmax_core(input_view)
        output_tensor[b_offset:, ...] = softmax_out
    return output_tensor


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def softmax_kernel_sim(
    input_tensor: pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32),
) -> pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32):
    output_tensor = pypto.tensor((B, N1, N2, DIM), pypto.DT_FP32)
    tile_b = 1  # Process one batch at a time
    b_loop = B // tile_b

    # Tiling shape setting for efficient execution
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        input_view = input_tensor[b_offset:b_offset_end, :N1, :N2, :DIM]
        softmax_out = softmax_core(input_view)
        output_tensor[b_offset:, ...] = softmax_out
    return output_tensor


def softmax(input_tensor: torch.Tensor, run_mode: str = "npu") -> torch.Tensor:
    if run_mode == "npu":
        return softmax_kernel_npu(input_tensor)
    else:
        return softmax_kernel_sim(input_tensor)


def test_softmax(device_id=None, run_mode: str = "npu"):
    """
    Test softmax implementation against PyTorch reference.

    Tests with shape [batch, n1, n2, dim] where batch is dynamic.
    """
    if not device_id:
        device_id = torch.npu.current_device()
    else:
        torch.npu.set_device(device_id)
    if run_mode == "npu":
        import torch_npu
        device = f'npu:{device_id}'
    else:
        device = 'cpu'


    b_concrete = 32
    shape = (b_concrete, N1, N2, DIM)
    x = torch.rand(shape, dtype=torch.float32, device=device)

    y = softmax(x, run_mode).cpu()
    golden = torch.softmax(x, dim=3).cpu()

    max_diff = np.abs(y.numpy() - golden.numpy()).max()
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Max difference: {max_diff:.6f}")

    if run_mode == "npu":
        assert_allclose(np.array(y), np.array(golden), rtol=3e-3, atol=3e-3)
    print("✓ Softmax test passed")
    print()

def main():
    """Run softmax example.

    Usage:
        python softmax.py          # Run example
        python softmax.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Softmax Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run the example
  %(prog)s --list       List all available examples
        """,
    )
    parser.add_argument(
        "example_id",
        type=int,
        nargs="?",
        help="Example ID to run (1). If not specified, the example will run.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available examples and exit"
    )

    args = parser.parse_args()

    # Define available examples
    examples = {
        1: {
            "name": "Softmax",
            "description": "Softmax implementation with dynamic batch size",
            "function": test_softmax,
            "requires_npu": True,
        }
    }

    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            npu_req = (
                " (Requires NPU)" if ex_info["requires_npu"] else " (No NPU required)"
            )
            print(f"  {ex_id}. {ex_info['name']}{npu_req}")
            print(f"     {ex_info['description']}\n")
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
    print("PyPTO Softmax Example")
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
            print("All softmax tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
