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
import torch_npu
import numpy as np
from numpy.testing import assert_allclose


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("ERROR: Environment variable TILE_FWK_DEVICE_ID is not set.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


B = pypto.frontend.dynamic("B")
N1 = 32
N2 = 1
DIM = 256


def softmax_core(input_tensor: pypto.Tensor((N1, N2, DIM), pypto.DT_FP32)) -> pypto.Tensor((N1, N2, DIM), pypto.DT_FP32):
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
    # Find maximum for numerical stability
    row_max = pypto.amax(input_tensor, dim=-1, keepdim=True)

    # Subtract maximum
    sub = pypto.sub(input_tensor, row_max)

    # Compute exponentials
    exp = pypto.exp(sub)

    # Sum exponentials
    esum = pypto.sum(exp, dim=-1, keepdim=True)

    return pypto.div(exp, esum)


# @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
@pypto.frontend.jit()
def softmax_npu(
    input_tensor: pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32),
    ) -> (
        pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32)
    ):
    """
    Softmax implementation with dynamic batch size support.

    This function processes input tensors in batches, applying softmax
    to each batch independently. The batch dimension is marked as dynamic,
    allowing variable batch sizes at runtime.

    Parameters
    ----------
    input_tensor : pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32)
        Input tensor [batch, n1, n2, dim]

    Returns
    -------
    pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32)
        Softmax normalized tensor [batch, n1, n2, dim]
    """
    output_tensor = pypto.tensor((B, N1, N2, DIM), pypto.DT_FP32)
    tile_b = 1  # Process one batch at a time
    b_loop = B // tile_b

    # Tiling shape setting for efficient execution
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b

        # Extract batch slice
        input_view = input_tensor[b_offset:b_offset_end, :N1, :N2, :DIM]

        # Apply softmax to batch slice
        softmax_out = softmax_core(input_view)

        # Assemble result back to output tensor
        pypto.assemble(softmax_out, [b_offset, 0, 0, 0], output_tensor)

    return output_tensor


# @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
# def softmax_sim(
#     input_tensor: pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32),
# ) -> pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32):
#     """
#     Softmax implementation with dynamic batch size support.

#     This function processes input tensors in batches, applying softmax
#     to each batch independently. The batch dimension is marked as dynamic,
#     allowing variable batch sizes at runtime.

#     Parameters
#     ----------
#     input_tensor : pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32)
#         Input tensor [batch, n1, n2, dim]

#     Returns
#     -------
#     pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32)
#         Softmax normalized tensor [batch, n1, n2, dim]
#     """
#     # After the dynamic axis of tensor is marked, get the tensor shape accordingly
#     tensor_shape = input_tensor.shape
#     b = tensor_shape[0]  # Dynamic batch size
#     n1, n2, dim = tensor_shape[1:]  # Static dimensions
#     tile_b = 1  # Process one batch at a time
#     b_loop = b // tile_b

#     # Tiling shape setting for efficient execution
#     pypto.set_vec_tile_shapes(1, 4, 1, 64)

#     # Create output tensor
#     output_tensor = pypto.tensor(tensor_shape, input_tensor.dtype)

#     for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
#         b_offset = idx * tile_b
#         b_offset_end = (idx + 1) * tile_b

#         # Extract batch slice
#         input_view = input_tensor[b_offset:b_offset_end, :n1, :n2, :dim]

#         # Apply softmax to batch slice
#         softmax_out = softmax_core(input_view)

#         # Assemble result back to output tensor
#         pypto.assemble(softmax_out, [b_offset, 0, 0, 0], output_tensor)

#     return output_tensor


def test_softmax(tensor_type, run_mode):
    """
    Test softmax implementation against PyTorch reference.

    Tests with shape [batch, n1, n2, dim] where batch is dynamic.
    """
    print("=" * 60)
    print("Test: Softmax")
    print("=" * 60)

    # Shape for verification: NCHW format, N can be any integer number as it is defined as dynamic axis
    shape = (32, 32, 1, 256)

    device_id = torch.npu.current_device() if tensor_type == 'npu' else 0

    # Prepare data
    input_tensor = torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}') if tensor_type == 'npu' else torch.rand(shape, dtype=torch.float32)
    
    if run_mode == 'npu':
        output_tensor = softmax_npu(input_tensor)
        pypto.runtime._device_synchronize()
    # else:
    #     output_tensor = softmax_sim(input_tensor)

    # Verify against PyTorch reference
    torch_softmax = torch.softmax(input_tensor, dim=3)
    npu_data = output_tensor.cpu()
    torch_data = torch_softmax.cpu()

    max_diff = np.abs(npu_data.numpy() - torch_data.numpy()).max()
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == 'npu':
        assert_allclose(np.array(npu_data), np.array(torch_data), rtol=3e-3, atol=3e-3)
    print("✓ Softmax test passed")
    print()


def main():
    """Run softmax example.

    Usage:
        python hello_world.py          # Run example
        python hello_world.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Softmax Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run the example on NPU
  %(prog)s --run_mode=sim       Run the example on simulator
  %(prog)s --tensor_type=cpu    Init tensor with CPU
        """
    )
    parser.add_argument("--run_mode", "--run-mode", nargs="?", type=str, default="npu",
                        choices=["npu", "sim"],
                        help="run mode, such as npu/sim etc.")
    parser.add_argument("--tensor_type", "--tensor-type", nargs="?", type=str, default="npu",
                        choices=["npu", "cpu"],
                        help="tensor type, such as npu/cpu etc.")

    args = parser.parse_args()

    # check run mode and tensor type
    if args.run_mode == "sim" and args.tensor_type == "npu":
        print(f"Error: Invalid parameters, tensor type(npu) and run mode(sim) can not be used at same time.")
        print(f"Tensor with npu can not be executed on sim.")
        sys.exit(1)

    # Get and validate device ID (needed for NPU examples)
    device_id = None
    # Check if any example requires NPU
    if args.tensor_type == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        # Set the device once for all examples
        torch.npu.set_device(device_id)

    try:
        test_softmax(args.tensor_type, args.run_mode)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

