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


def softmax_core(input_tensor: pypto.tensor) -> pypto.tensor:
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


@pypto.jit
def softmax(inputs, outputs):
    """
    Softmax implementation with dynamic batch size support.
    
    This function processes input tensors in batches, applying softmax
    to each batch independently. The batch dimension is marked as dynamic,
    allowing variable batch sizes at runtime.
    
    Parameters
    ----------
    inputs : list
        List containing input tensor [batch, n1, n2, dim]
    outputs : list
        List containing output tensor [batch, n1, n2, dim]
    """
    input_tensor = inputs[0]
    output_tensor = outputs[0]

    # After the dynamic axis of tensor is marked, get the tensor shape accordingly
    tensor_shape = input_tensor.shape
    b = tensor_shape[0]  # Dynamic batch size
    n1, n2, dim = tensor_shape[1:]  # Static dimensions
    tile_b = 1  # Process one batch at a time
    b_loop = b // tile_b

    # Tiling shape setting for efficient execution
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        
        # Extract batch slice
        input_view = input_tensor[b_offset:b_offset_end, :n1, :n2, :dim]
        
        # Apply softmax to batch slice
        softmax_out = softmax_core(input_view)
        
        # Assemble result back to output tensor
        pypto.assemble(softmax_out, [b_offset, 0, 0, 0], output_tensor)


def test_softmax():
    """
    Test softmax implementation against PyTorch reference.
    
    Tests with shape [batch, n1, n2, dim] where batch is dynamic.
    """
    print("=" * 60)
    print("Test: Softmax")
    print("=" * 60)
    
    # Shape for verification: NCHW format, N can be any integer number as it is defined as dynamic axis
    shape = (32, 32, 1, 256)

    device_id = torch.npu.current_device()

    # Prepare data
    input_data = torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}')
    output_data = torch.zeros(shape, dtype=torch.float32, device=f'npu:{device_id}')

    # Initialize PyPTO inputs and outputs
    # Mark dynamic axis: the actual size of the axis can be any integer number during runtime
    inputs = {
        input_data: [0]
    }
    outputs = {
        output_data: [0]
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    # Launch the kernel
    softmax(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    # Verify against PyTorch reference
    torch_softmax = torch.softmax(input_data, dim=3)
    npu_data = output_data.cpu()
    torch_data = torch_softmax.cpu()

    max_diff = np.abs(npu_data.numpy() - torch_data.numpy()).max()
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_data.shape}")
    print(f"Max difference: {max_diff:.6f}")
    
    assert_allclose(np.array(npu_data), np.array(torch_data), rtol=3e-3, atol=3e-3)
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
        """
    )
    parser.add_argument(
        'example_id',
        type=int,
        nargs='?',
        help='Example ID to run (1). If not specified, the example will run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available examples and exit'
    )
    
    args = parser.parse_args()
    
    # Define available examples
    examples = {
        1: {
            'name': 'Softmax',
            'description': 'Softmax implementation with dynamic batch size',
            'function': test_softmax,
            'requires_npu': True
        }
    }
    
    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            npu_req = " (Requires NPU)" if ex_info['requires_npu'] else " (No NPU required)"
            print(f"  {ex_id}. {ex_info['name']}{npu_req}")
            print(f"     {ex_info['description']}\n")
        return
    
    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
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
    requires_npu = any(ex_info['requires_npu'] for _, ex_info in examples_to_run)
    
    if requires_npu:
        device_id = get_device_id()
        if device_id is None:
            return
        # Set the device once for all examples
        torch.npu.set_device(device_id)
    
    try:
        for ex_id, ex_info in examples_to_run:
            if ex_info['requires_npu'] and device_id is None:
                print(f"Skipping example {ex_id} ({ex_info['name']}): NPU device not configured")
                continue
            
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function']()
        
        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All softmax tests passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

