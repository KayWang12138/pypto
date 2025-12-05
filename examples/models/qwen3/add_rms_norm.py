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
Add RMSNorm Example for Qwen3 Model

This example demonstrates how to implement Add + RMSNorm operation using PyPTO,
which is a common pattern in transformer architectures. It shows:
- Residual connection (add residual to hidden states)
- RMS normalization
- Dynamic batch size support
- Tiling for efficient execution

This operation combines a residual connection with RMS normalization, which is
more efficient than separate operations and commonly used in modern LLMs.
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


def add_rms_norm_golden(residual: torch.Tensor, hidden_states: torch.Tensor, 
                        gamma: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of Add + RMSNorm.
    
    Parameters
    ----------
    residual : torch.Tensor
        Residual tensor to add
    hidden_states : torch.Tensor
        Hidden states tensor
    gamma : torch.Tensor
        RMSNorm weight parameter
    eps : float
        Epsilon value for numerical stability
        
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (normalized output, residual output)
    """
    # Add residual connection
    x = residual + hidden_states
    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]
    
    # Convert to FP32 for computation
    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff
    
    # RMS normalization
    reduce_sum = mean_res.sum(dim=-1, keepdim=True) + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt
    res = res_div * gamma
    
    # Convert back to original dtype
    if x_dtype != torch.float32:
        res = res.to(x_dtype)
        x_out = x_f32.to(x_dtype)
    else:
        x_out = x_f32
    
    return res, x_out


@pypto.jit
def add_rms_norm(inputs: list, outputs: list, eps: float):
    """
    PyPTO implementation of Add + RMSNorm with dynamic batch size support.
    
    This function processes input tensors in tiles, applying add + RMSNorm
    to each tile independently. The batch dimension is marked as dynamic,
    allowing variable batch sizes at runtime.
    
    Parameters
    ----------
    inputs : list
        List containing [residual, hidden_states, weight]
    outputs : list
        List containing [output_hidden_states, output_residual]
    eps : float
        Epsilon value for numerical stability
    """
    # Enable dynamic unaligned support for code generation
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    pypto.set_host_options(only_codegen=True)
    
    # Extract input and output tensors
    residual = inputs[0]
    hidden_states = inputs[1]
    weight = inputs[2]
    
    output_hidden_states = outputs[0]
    output_residual = outputs[1]
    
    # Get tensor shapes
    calc_dtype = pypto.DT_FP32
    input_dtype = hidden_states.dtype
    m = hidden_states.shape[0]  # Dynamic batch size
    n = hidden_states.shape[1]  # Static hidden size
    
    # Define tiling configuration
    view_shape = (16, n)
    tile_shape = [16, 1024]
    bs_loop = (m + view_shape[0] - 1) // view_shape[0]
    
    # Define the computation graph
    def rms_inside_func():
        """Inner function to encapsulate kernel logic for automatic variable cleanup."""
        # Loop over dynamic batch axis
        for idx_loop in pypto.loop(bs_loop, name="LOOP_RMS_NORM_L0", idx_name="idx_loop"):
            def bs_loop_func(idx_loop):
                """Process one batch tile."""
                # Create views for current batch tile
                tile_residual = pypto.view(
                    residual, 
                    view_shape,
                    [idx_loop * view_shape[0], 0],
                    valid_shape=[(m - idx_loop * view_shape[0]).min(view_shape[0]), n]
                )
                tile_hidden_states = pypto.view(
                    hidden_states,
                    view_shape,
                    [idx_loop * view_shape[0], 0],
                    valid_shape=[(m - idx_loop * view_shape[0]).min(view_shape[0]), n]
                )
                
                # Configure tiling: use full UB but don't exceed UB size
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                
                mean_coff = 1.0 / tile_hidden_states.shape[-1]
                
                # Cast to computation dtype (FP32)
                tile_residual_fp32 = pypto.cast(tile_residual, calc_dtype)
                tile_hidden_states_fp32 = pypto.cast(tile_hidden_states, calc_dtype)
                
                # Reshape weight to match tensor dimensions
                weight_shape = [1] * len(tile_hidden_states_fp32.shape)
                weight_shape[-1] = weight.shape[0]
                weight_2d = pypto.reshape(weight, weight_shape)
                tile_weight_fp32 = pypto.cast(weight_2d, calc_dtype)
                
                # Add residual connection: x = residual + hidden_states
                x_f32 = pypto.add(tile_residual_fp32, tile_hidden_states_fp32)
                
                # Compute square: square = x^2
                square = pypto.mul(x_f32, x_f32)
                
                # Compute mean: mean_res = square * mean_coff
                mean_res = pypto.mul(square, mean_coff)
                
                # Reduce sum: reduce_asum = sum(mean_res, dim=-1, keepdim=True)
                reduce_asum = pypto.sum(mean_res, dim=-1, keepdim=True)
                
                # Add epsilon: reduce_sum = reduce_asum + eps
                reduce_sum = pypto.add(reduce_asum, eps)
                
                # Square root: reduce_sqrt = sqrt(reduce_sum)
                reduce_sqrt = pypto.sqrt(reduce_sum)
                
                res_div = pypto.div(x_f32, reduce_sqrt)
                
                res = pypto.mul(res_div, tile_weight_fp32)
                
                # Cast output back to input dtype
                y_output = pypto.cast(res, input_dtype)
                output_hidden_states[
                    idx_loop * pypto.symbolic_scalar(view_shape[0]):,
                    pypto.symbolic_scalar(0):
                ] = y_output
                
                x_output = pypto.cast(x_f32, input_dtype)
                output_residual[
                    idx_loop * pypto.symbolic_scalar(view_shape[0]):,
                    pypto.symbolic_scalar(0):
                ] = x_output
            
            bs_loop_func(idx_loop)
    
    rms_inside_func()
    
    # Type assertions for verification
    assert isinstance(output_hidden_states, pypto.tensor)
    assert isinstance(output_residual, pypto.tensor)


def test_add_rms_norm():
    """
    Test Add + RMSNorm implementation against PyTorch reference.
    
    Tests with different batch sizes to verify dynamic shape support.
    """
    print("=" * 60)
    print("Test: Add + RMSNorm (Dynamic Batch)")
    print("=" * 60)
    
    hidden_size = 2048
    eps = 1e-6
    
    device_id = torch.npu.current_device()
    
    # Test with different batch sizes
    for i in range(4):
        if i == 2:
            batch_size = 2  # Test with small batch size
        else:
            batch_size = 5
        
        print(f"\nTesting with batch size: {batch_size}")
        
        # Prepare test data
        residual_tensor = torch.rand(
            (batch_size, hidden_size), 
            dtype=torch.float16, 
            device=f'npu:{device_id}'
        )
        hidden_states_tensor = torch.rand(
            (batch_size, hidden_size), 
            dtype=torch.float16, 
            device=f'npu:{device_id}'
        )
        weight_tensor = torch.rand(
            hidden_size, 
            dtype=torch.float16, 
            device=f'npu:{device_id}'
        )
        
        output_hidden_states = torch.full(
            (batch_size, hidden_size), 
            9, 
            dtype=torch.float16, 
            device=f'npu:{device_id}'
        )
        output_residual = torch.full(
            (batch_size, hidden_size), 
            7, 
            dtype=torch.float16, 
            device=f'npu:{device_id}'
        )

        # Initialize PyPTO inputs and outputs, mark batch dimension (axis 0) as dynamic
        inputs = {
            residual_tensor: [0],
            hidden_states_tensor: [0],
            weight_tensor: []
        }
        outputs = {
            output_hidden_states: [0],
            output_residual: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

        # Execute PyPTO kernel
        add_rms_norm(pto_inputs, pto_outputs, eps)
        pypto.runtime._device_synchronize()
        
        # Compute reference using PyTorch
        golden_res, golden_x = add_rms_norm_golden(
            residual_tensor.cpu(),
            hidden_states_tensor.cpu(),
            weight_tensor.cpu(),
            eps
        )
        
        # Verify results
        assert_allclose(
            np.array(output_residual.cpu().flatten().tolist()),
            np.array(golden_x.flatten().tolist()),
            rtol=0.001,
            atol=0.001
        )
        assert_allclose(
            np.array(output_hidden_states.cpu().flatten().tolist()),
            np.array(golden_res.flatten().tolist()),
            rtol=0.001,
            atol=0.001
        )
        
        print(f"  ✓ Batch size {batch_size} passed")
    
    print("\n✓ All Add + RMSNorm tests passed")


def main():
    """Run Add + RMSNorm example.
    
    Usage:
        python add_rms_norm.py          # Run example
        python add_rms_norm.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Add + RMSNorm Example (Qwen3)",
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
            'name': 'Add + RMSNorm',
            'description': 'Add residual connection with RMS normalization',
            'function': test_add_rms_norm,
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
    print("PyPTO Add + RMSNorm Example (Qwen3)")
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
            print("\n" + "=" * 60)
            print("All tests completed successfully!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
