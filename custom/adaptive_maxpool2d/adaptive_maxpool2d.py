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
AdaptiveMaxPool2D Operator for PyPTO

This example demonstrates how to implement AdaptiveMaxPool2D using PyPTO APIs.
"""

import os
import sys
import argparse
import pypto
import torch
import torch.nn.functional as F
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


def adaptive_maxpool2d_golden(
    input_tensor: torch.Tensor,
    output_size: tuple,
) -> torch.Tensor:
    """
    PyTorch reference implementation of AdaptiveMaxPool2D.
    
    Args:
        input_tensor: Input tensor of shape [batch, channel, H, W]
        output_size: Output size as (output_H, output_W)
    
    Returns:
        Output tensor of shape [batch, channel, output_H, output_W]
    """
    return F.adaptive_max_pool2d(input_tensor, output_size)


def create_adaptive_maxpool2d_kernel(
    input_shape: tuple,
    output_size: tuple,
    dtype: pypto.DataType = pypto.DT_FP16,
    run_mode: str = "npu"
):
    """
    Create AdaptiveMaxPool2D kernel using PyPTO.
    
    Args:
        input_shape: Input tensor shape as (batch, channel, H, W)
        output_size: Output size as (output_H, output_W)
        dtype: Data type (DT_FP16, DT_BF16, DT_FP32)
        run_mode: Run mode ("npu" or "sim")
    
    Returns:
        JIT kernel function
    """
    batch, channel, H, W = input_shape
    output_H, output_W = output_size
    
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def adaptive_maxpool2d_kernel(
        input_tensor: pypto.Tensor(input_shape, dtype),
    ) -> pypto.Tensor((batch, channel, output_H, output_W), dtype):
        # Set tiling configuration for vector operations
        pypto.set_vec_tile_shapes(1, 1, 8, 8)
        
        # Create output tensor
        output_tensor = pypto.tensor((batch, channel, output_H, output_W), dtype)
        
        # Iterate over batch dimension
        for b_idx in pypto.loop(batch, name="batch_loop", idx_name="b_idx"):
            # Iterate over channel dimension
            for c_idx in pypto.loop(channel, name="channel_loop", idx_name="c_idx"):
                # Iterate over output height dimension
                for h_out_idx in pypto.loop(output_H, name="h_out_loop", idx_name="h_out_idx"):
                    # Iterate over output width dimension
                    for w_out_idx in pypto.loop(output_W, name="w_out_loop", idx_name="w_out_idx"):
                        # Calculate input window boundaries
                        h_start = (h_out_idx * H) // output_H
                        h_end = ((h_out_idx + 1) * H) // output_H
                        w_start = (w_out_idx * W) // output_W
                        w_end = ((w_out_idx + 1) * W) // output_W
                        
                        # Calculate window size
                        window_h = h_end - h_start
                        window_w = w_end - w_start
                        
                        # Extract the pooling window using view
                        # First extract from [batch, channel, H, W] to get [1, 1, window_h, window_w]
                        window = pypto.view(
                            input_tensor,
                            [1, 1, window_h, window_w],
                            [b_idx, c_idx, h_start, w_start]
                        )
                        
                        # Apply max pooling: reduce over H and W dimensions
                        # First reduce over W dimension (dim=-1)
                        max_w = pypto.amax(window, dim=-1, keepdim=True)
                        # Then reduce over H dimension (dim=-2)
                        max_val = pypto.amax(max_w, dim=-2, keepdim=True)
                        
                        # Assemble the result to output tensor
                        pypto.assemble(max_val, [b_idx, c_idx, h_out_idx, w_out_idx], output_tensor)
        
        return output_tensor
    
    return adaptive_maxpool2d_kernel


def test_basic_case(device_id=None, run_mode="npu"):
    """Test case 1: Basic case with small input (4x4 -> 2x2)"""
    print("=" * 60)
    print("Test Case 1: Basic Case (4x4 -> 2x2)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test parameters
    batch, channel = 1, 1
    H, W = 4, 4
    output_H, output_W = 2, 2
    dtype_torch = torch.float16
    dtype_pypto = pypto.DT_FP16
    
    # Create input tensor
    input_torch = torch.randn(batch, channel, H, W, dtype=dtype_torch, device=device)
    
    # PyTorch golden result
    golden = adaptive_maxpool2d_golden(input_torch, (output_H, output_W))
    
    # PyPTO implementation
    kernel = create_adaptive_maxpool2d_kernel(
        (batch, channel, H, W),
        (output_H, output_W),
        dtype_pypto,
        run_mode
    )
    output = kernel(input_torch)
    
    # Verify results
    print(f"Input shape: {input_torch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Golden shape: {golden.shape}")
    
    if run_mode == "npu":
        max_diff = (output - golden).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        
        assert_allclose(
            output.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3,
            atol=1e-3
        )
        print("✓ Test case 1 passed!")
    print()


def test_medium_case(device_id=None, run_mode="npu"):
    """Test case 2: Medium case (8x8 -> 3x3)"""
    print("=" * 60)
    print("Test Case 2: Medium Case (8x8 -> 3x3)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test parameters
    batch, channel = 2, 3
    H, W = 8, 8
    output_H, output_W = 3, 3
    dtype_torch = torch.float16
    dtype_pypto = pypto.DT_FP16
    
    # Create input tensor
    input_torch = torch.randn(batch, channel, H, W, dtype=dtype_torch, device=device)
    
    # PyTorch golden result
    golden = adaptive_maxpool2d_golden(input_torch, (output_H, output_W))
    
    # PyPTO implementation
    kernel = create_adaptive_maxpool2d_kernel(
        (batch, channel, H, W),
        (output_H, output_W),
        dtype_pypto,
        run_mode
    )
    output = kernel(input_torch)
    
    # Verify results
    print(f"Input shape: {input_torch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Golden shape: {golden.shape}")
    
    if run_mode == "npu":
        max_diff = (output - golden).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        
        assert_allclose(
            output.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3,
            atol=1e-3
        )
        print("✓ Test case 2 passed!")
    print()


def test_non_divisible_case(device_id=None, run_mode="npu"):
    """Test case 3: Non-divisible case (7x7 -> 3x3)"""
    print("=" * 60)
    print("Test Case 3: Non-divisible Case (7x7 -> 3x3)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test parameters
    batch, channel = 2, 4
    H, W = 7, 7
    output_H, output_W = 3, 3
    dtype_torch = torch.float16
    dtype_pypto = pypto.DT_FP16
    
    # Create input tensor
    input_torch = torch.randn(batch, channel, H, W, dtype=dtype_torch, device=device)
    
    # PyTorch golden result
    golden = adaptive_maxpool2d_golden(input_torch, (output_H, output_W))
    
    # PyPTO implementation
    kernel = create_adaptive_maxpool2d_kernel(
        (batch, channel, H, W),
        (output_H, output_W),
        dtype_pypto,
        run_mode
    )
    output = kernel(input_torch)
    
    # Verify results
    print(f"Input shape: {input_torch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Golden shape: {golden.shape}")
    
    if run_mode == "npu":
        max_diff = (output - golden).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        
        assert_allclose(
            output.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3,
            atol=1e-3
        )
        print("✓ Test case 3 passed!")
    print()


def test_bf16_case(device_id=None, run_mode="npu"):
    """Test case 4: BF16 data type (6x6 -> 2x2)"""
    print("=" * 60)
    print("Test Case 4: BF16 Data Type (6x6 -> 2x2)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test parameters
    batch, channel = 1, 2
    H, W = 6, 6
    output_H, output_W = 2, 2
    dtype_torch = torch.bfloat16
    dtype_pypto = pypto.DT_BF16
    
    # Create input tensor
    input_torch = torch.randn(batch, channel, H, W, dtype=dtype_torch, device=device)
    
    # PyTorch golden result
    golden = adaptive_maxpool2d_golden(input_torch, (output_H, output_W))
    
    # PyPTO implementation
    kernel = create_adaptive_maxpool2d_kernel(
        (batch, channel, H, W),
        (output_H, output_W),
        dtype_pypto,
        run_mode
    )
    output = kernel(input_torch)
    
    # Verify results
    print(f"Input shape: {input_torch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Golden shape: {golden.shape}")
    
    if run_mode == "npu":
        max_diff = (output - golden).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        
        assert_allclose(
            output.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-2,
            atol=1e-2
        )
        print("✓ Test case 4 passed!")
    print()


def main():
    """Run AdaptiveMaxPool2D tests.
    
    Usage:
        python adaptive_maxpool2d.py          # Run all tests
        python adaptive_maxpool2d.py --list   # List all available tests
        python adaptive_maxpool2d.py 1        # Run test 1 only
    """
    parser = argparse.ArgumentParser(
        description="PyPTO AdaptiveMaxPool2D Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all tests
  %(prog)s 1            Run test 1 only
  %(prog)s --list       List all available tests
        """
    )
    parser.add_argument(
        'test_id',
        type=str,
        nargs='?',
        help='Test ID to run (1-4). If not specified, all tests will run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available tests and exit'
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
    
    # Define available tests
    tests = {
        '1': {
            'name': 'Basic Case',
            'description': '4x4 -> 2x2',
            'function': test_basic_case
        },
        '2': {
            'name': 'Medium Case',
            'description': '8x8 -> 3x3',
            'function': test_medium_case
        },
        '3': {
            'name': 'Non-divisible Case',
            'description': '7x7 -> 3x3',
            'function': test_non_divisible_case
        },
        '4': {
            'name': 'BF16 Data Type',
            'description': '6x6 -> 2x2 with BF16',
            'function': test_bf16_case
        }
    }
    
    # List tests if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Tests")
        print("=" * 60 + "\n")
        for test_id, test_info in sorted(tests.items()):
            print(f"  ID: {test_id}")
            print(f"     name: {test_info['name']}")
            print(f"     description: {test_info['description']}\n")
        return
    
    # Validate test ID if provided
    if args.test_id is not None:
        if args.test_id not in tests:
            print(f"ERROR: Invalid test ID: {args.test_id}")
            print(f"Valid test IDs are: {', '.join(map(str, sorted(tests.keys())))}")
            print("\nUse --list to see all available tests.")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("PyPTO AdaptiveMaxPool2D Example")
    print("=" * 60 + "\n")
    
    # Get and validate device ID (needed for NPU tests)
    device_id = None
    tests_to_run = []
    
    if args.test_id is not None:
        # Run single test
        test = tests.get(args.test_id)
        if test is None:
            raise ValueError(f"Invalid test ID: {args.test_id}")
        tests_to_run = [(args.test_id, test)]
    else:
        # Run all tests
        tests_to_run = list(tests.items())
    
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running tests that require NPU hardware...")
        print("(Make sure CANN environment is configured and NPU is available)\n")
    
    try:
        for test_id, test_info in tests_to_run:
            print(f"Running Test {test_id}: {test_info['name']}")
            test_info['function'](device_id, args.run_mode)
        
        if len(tests_to_run) > 1:
            print("=" * 60)
            print("All AdaptiveMaxPool2D tests passed!")
            print("=" * 60)
    
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
