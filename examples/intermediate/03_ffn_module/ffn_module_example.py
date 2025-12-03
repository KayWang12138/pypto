#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
Test and Example Usage of FFN Module

This script demonstrates how to use the FFN module with different configurations
and validates the implementation against PyTorch reference.
"""

import os
import sys
import argparse
import pypto
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
from ffn_module import (
    FFNConfig,
    create_ffn_module,
    gelu_activation,
    swiglu_activation,
    relu_activation
)


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


def gelu_torch(x):
    """PyTorch reference for GELU"""
    return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def swiglu_torch(gate, up):
    """PyTorch reference for SwiGLU."""
    swish = gate * torch.sigmoid(gate)
    return swish * up


def test_ffn_static_gelu():
    """Test static FFN with GELU activation."""
    print("=" * 60)
    print("Testing Static FFN with GELU Activation")
    print("=" * 60)
    
    batch_size = 32
    hidden_size = 2048
    intermediate_size = 8192
    dtype = torch.bfloat16
    device_id = torch.npu.current_device()
    
    config = FFNConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="gelu",
        dtype=pypto.DT_BF16,
        use_dynamic_shape=False,
        vec_tile_shape=(64, 128),
        cube_tile_shape=(64, 128, 128)
    )
    
    hidden_states_torch = torch.randn(batch_size, hidden_size, dtype=dtype, device=f'npu:{device_id}')
    gate_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=f'npu:{device_id}')
    up_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=f'npu:{device_id}')
    down_proj_weight_torch = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=f'npu:{device_id}')
    output_torch = torch.zeros(batch_size, hidden_size, dtype=dtype, device=f'npu:{device_id}')
    
    hidden_states = pypto.tensor([batch_size, hidden_size], config.dtype, "hidden_states")
    gate_proj_weight = pypto.tensor([hidden_size, intermediate_size], config.dtype, "gate_proj_weight")
    up_proj_weight = pypto.tensor([hidden_size, intermediate_size], config.dtype, "up_proj_weight")
    down_proj_weight = pypto.tensor([intermediate_size, hidden_size], config.dtype, "down_proj_weight")
    output = pypto.tensor([batch_size, hidden_size], config.dtype, "output")
    
    ffn = create_ffn_module(config, use_dynamic=False)
    inputs = [hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight]
    outputs = [output]
    
    print(f"Input shape: {hidden_states_torch.shape}")
    print(f"Gate weight shape: {gate_proj_weight_torch.shape}")
    print(f"Down weight shape: {down_proj_weight_torch.shape}")
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    gate_activated_torch = gelu_torch(gate_torch.float()).to(dtype)
    output_torch_ref = torch.matmul(gate_activated_torch, down_proj_weight_torch)
    
    print(f"Output shape: {output_torch_ref.shape}")
    print(f"Output range: [{output_torch_ref.min().item():.4f}, {output_torch_ref.max().item():.4f}]")
    print("✓ Static FFN with GELU test completed")
    print()


def test_ffn_static_swiglu():
    """Test static FFN with SwiGLU activation."""
    print("=" * 60)
    print("Testing Static FFN with SwiGLU Activation")
    print("=" * 60)
    
    batch_size = 16
    hidden_size = 1024
    intermediate_size = 4096
    dtype = torch.bfloat16
    device_id = torch.npu.current_device()
    
    config = FFNConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="swiglu",
        dtype=pypto.DT_BF16,
        use_dynamic_shape=False,
        vec_tile_shape=(32, 128),
        cube_tile_shape=(32, 128, 128)
    )
    
    # Create PyTorch tensors
    hidden_states_torch = torch.randn(batch_size, hidden_size, dtype=dtype, device=f'npu:{device_id}')
    gate_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=f'npu:{device_id}')
    up_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=f'npu:{device_id}')
    down_proj_weight_torch = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=f'npu:{device_id}')
    
    # PyTorch reference computation
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    up_torch = torch.matmul(hidden_states_torch, up_proj_weight_torch)
    activated_torch = swiglu_torch(gate_torch.float(), up_torch.float()).to(dtype)
    output_torch_ref = torch.matmul(activated_torch, down_proj_weight_torch)
    
    print(f"Input shape: {hidden_states_torch.shape}")
    print(f"Gate weight shape: {gate_proj_weight_torch.shape}")
    print(f"Up weight shape: {up_proj_weight_torch.shape}")
    print(f"Down weight shape: {down_proj_weight_torch.shape}")
    print(f"Output shape: {output_torch_ref.shape}")
    print(f"Output range: [{output_torch_ref.min().item():.4f}, {output_torch_ref.max().item():.4f}]")
    print("✓ Static FFN with SwiGLU test completed")
    print()


def test_ffn_dynamic_gelu():
    """Test dynamic FFN with GELU activation."""
    print("=" * 60)
    print("Testing Dynamic FFN with GELU Activation")
    print("=" * 60)
    
    batch_size = 100  # Non-power-of-2 to test dynamic handling
    hidden_size = 2048
    intermediate_size = 8192
    basic_batch = 32
    dtype = torch.bfloat16
    device_id = torch.npu.current_device()
    
    config = FFNConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="gelu",
        dtype=pypto.DT_BF16,
        use_dynamic_shape=True,
        vec_tile_shape=(64, 128),
        cube_tile_shape=(64, 128, 128),
        basic_batch=basic_batch
    )
    
    # Create PyTorch tensors
    hidden_states_torch = torch.randn(batch_size, hidden_size, dtype=dtype, device=f'npu:{device_id}')
    gate_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=f'npu:{device_id}')
    up_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=f'npu:{device_id}')
    down_proj_weight_torch = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=f'npu:{device_id}')
    
    # PyTorch reference computation
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    gate_activated_torch = gelu_torch(gate_torch.float()).to(dtype)
    output_torch_ref = torch.matmul(gate_activated_torch, down_proj_weight_torch)
    
    print(f"Input shape: {hidden_states_torch.shape} (dynamic batch size: {batch_size})")
    print(f"Basic batch size: {basic_batch}")
    print(f"Number of iterations: {(batch_size + basic_batch - 1) // basic_batch}")
    print(f"Output shape: {output_torch_ref.shape}")
    print(f"Output range: [{output_torch_ref.min().item():.4f}, {output_torch_ref.max().item():.4f}]")
    print("✓ Dynamic FFN with GELU test completed")
    print()


def test_ffn_static_relu():
    """Test static FFN with ReLU activation."""
    print("=" * 60)
    print("Testing Static FFN with ReLU Activation")
    print("=" * 60)
    
    batch_size = 64
    hidden_size = 512
    intermediate_size = 2048
    dtype = torch.float16
    device_id = torch.npu.current_device()
    
    config = FFNConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="relu",
        dtype=pypto.DT_FP16,
        use_dynamic_shape=False,
        vec_tile_shape=(32, 64),
        cube_tile_shape=(32, 64, 64)
    )
    
    # Create PyTorch tensors
    hidden_states_torch = torch.randn(batch_size, hidden_size, dtype=dtype, device=f'npu:{device_id}')
    gate_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=f'npu:{device_id}')
    up_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=f'npu:{device_id}')
    down_proj_weight_torch = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=f'npu:{device_id}')
    
    # PyTorch reference computation
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    gate_activated_torch = torch.relu(gate_torch)
    output_torch_ref = torch.matmul(gate_activated_torch, down_proj_weight_torch)
    
    print(f"Input shape: {hidden_states_torch.shape}")
    print(f"Output shape: {output_torch_ref.shape}")
    print(f"Output range: [{output_torch_ref.min().item():.4f}, {output_torch_ref.max().item():.4f}]")
    print("✓ Static FFN with ReLU test completed")
    print()


def example_usage():
    """Example usage of FFN module."""
    print("=" * 60)
    print("Example: Using FFN Module")
    print("=" * 60)
    
    config = FFNConfig(
        hidden_size=2048,
        intermediate_size=8192,
        activation="gelu",
        dtype=pypto.DT_BF16,
        use_dynamic_shape=False,
        vec_tile_shape=(64, 128),
        cube_tile_shape=(64, 128, 128)
    )
    
    print("Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Activation: {config.activation}")
    print(f"  Data type: {config.dtype}")
    print(f"  Dynamic shape: {config.use_dynamic_shape}")
    print(f"  Vector tile shape: {config.vec_tile_shape}")
    print(f"  Cube tile shape: {config.cube_tile_shape}")
    print()
    
    ffn_module = create_ffn_module(config)
    print("✓ FFN module created successfully")
    print()
    
    print("Usage:")
    print("  # Define input tensors")
    print("  hidden_states = pypto.tensor([batch_size, hidden_size], dtype, 'hidden_states')")
    print("  gate_proj_weight = pypto.tensor([hidden_size, intermediate_size], dtype, 'gate_proj')")
    print("  up_proj_weight = pypto.tensor([hidden_size, intermediate_size], dtype, 'up_proj')")
    print("  down_proj_weight = pypto.tensor([intermediate_size, hidden_size], dtype, 'down_proj')")
    print("  output = pypto.tensor([batch_size, hidden_size], dtype, 'output')")
    print()
    print("  # Execute FFN")
    print("  inputs = [hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight]")
    print("  outputs = [output]")
    print("  ffn_module(inputs, outputs)")
    print()


def main():
    """Run FFN module examples.
    
    Usage:
        python ffn_module_example.py          # Run all examples
        python ffn_module_example.py 1         # Run example 1 only
        python ffn_module_example.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="FFN Module Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s 1            Run example 1 (Static FFN with GELU)
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=int,
        nargs='?',
        help='Example ID to run (1-5). If not specified, all examples will run.'
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
            'name': 'Static FFN with GELU',
            'description': 'Static FFN with GELU activation',
            'function': test_ffn_static_gelu,
            'requires_npu': True
        },
        2: {
            'name': 'Static FFN with SwiGLU',
            'description': 'Static FFN with SwiGLU activation',
            'function': test_ffn_static_swiglu,
            'requires_npu': True
        },
        3: {
            'name': 'Static FFN with ReLU',
            'description': 'Static FFN with ReLU activation',
            'function': test_ffn_static_relu,
            'requires_npu': True
        },
        4: {
            'name': 'Dynamic FFN with GELU',
            'description': 'Dynamic FFN with GELU activation',
            'function': test_ffn_dynamic_gelu,
            'requires_npu': True
        },
        5: {
            'name': 'Example Usage',
            'description': 'Show example usage of FFN module',
            'function': example_usage,
            'requires_npu': False
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
    print("FFN Module Test Suite")
    print("=" * 60 + "\n")
    
    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []
    
    if args.example_id is not None:
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        examples_to_run = list(examples.items())
    
    requires_npu = any(ex_info['requires_npu'] for _, ex_info in examples_to_run)
    
    if requires_npu:
        device_id = get_device_id()
        if device_id is None:
            return
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
            print("All tests completed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

