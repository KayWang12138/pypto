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
Custom Activation Functions Example for PyPTO

This example demonstrates how to implement custom activation functions by composing
PyPTO operations. It shows:
- SiLU (Swish) activation: x * sigmoid(x)
- GELU activation: x * sigmoid(1.702 * x) approximation
- SwiGLU activation: Swish(gate) * up
- GeGLU activation: GELU(gate) * up
- Custom activation composition patterns

These activations are commonly used in modern transformer architectures.
"""

import os
import sys
import argparse
import pypto
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
from dataclasses import dataclass
from typing import Literal


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


# Constants for element creation
F_1 = 1.0
F_NEGA_1 = -1.0


def silu_activation(x: pypto.tensor) -> pypto.tensor:
    """
    SiLU (Swish) activation function: x * sigmoid(x)
    
    SiLU is a smooth, non-monotonic activation function that has been shown
    to work well in deep networks.
    
    Formula: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    Parameters
    ----------
    x : pypto.tensor
        Input tensor
        
    Returns
    -------
    pypto.tensor
        SiLU activated tensor
    """
    # Configure tiling based on input shape
    if len(x.shape) >= 2:
        n_tile = 32
        tile_list = [n_tile for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_list)
    else:
        pypto.set_vec_tile_shapes(32, 128)
    
    # Compute sigmoid(x) = 1 / (1 + exp(-x))    
    # SiLU(x) = x * sigmoid(x)
    return pypto.mul(x, pypto.sigmoid(x))


def gelu_activation(x: pypto.tensor) -> pypto.tensor:
    """
    GELU (Gaussian Error Linear Unit) activation function.
    
    Uses approximation: x * sigmoid(1.702 * x)
    This is a fast approximation of the full GELU formula.
    
    Parameters
    ----------
    x : pypto.tensor
        Input tensor
        
    Returns
    -------
    pypto.tensor
        GELU activated tensor
    """
    # Configure tiling
    if len(x.shape) >= 2:
        n_tile = 32
        tile_list = [n_tile for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_list)
    else:
        pypto.set_vec_tile_shapes(32, 128)
    
    # GELU approximation: x * sigmoid(1.702 * x)
    coeff = float(1.702)
    x_scaled = pypto.mul(x, coeff)
    
    # GELU(x) = x * sigmoid(1.702 * x)
    return pypto.mul(x, pypto.sigmoid(x_scaled))


def swiglu_activation(gate: pypto.tensor, up: pypto.tensor) -> pypto.tensor:
    """
    SwiGLU activation function: Swish(gate) * up
    
    SwiGLU is a gated linear unit that uses Swish (SiLU) as the gating function.
    It's commonly used in modern LLMs like PaLM and LLaMA.
    
    Formula: SwiGLU(gate, up) = Swish(gate) * up = (gate * sigmoid(gate)) * up
    
    Parameters
    ----------
    gate : pypto.tensor
        Gate tensor
    up : pypto.tensor
        Up projection tensor
        
    Returns
    -------
    pypto.tensor
        SwiGLU activated tensor
    """
    # Configure tiling
    if len(gate.shape) >= 2:
        pypto.set_vec_tile_shapes(gate.shape[0], gate.shape[1])
    else:
        n_tile = 32
        tile_list = [n_tile for _ in range(len(gate.shape))]
        pypto.set_vec_tile_shapes(*tile_list)
    
    # Swish(gate) = gate * sigmoid(gate)
    sigmoid = pypto.sigmoid(gate)
    swish = pypto.mul(gate, sigmoid)
    
    # Multiply with up projection
    return pypto.mul(swish, up)


def geglu_activation(gate: pypto.tensor, up: pypto.tensor) -> pypto.tensor:
    """
    GeGLU activation function: GELU(gate) * up
    
    GeGLU is a gated linear unit that uses GELU as the gating function.
    It's an alternative to SwiGLU.
    
    Formula: GeGLU(gate, up) = GELU(gate) * up
    
    Parameters
    ----------
    gate : pypto.tensor
        Gate tensor
    up : pypto.tensor
        Up projection tensor
        
    Returns
    -------
    pypto.tensor
        GeGLU activated tensor
    """
    gelu_gate = gelu_activation(gate)
    
    # Multiply with up projection
    return pypto.mul(gelu_gate, up)


# Reference implementations for verification
def silu_golden(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of SiLU."""
    return x * torch.sigmoid(x)


def gelu_golden(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of GELU."""
    return torch.nn.functional.gelu(x)


def swiglu_golden(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of SwiGLU."""
    return (gate * torch.sigmoid(gate)) * up


def geglu_golden(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of GeGLU."""
    return torch.nn.functional.gelu(gate) * up


@pypto.jit
def apply_silu_activation(inputs, outputs):
    """Apply activation function."""
    x = inputs[0]
    out = outputs[0]
    n_tile = 32
    tile_list = [n_tile for _ in range(len(x.shape))]
    pypto.set_vec_tile_shapes(*tile_list)
    
    for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
        out[:] = silu_activation(x)


@pypto.jit
def apply_gelu_activation(inputs, outputs):
    """Apply activation function."""
    x = inputs[0]
    out = outputs[0]
    n_tile = 32
    tile_list = [n_tile for _ in range(len(x.shape))]
    pypto.set_vec_tile_shapes(*tile_list)
    
    for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
        out[:] = gelu_activation(x)


@pypto.jit
def apply_swiglu_activation(inputs, outputs):
    """Apply gated activation function SwiGLU."""
    gate = inputs[0]
    up = inputs[1]
    out = outputs[0]
    
    n_tile = 32
    tile_list = [n_tile for _ in range(len(gate.shape))]
    pypto.set_vec_tile_shapes(*tile_list)
    
    for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
        out[:] = swiglu_activation(gate, up)


@pypto.jit
def apply_geglu_activation(inputs, outputs):
    """Apply gated activation function GeGLU."""
    gate = inputs[0]
    up = inputs[1]
    out = outputs[0]
    
    n_tile = 32
    tile_list = [n_tile for _ in range(len(gate.shape))]
    pypto.set_vec_tile_shapes(*tile_list)
    
    for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
        out[:] = geglu_activation(gate, up)
            

def test_silu():
    """Test SiLU activation."""
    print("=" * 60)
    print("Test: SiLU Activation")
    print("=" * 60)
    
    # Get current device ID (set in main)
    device_id = torch.npu.current_device()
    
    shape = (32, 128)
    x_torch = torch.randn(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    out_torch = torch.zeros(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    
    # Execute
    inputs = [x_torch]
    outputs = [out_torch]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    apply_silu_activation(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    # Verify
    expected = silu_golden(x_torch)
    max_diff = (out_torch - expected).abs().max().item()
    
    print(f"Input shape: {x_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    print(f"Max difference: {max_diff:.6f}")
    assert max_diff < 1e-1, "Result mismatch!"
    print("✓ SiLU passed")
    print()


def test_gelu():
    """Test GELU activation."""
    print("=" * 60)
    print("Test: GELU Activation")
    print("=" * 60)
    
    # Get current device ID (set in main)
    device_id = torch.npu.current_device()
    
    shape = (32, 128)
    x_torch = torch.randn(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    out_torch = torch.zeros(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    
    # Execute
    inputs = [x_torch]
    outputs = [out_torch]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    apply_gelu_activation(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()
    
    # Verify
    expected = gelu_golden(x_torch)
    max_diff = (out_torch - expected).abs().max().item()
    
    print(f"Input shape: {x_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    print(f"Max difference: {max_diff:.6f}")
    assert max_diff < 1e-1, "Result mismatch!"
    print("✓ GELU passed")
    print()


def test_swiglu():
    """Test SwiGLU activation."""
    print("=" * 60)
    print("Test: SwiGLU Activation")
    print("=" * 60)
    
    # Get current device ID (set in main)
    device_id = torch.npu.current_device()
    
    shape = (32, 128)
    gate_torch = torch.randn(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    up_torch = torch.randn(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    out_torch = torch.zeros(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    
    # Execute
    inputs = [gate_torch, up_torch]
    outputs = [out_torch]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    apply_swiglu_activation(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    # Verify
    expected = swiglu_golden(gate_torch, up_torch)
    max_diff = (out_torch - expected).abs().max().item()
    
    print(f"Gate shape: {gate_torch.shape}")
    print(f"Up shape: {up_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    print(f"Max difference: {max_diff:.6f}")
    assert max_diff < 1e-1, "Result mismatch!"
    print("✓ SwiGLU passed")
    print()


def test_geglu():
    """Test GeGLU activation."""
    print("=" * 60)
    print("Test: GeGLU Activation")
    print("=" * 60)
    
    # Get current device ID (set in main)
    device_id = torch.npu.current_device()
    
    shape = (32, 128)
    gate_torch = torch.randn(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    up_torch = torch.randn(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    out_torch = torch.zeros(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    
    # Execute
    inputs = [gate_torch, up_torch]
    outputs = [out_torch]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    apply_geglu_activation(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    # Verify
    expected = geglu_golden(gate_torch, up_torch)
    max_diff = (out_torch - expected).abs().max().item()
    
    print(f"Gate shape: {gate_torch.shape}")
    print(f"Up shape: {up_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    print(f"Max difference: {max_diff:.6f}")
    assert max_diff < 1e-1, "Result mismatch!"
    print("✓ GeGLU passed")
    print()


def main():
    """Run custom activation examples.
    
    Usage:
        python custom_activation.py          # Run all examples
        python custom_activation.py 1         # Run example 1 only
        python custom_activation.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Custom Activation Functions Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s 1            Run example 1 (GELU)
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=int,
        nargs='?',
        help='Example ID to run (1-4). If not specified, all examples will run.'
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
            'name': 'GELU Activation',
            'description': 'Gaussian Error Linear Unit activation',
            'function': test_gelu,
            'requires_npu': True
        },
        2: {
            'name': 'SiLU Activation',
            'description': 'Sigmoid Linear Unit (Swish) activation',
            'function': test_silu,
            'requires_npu': True
        },
        3: {
            'name': 'SwiGLU Activation',
            'description': 'Swish-Gated Linear Unit activation',
            'function': test_swiglu,
            'requires_npu': True
        },
        4: {
            'name': 'GeGLU Activation',
            'description': 'GELU-Gated Linear Unit activation',
            'function': test_geglu,
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
    print("PyPTO Custom Activation Functions Examples")
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
            print("All custom activation tests passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()