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
Multi-Function Module Example for PyPTO

This example demonstrates how to use multiple `@pypto.jit` functions together
to build complex computation pipelines. It shows:
- Multiple JIT-compiled functions
- Data flow between functions
- Function composition patterns
- Reusing compiled functions
- Switching between different functions at runtime

This pattern is useful for building modular neural network components.
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
from typing import Optional


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


@dataclass
class ModuleConfig:
    """Configuration for multi-function module."""
    hidden_size: int = 128
    intermediate_size: int = 256
    dtype: pypto.DataType = pypto.DT_BF16
    use_dynamic_shape: bool = False


# Function 1: Layer Normalization
@pypto.jit
def layer_norm(x, gamma, beta, out, eps: float = 1e-6):
    """Layer normalization function."""
    hidden_size = x.shape[-1]

    pypto.set_vec_tile_shapes(64, 128)

    for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
        # Compute mean
        mean = pypto.sum(x, dim=-1, keepdim=True)
        mean = pypto.div(mean, float(hidden_size))

        # Center
        centered = pypto.sub(x, mean)

        # Compute variance
        squared = pypto.mul(centered, centered)
        var = pypto.sum(squared, dim=-1, keepdim=True)
        var = pypto.div(var, float(hidden_size))

        # Normalize
        var_eps = pypto.add(var, eps)
        std = pypto.sqrt(var_eps)
        normalized = pypto.div(centered, std)

        # Scale and shift
        scaled = pypto.mul(normalized, gamma)
        out[:] = pypto.add(scaled, beta)


# Function 2: Linear Projection
@pypto.jit
def linear_projection(x, weight, out):
    """Linear projection: y = x @ W + b"""
    bias = None

    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
        # Matrix multiplication
        if bias is not None:
            out[:] = pypto.add(pypto.matmul(x, weight, out_dtype=x.dtype), bias)
        else:
            out[:] = pypto.matmul(x, weight, out_dtype=x.dtype)


# Function 3: GELU Activation
@pypto.jit
def gelu_activation(x, out, simplify_express=False):
    """GELU activation function."""
    pypto.set_vec_tile_shapes(64, 128)

    if simplify_express:
        for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
            # GELU approximation: x * sigmoid(1.702 * x)
            coeff = float(1.702)
            x_scaled = pypto.mul(x, coeff)
            out[:] = pypto.mul(x, pypto.sigmoid(x_scaled))
    else:
        for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
            # GELU approximation: x * sigmoid(1.702 * x)
            coeff = float(1.702)
            x_scaled = pypto.mul(x, coeff)

            dtype = x_scaled.dtype
            x_scaled = pypto.cast(x_scaled, pypto.DT_FP32)
            x_scaled_neg = pypto.mul(x_scaled, -1.0)
            exp_neg = pypto.exp(x_scaled_neg)
            one = 1.0
            exp_neg_plus_one = pypto.add(exp_neg, one)
            ones = pypto.full(exp_neg_plus_one.shape, 1.0, pypto.DT_FP32, valid_shape=exp_neg_plus_one.shape)
            sigmoid = pypto.div(ones, exp_neg_plus_one)
            if dtype != pypto.DT_FP32:
                sigmoid = pypto.cast(sigmoid, dtype)
            out[:] = pypto.mul(x, sigmoid)


# Function 4: Residual Connection
@pypto.jit
def residual_add(x, residual, out):
    """Add residual connection: out = x + residual"""
    pypto.set_vec_tile_shapes(64, 128)

    for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
        out[:] = pypto.add(x, residual)


# Function 5: Attention (simplified)
@pypto.jit
def attention(q, k, v, out, scale: float):
    """Simplified attention mechanism."""
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])

    # Q @ K^T
    k_t = pypto.transpose(k, [0, 1, 3, 2])
    scores = pypto.matmul(q, k_t, out_dtype=q.dtype)

    # Scale
    scores_scaled = pypto.mul(scores, scale)

    # Softmax
    attn_weights = pypto.softmax(scores_scaled, dim=-1)

    # Apply to values
    out[:] = pypto.matmul(attn_weights, v, out_dtype=q.dtype)


# Reference implementations for verification
def layer_norm_golden(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch reference for layer norm."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized * gamma + beta


def gelu_golden(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for GELU."""
    return torch.nn.functional.gelu(x)


def test_sequential_functions():
    """Test multiple functions in sequence."""
    print("=" * 60)
    print("Test: Sequential Functions")
    print("=" * 60)

    # Get current device ID (set in main)
    device_id = torch.npu.current_device()
    atol_val = 1e-1

    batch_size, hidden_size = 32, 128

    # Create tensors
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    gamma = torch.ones(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    beta = torch.zeros(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    # Intermediate tensors
    normed = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    activated = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    inputs = [x, gamma, beta]
    outputs = [normed]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    # Step 1: Layer normalization
    layer_norm(*pto_inputs, *pto_outputs, eps=1e-6)
    pypto.runtime._device_synchronize()

    # Step 2: GELU activation
    inputs = [normed]
    outputs = [activated]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    gelu_activation(*pto_inputs, *pto_outputs)
    pypto.runtime._device_synchronize()

    # Verify
    expected_normed = layer_norm_golden(x, gamma, beta, 1e-6)
    expected_activated = gelu_golden(expected_normed)

    max_diff_norm = (normed - expected_normed).abs().max().item()
    max_diff_act = (activated - expected_activated).abs().max().item()

    print(f"Input shape: {x.shape}")
    print(f"Normalized max diff: {max_diff_norm:.6f}")
    print(f"Activated max diff: {max_diff_act:.6f}")
    assert max_diff_norm < atol_val, "Layer norm mismatch!"
    assert max_diff_act < atol_val, "GELU mismatch!"
    print("✓ Sequential functions passed")
    print()


def test_residual_connection():
    """Test residual connection pattern."""
    print("=" * 60)
    print("Test: Residual Connection")
    print("=" * 60)

    # Get current device ID (set in main)
    device_id = torch.npu.current_device()

    batch_size, hidden_size = 32, 128

    # Create tensors
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    residual = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    out = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    inputs = [x, residual]
    outputs = [out]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    # Apply residual connection
    residual_add(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    # Verify
    expected = x + residual
    max_diff = (out - expected).abs().max().item()

    print(f"Input shape: {x.shape}")
    print(f"Residual shape: {residual.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Max difference: {max_diff:.6f}")
    assert max_diff < 1e-2, "Residual connection mismatch!"
    print("✓ Residual connection passed")
    print()


def test_transformer_block():
    """Test a complete transformer block using multiple functions."""
    print("=" * 60)
    print("Test: Transformer Block (Multi-Function)")
    print("=" * 60)

    # Get current device ID (set in main)
    device_id = torch.npu.current_device()

    batch_size, hidden_size, intermediate_size = 32, 128, 256

    # Create input
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    # Layer norm parameters
    gamma = torch.ones(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    beta = torch.zeros(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    # FFN weights
    gate_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    up_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    down_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    # Intermediate tensors
    normed = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    gate = torch.zeros(batch_size, intermediate_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    up = torch.zeros(batch_size, intermediate_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    activated = torch.zeros(batch_size, intermediate_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    ffn_out = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    output = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    inputs = [x, gamma, beta]
    outputs = [normed]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    # Transformer block computation:
    # 1. Layer normalization
    layer_norm(*pto_inputs, *pto_outputs, eps=1e-6)
    pypto.runtime._device_synchronize()

    # 2. FFN: Gate and Up projections
    inputs = [normed, gate_weight]
    outputs = [gate]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    linear_projection(*pto_inputs, *pto_outputs)
    pypto.runtime._device_synchronize()

    inputs = [normed, up_weight]
    outputs = [up]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    linear_projection(*pto_inputs, *pto_outputs)
    pypto.runtime._device_synchronize()

    # 3. GELU activation on gate
    inputs = [gate]
    outputs = [activated]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    gelu_activation(*pto_inputs, *pto_outputs)
    pypto.runtime._device_synchronize()

    # 4. Multiply with up (SwiGLU-like)
    activated = activated * up  # PyTorch operation for simplicity

    # 5. Down projection
    inputs = [activated, down_weight]
    outputs = [ffn_out]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    linear_projection(*pto_inputs, *pto_outputs)
    pypto.runtime._device_synchronize()

    # 6. Residual connection
    inputs = [x, ffn_out]
    outputs = [output]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    residual_add(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("✓ Transformer block (multi-function) completed")
    print()


def test_function_reuse():
    """Test reusing the same function multiple times."""
    print("=" * 60)
    print("Test: Function Reuse")
    print("=" * 60)

    # Get current device ID (set in main)
    device_id = torch.npu.current_device()

    batch_size, hidden_size = 32, 128

    # Create multiple inputs
    x1 = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    x2 = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    x3 = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    gamma = torch.ones(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    beta = torch.zeros(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    # Outputs
    out1 = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    out2 = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    out3 = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    # Reuse the same function with different inputs
    inputs1 = [x1, gamma, beta]
    inputs2 = [x2, gamma, beta]
    inputs3 = [x3, gamma, beta]

    outputs1 = [out1]
    outputs2 = [out2]
    outputs3 = [out3]

    pto_inputs1 = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs1)]
    pto_inputs2 = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs2)]
    pto_inputs3 = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs3)]

    pto_outputs1 = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs1)]
    pto_outputs2 = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs2)]
    pto_outputs3 = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs3)]

    layer_norm(*pto_inputs1, *pto_outputs1, eps=1e-6)
    pypto.runtime._device_synchronize()
    layer_norm(*pto_inputs2, *pto_outputs2, eps=1e-6)
    pypto.runtime._device_synchronize()
    layer_norm(*pto_inputs3, *pto_outputs3, eps=1e-6)
    pypto.runtime._device_synchronize()

    # Verify
    expected1 = layer_norm_golden(x1, gamma, beta, 1e-6)
    expected2 = layer_norm_golden(x2, gamma, beta, 1e-6)
    expected3 = layer_norm_golden(x3, gamma, beta, 1e-6)

    max_diff1 = (out1 - expected1).abs().max().item()
    max_diff2 = (out2 - expected2).abs().max().item()
    max_diff3 = (out3 - expected3).abs().max().item()

    print(f"Function reused 3 times with different inputs")
    print(f"Max diff 1: {max_diff1:.6f}")
    print(f"Max diff 2: {max_diff2:.6f}")
    print(f"Max diff 3: {max_diff3:.6f}")
    assert max_diff1 < 1e-1 and max_diff2 < 1e-1 and max_diff3 < 1e-1, "Function reuse mismatch!"
    print("✓ Function reuse passed")
    print()


def main():
    """Run multi-function module examples.

    Usage:
        python multi_function_module.py          # Run all examples
        python multi_function_module.py 1         # Run example 1 only
        python multi_function_module.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Multi-Function Module Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s 1            Run example 1 (Sequential Functions)
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
            'name': 'Sequential Functions',
            'description': 'Using multiple functions in sequence',
            'function': test_sequential_functions,
            'requires_npu': True
        },
        2: {
            'name': 'Residual Connection',
            'description': 'Residual connection pattern',
            'function': test_residual_connection,
            'requires_npu': True
        },
        3: {
            'name': 'Transformer Block',
            'description': 'Complete transformer block with multiple functions',
            'function': test_transformer_block,
            'requires_npu': True
        },
        4: {
            'name': 'Function Reuse',
            'description': 'Reusing the same function with different inputs',
            'function': test_function_reuse,
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
    print("PyPTO Multi-Function Module Examples")
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
            print("All multi-function module tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

