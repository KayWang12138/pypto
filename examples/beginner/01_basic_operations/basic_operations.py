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
Basic Operations Example for PyPTO

This example demonstrates fundamental PyPTO operations including:
- Tensor creation
- Element-wise operations (add, mul, sub, div)
- Matrix multiplication
- View operations
- Activation functions

This is a beginner-friendly example that shows the core concepts of PyPTO.
"""

import os
import sys
import argparse
import pypto
import torch
import torch_npu


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


def example_tensor_creation():
    """Example 1: Creating tensors with different properties."""
    print("=" * 60)
    print("Example 1: Tensor Creation")
    print("=" * 60)

    # Create a tensor with shape [4, 4] and FP16 data type
    tensor = pypto.tensor([4, 4], pypto.DT_FP16, "my_tensor")

    print(f"Tensor name: {tensor.name}")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor format: {tensor.format}")
    print(f"Tensor dimensions: {tensor.dim}")
    print()


def example_element_wise_operations():
    """Example 2: Element-wise arithmetic operations."""
    print("=" * 60)
    print("Example 2: Element-wise Operations")
    print("=" * 60)

    device_id = torch.npu.current_device()
    shape = (8, 8)
    a_torch = torch.randn(shape, dtype=torch.float16, device=f'npu:{device_id}')
    b_torch = torch.randn(shape, dtype=torch.float16, device=f'npu:{device_id}')
    c_torch = torch.zeros(shape, dtype=torch.float16, device=f'npu:{device_id}')

    @pypto.jit
    def element_wise_ops(a, b, result):
        pypto.set_vec_tile_shapes(8, 8)

        for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
            add_result = pypto.add(a, b)
            mul_result = pypto.mul(add_result, 2.0)
            result[:] = mul_result
    inputs = [a_torch, b_torch]
    outputs = [c_torch]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    element_wise_ops(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    expected = (a_torch + b_torch) * 2.0
    max_diff = (c_torch - expected).abs().max().item()
    print(f"Input A shape: {a_torch.shape}")
    print(f"Input B shape: {b_torch.shape}")
    print(f"Output shape: {c_torch.shape}")
    print(f"Max difference from PyTorch: {max_diff:.6f}")
    assert max_diff < 1e-2, "Result mismatch!"
    print("✓ Element-wise operations completed successfully")
    print()


def example_matrix_multiplication():
    """Example 3: Matrix multiplication."""
    print("=" * 60)
    print("Example 3: Matrix Multiplication")
    print("=" * 60)

    device_id = torch.npu.current_device()
    M, K, N = 64, 128, 64
    A_torch = torch.randn(M, K, dtype=torch.bfloat16, device=f'npu:{device_id}')
    B_torch = torch.randn(K, N, dtype=torch.bfloat16, device=f'npu:{device_id}')
    C_torch = torch.zeros(M, N, dtype=torch.bfloat16, device=f'npu:{device_id}')

    @pypto.jit
    def matrix_multiply(A, B, C):
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])

        for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
            C[:] = pypto.matmul(A, B, out_dtype=pypto.DT_BF16)

    inputs = [A_torch, B_torch]
    outputs = [C_torch]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    matrix_multiply(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    expected = torch.matmul(A_torch, B_torch)
    max_diff = (C_torch - expected).abs().max().item()
    print(f"Matrix A shape: {A_torch.shape}")
    print(f"Matrix B shape: {B_torch.shape}")
    print(f"Output C shape: {C_torch.shape}")
    print(f"Max difference from PyTorch: {max_diff:.6f}")
    assert max_diff < 1e-1, "Result mismatch!"
    print("✓ Matrix multiplication completed successfully")
    print()


def example_activation_functions():
    """Example 4: Activation functions."""
    print("=" * 60)
    print("Example 4: Activation Functions")
    print("=" * 60)

    device_id = torch.npu.current_device()
    shape = (32, 64)
    input_torch = torch.randn(shape, dtype=torch.float16, device=f'npu:{device_id}')
    output_torch = torch.zeros(shape, dtype=torch.float16, device=f'npu:{device_id}')

    @pypto.jit
    def apply_activations(x, result):
        pypto.set_vec_tile_shapes(32, 64)

        for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
            result[:] = pypto.sigmoid(x)

    inputs = [input_torch]
    outputs = [output_torch]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    apply_activations(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    expected = torch.sigmoid(input_torch)
    max_diff = (output_torch - expected).abs().max().item()
    print(f"Input shape: {input_torch.shape}")
    print(f"Output shape: {output_torch.shape}")
    print(f"Output range: [{output_torch.min():.4f}, {output_torch.max():.4f}]")
    print(f"Max difference from PyTorch: {max_diff:.6f}")
    assert max_diff < 1e-2, "Result mismatch!"
    print("✓ Activation functions completed successfully")
    print()


def example_view_operations():
    """Example 5: View operations for tiling."""
    print("=" * 60)
    print("Example 5: View Operations")
    print("=" * 60)

    device_id = torch.npu.current_device()
    shape = (256, 512)
    input_torch = torch.randn(shape, dtype=torch.float16, device=f'npu:{device_id}')
    output_torch = torch.zeros(shape, dtype=torch.float16, device=f'npu:{device_id}')

    @pypto.jit
    def tiled_operation(input_tensor, output_tensor):
        # Get shape
        h, w = input_tensor.shape[0], input_tensor.shape[1]

        # Define tile size
        tile_h, tile_w = 32, 32

        # Calculate number of tiles
        h_tiles = (h + tile_h - 1) // tile_h
        w_tiles = (w + tile_w - 1) // tile_w

        pypto.set_vec_tile_shapes(tile_h, tile_w)

        for h_idx in pypto.loop(h_tiles, name="h_loop", idx_name="h_idx"):
            for w_idx in pypto.loop(w_tiles, name="w_loop", idx_name="w_idx"):
                # Calculate offsets
                h_offset = h_idx * tile_h
                w_offset = w_idx * tile_w

                # Create view for this tile
                view = pypto.view(
                    input_tensor,
                    [tile_h, tile_w],
                    [h_offset, w_offset]
                )

                # Process tile (simple operation: multiply by 2)
                result = pypto.mul(view, 2.0)

                # Assemble result back
                pypto.assemble(result, [h_offset, w_offset], output_tensor)

    # Execute
    inputs = [input_torch]
    outputs = [output_torch]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]

    tiled_operation(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    # Verify
    expected = input_torch * 2.0
    max_diff = (output_torch - expected).abs().max().item()
    print(f"Input shape: {input_torch.shape}")
    print(f"Tile size: (32, 32)")
    print(f"Number of tiles: ({shape[0]//32 + 1}, {shape[1]//32 + 1})")
    print(f"Output shape: {output_torch.shape}")
    print(f"Max difference from PyTorch: {max_diff:.6f}")
    assert max_diff < 1e-2, "Result mismatch!"
    print("✓ View operations completed successfully")
    print()


def example_combined_operations():
    """Example 6: Combining multiple operations."""
    print("=" * 60)
    print("Example 6: Combined Operations")
    print("=" * 60)

    # Get current device ID (set in main)
    device_id = torch.npu.current_device()

    # Simple neural network layer: y = sigmoid(x @ W + b)
    batch, in_features, out_features = 32, 64, 32
    x_torch = torch.randn(batch, in_features, dtype=torch.bfloat16, device=f'npu:{device_id}')
    W_torch = torch.randn(in_features, out_features, dtype=torch.bfloat16, device=f'npu:{device_id}')
    b_torch = torch.randn(out_features, dtype=torch.bfloat16, device=f'npu:{device_id}')
    y_torch = torch.zeros(batch, out_features, dtype=torch.bfloat16, device=f'npu:{device_id}')

    @pypto.jit
    def linear_layer_with_activation(x, W, b, y):
        pypto.set_vec_tile_shapes(32, 64)
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])

        for _ in pypto.loop(1, name="dummy_loop", idx_name="dummy_idx"):
            linear = pypto.matmul(x, W, out_dtype=pypto.DT_BF16)
            biased = pypto.add(linear, b)
            y[:] = pypto.sigmoid(biased)

    inputs = [x_torch, W_torch, b_torch]
    outputs = [y_torch]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    linear_layer_with_activation(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    expected = torch.sigmoid(torch.matmul(x_torch, W_torch) + b_torch)
    max_diff = (y_torch - expected).abs().max().item()
    print(f"Input x shape: {x_torch.shape}")
    print(f"Weight W shape: {W_torch.shape}")
    print(f"Bias b shape: {b_torch.shape}")
    print(f"Output y shape: {y_torch.shape}")
    print(f"Max difference from PyTorch: {max_diff:.6f}")
    assert max_diff < 1e-1, "Result mismatch!"
    print("✓ Combined operations completed successfully")
    print()


def main():
    """Run basic operation examples.

    Usage:
        python basic_operations.py          # Run all examples
        python basic_operations.py 2         # Run example 2 only
        python basic_operations.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Basic Operations Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s 2            Run example 2 (Element-wise operations)
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=int,
        nargs='?',
        help='Example ID to run (1-6). If not specified, all examples will run.'
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
            'name': 'Tensor Creation',
            'description': 'Creating tensors with different properties',
            'function': example_tensor_creation,
            'requires_npu': False
        },
        2: {
            'name': 'Element-wise Operations',
            'description': 'Element-wise arithmetic operations (add, mul)',
            'function': example_element_wise_operations,
            'requires_npu': True
        },
        3: {
            'name': 'Matrix Multiplication',
            'description': 'Matrix multiplication operations',
            'function': example_matrix_multiplication,
            'requires_npu': True
        },
        4: {
            'name': 'Activation Functions',
            'description': 'Activation functions (sigmoid)',
            'function': example_activation_functions,
            'requires_npu': True
        },
        5: {
            'name': 'View Operations',
            'description': 'View operations for tiling',
            'function': example_view_operations,
            'requires_npu': True
        },
        6: {
            'name': 'Combined Operations',
            'description': 'Combining multiple operations',
            'function': example_combined_operations,
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
    print("PyPTO Basic Operations Examples")
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
        print("Running examples that require NPU hardware...")
        print("(Make sure CANN environment is configured and NPU is available)\n")

    try:
        for ex_id, ex_info in examples_to_run:
            if ex_info['requires_npu'] and device_id is None:
                print(f"Skipping example {ex_id} ({ex_info['name']}): NPU device not configured")
                continue

            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function']()

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All examples completed successfully!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure CANN environment is sourced")
        print("2. Check NPU device is available: npu-smi info")
        print("3. Verify PyTorch and torch_npu are installed")
        raise


if __name__ == "__main__":
    main()

