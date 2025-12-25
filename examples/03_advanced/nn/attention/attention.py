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
Attention Example for PyPTO

This example demonstrates:
- Scaled dot-product attention
- Attention with projections (Q, K, V + output)
- Static shape runs using the PyPTO frontend
- Comparison against PyTorch golden results
"""

import os
import sys
import argparse
import pypto
import torch
import numpy as np

# Module-level shapes for frontend.jit annotations
BATCH_SIZE = 2
SEQ_LEN_Q = 16
SEQ_LEN_KV = 16
SEQ_LEN = 32
NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_SIZE = 512

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


def scaled_dot_product_attention_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """PyTorch reference implementation of scaled dot-product attention."""
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


def attention_with_projection_golden(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    out_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """PyTorch reference implementation for attention with projections."""
    q = torch.matmul(hidden_states, q_weight)
    k = torch.matmul(hidden_states, k_weight)
    v = torch.matmul(hidden_states, v_weight)

    batch_size, seq_len, _ = q.shape
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(attn_weights, v)
    context = context.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
    output = torch.matmul(context, out_weight)
    return output


@pypto.frontend.jit()
def scaled_dot_product_attention(
    q: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    k: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    v: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
) -> pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16):
    scale = 1.0 / (HEAD_DIM ** 0.5)
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 8, 16, HEAD_DIM)
    scores = pypto.matmul(q, pypto.transpose(k, 2, 3), out_dtype=pypto.DT_BF16)
    scores_scaled = pypto.mul(scores, scale)
    attn_weights = pypto.softmax(scores_scaled, dim=-1)
    output = pypto.matmul(attn_weights, v, out_dtype=pypto.DT_BF16)
    return output


@pypto.frontend.jit()
def attention_with_projection(
    hidden_states: pypto.Tensor((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), pypto.DT_BF16),
    q_weight: pypto.Tensor((1, HIDDEN_SIZE, NUM_HEADS * HEAD_DIM), pypto.DT_BF16),
    k_weight: pypto.Tensor((1, HIDDEN_SIZE, NUM_HEADS * HEAD_DIM), pypto.DT_BF16),
    v_weight: pypto.Tensor((1, HIDDEN_SIZE, NUM_HEADS * HEAD_DIM), pypto.DT_BF16),
    out_weight: pypto.Tensor((1, NUM_HEADS * HEAD_DIM, HIDDEN_SIZE), pypto.DT_BF16),
) -> pypto.Tensor((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), pypto.DT_BF16):
    output_tensor = pypto.tensor((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), pypto.DT_BF16)
    tile_b = 1
    b_loop = BATCH_SIZE // tile_b

    scale = 1.0 / (HEAD_DIM ** 0.5)
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 16, 8, HEAD_DIM)

    q_flat = pypto.matmul(hidden_states, q_weight, out_dtype=pypto.DT_BF16)
    k_flat = pypto.matmul(hidden_states, k_weight, out_dtype=pypto.DT_BF16)
    v_flat = pypto.matmul(hidden_states, v_weight, out_dtype=pypto.DT_BF16)

    q = pypto.reshape(q_flat, [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM])
    k = pypto.reshape(k_flat, [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM])
    v = pypto.reshape(v_flat, [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM])

    q = pypto.transpose(q, 1, 2)
    k = pypto.transpose(k, 1, 2)
    v = pypto.transpose(v, 1, 2)

    for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
        b_offset = idx * tile_b
        b_offset_end = pypto.min((idx + 1) * tile_b, BATCH_SIZE)
        view_shape = [tile_b, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        valid_shape = [b_offset_end - b_offset, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        q_view = pypto.view(q, view_shape, [b_offset, 0, 0, 0], valid_shape=valid_shape)
        k_view = pypto.view(k, view_shape, [b_offset, 0, 0, 0], valid_shape=valid_shape)
        v_view = pypto.view(v, view_shape, [b_offset, 0, 0, 0], valid_shape=valid_shape)

        scores = pypto.matmul(q_view, pypto.transpose(k_view, 2, 3), out_dtype=pypto.DT_BF16)
        scores_scaled = pypto.mul(scores, scale)
        attn_weights = pypto.softmax(scores_scaled, dim=-1)
        context = pypto.matmul(attn_weights, v_view, out_dtype=pypto.DT_BF16)

        context = pypto.transpose(context, 1, 2)
        context_flat = pypto.reshape(context, [tile_b, SEQ_LEN, NUM_HEADS * HEAD_DIM])
        output_view = pypto.matmul(context_flat, out_weight, out_dtype=pypto.DT_BF16)
        output_tensor[b_offset:, ...] = output_view

    return output_tensor


def test_scaled_dot_product_attention(run_mode="npu"):
    """Test scaled dot-product attention using frontend.jit."""
    print("=" * 60)
    print("Test: Scaled Dot-Product Attention")
    print("=" * 60)

    if run_mode == "npu":
        import torch_npu  # noqa: F401
        device = f'npu:{torch.npu.current_device()}'
    else:
        device = 'cpu'

    q_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)

    out = scaled_dot_product_attention(q_torch, k_torch, v_torch)
    pypto.runtime._device_synchronize()

    scale = 1.0 / (HEAD_DIM ** 0.5)
    golden = scaled_dot_product_attention_golden(q_torch, k_torch, v_torch, scale)

    max_diff = (out - golden).abs().max().item()
    print(f"Batch={BATCH_SIZE}, SeqQ={SEQ_LEN_Q}, SeqKV={SEQ_LEN_KV}, Max diff: {max_diff:.6f}")
    print(f"Input shape: {q_torch.shape}")
    print(f"Output shape: {out.shape}")
    torch.allclose(out, golden, rtol=3e-3, atol=3e-3)
    print("✓ Scaled dot-product attention passed for the test case")
    print()


def test_attention_with_projection(run_mode="npu"):
    """Test attention with input/output projections using frontend.jit."""
    print("=" * 60)
    print("Test: Attention with Projections")
    print("=" * 60)

    if run_mode == "npu":
        import torch_npu  # noqa: F401
        device = f'npu:{torch.npu.current_device()}'
    else:
        device = 'cpu'

    hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    q_weight = torch.randn(1, HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(1, HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_weight = torch.randn(1, HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
    out_weight = torch.randn(1, NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

    out = attention_with_projection(hidden_states, q_weight, k_weight, v_weight, out_weight)
    pypto.runtime._device_synchronize()

    golden = attention_with_projection_golden(
        hidden_states, q_weight, k_weight, v_weight, out_weight, NUM_HEADS, HEAD_DIM
    )

    max_diff = (out - golden).abs().max().item()
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Max difference: {max_diff:.6f}")
    torch.allclose(out, golden, rtol=3e-3, atol=3e-3)
    print("✓ Attention with projections passed for the test case")
    print()


def main():
    """Run attention examples.

    Usage:
        python attention.py          # Run all examples
        python attention.py 2         # Run example 2 only
        python attention.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Attention Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s 1            Run example 1 (Scaled Dot-Product Attention)
  %(prog)s 2            Run example 2 (Attention with Projections)
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=int,
        nargs='?',
        help='Example ID to run (1-2). If not specified, all examples will run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available examples and exit'
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

    examples = {
        1: {
            'name': 'Scaled Dot-Product Attention',
            'description': 'Scaled dot-product attention with frontend.jit',
            'function': test_scaled_dot_product_attention
        },
        2: {
            'name': 'Attention with Projections',
            'description': 'Complete attention with input/output projections',
            'function': test_attention_with_projection
        }
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  {ex_id}. {ex_info['name']}")
            print(f"     {ex_info['description']}\n")
        return

    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO Attention Examples")
    print("=" * 60 + "\n")

    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        examples_to_run = list(examples.items())

    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        torch.npu.set_device(device_id)
        print("Running examples that require NPU hardware...")
        print("Make sure CANN environment is configured and NPU is available\n")

    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](args.run_mode)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All attention tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
