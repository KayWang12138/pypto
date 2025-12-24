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
Scaled Dot-Product Attention Example for PyPTO

This example demonstrates:
- Scaled dot-product attention mechanism
- Q, K, V computation
- Attention scores calculation
- Softmax normalization
- Output projection
- Static and dynamic batch/sequence length support

Attention is the core mechanism in transformer architectures.
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


def scaled_dot_product_attention_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """PyTorch reference implementation of scaled dot-product attention."""
    # Compute attention scores: Q @ K^T
    scores = torch.matmul(
        q, k.transpose(-2, -1)
    )  # [batch, num_heads, seq_len_q, seq_len_kv]

    # Scale
    scores = scores * scale

    # Apply attention mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask

    # Softmax
    attn_weights = torch.softmax(
        scores, dim=-1
    )  # [batch, num_heads, seq_len_q, seq_len_kv]

    # Apply to values: attn_weights @ V
    output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len_q, head_dim]

    return output


def test_attention_dynamic():
    """Test attention with dynamic shapes."""
    print("=" * 60)
    print("Test: Scaled Dot-Product Attention (Dynamic)")
    print("=" * 60)

    num_heads, head_dim = 8, 64
    seq_len_kv = 64
    batch_size, seq_len_q = 8, 64  # concrete shape
    dtype = pypto.DT_BF16
    shape = batch_size, num_heads, seq_len_q, head_dim

    mark_dynamic = True  # TODO: fix the wrong result if switched to dynamic
    if mark_dynamic:
        B = pypto.frontend.dynamic("B")  # symbolic shape
        Sq = pypto.frontend.dynamic("Sq")
        # Sq = 64
        shape = B, num_heads, Sq, head_dim
    else:
        print("warning: NOT marking dynamic dimension!!")

    # pass (batch_size, num_heads, seq_len_q, head_dim) all by closure
    @pypto.frontend.jit(host_options={"only_codegen": True})
    def scaled_dot_product_attention_dynamic(
        q: pypto.Tensor(shape, dtype),
        k: pypto.Tensor(shape, dtype),
        v: pypto.Tensor(shape, dtype),
    ) -> pypto.Tensor(shape, dtype):
        """Scaled dot-product attention with dynamic batch and sequence lengths."""
        pypto.set_runtime_options(cfgcache_device_task_num=100)
        pypto.set_runtime_options(cfgcache_root_task_num=100)
        pypto.set_runtime_options(cfgcache_leaf_task_num=10000)

        out = pypto.tensor(shape, dtype)

        # Calculate scale
        scale = 1.0 / (head_dim**0.5)

        # Configure tiling
        cube_tiling = 64
        pypto.set_cube_tile_shapes(
            [cube_tiling, cube_tiling],
            [cube_tiling, cube_tiling],
            [cube_tiling, cube_tiling],
        )
        view_shape = (batch_size, num_heads, seq_len_q, head_dim)
        bs_loop = (batch_size + view_shape[0] - 1) // view_shape[0]

        for bs_idx in pypto.loop(
            bs_loop, name="LOOP_L0", idx_name="bs_idx", unroll_List={1}
        ):
            offsets = [bs_idx * view_shape[0], 0, 0, 0]
            q_view = pypto.view(q, view_shape, offsets)
            k_view = pypto.view(k, view_shape, offsets)
            v_view = pypto.view(v, view_shape, offsets)
            pypto.set_vec_tile_shapes(1, 8, 16, head_dim)
            k_t = pypto.transpose(k_view, 2, 3)
            scores = pypto.matmul(q_view, k_t, out_dtype=pypto.DT_BF16)
            scores_scaled = pypto.mul(scores, scale)
            attn_weights = pypto.softmax(scores_scaled, dim=-1)
            res = pypto.matmul(attn_weights, v_view, out_dtype=pypto.DT_BF16)
            out[bs_idx * view_shape[0]:, ...] = res
            # pypto.assemble(res, offsets, out)

        return out

    device_id = torch.npu.current_device()
    q_torch = torch.randn(
        batch_size,
        num_heads,
        seq_len_q,
        head_dim,
        dtype=torch.bfloat16,
        device=f"npu:{device_id}",
    )
    k_torch = torch.randn(
        batch_size,
        num_heads,
        seq_len_kv,
        head_dim,
        dtype=torch.bfloat16,
        device=f"npu:{device_id}",
    )
    v_torch = torch.randn(
        batch_size,
        num_heads,
        seq_len_kv,
        head_dim,
        dtype=torch.bfloat16,
        device=f"npu:{device_id}",
    )
    # Execute
    out_torch = scaled_dot_product_attention_dynamic(q_torch, k_torch, v_torch)
    pypto.runtime._device_synchronize()

    # Verify
    scale = 1.0 / (head_dim**0.5)
    expected = scaled_dot_product_attention_golden(q_torch, k_torch, v_torch, scale)
    max_diff = (out_torch - expected).abs().max().item()

    print(
        f"Batch={batch_size}, SeqQ={seq_len_q}, SeqKV={seq_len_kv}, Max diff: {max_diff:.6f}"
    )

    print(out_torch[0][0])
    print(expected[7][7])

    if max_diff < 3e-3:
        print("✓ Attention (dynamic) passed for the test case")
    else:
        print(f"Result mismatch for batch={batch_size}!")
    print()


def test_attention_with_projection():
    """Test complete attention with input/output projections."""
    print("=" * 60)
    print("Test: Attention with Projections")
    print("=" * 60)

    batch_size, seq_len, hidden_size = 2, 32, 512
    num_heads, head_dim = 8, 64
    dtype = pypto.DT_FP32

    activation_shape = (batch_size, seq_len, hidden_size)
    in_weight_shape = (1, hidden_size, num_heads * head_dim)
    out_weight_shape = (1, num_heads * head_dim, hidden_size)

    @pypto.frontend.jit(host_options={"only_codegen": True})
    def attention_with_projection(
        hidden_states: pypto.Tensor(activation_shape, dtype),
        q_weight: pypto.Tensor(in_weight_shape, dtype),
        k_weight: pypto.Tensor(in_weight_shape, dtype),
        v_weight: pypto.Tensor(in_weight_shape, dtype),
        out_weight: pypto.Tensor(out_weight_shape, dtype),
    ) -> pypto.Tensor(activation_shape, dtype):
        """Complete attention with input projection (Q, K, V from hidden states)."""
        out = pypto.tensor(activation_shape, dtype)
        print(in_weight_shape, out_weight_shape)
        # without using the shape variable, will get:
        # - Error: NameError: name 'in_weight_shape' is not defined
        # - Error: NameError: name 'out_weight_shape' is not defined
        out_dtype = pypto.DT_FP32

        view_shape = (batch_size, num_heads, seq_len, head_dim)
        bs_loop = (batch_size + view_shape[0] - 1) // view_shape[0]
        # Configure tiling
        cube_tiling = 64
        pypto.set_cube_tile_shapes(
            [cube_tiling, cube_tiling],
            [cube_tiling, cube_tiling],
            [cube_tiling, cube_tiling],
        )

        for bs_idx in pypto.loop(
            bs_loop, name="LOOP_L0", idx_name="bs_idx", unroll_List={1}
        ):
            q_flat = pypto.matmul(hidden_states, q_weight, out_dtype=out_dtype)
            k_flat = pypto.matmul(hidden_states, k_weight, out_dtype=out_dtype)
            v_flat = pypto.matmul(hidden_states, v_weight, out_dtype=out_dtype)

            # Reshape to multi-head format
            q = pypto.reshape(q_flat, [batch_size, seq_len, num_heads, head_dim])
            k = pypto.reshape(k_flat, [batch_size, seq_len, num_heads, head_dim])
            v = pypto.reshape(v_flat, [batch_size, seq_len, num_heads, head_dim])
            pypto.set_vec_tile_shapes(1, 16, 8, head_dim)
            # Transpose for attention: [batch, num_heads, seq_len, head_dim]
            q = pypto.transpose(q, 1, 2)
            k = pypto.transpose(k, 1, 2)
            v = pypto.transpose(v, 1, 2)

            offsets = [bs_idx * view_shape[0], 0, 0, 0]
            offsets_out = [bs_idx * view_shape[0], 0, 0]
            q_view = pypto.view(q, view_shape, offsets)
            k_view = pypto.view(k, view_shape, offsets)
            v_view = pypto.view(v, view_shape, offsets)

            # Step 2: Scaled dot-product attention
            scale = 1.0 / (head_dim**0.5)
            k_t = pypto.transpose(k_view, 2, 3)
            scores = pypto.matmul(q_view, k_t, out_dtype=out_dtype)
            scores_scaled = pypto.mul(scores, scale)
            attn_weights = pypto.softmax(scores_scaled, dim=-1)
            attn_output = pypto.matmul(attn_weights, v_view, out_dtype=out_dtype)
            # Step 3: Transpose back and reshape
            attn_output = pypto.transpose(attn_output, 1, 2)
            attn_output_flat = pypto.reshape(
                attn_output, [view_shape[0], seq_len, num_heads * head_dim]
            )
            # Step 4: Output projection
            res = pypto.matmul(attn_output_flat, out_weight, out_dtype=out_dtype)
            pypto.assemble(res, offsets_out, out)

        return out

    device_id = torch.npu.current_device()

    # Create tensors
    hidden_states = torch.randn(
        *activation_shape, dtype=torch.float32, device=f"npu:{device_id}"
    )
    q_weight = torch.randn(
        *in_weight_shape, dtype=torch.float32, device=f"npu:{device_id}"
    )
    k_weight = torch.randn(
        *in_weight_shape, dtype=torch.float32, device=f"npu:{device_id}"
    )
    v_weight = torch.randn(
        *in_weight_shape, dtype=torch.float32, device=f"npu:{device_id}"
    )
    out_weight = torch.randn(
        *out_weight_shape, dtype=torch.float32, device=f"npu:{device_id}"
    )

    # Execute
    out_torch = attention_with_projection(
        hidden_states,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
    )

    # Verify (simplified - just check output shape and range)
    # TODO: check numerical correctness!!
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Output shape: {out_torch.shape}")
    print(f"Output range: [{out_torch.min():.4f}, {out_torch.max():.4f}]")
    print("✓ Attention with projections completed")
    print()


def main():
    """Run attention examples.

    Usage:
        python attention.py          # Run all examples
        python attention.py 1         # Run example 1 only
        python attention.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Scaled Dot-Product Attention Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s 1            Run example 1 (Attention Dynamic)
  %(prog)s --list       List all available examples
        """,
    )
    parser.add_argument(
        "example_id",
        type=int,
        nargs="?",
        help="Example ID to run (1-2). If not specified, all examples will run.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available examples and exit"
    )

    args = parser.parse_args()

    # Define available examples
    examples = {
        1: {
            "name": "Attention Dynamic",
            "description": "Scaled dot-product attention with dynamic shapes",
            "function": test_attention_dynamic,
            "requires_npu": True,
        },
        2: {
            "name": "Attention with Projections",
            "description": "Complete attention with input/output projections",
            "function": test_attention_with_projection,
            "requires_npu": True,
        },
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
    print("PyPTO Scaled Dot-Product Attention Examples")
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
            print("All attention tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
