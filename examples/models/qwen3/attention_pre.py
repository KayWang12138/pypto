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
Attention Pre-processing Example for Qwen3 Model

This example demonstrates attention pre-processing operations using PyPTO, including:
- Linear projection to generate Q, K, V
- RMS normalization for Q and K
- Rotary Position Embedding (RoPE) application
- Dynamic batch size support
- Tiling for efficient execution

This is a key component in transformer architectures, preparing queries and keys
for attention computation with positional information.
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


def rms_norm_golden(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """
    PyTorch reference implementation of RMSNorm.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    gamma : torch.Tensor
        RMSNorm weight parameter
    eps : float
        Epsilon value for numerical stability

    Returns
    -------
    torch.Tensor
        RMS normalized tensor
    """
    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]
    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff

    reduce_sum = torch.sum(mean_res, dim=-1, keepdim=True) + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt

    res = res_div * gamma

    if x_dtype != torch.float32:
        res = res.to(x_dtype)
    return res


def _apply_rotary_emb_neuron(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embedding to a single tensor (internal helper).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to apply RoPE to
    cos : torch.Tensor
        Cosine values for RoPE
    sin : torch.Tensor
        Sine values for RoPE

    Returns
    -------
    torch.Tensor
        Tensor with rotary embedding applied
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


def apply_rotary_pos_emb_v2(q: torch.Tensor, k: torch.Tensor,
                            cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of Rotary Position Embedding (RoPE).

    Parameters
    ----------
    q : torch.Tensor
        Query tensor
    k : torch.Tensor
        Key tensor
    cos : torch.Tensor
        Cosine values for RoPE
    sin : torch.Tensor
        Sine values for RoPE

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (q_embed, k_embed) with RoPE applied
    """
    x_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    q_embed = _apply_rotary_emb_neuron(q, cos, sin)
    k_embed = _apply_rotary_emb_neuron(k, cos, sin)

    if x_dtype != torch.float32:
        q_embed = q_embed.to(x_dtype)
        k_embed = k_embed.to(x_dtype)
    return q_embed, k_embed


def rms_norm(tensor_value: pypto.tensor, gamma: pypto.tensor, eps: float, tile_shape: tuple) -> pypto.tensor:
    """
    PyPTO implementation of RMSNorm.

    Parameters
    ----------
    tensor_value : pypto.tensor
        Input tensor to normalize
    gamma : pypto.tensor
        RMSNorm weight parameter
    eps : float
        Epsilon value for numerical stability
    tile_shape : tuple
        Tiling shape for vector operations

    Returns
    -------
    pypto.tensor
        RMS normalized tensor
    """
    input_dtype = tensor_value.dtype

    # Cast to FP32 for computation
    pypto.set_vec_tile_shapes(*tile_shape)
    tensor_value_fp32 = pypto.cast(tensor_value, pypto.DT_FP32)

    # Reshape gamma to match tensor dimensions
    pypto.set_vec_tile_shapes(tile_shape[-1])
    gamma_shape = [1] * len(tensor_value_fp32.shape)
    gamma_shape[-1] = gamma.shape[0]
    gamma_3d = pypto.reshape(gamma, gamma_shape)

    # Cast gamma to FP32
    pypto.set_vec_tile_shapes(*tile_shape)
    gamma_fp32 = pypto.cast(gamma_3d, pypto.DT_FP32)

    # 【Compute square】: square = x^2
    square = pypto.mul(tensor_value_fp32, tensor_value_fp32)

    # 【Compute mean】: mean_res = square * mean_coff
    mean_coff = 1.0 / tensor_value_fp32.shape[-1]
    mean_res = pypto.mul(square, mean_coff)

    # 【Reduce sum】: reduce_asum = sum(mean_res, dim=-1, keepdim=True)
    reduce_asum = pypto.sum(mean_res, dim=-1, keepdim=True)

    # 【Add epsilon】: reduce_sum = reduce_asum + eps
    reduce_sum = pypto.add(reduce_asum, eps)

    # 【Square root】: reduce_sqrt = sqrt(reduce_sum)
    reduce_sqrt = pypto.sqrt(reduce_sum)

    res_div = pypto.div(tensor_value_fp32, reduce_sqrt)

    res = pypto.mul(res_div, gamma_fp32)

    # Cast back to input dtype
    y_bf16 = pypto.cast(res, input_dtype)
    return y_bf16


def rope_data(x1: pypto.tensor, x2: pypto.tensor, cos: pypto.tensor,
              sin: pypto.tensor, tile_shape: tuple) -> pypto.tensor:
    """
    PyPTO implementation of Rotary Position Embedding (RoPE).

    Parameters
    ----------
    x1 : pypto.tensor
        First half of input tensor
    x2 : pypto.tensor
        Second half of input tensor
    cos : pypto.tensor
        Cosine values for RoPE
    sin : pypto.tensor
        Sine values for RoPE
    tile_shape : tuple
        Tiling shape for vector operations

    Returns
    -------
    pypto.tensor
        Tensor with RoPE applied
    """
    pypto.set_vec_tile_shapes(*tile_shape)

    # 【Apply RoPE】: o1 = x1 * cos - x2 * sin, o2 = x2 * cos + x1 * sin
    o1 = pypto.sub(pypto.mul(x1, cos), pypto.mul(x2, sin))
    o2 = pypto.add(pypto.mul(x2, cos), pypto.mul(x1, sin))

    # Concatenate results
    res = pypto.concat([o1, o2], dim=2)

    # Cast to BF16
    y_bf16 = pypto.cast(res, pypto.DT_BF16)
    return y_bf16


@pypto.jit
def attention_pre(x, weight, q_gamma, k_gamma, cos, sin, q, k, v):
    """
    PyPTO implementation of attention pre-processing with dynamic batch size support.

    This function performs:
    1. Linear projection to generate Q, K, V
    2. RMS normalization for Q and K
    3. Rotary Position Embedding (RoPE) application
    4. Output reshaping

    Parameters
    ----------
    inputs : list
        List containing [x, weight, q_gamma, k_gamma, cos, sin]
    outputs : list
        List containing [q, k, v]
    """
    # Enable dynamic unaligned support for code generation
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    pypto.set_host_options(only_codegen=True)

    # Get tensor shapes
    bs = x.shape[0]  # Dynamic batch size
    hidden_size = x.shape[1]  # Static hidden size
    total_hidden_size = weight.shape[0]  # Static total hidden size

    # Attention configuration
    head_size = 128
    half_head_size = head_size // 2
    q_size = q.shape[-1]
    kv_size = k.shape[-1]
    q_num_head = q_size // head_size
    kv_num_head = kv_size // head_size
    kv_index = q_num_head + kv_num_head
    eps = 1e-6
    bs_tile = 1
    bs_loop = (bs + bs_tile - 1) // bs_tile

    # Define the computation graph
    def inside_attention_pre_func():
        """Inner function to encapsulate kernel logic for automatic variable cleanup."""
        # Loop over dynamic batch axis
        for bs_idx in pypto.loop(bs_loop, name="LOOP_ATT_PRE_L0", idx_name="bs_idx"):
            def bs_loop_func(bs_idx):
                """Process one batch element."""
                # Create views for current batch element
                x_tile = pypto.view(x, [bs_tile, hidden_size], [bs_idx * bs_tile, 0])
                cos_tile = pypto.view(cos, [bs_tile, 1, half_head_size], [bs_idx * bs_tile, 0, 0])
                sin_tile = pypto.view(sin, [bs_tile, 1, half_head_size], [bs_idx * bs_tile, 0, 0])

                # Step 1: Linear projection to generate Q, K, V
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                mm_res = pypto.matmul(x_tile, weight, out_dtype=pypto.DT_BF16, a_trans=False, b_trans=True)

                # Reshape to 3D: [batch, num_heads, head_size]
                pypto.set_vec_tile_shapes(128, 128, 128)
                mm_3d = pypto.reshape(mm_res, [bs_tile, total_hidden_size // head_size, head_size])

                # Split into Q, K, V
                q_tile = pypto.view(mm_3d, [bs_tile, q_num_head, head_size], [0, 0, 0])
                k_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, q_num_head, 0])
                v_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, kv_index, 0])

                # Step 2: RMS normalization for Q and K
                q_norm = rms_norm(q_tile, q_gamma, eps, [bs_tile, q_num_head, head_size])
                k_norm = rms_norm(k_tile, k_gamma, eps, [bs_tile, kv_num_head, head_size])

                # Step 3: Apply Rotary Position Embedding (RoPE)
                # Cast to FP32 for RoPE computation
                pypto.set_vec_tile_shapes(bs_tile, q_num_head, head_size)
                q_fp32 = pypto.cast(q_norm, pypto.DT_FP32)
                k_fp32 = pypto.cast(k_norm, pypto.DT_FP32)

                pypto.set_vec_tile_shapes(bs_tile, kv_num_head, half_head_size)
                cos_fp32 = pypto.cast(cos_tile, pypto.DT_FP32)
                sin_fp32 = pypto.cast(sin_tile, pypto.DT_FP32)

                # Split Q into two halves for RoPE
                q1 = pypto.view(q_fp32, [bs_tile, q_num_head, half_head_size], [0, 0, 0])
                q2 = pypto.view(q_fp32, [bs_tile, q_num_head, half_head_size], [0, 0, half_head_size])
                q_rope = rope_data(q1, q2, cos_fp32, sin_fp32, [bs_tile, q_num_head, half_head_size])

                # Split K into two halves for RoPE
                k1 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_head_size], [0, 0, 0])
                k2 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_head_size], [0, 0, half_head_size])
                k_rope = rope_data(k1, k2, cos_fp32, sin_fp32, [bs_tile, kv_num_head, half_head_size])

                # Step 4: Reshape outputs
                q_res = pypto.reshape(q_rope, [bs_tile, q_size])
                k_res = pypto.reshape(k_rope, [bs_tile, kv_size])
                v_res = pypto.reshape(v_tile, [bs_tile, kv_size])

                # Assemble results back to output tensors
                q[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = q_res
                k[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = k_res
                v[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = v_res

            bs_loop_func(bs_idx)

    inside_attention_pre_func()


def test_attention_pre():
    """
    Test attention pre-processing implementation against PyTorch reference.

    Tests with different batch sizes to verify dynamic shape support.
    """
    print("=" * 60)
    print("Test: Attention Pre-processing (Dynamic Batch)")
    print("=" * 60)

    # Configuration
    hidden_size = 2048
    total_hidden_size = 768
    head_size = 128
    q_size = 512
    kv_size = 128
    half_head_size = head_size // 2

    device_id = torch.npu.current_device()
    eps = 1e-6

    # Test with different batch sizes
    for i in range(4):
        if i == 2:
            batch_size = 5  # Test with small batch size
        else:
            batch_size = 48

        print(f"\nTesting with batch size: {batch_size}")

        # Prepare test data
        np.random.seed(0)
        x = torch.rand(batch_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        weight = torch.rand(total_hidden_size, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        q_gamma = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        k_gamma = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        cos = torch.rand(batch_size, 1, half_head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        sin = torch.rand(batch_size, 1, half_head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

        # Output tensors
        q = torch.zeros((batch_size, q_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
        k = torch.zeros((batch_size, kv_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
        v = torch.zeros((batch_size, kv_size), dtype=torch.bfloat16, device=f'npu:{device_id}')

        # Initialize PyPTO inputs and outputs, mark batch dimension (axis 0) as dynamic
        inputs = {
            x: [0],
            weight: [],
            q_gamma: [],
            k_gamma: [],
            cos: [0],
            sin: [0]
        }
        outputs = {
            q: [0],
            k: [0],
            v: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

        # Execute PyPTO kernel
        attention_pre(*pto_inputs, *pto_outputs)
        pypto.runtime._device_synchronize()

        # Compute reference using PyTorch
        mm_golden = torch.matmul(x, weight.T)
        q_g, k_g, v_g = mm_golden.split([q_size, kv_size, kv_size], dim=-1)

        # RMS normalization
        q_by_head = q_g.view(*q_g.shape[:-1], q_g.shape[-1] // head_size, head_size)
        q_by_head = rms_norm_golden(q_by_head, q_gamma, eps)
        k_by_head = k_g.view(*k_g.shape[:-1], k_g.shape[-1] // head_size, head_size)
        k_by_head = rms_norm_golden(k_by_head, k_gamma, eps)

        # Apply RoPE
        q_r, k_r = apply_rotary_pos_emb_v2(q_by_head, k_by_head, cos, sin)

        # Reshape outputs
        q_r = q_r.view(batch_size, q_size)
        k_r = k_r.view(batch_size, kv_size)

        # Verify results
        assert_allclose(
            np.array(q_r.cpu().flatten().tolist()),
            np.array(q.cpu().flatten().tolist()),
            rtol=0.001,
            atol=0.001
        )
        assert_allclose(
            np.array(k_r.cpu().flatten().tolist()),
            np.array(k.cpu().flatten().tolist()),
            rtol=0.001,
            atol=0.001
        )
        assert_allclose(
            np.array(v_g.cpu().flatten().tolist()),
            np.array(v.cpu().flatten().tolist()),
            rtol=0.001,
            atol=0.001
        )

        print(f"  ✓ Batch size {batch_size} passed")

    print("\n✓ All attention pre-processing tests passed")


def main():
    """Run attention pre-processing example.

    Usage:
        python attention_pre.py          # Run example
        python attention_pre.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Attention Pre-processing Example (Qwen3)",
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
            'name': 'Attention Pre-processing',
            'description': 'Attention pre-processing with RMSNorm and RoPE',
            'function': test_attention_pre,
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
    print("PyPTO Attention Pre-processing Example (Qwen3)")
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
