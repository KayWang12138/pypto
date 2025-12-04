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
MoE FFN MLP Example for Qwen3 Model

This example demonstrates a Mixture-of-Experts (MoE) Feed-Forward Network using PyPTO:
- Expert routing based on token assignments
- SwiGLU activation function
- Dynamic token processing per expert
- Efficient tiling for multiple experts
- Token accumulation table usage

This is a key component in MoE transformer architectures, where different
experts process different subsets of tokens for improved efficiency.
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


@dataclass
class ExpertInferConfig:
    exp_idx: pypto.symbolic_scalar
    loop_base: int
    token_loop_idx: pypto.symbolic_scalar
    expand_x: pypto.tensor
    expert_tokens: pypto.tensor
    token_acc_table: pypto.tensor
    weight_gate_upper: pypto.tensor
    weight_down_proj: pypto.tensor
    vec_tile_shape: tuple
    cube_tile_shape: tuple
    ffn_out: pypto.tensor
    

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


# Tiling configuration constants
VEC_TILE_SHAPE = (64, 128)
CUBE_TILE_SHAPE = (64, 128, 128)
LOOP_BASE = 16  # Number of tokens processed per expert per iteration


def get_token_acc_table(expert_tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute token accumulation table (cumulative sum).
    
    This helper function computes the cumulative sum of token counts per expert,
    which is used to determine token offsets for each expert.
    
    Parameters
    ----------
    expert_tokens : torch.Tensor
        Tensor containing token counts per expert
        
    Returns
    -------
    torch.Tensor
        Token accumulation table (cumulative sum)
    """
    assert len(expert_tokens.shape) == 1
    token_acc_table = torch.zeros_like(expert_tokens)
    for i in range(1, expert_tokens.shape[0]):
        token_acc_table[i] = torch.sum(expert_tokens[0:i])
    return token_acc_table


def ffn_golden_torch(topk: int, expand_x: torch.Tensor, expert_tokens: torch.Tensor,
                     weight_gate_upper: torch.Tensor, weight_down_proj: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of MoE FFN.
    
    Parameters
    ----------
    topk : int
        Number of top experts selected per token
    expand_x : torch.Tensor
        Expanded input tensor [batch * seq * topk, hidden_size]
    expert_tokens : torch.Tensor
        Token counts per expert
    weight_gate_upper : torch.Tensor
        Gate and up projection weights [num_experts, intermediate_size * 2, hidden_size]
    weight_down_proj : torch.Tensor
        Down projection weights [num_experts, hidden_size, intermediate_size]
        
    Returns
    -------
    torch.Tensor
        Output tensor after MoE FFN processing
    """
    batch_size = expand_x.shape[0] // topk
    hidden_size = expand_x.shape[1]
    intermediate_size = weight_down_proj.shape[2]
    out = torch.zeros_like(expand_x)
    
    start_idx = 0
    for i in range(expert_tokens.shape[0]):
        token_count = expert_tokens[i].item()
        if token_count <= 0:
            continue  # Skip experts with no tokens
        
        end_idx = start_idx + token_count
        
        # Safety check: ensure we don't exceed tensor bounds
        if end_idx > expand_x.shape[0]:
            end_idx = expand_x.shape[0]
        
        # Select tokens for current expert
        selected_x = expand_x[start_idx:end_idx, :]
        
        # Get current expert's weights
        current_gate_weight = weight_gate_upper[i]  # [intermediate_size * 2, hidden_size]
        current_down_weight = weight_down_proj[i]   # [hidden_size, intermediate_size]
        
        # Gate projection: [token_count, hidden_size] @ [hidden_size, intermediate_size * 2]
        gate_output = torch.matmul(selected_x.float(), current_gate_weight.float().T)
        
        # Split gate output into left (gate) and right (up) parts
        split_dim = gate_output.shape[-1] // 2
        left, right = torch.split(gate_output, split_dim, dim=-1)
        
        # SwiGLU activation: Swish(gate) * up
        # Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
        swiglu = left * torch.sigmoid(left)
        
        # Multiply Swish(gate) with up projection
        swiglu_right = swiglu * right
        
        # Down projection: [token_count, intermediate_size] @ [intermediate_size, hidden_size]
        expert_output = torch.matmul(
            swiglu_right.to(current_down_weight.dtype).float(),
            current_down_weight.float().T
        )
        
        # Store results
        out[start_idx:end_idx, :] = expert_output.to(expand_x.dtype)
        
        # Update start index for next expert
        start_idx = end_idx
    
    return out


def gen_input(batch_size: int, seq_len: int, topk: int, per_expert_num: int,
              hidden_size: int, intermediate_size: int, dtype: torch.dtype,
              device_id: int) -> tuple:
    """
    Generate test input data for MoE FFN.
    
    Parameters
    ----------
    batch_size : int
        Batch size
    seq_len : int
        Sequence length
    topk : int
        Number of top experts per token
    per_expert_num : int
        Number of experts
    hidden_size : int
        Hidden dimension size
    intermediate_size : int
        Intermediate dimension size
    dtype : torch.dtype
        Data type for tensors
    device_id : int
        NPU device ID
        
    Returns
    -------
    tuple
        Tuple of (expand_x, expert_tokens, token_acc_table, weight_gate_upper, weight_down_proj, out)
    """
    expand_x_tensor = torch.randn(
        (batch_size * seq_len * topk, hidden_size),
        dtype=dtype,
        device=f'npu:{device_id}'
    ) * 0.01 * 2 - 0.01
    
    expert_tokens_tensor = torch.randint(
        0, 2,
        (per_expert_num,),
        dtype=torch.int32,
        device=f'npu:{device_id}'
    )
    
    token_acc_table_tensor = get_token_acc_table(expert_tokens_tensor).to(torch.int32)
    
    weight_gate_upper_tensor = torch.randn(
        (per_expert_num, intermediate_size * 2, hidden_size),
        dtype=dtype,
        device=f'npu:{device_id}'
    ) * 0.01 * 2 - 0.01
    
    weight_down_proj_tensor = torch.randn(
        (per_expert_num, hidden_size, intermediate_size),
        dtype=dtype,
        device=f'npu:{device_id}'
    ) * 0.01 * 2 - 0.01
    
    out_tensor = torch.zeros_like(expand_x_tensor, device=f'npu:{device_id}')
    input_datas = (expand_x_tensor, expert_tokens_tensor, token_acc_table_tensor,
            weight_gate_upper_tensor, weight_down_proj_tensor, out_tensor)
    return input_datas


def expert_infer_base(config: ExpertInferConfig):
    """
    Base expert inference function for processing tokens assigned to a specific expert.
    
    This function processes a batch of tokens for one expert, applying:
    1. Gate and up projection
    2. SwiGLU activation
    3. Down projection
    
    Parameters
    ----------
    exp_idx : pypto.symbolic_scalar
        Expert index
    loop_base : int
        Base number of tokens processed per iteration
    token_loop_idx : pypto.symbolic_scalar
        Token loop index within expert
    expand_x : pypto.tensor
        Expanded input tensor
    expert_tokens : pypto.tensor
        Token counts per expert
    token_acc_table : pypto.tensor
        Token accumulation table
    weight_gate_upper : pypto.tensor
        Gate and up projection weights
    weight_down_proj : pypto.tensor
        Down projection weights
    vec_tile_shape : tuple
        Vector tiling shape
    cube_tile_shape : tuple
        Cube tiling shape
    ffn_out : pypto.tensor
        Output tensor
    """
    exp_idx = config.exp_idx
    loop_base = config.loop_base
    token_loop_idx = config.token_loop_idx
    expand_x = config.expand_x
    expert_tokens = config.expert_tokens
    token_acc_table = config.token_acc_table
    weight_gate_upper = config.weight_gate_upper
    weight_down_proj = config.weight_down_proj
    vec_tile_shape = config.vec_tile_shape
    cube_tile_shape = config.cube_tile_shape
    ffn_out = config.ffn_out
    
    hidden_size = expand_x.shape[1]
    intermediate_size = weight_down_proj.shape[1]
    x_dtype = expand_x.dtype
    
    # Compute offset for current expert's tokens
    pypto.set_vec_tile_shapes(32)
    token_num = expert_tokens[exp_idx]
    expand_x_offset_start = token_acc_table[exp_idx]
    expand_x_offset = [expand_x_offset_start + token_loop_idx * loop_base, 0]
    
    # Compute weight offsets for current expert
    weight_13_offset = [exp_idx * (intermediate_size * 2), 0]
    weight_2_offset = [exp_idx * hidden_size, 0]
    
    # Get current valid token size (may be less than loop_base for last iteration)
    cur_valid_size = (token_num - token_loop_idx * loop_base).min(loop_base)
    
    # Create view of input tokens for current expert and iteration
    pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
    x = pypto.view(
        expand_x,
        [loop_base, hidden_size],
        expand_x_offset,
        valid_shape=[cur_valid_size, hidden_size]
    )
    
    # Get current expert's gate and up projection weights
    ffn_weight_2d = pypto.view(
        weight_gate_upper,
        [intermediate_size * 2, hidden_size],
        weight_13_offset
    )
    
    # Get current expert's down projection weights
    down_proj_2d = pypto.view(
        weight_down_proj,
        [hidden_size, intermediate_size],
        weight_2_offset
    )
    
    # Step 1: Gate and up projection
    pypto.set_cube_tile_shapes(
        [cube_tile_shape[0], cube_tile_shape[0]],
        [cube_tile_shape[1], cube_tile_shape[1]],
        [cube_tile_shape[2], cube_tile_shape[2]]
    )
    pypto.set_matrix_size({loop_base, ffn_weight_2d.shape[1], ffn_weight_2d.shape[0]})
    gate = pypto.matmul(x, ffn_weight_2d, out_dtype=pypto.DT_FP32, b_trans=True)
    
    # Split gate output into left (gate) and right (up) parts
    pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
    gate_left = pypto.view(gate, [loop_base, intermediate_size], [0, 0])
    gate_right = pypto.view(gate, [loop_base, intermediate_size], [0, intermediate_size])
    
    # Step 2: SwiGLU activation
    # Swish(gate) = gate / (1 + exp(-gate))
    swiglu_a = pypto.mul(gate_left, -1.0)
    swiglu_b = pypto.exp(swiglu_a)
    swiglu_c = pypto.add(swiglu_b, 1.0)
    swiglu_out = pypto.div(gate_left, swiglu_c)
    
    # Multiply Swish(gate) with up projection
    swiglu = pypto.mul(swiglu_out, gate_right)
    
    # Step 3: Down projection
    swish_fp16 = pypto.cast(swiglu, x_dtype)
    pypto.set_cube_tile_shapes(
        [cube_tile_shape[0], cube_tile_shape[0]],
        [cube_tile_shape[1], cube_tile_shape[1]],
        [cube_tile_shape[2], cube_tile_shape[2]]
    )
    pypto.set_matrix_size({loop_base, down_proj_2d.shape[1], down_proj_2d.shape[0]})
    out = pypto.matmul(swish_fp16, down_proj_2d, out_dtype=x_dtype, b_trans=True)
    
    # Assemble result back to output tensor
    pypto.assemble(out, expand_x_offset, ffn_out)


@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_unaligned": True}
)
def moe_ffn(inputs: list, outputs: list):
    """
    PyPTO implementation of MoE FFN with dynamic token support.
    
    This function processes tokens through multiple experts, where each expert
    processes a subset of tokens based on routing decisions.
    
    Parameters
    ----------
    inputs : list
        List containing [expand_x, expert_tokens, token_acc_table, weight_gate_upper, weight_down_proj]
    outputs : list
        List containing [ffn_out]
    """
    # Mark dynamic dimension
    pypto.mark_dynamic(inputs[0], 0)
    
    # Extract input tensors
    expand_x = inputs[0]
    expert_tokens = inputs[1]
    token_acc_table = inputs[2]
    weight_gate_upper = inputs[3]
    weight_down_proj = inputs[4]
    ffn_out = outputs[0]
    
    # Define the computation graph
    def inside_main_function():
        """Inner function to encapsulate kernel logic for automatic variable cleanup."""
        # Get number of experts
        expert_num = expert_tokens.shape[0]
        
        # Reshape weights to 2D for efficient processing
        w1_2d_shape = (weight_gate_upper.shape[0] * weight_gate_upper.shape[1], weight_gate_upper.shape[2])
        w2_2d_shape = (weight_down_proj.shape[0] * weight_down_proj.shape[1], weight_down_proj.shape[2])
        
        for _ in pypto.loop(0, 1, 1, name="LOOP_RESHAPE", idx_name="reshape_inplace_1"):
            w1_2d = pypto.reshape(weight_gate_upper, w1_2d_shape, inplace=True)
            w2_2d = pypto.reshape(weight_down_proj, w2_2d_shape, inplace=True)
        
        # Loop over experts
        for exp_idx in pypto.loop(0, expert_num, 1, name="LOOP_FFN_L0", idx_name="exp_idx"):
            def loop_expert(exp_idx):
                """Process one expert."""
                # Get token count for current expert
                token_num = expert_tokens[exp_idx]
                
                # Calculate number of iterations needed (process LOOP_BASE tokens per iteration)
                exp_loop_times = (token_num + LOOP_BASE - 1) / LOOP_BASE
                
                # Loop over token batches for current expert
                for token_loop_idx in pypto.loop(
                    0, exp_loop_times, 1, name="LOOP_FFN_L1", idx_name="token_loop_idx"):
                    def loop_token(exp_idx, token_loop_idx):
                        """Process one token batch for current expert."""
                        expert_infer_base(
                            ExpertInferConfig(
                                exp_idx=exp_idx,
                                token_loop_idx=token_loop_idx,
                                loop_base=LOOP_BASE,
                                expand_x=expand_x,
                                expert_tokens=expert_tokens,
                                token_acc_table=token_acc_table,
                                weight_gate_upper=w1_2d,
                                weight_down_proj=w2_2d,
                                ffn_out=ffn_out,
                                vec_tile_shape=VEC_TILE_SHAPE,
                                cube_tile_shape=CUBE_TILE_SHAPE,
                            )
                        )
                    loop_token(exp_idx, token_loop_idx)
            
            loop_expert(exp_idx)
    
    inside_main_function()


def test_qwen3_ffn():
    """
    Test MoE FFN implementation against PyTorch reference.
    """
    print("=" * 60)
    print("Test: MoE FFN (Qwen3)")
    print("=" * 60)
    
    # Configuration
    dtype = torch.bfloat16
    batch_size = 2
    seq_len = 1
    intermediate_size = 768
    hidden_size = 2048
    per_expert_num = 16
    topk = 8
    
    device_id = torch.npu.current_device()
    
    # Generate test inputs
    inputs_list = gen_input(
        batch_size, seq_len, topk, per_expert_num,
        hidden_size, intermediate_size, dtype, device_id
    )
    
    # Execute PyPTO kernel
    inputs = [inputs_list[0], inputs_list[1], inputs_list[2], inputs_list[3], inputs_list[4]]
    outputs = [inputs_list[5]]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    moe_ffn(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()
    
    # Compute reference using PyTorch
    golden = ffn_golden_torch(
        topk,
        inputs_list[0],
        inputs_list[1],
        inputs_list[3],
        inputs_list[4]
    )
    
    # Verify results
    assert_allclose(
        np.array(inputs_list[5].cpu().flatten().tolist()),
        np.array(golden.cpu().flatten().tolist()),
        rtol=0.005,
        atol=0.005
    )
    
    print(f"Input shape: {inputs_list[0].shape}")
    print(f"Output shape: {inputs_list[5].shape}")
    print(f"Number of experts: {per_expert_num}")
    print("✓ MoE FFN test passed")


def main():
    """Run MoE FFN example.
    
    Usage:
        python ffn_mlp.py          # Run example
        python ffn_mlp.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO MoE FFN Example (Qwen3)",
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
            'name': 'MoE FFN',
            'description': 'Qwen3 Feed-Forward Network with MoE support',
            'function': test_qwen3_ffn,
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
    print("PyPTO MoE FFN Example (Qwen3)")
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
