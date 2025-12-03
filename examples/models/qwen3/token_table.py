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
Token Accumulation Table Example for Qwen3 MoE Model

This example demonstrates how to compute token accumulation tables for MoE models using PyPTO:
- Computing cumulative token counts per expert
- Dynamic expert number support
- Efficient prefix sum computation

This is a utility function used in MoE architectures to track which tokens
belong to which experts, enabling efficient expert routing.
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


@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_unaligned": True}
)
def get_token_acc_table(inputs: list, outputs: list):
    """
    PyPTO implementation of token accumulation table computation.
    
    This function computes the cumulative sum of token counts per expert,
    which is used to determine token offsets for each expert in MoE models.
    
    Parameters
    ----------
    inputs : list
        List containing [expert_tokens]
    outputs : list
        List containing [expert_offset]
    """
    expert_tokens = inputs[0]
    expert_offset = outputs[0]
    
    # Mark expert dimension as dynamic
    pypto.mark_dynamic(expert_tokens, 0)
    
    # Define the computation graph
    with pypto.function("GET_TOKEN_TABLE", [expert_tokens], [expert_offset]):
        def inside_main_function():
            """Inner function to encapsulate kernel logic for automatic variable cleanup."""
            expert_num = expert_tokens.shape[0]
            pypto.set_vec_tile_shapes(32)
            
            # Initialize first element to 0
            for _ in pypto.loop(0, 1, 1, name="LOOP_init", idx_name="idx"):
                def loop_for_init_offset():
                    """Initialize offset table."""
                    pypto.set_vec_tile_shapes(32)
                    tmp = pypto.full([32], 0, pypto.DT_INT32)
                    pypto.assemble(tmp, [0], expert_offset)
                loop_for_init_offset()
            
            # Compute cumulative sum for remaining experts
            for exp_idx in pypto.loop(1, expert_num, 1, 
                                      name="LOOP_expert", idx_name="exp_idx", submit_before_loop=True):
                def loop_for_offset(exp_idx):
                    """Compute offset for expert at index exp_idx."""
                    pypto.set_vec_tile_shapes(32)
                    
                    # Create view of tokens up to current expert
                    view_shape = [pypto.min(exp_idx, expert_num)]
                    tmp_view = pypto.view(
                        expert_tokens,
                        [16],
                        [0],
                        valid_shape=view_shape
                    )
                    
                    # Cast to FP32 for sum computation
                    tmp_cast = pypto.cast(tmp_view, pypto.DT_FP32)
                    
                    # Compute sum: sum of tokens from expert 0 to exp_idx-1
                    tmp_acc = pypto.sum(tmp_cast, dim=-1, keepdim=True)
                    
                    # Cast back to INT32
                    tmp_int = pypto.cast(tmp_acc, pypto.DT_INT32)
                    
                    # Store result at position exp_idx
                    pypto.assemble(tmp_int, [exp_idx], expert_offset)
                
                loop_for_offset(exp_idx)
        
        inside_main_function()


def get_token_acc_table_golden(expert_tokens: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of token accumulation table.
    
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


def test_token_acc_table():
    """
    Test token accumulation table implementation against PyTorch reference.
    """
    print("=" * 60)
    print("Test: Token Accumulation Table")
    print("=" * 60)
    
    # Configuration
    batch_size = 16
    per_expert_num = 8
    
    # Get current device ID (set in main)
    device_id = torch.npu.current_device()
    
    # Prepare test data
    np.random.seed(0)
    expert_tokens = torch.randint(
        0, batch_size,
        (per_expert_num,),
        dtype=torch.int32,
        device=f'npu:{device_id}'
    )
    expert_offset = torch.zeros_like(expert_tokens, device=f'npu:{device_id}')
    
    # Execute PyPTO kernel
    inputs = [expert_tokens]
    outputs = [expert_offset]
    get_token_acc_table(inputs, outputs)
    pypto.runtime._device_synchronize()
    
    # Compute reference using PyTorch
    token_acc_table_tensor = get_token_acc_table_golden(expert_tokens)
    
    # Verify results
    assert_allclose(
        np.array(expert_offset.cpu().flatten().tolist()),
        np.array(token_acc_table_tensor.cpu().flatten().tolist()),
        rtol=0.005,
        atol=0.005
    )
    
    print(f"Expert tokens: {expert_tokens.cpu().tolist()}")
    print(f"Token accumulation table: {expert_offset.cpu().tolist()}")
    print("✓ Token accumulation table test passed")


def main():
    """Run token accumulation table example.
    
    Usage:
        python token_table.py          # Run example
        python token_table.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Token Accumulation Table Example (Qwen3 MoE)",
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
            'name': 'Token Accumulation Table',
            'description': 'Compute token accumulation table for MoE models',
            'function': test_token_acc_table,
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
    print("PyPTO Token Accumulation Table Example (Qwen3 MoE)")
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
