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
Expert Selection Example for Qwen3 MoE Model

This example demonstrates expert selection in Mixture-of-Experts (MoE) models using PyPTO:
- Softmax computation over expert logits
- Top-K expert selection
- Optional weight renormalization
- Dynamic batch size support
- Tiling for efficient execution

This is a key component in MoE architectures, where tokens are routed to different
experts based on their logits.
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
def select_experts(inputs: list, outputs: list, renormalize_flag: bool):
    """
    PyPTO implementation of expert selection with dynamic batch size support.
    
    This function performs:
    1. Softmax over expert logits
    2. Top-K expert selection
    3. Optional weight renormalization
    
    Parameters
    ----------
    inputs : list
        List containing [router_logits]
    outputs : list
        List containing [topk_ids, topk_weights]
    renormalize_flag : bool
        Whether to renormalize top-k weights
    """
    # Extract input and output tensors
    logits_input = inputs[0]
    ids_k = outputs[0]
    weight_k = outputs[1]
    
    # Mark batch dimension (axis 0) as dynamic
    pypto.mark_dynamic(logits_input, 0)
    pypto.mark_dynamic(ids_k, 0)
    pypto.mark_dynamic(weight_k, 0)
    
    # Get tensor shapes
    bs = logits_input.shape[0]  # Dynamic batch size
    ne = logits_input.shape[1]  # Static number of experts
    idx_k_shape = ids_k.shape
    topk = idx_k_shape[1]  # Static top-k value
    
    # Define tiling configuration
    view_shape = (1024, ne)
    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]
    
    # Define the computation graph
    with pypto.function("MOEGATE", [logits_input], [ids_k, weight_k]):
        def inside_select_experts():
            """Inner function to encapsulate kernel logic for automatic variable cleanup."""
            # Loop over dynamic batch axis
            for bs_idx in pypto.loop(bs_loop, name="LOOP_MOEGATE_L0", idx_name="bs_idx", unroll_list={1}):
                def bs_loop_func(bs_idx):
                    """Process one batch tile."""
                    # Create view for current batch tile
                    tile_logits = pypto.view(
                        logits_input,
                        view_shape,
                        [bs_idx * view_shape[0], 0],
                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), ne]
                    )
                    
                    # Configure tiling: use full UB but don't exceed UB size
                    pypto.set_vec_tile_shapes(64, 128)
                    
                    # Step 1: Cast to FP32 for softmax computation
                    tile_logits_fp32 = pypto.cast(tile_logits, pypto.DT_FP32)
                    
                    # Step 2: Apply softmax
                    softmax_out = pypto.softmax(tile_logits_fp32, dim=-1)
                    
                    # Step 3: Select top-K experts
                    topk_weight_tmp, topk_ids_tmp = pypto.topk(softmax_out, topk, dim=-1, largest=True)
                    
                    # Step 4: Optional weight renormalization
                    pypto.set_vec_tile_shapes(128, 8)
                    if pypto.cond(pypto.symbolic_scalar(1 if renormalize_flag else 0)):
                        # Renormalize: divide by sum of top-k weights
                        denominator = pypto.sum(topk_weight_tmp, dim=-1, keepdim=True)
                        topk_weight2 = pypto.div(topk_weight_tmp, denominator)
                    else:
                        # No renormalization
                        topk_weight2 = topk_weight_tmp
                    
                    # Step 5: Cast weights back to input dtype
                    topk_weight2_f16 = pypto.cast(topk_weight2, weight_k.dtype)
                    
                    # Assemble results back to output tensors
                    weight_k[
                        bs_idx * pypto.symbolic_scalar(view_shape[0]):,
                        pypto.symbolic_scalar(0):
                    ] = topk_weight2_f16
                    ids_k[
                        bs_idx * pypto.symbolic_scalar(view_shape[0]):,
                        pypto.symbolic_scalar(0):
                    ] = topk_ids_tmp
                
                bs_loop_func(bs_idx)
        
        inside_select_experts()


def test_select_experts():
    """
    Test expert selection implementation against PyTorch reference.
    
    Tests with different batch sizes to verify dynamic shape support.
    """
    print("=" * 60)
    print("Test: Expert Selection (Dynamic Batch)")
    print("=" * 60)
    
    # Configuration
    num_experts = 128
    top_k = 8
    renormalize = True
    
    device_id = torch.npu.current_device()
    
    # Test with different batch sizes
    for i in range(2):
        if i == 1:
            batch_size = 1  # Test with small batch size
        else:
            batch_size = 4959
        
        print(f"\nTesting with batch size: {batch_size}")
        
        # Prepare test data
        np.random.seed(0)
        router_logits = torch.rand(
            (batch_size, num_experts),
            dtype=torch.float16,
            device=f'npu:{device_id}'
        )
        topk_weights = torch.zeros(
            (batch_size, top_k),
            dtype=torch.float16,
            device=f'npu:{device_id}'
        )
        topk_ids = torch.zeros(
            (batch_size, top_k),
            dtype=torch.int32,
            device=f'npu:{device_id}'
        )
        
        # Execute PyPTO kernel
        inputs = [router_logits]
        outputs = [topk_ids, topk_weights]
        select_experts(inputs, outputs, renormalize)
        pypto.runtime._device_synchronize()
        
        # Compute reference using PyTorch
        result = torch.softmax(router_logits.to(torch.float32), dim=-1)
        topk_weight_tensor, topk_ids_tensor = torch.topk(
            result, top_k, dim=-1, largest=True, sorted=True
        )
        topk_ids_tensor_list = topk_ids_tensor.flatten().tolist()
        
        # Renormalize if needed
        denominator_g = torch.sum(topk_weight_tensor, dim=-1, keepdim=True)
        topk_weight_2_tensor = torch.div(topk_weight_tensor, denominator_g).to(torch.float16)
        topk_weight_2_tensor_list = topk_weight_2_tensor.flatten().tolist()
        
        # Verify results
        assert_allclose(
            np.array(topk_ids.cpu().flatten().tolist()),
            np.array(topk_ids_tensor_list),
            rtol=5e-3,
            atol=5e-3
        )
        
        assert_allclose(
            np.array(topk_weights.cpu().flatten().tolist()),
            np.array(topk_weight_2_tensor_list),
            rtol=5e-3,
            atol=5e-3
        )
        
        print(f"  ✓ Batch size {batch_size} passed")
    
    print("\n✓ All expert selection tests passed")


def main():
    """Run expert selection example.
    
    Usage:
        python select_experts.py          # Run example
        python select_experts.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Expert Selection Example (Qwen3 MoE)",
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
            'name': 'Expert Selection',
            'description': 'Expert selection with top-K and renormalization',
            'function': test_select_experts,
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
    print("PyPTO Expert Selection Example (Qwen3 MoE)")
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
