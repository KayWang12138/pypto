#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import argparse
import math
import pypto
import torch
import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("If no NPU environment is available, set --run_mode sim to run in simulation mode;")
        print("otherwise, set the environment variable TILE_FWK_DEVICE_ID.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def check_cond(cond, msg):
    if not cond:
        raise ValueError(msg)


def powers_of_2(n: int) -> set[int]:
    check_cond(n > 0, "n must be positive")
    result = set()
    power = 0
    while True:
        current = 1 << power
        if current > n:
            break
        result.add(current)
        power += 1
    return result


def create_scatter_nd_sub_kernel(target_shape: tuple, indices_shape: tuple, updates_shape: tuple, strides_shape: tuple, 
                                 run_mode: str = "npu", dynamic: bool = True):
    if dynamic:
        target_first = target_shape[0]
        target_second = target_shape[1]
        indices_first = pypto.frontend.dynamic("indices_first")
        indices_second = indices_shape[1]
    else:
        target_first = target_shape[0]
        target_second = target_shape[1]
        indices_first = indices_shape[0]
        indices_second = indices_shape[1]
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(runtime_options={"run_mode": mode, "stitch_function_num_initial": 128, 
                        "stitch_function_outcast_memory": 1024, "stitch_function_inner_memory": 1024})
    def scatter_nd_sub_kernel(
        target: pypto.Tensor((target_first, target_second), pypto.DT_FP32),
        indices: pypto.Tensor((indices_first, indices_second), pypto.DT_INT32),
        updates: pypto.Tensor((indices_first, target_second), pypto.DT_FP32),
        strides: pypto.Tensor(strides_shape, pypto.DT_INT32),
    ) -> pypto.Tensor((target_first, target_second), pypto.DT_FP32):        
        k = indices.shape[-1]

        pypto.set_vec_tile_shapes(64, 512)
        front_prod = math.prod(target.shape[:k])
        target_new_shape = [front_prod] + target.shape[k:]
        target_reshaped = pypto.reshape(target, target_new_shape, inplace=True)

        batch_shape = indices.shape[:-1]
        num_batch = math.prod(batch_shape)

        updates_flat = pypto.reshape(updates, [num_batch, *target.shape[k:]], inplace=True)

        result_reshaped = pypto.tensor(target_new_shape, pypto.DT_FP32)

        for bs_idx, tile_batch in pypto.loop_unroll(0, indices_first, 1, name="LOOP_SCATTER_ND_SUB_L0", 
                                                    idx_name="bs_idx", unroll_list=powers_of_2(512)):
            b_offset = bs_idx
            b_offset_end = bs_idx + tile_batch
            flat_indices = indices[b_offset:b_offset_end, ...] * strides
            flat_indices_fp32 = pypto.cast(flat_indices, pypto.DT_FP32)
            flat_indices_temp = pypto.sum(flat_indices_fp32, -1)
            flat_indices_int32 = pypto.cast(flat_indices_temp, pypto.DT_INT32)
            if target_reshaped.dim > 1:
                num_missing_dims = target_reshaped.dim - 1
                new_shape = flat_indices_int32.shape + [1] * num_missing_dims
                flat_indices_int32 = pypto.reshape(flat_indices_int32, new_shape, inplace=True)

                expand_shape = (tile_batch, *target_reshaped.shape[1:])
                pypto.set_vec_tile_shapes(64, 512)
                flat_indices_int32 = pypto.expand_clone(flat_indices_int32, expand_shape)
            tile_shape_dim0 = max(target_reshaped.shape[0], flat_indices_int32.shape[0])
            pypto.set_vec_tile_shapes(tile_shape_dim0, 8)

            neg_updates_flat = pypto.mul(updates_flat[b_offset:b_offset_end, ...], -1)
            pypto.scatter_(target_reshaped, 0, flat_indices_int32, neg_updates_flat, reduce='add')
        final_result = pypto.reshape(target_reshaped, target.shape, inplace=True)
        return final_result
    return scatter_nd_sub_kernel


def calculate_strides(indices: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    k = indices.shape[-1]
    dims = target.shape[:k]
    dims_reversed = list(dims[::-1])    
    dims_truncated = dims_reversed[:-1] if len(dims_reversed) > 0 else []

    tmp_list = [1] + dims_truncated
    cumulative_prod = []
    current_prod = 1
    for num in tmp_list:
        current_prod *= num
        cumulative_prod.append(current_prod)
    strides_list = cumulative_prod[::-1]
    device = indices.device
    strides = torch.tensor(strides_list, dtype=torch.int32, device=device)
    strides_shape = strides.shape
    return strides, strides_shape


def test_scatter_nd_sub(device_id: int = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    # Get current device ID (set in main)
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    # 1. prepare pypto data
    target_shape = (10, 16)
    target_max_value = 10
    target = torch.rand(target_shape, dtype=torch.float32, device=device) * target_max_value

    indices_shape = (1024, 1)
    indices_min_value = 0
    indices_max_value = target_max_value
    indices = torch.randint(
        low=indices_min_value,
        high=indices_max_value,
        size=indices_shape,
        dtype=torch.int32,
        device=device
    )

    updates_shape = (1024, 16)
    updates_max_value = 2
    updates = torch.rand(updates_shape, dtype=torch.float32, device=device) * updates_max_value

    target_shape = target.shape
    indices_shape = indices.shape
    updates_shape = updates.shape
    strides, strides_shape = calculate_strides(indices, target)
    pypto_output = create_scatter_nd_sub_kernel(target_shape, indices_shape, updates_shape, strides_shape, 
                                                run_mode)(target, indices, updates, strides)
    print("pypto output:", pypto_output)

    # 2. prepare tensorflow data
    target_tf = tf.convert_to_tensor(target.cpu().numpy())
    indices_tf = tf.convert_to_tensor(indices.cpu().numpy())
    updates_tf = tf.convert_to_tensor(updates.cpu().numpy())

    target_var = tf.compat.v1.Variable(target_tf)
    indices_var = tf.compat.v1.Variable(indices_tf)
    updates_var = tf.compat.v1.Variable(updates_tf)
    tf_output = tf.compat.v1.scatter_nd_sub(target_var, indices_var, updates_var)
    print("tensorflow output:", tf_output)

    # 3. compare pypto vs tensorflow output
    max_diff = np.abs(pypto_output.cpu().numpy() - tf_output.numpy()).max().item()
    if run_mode == "npu":
        print(f"Max difference from Tensorflow: {max_diff:.6f}")
        assert max_diff < 1e-1, "Result mismatch!"
    print("✓ Combined operations completed successfully")
    print("finished")


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO scatter_nd_sub Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scatter_nd_sub::test_scatter_nd_sub
            Run the scatter_nd_sub::test_scatter_nd_sub example
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs='?',
        help='Example ID to run (1). If not specified, the example will run.'
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

    # Define available examples
    examples = {
        "scatter_nd_sub::test_scatter_nd_sub": {
            'name': 'scatter_nd_sub',
            'description': 'scatter_nd_sub implementation',
            'function': test_scatter_nd_sub
        }
    }

    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  ID: {ex_id}")
            print(f"     name: {ex_info['name']}")
            print(f"     description: {ex_info['description']}\n")
        return

    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO test_scatter_nd_sub Example")
    print("=" * 60 + "\n")

    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        # Run single example
        example = examples.get(args.example_id)
        if example is None:
            raise ValueError(f"Invalid example ID: {args.example_id}")
        examples_to_run = [(args.example_id, example)]
    else:
        # Run all examples
        examples_to_run = list(examples.items())

    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running examples that require NPU hardware...")
        print("(Make sure CANN environment is configured and NPU is available)\n")

    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id, args.run_mode)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All scatter_nd_sub tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()