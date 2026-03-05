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

import os
import sys
import argparse
import pypto
import torch
import torch.nn.functional as F
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


def create_maxpool2d_kernel(
    input_shape: tuple,
    kernel_size: int = 2,
    stride: int = 2,
    dtype: pypto.DataType = pypto.DT_FP16,
    run_mode: str = "npu"
):
    """
    MaxPool2D kernel with fixed kernel size and stride.
    
    Args:
        input_shape: (batch, channel, H, W)
        kernel_size: Pooling window size (square)
        stride: Stride of the pooling operation
        dtype: Data type
        run_mode: "npu" or "sim"
    """
    batch, channel, H, W = input_shape
    output_H = (H - kernel_size) // stride + 1
    output_W = (W - kernel_size) // stride + 1
    
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def maxpool2d_kernel(
        input_tensor: pypto.Tensor(input_shape, dtype),
    ) -> pypto.Tensor((batch, channel, output_H, output_W), dtype):
        align_size = 16
        pypto.set_vec_tile_shapes(1, 1, kernel_size, align_size)
        
        output_tensor = pypto.tensor((batch, channel, output_H, output_W), dtype)
        
        for b_idx in pypto.loop(batch, name="batch_loop"):
            for c_idx in pypto.loop(channel, name="channel_loop"):
                for h_out_idx in pypto.loop(output_H, name="h_out_loop"):
                    for w_out_idx in pypto.loop(output_W, name="w_out_loop"):
                        h_start = h_out_idx * stride
                        w_start = w_out_idx * stride
                        
                        window = pypto.view(
                            input_tensor,
                            [1, 1, kernel_size, align_size],
                            [b_idx, c_idx, h_start, w_start],
                            valid_shape=[1, 1, kernel_size, kernel_size]
                        )
                        
                        max_w = pypto.amax(window, dim=-1, keepdim=True)
                        max_val = pypto.amax(max_w, dim=-2, keepdim=True)
                        
                        result = pypto.view(max_val, [1, 1, 1, 1], [0, 0, 0, 0])
                        pypto.assemble(result, [b_idx, c_idx, h_out_idx, w_out_idx], output_tensor)
        
        return output_tensor
    
    return maxpool2d_kernel


def test_basic_case(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test Case 1: Basic MaxPool2D (4x4 -> 2x2 with kernel=2, stride=2)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    batch, channel = 1, 1
    H, W = 4, 4
    kernel_size, stride = 2, 2
    dtype_torch = torch.float16
    dtype_pypto = pypto.DT_FP16
    
    input_torch = torch.randn(batch, channel, H, W, dtype=dtype_torch, device=device)
    
    golden = F.max_pool2d(input_torch, kernel_size=kernel_size, stride=stride)
    
    kernel = create_maxpool2d_kernel(
        (batch, channel, H, W),
        kernel_size,
        stride,
        dtype_pypto,
        run_mode
    )
    output = kernel(input_torch)
    
    print(f"Input shape: {input_torch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Golden shape: {golden.shape}")
    
    if run_mode == "npu":
        max_diff = (output - golden).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        
        assert_allclose(
            output.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3,
            atol=1e-3
        )
        print("✓ Test case 1 passed!")
    print()


def test_medium_case(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test Case 2: Medium Case (8x8 -> 4x4 with kernel=2, stride=2)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    batch, channel = 2, 3
    H, W = 8, 8
    kernel_size, stride = 2, 2
    dtype_torch = torch.float16
    dtype_pypto = pypto.DT_FP16
    
    input_torch = torch.randn(batch, channel, H, W, dtype=dtype_torch, device=device)
    
    golden = F.max_pool2d(input_torch, kernel_size=kernel_size, stride=stride)
    
    kernel = create_maxpool2d_kernel(
        (batch, channel, H, W),
        kernel_size,
        stride,
        dtype_pypto,
        run_mode
    )
    output = kernel(input_torch)
    
    print(f"Input shape: {input_torch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Golden shape: {golden.shape}")
    
    if run_mode == "npu":
        max_diff = (output - golden).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        
        assert_allclose(
            output.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3,
            atol=1e-3
        )
        print("✓ Test case 2 passed!")
    print()


def test_bf16_case(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test Case 3: BF16 Data Type (6x6 -> 3x3 with kernel=2, stride=2)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    batch, channel = 1, 2
    H, W = 6, 6
    kernel_size, stride = 2, 2
    dtype_torch = torch.bfloat16
    dtype_pypto = pypto.DT_BF16
    
    input_torch = torch.randn(batch, channel, H, W, dtype=dtype_torch, device=device)
    
    golden = F.max_pool2d(input_torch, kernel_size=kernel_size, stride=stride)
    
    kernel = create_maxpool2d_kernel(
        (batch, channel, H, W),
        kernel_size,
        stride,
        dtype_pypto,
        run_mode
    )
    output = kernel(input_torch)
    
    print(f"Input shape: {input_torch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Golden shape: {golden.shape}")
    
    if run_mode == "npu":
        max_diff = (output - golden).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        
        assert_allclose(
            output.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-2,
            atol=1e-2
        )
        print("✓ Test case 3 passed!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO MaxPool2D (Fixed Kernel) Example",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'test_id',
        type=str,
        nargs='?',
        help='Test ID to run (1-3). If not specified, all tests will run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available tests and exit'
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        nargs='?',
        default="npu",
        choices=["npu", "sim"],
        help='Run mode: npu or sim'
    )
    
    args = parser.parse_args()
    
    tests = {
        '1': {
            'name': 'Basic Case',
            'description': '4x4 -> 2x2 with kernel=2, stride=2',
            'function': test_basic_case
        },
        '2': {
            'name': 'Medium Case',
            'description': '8x8 -> 4x4 with kernel=2, stride=2',
            'function': test_medium_case
        },
        '3': {
            'name': 'BF16 Data Type',
            'description': '6x6 -> 3x3 with kernel=2, stride=2, BF16',
            'function': test_bf16_case
        }
    }
    
    if args.list:
        print("\n" + "=" * 60)
        print("Available Tests")
        print("=" * 60 + "\n")
        for test_id, test_info in sorted(tests.items()):
            print(f"  ID: {test_id}")
            print(f"     name: {test_info['name']}")
            print(f"     description: {test_info['description']}\n")
        return
    
    if args.test_id is not None:
        if args.test_id not in tests:
            print(f"ERROR: Invalid test ID: {args.test_id}")
            print(f"Valid test IDs are: {', '.join(map(str, sorted(tests.keys())))}")
            print("\nUse --list to see all available tests.")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("PyPTO MaxPool2D (Fixed Kernel) Example")
    print("=" * 60 + "\n")
    
    device_id = None
    tests_to_run = []
    
    if args.test_id is not None:
        test = tests.get(args.test_id)
        if test is None:
            raise ValueError(f"Invalid test ID: {args.test_id}")
        tests_to_run = [(args.test_id, test)]
    else:
        tests_to_run = list(tests.items())
    
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running tests that require NPU hardware...")
        print("(Make sure CANN environment is configured and NPU is available)\n")
    
    try:
        for test_id, test_info in tests_to_run:
            print(f"Running Test {test_id}: {test_info['name']}")
            test_info['function'](device_id, args.run_mode)
        
        if len(tests_to_run) > 1:
            print("=" * 60)
            print("All MaxPool2D tests passed!")
            print("=" * 60)
    
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
