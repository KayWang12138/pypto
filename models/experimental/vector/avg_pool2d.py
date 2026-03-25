#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE; IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER; EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
AvgPool2d Example for PyPTO

This example demonstrates average pooling 2D operation.
"""
import os
import sys
import argparse
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose
import torch.nn.functional as F


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
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


def avg_pool_2d(shape, kernel_size, stride=None, padding_mode=None, run_mode="npu", dynamic=True):
    batch_size, channels, in_h, in_w = shape
    k_h, k_w = kernel_size

    if stride is None:
        stride = kernel_size
    s_h, s_w = stride

    if padding_mode.upper() == 'VALID':
        t_pad = b_pad = l_pad = r_pad = 0
    elif padding_mode.upper() == 'SAME':
        out_h = (in_h + s_h - 1) // s_h
        out_w = (in_w + s_w - 1) // s_w
        pad_h = max(0, (out_h - 1) * s_h + k_h - in_h)
        pad_w = max(0, (out_w - 1) * s_w + k_w - in_w)
        t_pad = pad_h // 2
        b_pad = pad_h - t_pad
        l_pad = pad_w // 2
        r_pad = pad_w - l_pad
    else:
        raise ValueError(f"Invalid padding_mode: {padding_mode}. Must be 'VALID' or 'SAME'")

    out_h = (in_h + t_pad + b_pad - k_h) // s_h + 1
    out_w = (in_w + l_pad + r_pad - k_w) // s_w + 1

    if dynamic:
        batch_size = pypto.frontend.dynamic("batch_size")
        channels = pypto.frontend.dynamic("channels")

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(pass_options={"vec_nbuffer_setting": {-1: 2, 0: 8}}, 
                        runtime_options={"run_mode": mode, "stitch_function_num_initial": 128, 
                        "stitch_function_outcast_memory": 1024, "stitch_function_inner_memory": 1024}, 
                        debug_options=dict(runtime_debug_mode=1, compile_debug_mode=1))
    def avg_pool_2d_kernel(
        input_tensor: pypto.Tensor((batch_size, channels, in_h, in_w), pypto.DT_FP32),
        output_result: pypto.Tensor((batch_size, channels, out_h, out_w), pypto.DT_FP32),
    ):
        bc_total = batch_size * channels
        pypto.set_vec_tile_shapes(16, 16, 4, 128)
        input_reshaped = pypto.reshape(input_tensor, [batch_size * channels, in_h, in_w], inplace=True)
        output_tmp = pypto.tensor((bc_total, out_h, out_w), pypto.DT_FP32)

        for bc_idx, unroll_length in pypto.loop_unroll(0, bc_total, 1, name="LOOP_BC", 
                                                       idx_name="bc_idx", unroll_list=[16, 4, 2, 1]):
            input_cur = input_reshaped[bc_idx: bc_idx + unroll_length, :, :]
            for oh in range(out_h):
                h_start = oh * s_h - t_pad
                h_end = h_start + k_h
                h_start_clamped = max(h_start, 0)
                h_end_clamped = min(h_end, in_h)

                cur_k_h = h_end_clamped - h_start_clamped

                pypto.set_vec_tile_shapes(16, 16, 128)
                if cur_k_h > 0:
                    input_single_row = input_cur[:, h_start_clamped:h_end_clamped, :]
                else:
                    input_single_row = None

                input_single_row_1 = pypto.sum(input_single_row, 1, keepdim=True)
                
                for ow in range(out_w):
                    w_start = ow * s_w - l_pad
                    w_end = w_start + k_w
                    w_start_clamped = max(w_start, 0)
                    w_end_clamped = min(w_end, in_w)

                    cur_k_w = w_end_clamped - w_start_clamped

                    if cur_k_h > 0 and cur_k_w > 0:
                        window = input_single_row_1[:, :, w_start_clamped:w_end_clamped]
                        sum_val = pypto.sum(window, dim=2, keepdim=True)
                        avg_val = sum_val / (k_h * k_w)
                        pypto.set_vec_tile_shapes(unroll_length, 4, 128)
                        pypto.assemble(avg_val, [bc_idx, oh, ow], output_tmp)
                    else:
                        zero_val = pypto.zeros([unroll_length, 1, 1], dtype=pypto.DT_FP32)
                        pypto.set_vec_tile_shapes(unroll_length, 4, 128)
                        pypto.assemble(zero_val, [bc_idx, oh, ow], output_tmp)
            pypto.set_vec_tile_shapes(unroll_length, 4, 128)
            output_result.move(pypto.reshape(output_tmp, [batch_size, channels, out_h, out_w], inplace=True))
    return avg_pool_2d_kernel


def avg_pool_2d_golden(x, kernel_size, stride, padding_mode):
    """
    Compute golden output using numpy (equivalent to tf.compat.v1.nn.avg_pool).
    
    Args:
        x: Input tensor (input_n, input_c, input_h, input_w)
        kernel_size: (k_h, k_w)
        stride: (s_h, s_w)
        padding_mode: 'VALID' or 'SAME'
    
    Returns:
        Output tensor (input_n, input_c, out_h, out_w)
    """
    input_n, input_c, input_h, input_w = x.shape
    k_h, k_w = kernel_size
    s_h, s_w = stride
    
    if padding_mode == 'VALID':
        t_pad = b_pad = l_pad = r_pad = 0
    elif padding_mode == 'SAME':
        out_h = (input_h + s_h - 1) // s_h
        out_w = (input_w + s_w - 1) // s_w
        pad_h = max(0, (out_h - 1) * s_h + k_h - input_h)
        pad_w = max(0, (out_w - 1) * s_w + k_w - input_w)
        t_pad = pad_h // 2
        b_pad = pad_h - t_pad
        l_pad = pad_w // 2
        r_pad = pad_w - l_pad
    else:
        raise ValueError(f"Invalid padding_mode: {padding_mode}")
    
    out_h = (input_h + t_pad + b_pad - k_h) // s_h + 1
    out_w = (input_w + l_pad + r_pad - k_w) // s_w + 1
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (t_pad, b_pad), (l_pad, r_pad)), mode='constant', constant_values=0)
    
    output = np.zeros((input_n, input_c, out_h, out_w), dtype=x.dtype)
    
    for n in range(input_n):
        for c in range(input_c):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * s_h
                    w_start = ow * s_w
                    h_end = h_start + k_h
                    w_end = w_start + k_w
                    
                    window = x_padded[n, c, h_start:h_end, w_start:w_end]
                    output[n, c, oh, ow] = np.mean(window)
    
    return output


def test_avg_pool_2d(device_id=None, run_mode: str = "npu", dynamic=True) -> None:
    """Test avg_pool_2d implementation with VALID padding."""
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (8, 8, 16, 16)
    kernel_size = (3, 3)
    stride = (2, 2)
    padding_mode = 'VALID'

    np.random.seed(42)
    x_np = np.random.randn(*shape).astype(np.float32)
    x = torch.from_numpy(x_np).to(device)

    batch_size, channels, in_h, in_w = shape
    k_h, k_w = kernel_size

    if stride is None:
        stride = kernel_size
    s_h, s_w = stride

    if padding_mode.upper() == 'VALID':
        t_pad = b_pad = l_pad = r_pad = 0
    elif padding_mode.upper() == 'SAME':
        out_h = (in_h + s_h - 1) // s_h
        out_w = (in_w + s_w - 1) // s_w
        pad_h = max(0, (out_h - 1) * s_h + k_h - in_h)
        pad_w = max(0, (out_w - 1) * s_w + k_w - in_w)
        t_pad = pad_h // 2
        b_pad = pad_h - t_pad
        l_pad = pad_w // 2
        r_pad = pad_w - l_pad
    else:
        raise ValueError(f"Invalid padding_mode: {padding_mode}. Must be 'VALID' or 'SAME'")

    out_h = (in_h + t_pad + b_pad - k_h) // s_h + 1
    out_w = (in_w + l_pad + r_pad - k_w) // s_w + 1
    
    y = torch.empty((batch_size, channels,out_h, out_w), dtype=torch.float32, device=device)
    avg_pool_2d(x.shape, kernel_size, stride, padding_mode=padding_mode, run_mode=run_mode, dynamic=dynamic)(x, y)
    y = y.cpu().numpy()

    golden_output = avg_pool_2d_golden(x_np, kernel_size, stride, padding_mode)

    max_diff = np.abs(y - golden_output).max()
    
    if run_mode == "npu":
        assert_allclose(y, golden_output, rtol=1e-3, atol=1e-3)
    print("✓ test_avg_pool_2d (VALID) example passed")
    print()


def test_avg_pool_2d_with_padding(device_id=None, run_mode: str = "npu", dynamic=True) -> None:
    """Test avg_pool_2d with SAME padding."""
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (2, 3, 6, 6)
    kernel_size = (2, 2)
    stride = (2, 2)
    padding_mode = 'SAME'

    np.random.seed(42)
    x_np = np.random.randn(*shape).astype(np.float32)
    x = torch.from_numpy(x_np).to(device)

    batch_size, channels, in_h, in_w = shape
    k_h, k_w = kernel_size

    if stride is None:
        stride = kernel_size
    s_h, s_w = stride

    if padding_mode.upper() == 'VALID':
        t_pad = b_pad = l_pad = r_pad = 0
    elif padding_mode.upper() == 'SAME':
        out_h = (in_h + s_h - 1) // s_h
        out_w = (in_w + s_w - 1) // s_w
        pad_h = max(0, (out_h - 1) * s_h + k_h - in_h)
        pad_w = max(0, (out_w - 1) * s_w + k_w - in_w)
        t_pad = pad_h // 2
        b_pad = pad_h - t_pad
        l_pad = pad_w // 2
        r_pad = pad_w - l_pad
    else:
        raise ValueError(f"Invalid padding_mode: {padding_mode}. Must be 'VALID' or 'SAME'")

    out_h = (in_h + t_pad + b_pad - k_h) // s_h + 1
    out_w = (in_w + l_pad + r_pad - k_w) // s_w + 1
    
    y = torch.empty((batch_size, channels,out_h, out_w), dtype=torch.float32, device=device)
    avg_pool_2d(x.shape, kernel_size, stride, padding_mode=padding_mode, run_mode=run_mode, dynamic=dynamic)(x, y)
    y = y.cpu().numpy()

    golden_output = avg_pool_2d_golden(x_np, kernel_size, stride, padding_mode)

    max_diff = np.abs(y - golden_output).max()
    
    if run_mode == "npu":
        assert_allclose(y, golden_output, rtol=1e-3, atol=1e-3)
    else:
        assert_allclose(y, golden_output, rtol=1e-3, atol=1e-3)
    print("✓ test_avg_pool_2d_with_padding (SAME) example passed")
    print()


def main():
    """Run avg_pool_2d example.

    Usage:
        python avg_pool_2d.py          # Run example
        python avg_pool_2d.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO avg_pool_2d Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s avg_pool_2d::test_avg_pool_2d
            Run test_avg_pool_2d example
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs='?',
        help='Example ID to run. If not specified, all examples will run.'
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
        "avg_pool_2d::test_avg_pool_2d": {
            'name': 'avg_pool_2d',
            'description': 'avg_pool_2d implementation',
            'function': test_avg_pool_2d
        },
        "avg_pool_2d::test_avg_pool_2d_with_padding": {
            'name': 'avg_pool_2d_with_padding',
            'description': 'avg_pool_2d with padding',
            'function': test_avg_pool_2d_with_padding
        }
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  ID: {ex_id}")
            print(f"     name: {ex_info['name']}")
            print(f"     description: {ex_info['description']}\n")
        return

    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO avg_pool_2d Example")
    print("=" * 60 + "\n")

    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        example = examples.get(args.example_id)
        if example is None:
            raise ValueError(f"Invalid example ID: {args.example_id}")
        examples_to_run = [(args.example_id, example)]
    else:
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
            print("All avg_pool_2d tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
