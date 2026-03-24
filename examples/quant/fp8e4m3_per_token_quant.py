#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
FP8E4M3 Per-Token Quantization Example for PyPTO

This example demonstrates per-token quantization to FP8E4M3 format.
Per-token quantization computes a scale for each token (row) independently.

Input: BF16 tensor of shape (m, n)
Output: FP8E4M3 quantized tensor of shape (m, n) and FP8E8M0 scale tensor of shape (m, 1)
"""
import os
import sys
import argparse
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


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


def create_quant_kernel(shape: tuple, run_mode: str = "npu"):
    """
    Create per-token FP8E4M3 quantization kernel.
    
    Args:
        shape: Input tensor shape (m, n)
        run_mode: Run mode (npu or sim)
    
    Returns:
        JIT compiled quantization kernel
    """
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    m, n = shape
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def quant_kernel(
        x: pypto.Tensor([m, n], pypto.DT_BF16),
        out_quant: pypto.Tensor([m, n], pypto.DT_FP8_E4M3),
        out_scale: pypto.Tensor([m, 1], pypto.DT_FP8_E8M0),
    ):
        pypto.set_vec_tile_shapes(m, n, 1, 1)
        
        x_abs = pypto.abs(x)
        scale = pypto.max(x_abs, axis=1, keepdims=True)
        scale = pypto.maximum(scale, 1e-6)
        
        out_scale[:] = scale
        out_quant[:] = x / scale

    return quant_kernel


def golden_per_token_quantize(x: torch.Tensor) -> tuple:
    """
    Golden reference implementation of per-token FP8E4M3 quantization.
    
    Args:
        x: Input tensor of shape (m, n) with dtype torch.bfloat16
    
    Returns:
        tuple: (quantized_tensor, scale)
            - quantized_tensor: Quantized tensor in FP8E4M3 format, shape (m, n)
            - scale: Scale tensor in FP8E8M0 format, shape (m, 1)
    """
    m, n = x.shape
    
    x_abs = torch.abs(x)
    scale = torch.max(x_abs, dim=1, keepdim=True)[0]
    
    scale = torch.clamp(scale, min=1e-6)
    
    x_scaled = x / scale
    
    quantized = x_scaled.to(torch.float8_e4m3fn)
    scale_fp8 = scale.to(torch.float8_e8m0fnu)
    
    return quantized, scale_fp8


def test_per_token_quantize(device_id=None, run_mode: str = "npu") -> None:
    """
    Test per-token FP8E4M3 quantization.
    
    Args:
        device_id: NPU device ID
        run_mode: Run mode (npu or sim)
    """
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (16, 128)
    
    input_data = torch.randn(shape, dtype=torch.bfloat16, device=device)
    
    golden_quantized, golden_scale = golden_per_token_quantize(input_data)
    
    output_quant = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    output_scale = torch.empty((shape[0], 1), dtype=torch.float8_e8m0fnu, device=device)
    
    create_quant_kernel(shape, run_mode)(input_data, output_quant, output_scale)
    
    print(f"Input shape: {input_data.shape}, dtype: {input_data.dtype}")
    print(f"Output quant shape: {output_quant.shape}, dtype: {output_quant.dtype}")
    print(f"Output scale shape: {output_scale.shape}, dtype: {output_scale.dtype}")
    
    dequant_output = output_quant.to(torch.float32) * output_scale.to(torch.float32)
    dequant_golden = golden_quantized.to(torch.float32) * golden_scale.to(torch.float32)
    
    max_diff = torch.max(torch.abs(dequant_output.cpu() - dequant_golden.cpu())).item()
    mean_diff = torch.mean(torch.abs(dequant_output.cpu() - dequant_golden.cpu())).item()
    
    print(f"Max difference (dequantized): {max_diff:.6f}")
    print(f"Mean difference (dequantized): {mean_diff:.6f}")
    
    if run_mode == "npu":
        assert_allclose(dequant_output.cpu().numpy(), dequant_golden.cpu().numpy(), rtol=1e-2, atol=1e-2)
    
    print("✓ Per-token FP8E4M3 quantization example passed")
    print()


def main():
    """Run quantization example.

    Usage:
        python fp8e4m3_per_token_quant.py          # Run example
        python fp8e4m3_per_token_quant.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO FP8E4M3 Per-Token Quantization Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s quant::test_per_token_quantize
            Run the per-token quantization example
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
        "quant::test_per_token_quantize": {
            'name': 'per_token_quantize',
            'description': 'per-token FP8E4M3 quantization',
            'function': test_per_token_quantize
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
    print("PyPTO FP8E44M3 Per-Token Quantization Example")
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
            print("All quantization tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
