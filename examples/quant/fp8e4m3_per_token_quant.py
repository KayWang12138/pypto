#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
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

Reference: models/deepseek_v32_exp/deepseekv32_lightning_indexer_quant.py
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
    
    This implementation follows the same pattern as INT8 quantization in
    models/deepseek_v32_exp/deepseekv32_lightning_indexer_quant.py
    
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


def compare_fp8_tensors(npu_quant, cpu_quant, npu_scale, cpu_scale, rtol=1e-2, atol=1e-2):
    """
    Compare FP8 tensors from NPU and CPU.
    
    Since FP8 tensors have limited precision, we compare them by:
    1. Converting both to float32
    2. Computing relative and absolute differences
    3. Checking if differences are within tolerance
    
    Args:
        npu_quant: NPU quantized tensor (FP8E4M3)
        cpu_quant: CPU quantized tensor (FP8E4M3)
        npu_scale: NPU scale tensor (FP8E8M0)
        cpu_scale: CPU scale tensor (FP8E8M0)
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        dict: Comparison results with statistics
    """
    npu_quant_f32 = npu_quant.to(torch.float32)
    cpu_quant_f32 = cpu_quant.to(torch.float32)
    
    npu_scale_f32 = npu_scale.to(torch.float32)
    cpu_scale_f32 = cpu_scale.to(torch.float32)
    
    quant_diff = torch.abs(npu_quant_f32 - cpu_quant_f32)
    scale_diff = torch.abs(npu_scale_f32 - cpu_scale_f32)
    
    quant_max_diff = quant_diff.max().item()
    quant_mean_diff = quant_diff.mean().item()
    quant_max_rel_diff = (quant_diff / (torch.abs(cpu_quant_f32) + 1e-6)).max().item()
    
    scale_max_diff = scale_diff.max().item()
    scale_mean_diff = scale_diff.mean().item()
    scale_max_rel_diff = (scale_diff / (torch.abs(cpu_scale_f32) + 1e-6)).max().item()
    
    results = {
        'quant_max_abs_diff': quant_max_diff,
        'quant_mean_abs_diff': quant_mean_diff,
        'quant_max_rel_diff': quant_max_rel_diff,
        'scale_max_abs_diff': scale_max_diff,
        'scale_mean_abs_diff': scale_mean_diff,
        'scale_max_rel_diff': scale_max_rel_diff,
        'passed': (quant_max_diff <= atol or quant_max_rel_diff <= rtol) and 
                  (scale_max_diff <= atol or scale_max_rel_diff <= rtol)
    }
    
    return results


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
    
    comparison_results = compare_fp8_tensors(output_quant, golden_quantized, 
                                        output_scale, golden_scale)
    
    print(f"\nQuantized tensor comparison:")
    print(f"  Max absolute difference: {comparison_results['quant_max_abs_diff']:.6f}")
    print(f"  Mean absolute difference: {comparison_results['quant_mean_abs_diff']:.6f}")
    print(f"  Max relative difference: {comparison_results['quant_max_rel_diff']:.6f}")
    
    print(f"\nScale tensor comparison:")
    print(f"  Max absolute difference: {comparison_results['scale_max_abs_diff']:.6f}")
    print(f"  Mean absolute difference: {comparison_results['scale_mean_abs_diff']:.6f}")
    print(f"  Max relative difference: {comparison_results['scale_max_rel_diff']:.6f}")
    
    if comparison_results['passed']:
        print("\n✓ FP8 tensor comparison passed")
    else:
        print("\n✗ FP8 tensor comparison failed")
        if run_mode == "npu":
            raise AssertionError("FP8 tensor comparison failed")
    
    print()


def main():
    """Run quantization example.

    Usage:
        python fp8e4m3_per_token_quant.py          # Run example
        python fp8e4m3_per_token_quant.py --list 
