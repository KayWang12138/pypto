#!/usr/bin/env python3
"""
Performance profiling script for QAT asymmetric quantization operator.
Generates swimlane graphs and performance traces.
"""
import os
import pypto
import torch
import torch_npu
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from qat_asymmetric import qat_asymmetric_golden

def create_profiled_kernel(orig_shape, num_groups, group_size, bit):
    total_elements = num_groups * group_size
    n_levels = 2 ** (bit - 1)
    shift = 0.5
    neg_clip_val = -0.99
    clip_val = 0.99
    eps = 1e-4

    @pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU},
        debug_options={"runtime_debug_mode": 1}
    )
    def qat_asymmetric_kernel(
        weight: pypto.Tensor((total_elements,), pypto.DT_FP32),
        scale: pypto.Tensor((num_groups,), pypto.DT_FP32),
        offset: pypto.Tensor((num_groups,), pypto.DT_FP32),
    ) -> pypto.Tensor((total_elements,), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(32, 32)
        
        protected_scale = pypto.maximum(scale, eps)
        alpha = pypto.mul(protected_scale, n_levels)
        
        weight_2d = pypto.reshape(weight, [num_groups, group_size])
        
        offset_2d = pypto.reshape(offset, [num_groups, 1])
        offset_expanded = pypto.expand_clone(offset_2d, [num_groups, group_size])
        
        alpha_2d = pypto.reshape(alpha, [num_groups, 1])
        alpha_expanded = pypto.expand_clone(alpha_2d, [num_groups, group_size])
        
        weight_shifted = pypto.sub(weight_2d, offset_expanded)
        
        weight_norm = pypto.div(weight_shifted, alpha_expanded)
        weight_clipped = pypto.clip(weight_norm, neg_clip_val, clip_val)
        
        weight_scaled = pypto.mul(weight_clipped, n_levels)
        weight_shifted2 = pypto.sub(weight_scaled, shift)
        
        weight_rounded = pypto.round(weight_shifted2, decimals=0)
        
        weight_unshifted = pypto.add(weight_rounded, shift)
        weight_denorm = pypto.div(weight_unshifted, n_levels)
        
        weight_rescaled = pypto.mul(weight_denorm, alpha_expanded)
        output_2d = pypto.add(weight_rescaled, offset_expanded)
        
        output = pypto.reshape(output_2d, [total_elements])
        
        return output

    return qat_asymmetric_kernel

def main():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("ERROR: TILE_FWK_DEVICE_ID not set")
        return
    device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
    torch.npu.set_device(device_id)

    if 'PTO_TILE_LIB_CODE_PATH' not in os.environ:
        print("ERROR: PTO_TILE_LIB_CODE_PATH not set")
        return

    shape = (256, 256)
    group_size = 128
    bit = 4
    total_elements = shape[0] * shape[1]
    num_groups = total_elements // group_size
    device = f'npu:{device_id}'

    print("Creating kernel with performance profiling enabled...")
    kernel = create_profiled_kernel(shape, num_groups, group_size, bit)

    print("Generating test data...")
    torch.manual_seed(42)
    weight_torch = torch.randn(shape, dtype=torch.float32, device=device)
    scale_torch = torch.rand(num_groups, dtype=torch.float32, device=device) * 0.1 + 0.01
    offset_torch = torch.randn(num_groups, dtype=torch.float32, device=device) * 0.1

    weight_flat = weight_torch.view(-1)

    print("Running kernel with profiling...")
    output_flat = kernel(weight_flat, scale_torch, offset_torch)

    print("Verifying correctness...")
    output_torch = output_flat.view(shape)
    expected = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    max_diff = (output_torch - expected).abs().max().item()
    print(f"Max error: {max_diff:.6f}")

    print("\nPerformance profiling data generated in output/ directory")
    print("Check for:")
    print("  - merged_swimlane.json")
    print("  - machine_runtime_operator_trace.json")
    print("  - bubble_analysis.log")

if __name__ == "__main__":
    main()