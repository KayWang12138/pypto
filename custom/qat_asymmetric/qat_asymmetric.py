#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
QAT (Quantization-Aware Training) Asymmetric Quantization Operator

This module implements Enhanced LSQ+ asymmetric quantization for weight quantization.
It supports group-wise quantization with learnable scale and offset parameters.

Mathematical Formula:
    n_levels = 2^(bit-1)
    shift = 0.5
    weight = weight - offset
    alpha = scale * n_levels
    weight = clamp(weight / alpha, -clip_val, clip_val) * n_levels - shift
    weight = round(weight)  # STE forward
    weight = (weight + shift) / n_levels
    weight = weight * alpha + offset

Key Features:
    - Group-wise quantization support
    - Asymmetric quantization with offset
    - Configurable bit-width (4, 8, etc.)
    - STE (Straight-Through Estimator) compatible forward pass
"""
import os
import pypto
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose


# ===== Golden Function (PyTorch Reference) =====
def qat_asymmetric_golden(weight, scale, offset, group_size, bit, eps=1e-4, clip_val=0.99):
    """PyTorch reference implementation for Enhanced LSQ+ asymmetric quantization."""
    eps_tensor = torch.tensor(eps, device=scale.device).float()
    scale = torch.where(scale > eps_tensor, scale, eps_tensor)
    
    orig_shape = weight.shape
    num_groups = weight.numel() // group_size
    
    weight = weight.view(num_groups, group_size)
    
    n_levels = 2 ** (bit - 1)
    shift = 0.5

    offset_expanded = offset.unsqueeze(1)
    weight = weight - offset_expanded
    alpha = scale * n_levels
    alpha_expanded = alpha.unsqueeze(1)

    # Quantization and de-quantize with STE
    weight = torch.clamp(weight / alpha_expanded, -clip_val, clip_val) * n_levels - shift
    weight = (weight.round() - weight).detach() + weight
    weight = (weight + shift) / n_levels
    weight = weight * alpha_expanded + offset_expanded
    
    # Reshape back to original shape
    weight = weight.view(orig_shape)
    
    return weight


# ===== JIT Kernel Implementation =====
def create_qat_asymmetric_kernel(weight_shape, num_groups, group_size, bit=4, 
                                  eps=1e-4, clip_val=0.99, run_mode="npu"):
    """Create JIT-compiled QAT asymmetric quantization kernel.

    Args:
        weight_shape: Original weight tensor shape (will be flattened internally)
        num_groups: Number of quantization groups
        group_size: Number of elements per group
        bit: Quantization bit-width (default: 4)
        eps: Minimum scale threshold (default: 1e-4)
        clip_val: Clipping value (default: 0.99)
        run_mode: Execution mode - "npu" or "sim" (default: "npu")

    Returns:
        JIT-compiled kernel function
    """
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")

    # Compute total elements
    total_elements = num_groups * group_size
    
    # Pre-compute constants
    n_levels = 2 ** (bit - 1)
    shift = 0.5
    neg_clip_val = -clip_val

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def qat_asymmetric_kernel(
        weight: pypto.Tensor((total_elements,), pypto.DT_FP32),
        scale: pypto.Tensor((num_groups,), pypto.DT_FP32),
        offset: pypto.Tensor((num_groups,), pypto.DT_FP32),
    ) -> pypto.Tensor((total_elements,), pypto.DT_FP32):
        """
        QAT Asymmetric Quantization Kernel
        
        Process:
        1. Protect scale from being too small
        2. Reshape weight to (num_groups, group_size)
        3. Apply asymmetric quantization
        4. Reshape back to original shape
        """
        # Set tile shapes for vector operations
        # Using larger tile for better NPU utilization
        pypto.set_vec_tile_shapes(32, 32)
        
        # Step 1: Scale protection (avoid scale <= eps)
        protected_scale = pypto.maximum(scale, eps)
        
        # Step 2: Compute alpha = scale * n_levels
        alpha = pypto.mul(protected_scale, n_levels)
        
        # Step 3: Reshape weight to (num_groups, group_size) for group-wise operations
        weight_2d = pypto.reshape(weight, [num_groups, group_size])
        
        # Step 4: Expand offset and alpha for broadcasting
        # offset: (num_groups,) -> (num_groups, 1) -> (num_groups, group_size)
        offset_2d = pypto.reshape(offset, [num_groups, 1])
        offset_expanded = pypto.expand_clone(offset_2d, [num_groups, group_size])
        
        # alpha: (num_groups,) -> (num_groups, 1) -> (num_groups, group_size)
        alpha_2d = pypto.reshape(alpha, [num_groups, 1])
        alpha_expanded = pypto.expand_clone(alpha_2d, [num_groups, group_size])
        
        # Step 5: Subtract offset
        weight_shifted = pypto.sub(weight_2d, offset_expanded)
        
        # Step 6: Normalize by alpha and apply clipping
        weight_norm = pypto.div(weight_shifted, alpha_expanded)
        weight_clipped = pypto.clip(weight_norm, neg_clip_val, clip_val)
        
        # Step 7: Scale to quantization levels and apply shift
        weight_scaled = pypto.mul(weight_clipped, n_levels)
        weight_shifted2 = pypto.sub(weight_scaled, shift)
        
        # Step 8: Round (STE forward pass - just rounding)
        weight_rounded = pypto.round(weight_shifted2, decimals=0)
        
        # Step 9: De-quantize
        weight_unshifted = pypto.add(weight_rounded, shift)
        weight_denorm = pypto.div(weight_unshifted, n_levels)
        
        # Step 10: Rescale by alpha and add offset back
        weight_rescaled = pypto.mul(weight_denorm, alpha_expanded)
        output_2d = pypto.add(weight_rescaled, offset_expanded)
        
        # Step 10: Reshape back to original shape
        output = pypto.reshape(output_2d, [total_elements])
        
        return output

    return qat_asymmetric_kernel


# ===== Test Cases =====
def test_qat_asymmetric_basic(device_id=None, run_mode="npu"):
    """Level 0: Small tensor test - Basic functionality verification."""
    print("Test: Basic functionality (small tensor)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test parameters
    orig_shape = (16, 16)  # 256 elements
    group_size = 32
    bit = 4
    num_groups = 256 // 32  # 8 groups
    
    # Generate test data
    weight_torch = torch.randn(orig_shape, dtype=torch.float32, device=device)
    scale_torch = torch.rand(num_groups, dtype=torch.float32, device=device) * 0.1 + 0.01
    offset_torch = torch.randn(num_groups, dtype=torch.float32, device=device) * 0.1
    
    # Flatten weight for kernel
    weight_flat = weight_torch.view(-1)
    
    # Run PyPTO kernel
    kernel = create_qat_asymmetric_kernel(orig_shape, num_groups, group_size, bit, run_mode=run_mode)
    output_flat = kernel(weight_flat, scale_torch, offset_torch)
    output_torch = output_flat.view(orig_shape)
    
    # Run golden reference
    expected = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    
    # Verify
    max_diff = (output_torch - expected).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Output shape: {output_torch.shape}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("  PASSED")


def test_qat_asymmetric_level1(device_id=None, run_mode="npu"):
    """Level 1: Typical size (1K elements) - Realistic use case."""
    print("Test: Typical size (1K elements)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test parameters - typical weight size
    orig_shape = (128, 64)  # 8192 elements
    group_size = 128
    bit = 4
    num_groups = 8192 // 128  # 64 groups
    
    # Generate test data
    weight_torch = torch.randn(orig_shape, dtype=torch.float32, device=device)
    scale_torch = torch.rand(num_groups, dtype=torch.float32, device=device) * 0.1 + 0.01
    offset_torch = torch.randn(num_groups, dtype=torch.float32, device=device) * 0.1
    
    # Flatten weight for kernel
    weight_flat = weight_torch.view(-1)
    
    # Run PyPTO kernel
    kernel = create_qat_asymmetric_kernel(orig_shape, num_groups, group_size, bit, run_mode=run_mode)
    output_flat = kernel(weight_flat, scale_torch, offset_torch)
    output_torch = output_flat.view(orig_shape)
    
    # Run golden reference
    expected = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    
    # Verify
    max_diff = (output_torch - expected).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("  PASSED")


def test_qat_asymmetric_edge_cases(device_id=None, run_mode="npu"):
    """Level 2: Edge cases - Boundary conditions."""
    print("Test: Edge cases")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    orig_shape = (8, 8)  # 64 elements
    group_size = 16
    bit = 4
    num_groups = 64 // 16  # 4 groups

    # Test case 1: Very small scale (should be protected by eps)
    print("  Test case 1: Very small scale")
    weight_torch = torch.randn(orig_shape, dtype=torch.float32, device=device)
    scale_torch = torch.full((num_groups,), 1e-6, dtype=torch.float32, device=device)
    offset_torch = torch.zeros(num_groups, dtype=torch.float32, device=device)
    
    weight_flat = weight_torch.view(-1)
    kernel = create_qat_asymmetric_kernel(orig_shape, num_groups, group_size, bit, run_mode=run_mode)
    output_flat = kernel(weight_flat, scale_torch, offset_torch)
    output_torch = output_flat.view(orig_shape)
    
    expected = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    
    max_diff = (output_torch - expected).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("    PASSED: Small scale case")

    # Test case 2: Zero offset
    print("  Test case 2: Zero offset")
    weight_torch = torch.randn(orig_shape, dtype=torch.float32, device=device)
    scale_torch = torch.rand(num_groups, dtype=torch.float32, device=device) * 0.1 + 0.01
    offset_torch = torch.zeros(num_groups, dtype=torch.float32, device=device)
    
    weight_flat = weight_torch.view(-1)
    output_flat = kernel(weight_flat, scale_torch, offset_torch)
    output_torch = output_flat.view(orig_shape)
    
    expected = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    
    max_diff = (output_torch - expected).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("    PASSED: Zero offset case")

    # Test case 3: Zero weight
    print("  Test case 3: Zero weight")
    weight_torch = torch.zeros(orig_shape, dtype=torch.float32, device=device)
    scale_torch = torch.rand(num_groups, dtype=torch.float32, device=device) * 0.1 + 0.01
    offset_torch = torch.randn(num_groups, dtype=torch.float32, device=device) * 0.1
    
    weight_flat = weight_torch.view(-1)
    output_flat = kernel(weight_flat, scale_torch, offset_torch)
    output_torch = output_flat.view(orig_shape)
    
    expected = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    
    max_diff = (output_torch - expected).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("    PASSED: Zero weight case")

    # Test case 4: Different bit-widths
    print("  Test case 4: Different bit-widths (8-bit)")
    weight_torch = torch.randn(orig_shape, dtype=torch.float32, device=device)
    scale_torch = torch.rand(num_groups, dtype=torch.float32, device=device) * 0.1 + 0.01
    offset_torch = torch.randn(num_groups, dtype=torch.float32, device=device) * 0.1
    
    bit_8 = 8
    weight_flat = weight_torch.view(-1)
    kernel_8bit = create_qat_asymmetric_kernel(orig_shape, num_groups, group_size, bit_8, run_mode=run_mode)
    output_flat = kernel_8bit(weight_flat, scale_torch, offset_torch)
    output_torch = output_flat.view(orig_shape)
    
    expected = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit_8)
    
    max_diff = (output_torch - expected).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("    PASSED: 8-bit quantization case")


def test_qat_asymmetric_large(device_id=None, run_mode="npu"):
    """Level 3: Large tensor for performance - Verify NPU performance."""
    print("Test: Large tensor (performance test)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Large tensor for performance test
    orig_shape = (1024, 1024)  # 1M elements
    group_size = 128
    bit = 4
    num_groups = 1024 * 1024 // 128  # 8192 groups
    
    # Generate test data
    weight_torch = torch.randn(orig_shape, dtype=torch.float32, device=device)
    scale_torch = torch.rand(num_groups, dtype=torch.float32, device=device) * 0.1 + 0.01
    offset_torch = torch.randn(num_groups, dtype=torch.float32, device=device) * 0.1
    
    # Flatten weight for kernel
    weight_flat = weight_torch.view(-1)
    
    # Run PyPTO kernel
    kernel = create_qat_asymmetric_kernel(orig_shape, num_groups, group_size, bit, run_mode=run_mode)
    output_flat = kernel(weight_flat, scale_torch, offset_torch)
    output_torch = output_flat.view(orig_shape)
    
    # Run golden reference
    expected = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    
    # Verify
    max_diff = (output_torch - expected).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Total elements: {orig_shape[0] * orig_shape[1]}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("  PASSED")


def test_qat_asymmetric_various_shapes(device_id=None, run_mode="npu"):
    """Level 4: Various input shapes - Verify flexibility."""
    print("Test: Various input shapes")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test different shapes
    test_cases = [
        ((128,), 32, 4),      # 1D tensor
        ((64, 64), 64, 4),    # 2D square
        ((32, 128), 32, 4),   # 2D rectangular
        ((16, 32, 64), 64, 4), # 3D tensor
    ]
    
    for orig_shape, group_size, bit in test_cases:
        total_elements = 1
        for dim in orig_shape:
            total_elements *= dim
        num_groups = total_elements // group_size
        
        print(f"  Shape: {orig_shape}, group_size: {group_size}, num_groups: {num_groups}")
        
        weight_torch = torch.randn(orig_shape, dtype=torch.float32, device=device)
        scale_torch = torch.rand(num_groups, dtype=torch.float32, device=device) * 0.1 + 0.01
        offset_torch = torch.randn(num_groups, dtype=torch.float32, device=device) * 0.1
        
        weight_flat = weight_torch.view(-1)
        kernel = create_qat_asymmetric_kernel(orig_shape, num_groups, group_size, bit, run_mode=run_mode)
        output_flat = kernel(weight_flat, scale_torch, offset_torch)
        output_torch = output_flat.view(orig_shape)
        
        expected = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
        
        max_diff = (output_torch - expected).abs().max().item()
        print(f"    Max difference: {max_diff:.6f}")
        assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=1e-2, atol=1e-2)
        print(f"    PASSED")


# ===== Main Test Runner =====
def main():
    """Run all test cases."""
    import argparse

    parser = argparse.ArgumentParser(description="QAT Asymmetric Quantization Operator Tests")
    parser.add_argument('--run_mode', type=str, default="npu", choices=["npu", "sim"],
                        help="Execution mode: npu (hardware) or sim (simulation)")
    parser.add_argument('--test_level', type=int, default=-1,
                        help="Test level: 0=basic, 1=typical, 2=edge, 3=large, 4=shapes, -1=all")
    args = parser.parse_args()

    # Get device ID
    device_id = None
    if args.run_mode == "npu":
        if 'TILE_FWK_DEVICE_ID' not in os.environ:
            print("ERROR: TILE_FWK_DEVICE_ID not set. Please set environment variable:")
            print("  export TILE_FWK_DEVICE_ID=0")
            print("\nOr use simulation mode:")
            print("  python3 qat_asymmetric.py --run_mode sim")
            return
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        import torch_npu
        torch.npu.set_device(device_id)
        print(f"Using NPU device: {device_id}")
    else:
        print("Using simulation mode (CPU)")

    print("=" * 60)
    print("QAT Asymmetric Quantization Operator Tests")
    print("Enhanced LSQ+ Group-wise Quantization")
    print("=" * 60)
    print()

    # Run tests based on level
    if args.test_level == -1 or args.test_level == 0:
        test_qat_asymmetric_basic(device_id, args.run_mode)
        print()

    if args.test_level == -1 or args.test_level == 1:
        test_qat_asymmetric_level1(device_id, args.run_mode)
        print()

    if args.test_level == -1 or args.test_level == 2:
        test_qat_asymmetric_edge_cases(device_id, args.run_mode)
        print()

    if args.test_level == -1 or args.test_level == 3:
        test_qat_asymmetric_large(device_id, args.run_mode)
        print()

    if args.test_level == -1 or args.test_level == 4:
        test_qat_asymmetric_various_shapes(device_id, args.run_mode)
        print()

    print("=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()