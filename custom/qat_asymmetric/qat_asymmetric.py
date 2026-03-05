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
    """PyTorch reference implementation for Enhanced LSQ+ asymmetric quantization (BF16 I/O, FP32 compute).
    
    Args:
        weight: Input weight tensor (N, M) in BF16
        scale: Quantization scale tensor (num_groups, 1) in BF16
        offset: Quantization offset tensor (num_groups, 1) in BF16
        group_size: Number of elements per group (default: 128)
        bit: Quantization bit-width (2, 3, or 4)
    """
    weight_fp32 = weight.float()
    scale_fp32 = scale.float()
    offset_fp32 = offset.float()
    
    eps_tensor = torch.tensor(eps, device=scale_fp32.device, dtype=torch.float32)
    protected_scale = torch.where(scale_fp32 > eps_tensor, scale_fp32, eps_tensor)
    
    orig_shape = weight.shape
    num_groups = weight.numel() // group_size
    
    weight_2d = weight_fp32.view(num_groups, group_size)
    
    n_levels = 2 ** (bit - 1)
    shift = 0.5

    weight_shifted = weight_2d - offset_fp32
    alpha = protected_scale * n_levels

    weight_clipped = torch.clamp(weight_shifted / alpha, -clip_val, clip_val) * n_levels - shift
    weight_rounded = (weight_clipped.round() - weight_clipped).detach() + weight_clipped
    weight_unshifted = weight_rounded + shift
    weight_denorm = weight_unshifted / n_levels
    output_2d = weight_denorm * alpha + offset_fp32
    
    output = output_2d.view(orig_shape)
    weight_denorm_out = weight_denorm.view(orig_shape)
    alpha_expanded_out = alpha.expand(-1, group_size).contiguous().view(orig_shape)
    
    return output.to(torch.bfloat16), weight_denorm_out.to(torch.bfloat16), alpha_expanded_out.to(torch.bfloat16)


# ===== JIT Kernel Implementation =====
def create_qat_asymmetric_kernel(weight_shape, num_groups, group_size, bit=4, 
                                  eps=1e-4, clip_val=0.99, run_mode="npu"):
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")

    total_elements = num_groups * group_size
    n_levels = 2 ** (bit - 1)
    shift = 0.5
    neg_clip_val = -clip_val

    runtime_opts = {"run_mode": mode}
    if run_mode == "npu":
        runtime_opts.update({
            "stitch_function_inner_memory": 512,
            "stitch_function_outcast_memory": 512,
            "stitch_function_num_initial": 128,
            "stitch_function_max_num": 128,
            "stitch_function_num_step": 20
        })

    @pypto.frontend.jit(runtime_options=runtime_opts)
    def qat_asymmetric_kernel(
        weight: pypto.Tensor(weight_shape, pypto.DT_BF16),
        scale: pypto.Tensor((num_groups, 1), pypto.DT_BF16),
        offset: pypto.Tensor((num_groups, 1), pypto.DT_BF16),
    ) -> (
        pypto.Tensor(weight_shape, pypto.DT_BF16),
        pypto.Tensor(weight_shape, pypto.DT_BF16),
        pypto.Tensor(weight_shape, pypto.DT_BF16),
    ):
        pypto.set_vec_tile_shapes(128, 128)
        
        weight_fp32 = pypto.cast(weight, pypto.DT_FP32)
        scale_fp32 = pypto.cast(scale, pypto.DT_FP32)
        offset_fp32 = pypto.cast(offset, pypto.DT_FP32)
        
        protected_scale = pypto.maximum(scale_fp32, eps)
        alpha = pypto.mul(protected_scale, n_levels)
        
        weight_2d = pypto.reshape(weight_fp32, [num_groups, group_size])
        
        offset_expanded = pypto.expand_clone(offset_fp32, [num_groups, group_size])
        alpha_expanded = pypto.expand_clone(alpha, [num_groups, group_size])
        
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
        
        output = pypto.reshape(output_2d, list(weight_shape))
        weight_denorm_out = pypto.reshape(weight_denorm, list(weight_shape))
        alpha_expanded_out = pypto.reshape(alpha_expanded, list(weight_shape))
        
        output_bf16 = pypto.cast(output, pypto.DT_BF16)
        weight_denorm_bf16 = pypto.cast(weight_denorm_out, pypto.DT_BF16)
        alpha_expanded_bf16 = pypto.cast(alpha_expanded_out, pypto.DT_BF16)
        
        return output_bf16, weight_denorm_bf16, alpha_expanded_bf16

    return qat_asymmetric_kernel


# ===== Test Cases =====
def test_qat_asymmetric_basic(device_id=None, run_mode="npu"):
    print("Test: Basic functionality (small tensor)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    weight_shape = (16, 32)
    group_size = 128
    bit = 4
    num_groups = (16 * 32) // group_size
    
    weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device)).to(torch.bfloat16)
    scale_torch = (torch.rand(num_groups, 1, dtype=torch.float32, device=device) * 0.1 + 0.01).to(torch.bfloat16)
    offset_torch = (torch.randn(num_groups, 1, dtype=torch.float32, device=device) * 0.1).to(torch.bfloat16)
    
    kernel = create_qat_asymmetric_kernel(weight_shape, num_groups, group_size, bit, run_mode=run_mode)
    output_torch, weight_denorm_torch, alpha_expanded_torch = kernel(weight_torch, scale_torch, offset_torch)
    
    expected_out, expected_denorm, expected_alpha = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    
    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("  PASSED")


def test_qat_asymmetric_level1(device_id=None, run_mode="npu"):
    print("Test: Typical size (1024, 2048)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    weight_shape = (1024, 2048)
    group_size = 128
    bit = 4
    num_groups = (1024 * 2048) // group_size
    
    weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device)).to(torch.bfloat16)
    scale_torch = (torch.rand(num_groups, 1, dtype=torch.float32, device=device) * 0.1 + 0.01).to(torch.bfloat16)
    offset_torch = (torch.randn(num_groups, 1, dtype=torch.float32, device=device) * 0.1).to(torch.bfloat16)
    
    kernel = create_qat_asymmetric_kernel(weight_shape, num_groups, group_size, bit, run_mode=run_mode)
    output_torch, _, _ = kernel(weight_torch, scale_torch, offset_torch)
    
    expected_out, _, _ = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    
    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("  PASSED")


def test_qat_asymmetric_edge_cases(device_id=None, run_mode="npu"):
    print("Test: Edge cases")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    weight_shape = (128, 256)
    group_size = 128
    bit = 4
    num_groups = (128 * 256) // group_size

    print("  Test case 1: Very small scale")
    weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device)).to(torch.bfloat16)
    scale_torch = torch.full((num_groups, 1), 1e-6, dtype=torch.bfloat16, device=device)
    offset_torch = torch.zeros((num_groups, 1), dtype=torch.bfloat16, device=device)
    
    kernel = create_qat_asymmetric_kernel(weight_shape, num_groups, group_size, bit, run_mode=run_mode)
    output_torch, _, _ = kernel(weight_torch, scale_torch, offset_torch)
    
    expected_out, _, _ = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    
    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("    PASSED: Small scale case")

    print("  Test case 2: Zero offset")
    weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device)).to(torch.bfloat16)
    scale_torch = (torch.rand(num_groups, 1, dtype=torch.float32, device=device) * 0.1 + 0.01).to(torch.bfloat16)
    offset_torch = torch.zeros((num_groups, 1), dtype=torch.bfloat16, device=device)
    
    output_torch, _, _ = kernel(weight_torch, scale_torch, offset_torch)
    expected_out, _, _ = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    
    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("    PASSED: Zero offset case")

    print("  Test case 3: Zero weight")
    weight_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
    scale_torch = (torch.rand(num_groups, 1, dtype=torch.float32, device=device) * 0.1 + 0.01).to(torch.bfloat16)
    offset_torch = (torch.randn(num_groups, 1, dtype=torch.float32, device=device) * 0.1).to(torch.bfloat16)
    
    output_torch, _, _ = kernel(weight_torch, scale_torch, offset_torch)
    expected_out, _, _ = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    
    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("    PASSED: Zero weight case")

    print("  Test case 4: Different bit-widths (2, 3)")
    weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device)).to(torch.bfloat16)
    scale_torch = (torch.rand(num_groups, 1, dtype=torch.float32, device=device) * 0.1 + 0.01).to(torch.bfloat16)
    offset_torch = (torch.randn(num_groups, 1, dtype=torch.float32, device=device) * 0.1).to(torch.bfloat16)
    
    for test_bit in [2, 3]:
        kernel_bit = create_qat_asymmetric_kernel(weight_shape, num_groups, group_size, test_bit, run_mode=run_mode)
        output_torch, _, _ = kernel_bit(weight_torch, scale_torch, offset_torch)
        expected_out, _, _ = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, test_bit)
        
        max_diff = (output_torch - expected_out).abs().max().item()
        print(f"    Bit={test_bit} max difference: {max_diff:.6f}")
        assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("    PASSED: Different bit-widths case")


def test_qat_asymmetric_large(device_id=None, run_mode="npu"):
    print("Test: Large tensor (4096, 2048)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    weight_shape = (4096, 2048)
    group_size = 128
    bit = 4
    num_groups = (4096 * 2048) // group_size
    
    weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device)).to(torch.bfloat16)
    scale_torch = (torch.rand(num_groups, 1, dtype=torch.float32, device=device) * 0.1 + 0.01).to(torch.bfloat16)
    offset_torch = (torch.randn(num_groups, 1, dtype=torch.float32, device=device) * 0.1).to(torch.bfloat16)
    
    kernel = create_qat_asymmetric_kernel(weight_shape, num_groups, group_size, bit, run_mode=run_mode)
    output_torch, weight_denorm_torch, alpha_expanded_torch = kernel(weight_torch, scale_torch, offset_torch)
    
    expected_out, _, _ = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    
    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=1e-2, atol=1e-2)
    print("  PASSED")


def test_qat_asymmetric_level4(device_id=None, run_mode="npu"):
    """Test performance benchmark shapes for accuracy."""
    print("Test: Performance benchmark shapes (Level 4)")
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    group_size = 128
    bit = 4
    
    # 4 performance test cases
    perf_shapes = [
        (1024, 2048),
        (768, 2048),
        (2048, 768),
        (4096, 2048),
    ]
    
    for weight_shape in perf_shapes:
        print(f"  Testing shape: {weight_shape}")
        num_groups = (weight_shape[0] * weight_shape[1]) // group_size
        
        weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device)).to(torch.bfloat16)
        scale_torch = (torch.rand(num_groups, 1, dtype=torch.float32, device=device) * 0.1 + 0.01).to(torch.bfloat16)
        offset_torch = (torch.randn(num_groups, 1, dtype=torch.float32, device=device) * 0.1).to(torch.bfloat16)
        
        kernel = create_qat_asymmetric_kernel(weight_shape, num_groups, group_size, bit, run_mode=run_mode)
        output_torch, _, _ = kernel(weight_torch, scale_torch, offset_torch)
        
        expected_out, _, _ = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
        
        max_diff = (output_torch - expected_out).abs().max().item()
        print(f"    Max difference: {max_diff:.6f}")
        assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=1e-2, atol=1e-2)
        print(f"    PASSED")


# ===== Main Test Runner =====
def main():
    import argparse

    parser = argparse.ArgumentParser(description="QAT Asymmetric Quantization Operator Tests")
    parser.add_argument('--run_mode', type=str, default="npu", choices=["npu", "sim"])
    parser.add_argument('--test_level', type=int, default=-1)
    args = parser.parse_args()

    device_id = None
    if args.run_mode == "npu":
        if 'TILE_FWK_DEVICE_ID' not in os.environ:
            print("ERROR: TILE_FWK_DEVICE_ID not set")
            print("  export TILE_FWK_DEVICE_ID=0")
            return
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        torch.npu.set_device(device_id)
        print(f"Using NPU device: {device_id}")
    else:
        print("Using simulation mode (CPU)")

    print("=" * 60)
    print("QAT Asymmetric Quantization Operator Tests")
    print("weight: (N, M), scale/offset: (num_groups, 1)")
    print("group_size: 128, bit: 2, 3, 4")
    print("=" * 60)
    print()

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
        test_qat_asymmetric_level4(device_id, args.run_mode)
        print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()