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
def asymmetric_qat_golden(weight, scale, offset, group_size, bit, eps=1e-4, clip_val=0.99):
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

    return output_2d.view(orig_shape).to(torch.bfloat16)
 
import torch
 
def asymmetric_qat_backward_golden(grad_output, weight, scale, offset, group_size, bit, eps=1e-4, clip_val=0.99):
    """
    PyTorch reference backward implementation for Enhanced LSQ+ asymmetric quantization.
    Computes gradients without forward caching by recomputing necessary intermediates.
    
    Args:
        grad_output: Gradient of the loss w.r.t. the output of the forward pass (N, M)
        weight: Original input weight tensor (N, M)
        scale: Quantization scale tensor (num_groups, 1)
        offset: Quantization offset tensor (num_groups, 1)
        group_size: Number of elements per group (default: 128)
        bit: Quantization bit-width (2, 3, or 4)
        eps: Epsilon for numerical stability
        clip_val: Clipping value (default: 0.99)
        
    Returns:
        Tuple of gradients: (grad_weight, grad_scale, grad_offset)
    """
    # 1. 精度对齐：全部转换到 FP32 以防止梯度计算中的下溢/溢出
    grad_out_fp32 = grad_output.float()
    weight_fp32 = weight.float()
    scale_fp32 = scale.float()
    offset_fp32 = offset.float()
    
    # 2. 状态重计算 (Recompute Intermediates)
    eps_tensor = torch.tensor(eps, device=scale_fp32.device, dtype=torch.float32)
    
    # 记录 scale 是否大于 eps，用于最后 scale 梯度的掩码处理
    scale_mask = scale_fp32 > eps_tensor
    protected_scale = torch.where(scale_mask, scale_fp32, eps_tensor)
    
    orig_shape = weight.shape
    num_groups = weight.numel() // group_size
    
    weight_2d = weight_fp32.view(num_groups, group_size)
    grad_out_2d = grad_out_fp32.view(num_groups, group_size)
    
    n_levels = 2 ** (bit - 1)
    shift = 0.5
    
    alpha = protected_scale * n_levels
    weight_shifted = weight_2d - offset_fp32
    weight_scaled = weight_shifted / alpha
    
    # 计算截断掩码 (STE active region: 1 if inside clip range else 0)
    mask = (weight_scaled >= -clip_val) & (weight_scaled <= clip_val)
    mask_f32 = mask.float()
    
    # 重新计算 weight_denorm 用于 scale 的梯度
    weight_clipped = torch.clamp(weight_scaled, -clip_val, clip_val) * n_levels - shift
    weight_rounded = weight_clipped.round() # 反向传播中不需要 STE 的 detach() trick
    weight_unshifted = weight_rounded + shift
    weight_denorm = weight_unshifted / n_levels
    
    # 3. 反向梯度计算 (Backprop via LSQ+ formulas)
    
    # -> grad_weight (N, M)
    grad_weight_2d = grad_out_2d * mask_f32
    grad_weight = grad_weight_2d.view(orig_shape).to(weight.dtype)
    
    # -> grad_offset (num_groups, 1)
    grad_offset_2d = grad_out_2d * (1.0 - mask_f32)
    grad_offset = grad_offset_2d.sum(dim=1, keepdim=True).to(offset.dtype)
    
    # -> grad_scale (num_groups, 1)
    grad_alpha_2d = grad_out_2d * (weight_denorm - weight_scaled * mask_f32)
    grad_alpha = grad_alpha_2d.sum(dim=1, keepdim=True)
    
    # 只有当 scale > eps 时才会回传梯度 (通过 scale_mask)
    grad_scale = grad_alpha * n_levels * scale_mask.float()
    grad_scale = grad_scale.to(scale.dtype)
    
    return grad_weight, grad_scale, grad_offset
 
# ===== JIT Kernel Implementation =====
def create_asymmetric_qat_kernel(group_size=128, bit=4, 
                                  eps=1e-4, clip_val=0.99, run_mode="npu", unroll_list=None):
    if unroll_list is None:
        unroll_list = [512, 256, 128]
    
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")
 
    n_levels = 2 ** (bit - 1)
    shift = 0.5
    neg_clip_val = -clip_val
 
    runtime_opts = {"run_mode": mode}
    if run_mode == "npu":
        runtime_opts.update({
            "stitch_function_max_num": 128,
        })
    G = pypto.frontend.dynamic("G")
 
    @pypto.frontend.jit(runtime_options=runtime_opts)
    def asymmetric_qat_kernel(
        weight: pypto.Tensor((G, group_size), pypto.DT_BF16),
        scale: pypto.Tensor((G, 1), pypto.DT_BF16),
        offset: pypto.Tensor((G, 1), pypto.DT_BF16),
        output_bf16: pypto.Tensor((G, group_size), pypto.DT_BF16),
    ):
        num_groups = scale.shape[0]
        pypto.set_vec_tile_shapes(128, 128)
        
        for g_offset, unroll_length in pypto.loop_unroll(
            0, num_groups, 1,
            name="LOOP_GROUPS",
            idx_name="g_offset",
            unroll_list=unroll_list
        ):
            tile_groups = unroll_length
            
            weight_tile = pypto.view(weight, [tile_groups, group_size], [g_offset, 0])
            weight_fp32 = pypto.cast(weight_tile, pypto.DT_FP32)
            
            scale_tile = pypto.view(scale, [tile_groups, 1], [g_offset, 0])
            offset_tile = pypto.view(offset, [tile_groups, 1], [g_offset, 0])
            
            scale_fp32 = pypto.cast(scale_tile, pypto.DT_FP32)
            offset_fp32 = pypto.cast(offset_tile, pypto.DT_FP32)
            
            protected_scale = pypto.maximum(scale_fp32, eps)
            alpha = pypto.mul(protected_scale, n_levels)
            
            weight_shifted = pypto.sub(weight_fp32, offset_fp32)
            weight_norm = pypto.div(weight_shifted, alpha)
            weight_clipped = pypto.clip(weight_norm, neg_clip_val, clip_val)
            
            weight_scaled = pypto.mul(weight_clipped, n_levels)
            weight_shifted2 = pypto.sub(weight_scaled, shift)
            weight_rounded = pypto.round(weight_shifted2, decimals=0)
            
            weight_unshifted = pypto.add(weight_rounded, shift)
            weight_denorm = pypto.div(weight_unshifted, n_levels)
            
            weight_rescaled = pypto.mul(weight_denorm, alpha)
            output = pypto.add(weight_rescaled, offset_fp32)
            
            output_tile = pypto.cast(output, pypto.DT_BF16)
 
            pypto.assemble(output_tile, [g_offset, 0], output_bf16)
    return asymmetric_qat_kernel
 
def create_asymmetric_qat_backward_kernel(group_size=128, bit=4, 
                                          eps=1e-4, clip_val=0.99, run_mode="npu", unroll_list=None, enable_perf=True):
    if unroll_list is None:
        unroll_list = [512, 256, 128]
    
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")
 
    n_levels = 2 ** (bit - 1)
    shift = 0.5
    neg_clip_val = -clip_val
 
    runtime_opts = {"run_mode": mode}
    if run_mode == "npu":
        runtime_opts.update({
            "stitch_function_max_num": 128,
        })

    debug_opts = {"runtime_debug_mode": 1} if enable_perf else {}
    G = pypto.frontend.dynamic("G")
 
    @pypto.frontend.jit(runtime_options=runtime_opts, debug_options=debug_opts)
    def asymmetric_qat_backward_kernel(
        grad_output: pypto.Tensor((G, group_size), pypto.DT_BF16),
        weight: pypto.Tensor((G, group_size), pypto.DT_BF16),
        scale: pypto.Tensor((G, 1), pypto.DT_BF16),
        offset: pypto.Tensor((G, 1), pypto.DT_BF16),
    ) -> (pypto.Tensor((G, group_size), pypto.DT_BF16),
          pypto.Tensor((G, 1), pypto.DT_BF16),
          pypto.Tensor((G, 1), pypto.DT_BF16),
    ):
        grad_weight_out = pypto.Tensor((G, group_size), pypto.DT_BF16)
        grad_scale_out = pypto.Tensor((G, 1), pypto.DT_BF16)
        grad_offset_out = pypto.Tensor((G, 1), pypto.DT_BF16)
        num_groups = scale.shape[0]
        
        pypto.experimental.set_operation_options(combine_axis=True)
        for g_offset, unroll_length in pypto.loop_unroll(
            0, num_groups, 1,
            name="LOOP_GROUPS",
            idx_name="g_offset",
            unroll_list=unroll_list
        ):
            tile_groups = unroll_length
            pypto.set_vec_tile_shapes(128, 128)
            # --- 1. 数据加载与类型转换 (Load & Cast) ---
            grad_out_tile = pypto.view(grad_output, [tile_groups, group_size], [g_offset, 0])
            weight_tile = pypto.view(weight, [tile_groups, group_size], [g_offset, 0])
            scale_tile = pypto.view(scale, [tile_groups, 1], [g_offset, 0])
            offset_tile = pypto.view(offset, [tile_groups, 1], [g_offset, 0])
            
            grad_out_fp32 = pypto.cast(grad_out_tile, pypto.DT_FP32)
            weight_fp32 = pypto.cast(weight_tile, pypto.DT_FP32)
            scale_fp32 = pypto.cast(scale_tile, pypto.DT_FP32)
            offset_fp32 = pypto.cast(offset_tile, pypto.DT_FP32)
            
            # --- 2. 前向状态重计算 (Recompute Intermediates) ---
            protected_scale = pypto.maximum(scale_fp32, eps)
            alpha = pypto.mul(protected_scale, n_levels)
            
            weight_shifted = pypto.sub(weight_fp32, offset_fp32)
            weight_norm = pypto.div(weight_shifted, alpha)  # 对应前向的 weight_scaled，未截断
            
            # 重新计算 weight_denorm 用于求解 scale 梯度
            weight_clipped = pypto.clip(weight_norm, neg_clip_val, clip_val)
            weight_scaled = pypto.mul(weight_clipped, n_levels)
            weight_shifted2 = pypto.sub(weight_scaled, shift)
            weight_rounded = pypto.round(weight_shifted2, decimals=0)
            weight_unshifted = pypto.add(weight_rounded, shift)
            weight_denorm = pypto.div(weight_unshifted, n_levels)
            
            # --- 3. 掩码生成 (Mask Generation) ---
            # 判断元素是否在 clip 范围内 (-clip_val <= w <= clip_val)
            diff = pypto.sub(weight_norm, weight_clipped)
            abs_diff = pypto.abs(diff)
            clipped_diff = pypto.clip(abs_diff, 0.0, 1.0)
            is_out = pypto.ceil(clipped_diff)
            one = pypto.full(is_out.shape, 1.0, is_out.dtype)
            mask_f32 = pypto.sub(one, is_out)
            
            one_tileG_gs = pypto.full([tile_groups, group_size], 1.0, pypto.DT_FP32)
            inv_mask_f32 = pypto.sub(one_tileG_gs, mask_f32)
            
            # 判断 scale 是否合法（> eps），用于过滤 scale 梯度
            scale_diff = pypto.sub(scale_fp32, eps)
            diff_pos = pypto.maximum(scale_diff, 0.0)
            amplified_diff = pypto.mul(diff_pos, 1000000.0)
            scale_mask_f32 = pypto.clip(amplified_diff, 0.0, 1.0)
            
            # --- 4. 梯度计算 (Gradient Computation) ---
            
            # Grad Weight: 只有在 clip 范围内的元素有梯度
            grad_weight_fp32 = pypto.mul(grad_out_fp32, mask_f32)
            
            # Grad Offset: 截断元素的梯度累加
            grad_offset_pre = pypto.mul(grad_out_fp32, inv_mask_f32)
            grad_offset_fp32 = pypto.sum(grad_offset_pre, dim=1, keepdim=True)
            
            # Grad Scale: grad_y * (weight_denorm - weight_norm * mask) * n_levels
            term_w_norm = pypto.mul(weight_norm, mask_f32)
            term_diff = pypto.sub(weight_denorm, term_w_norm)
            grad_alpha_pre = pypto.mul(grad_out_fp32, term_diff)
            grad_alpha_fp32 = pypto.sum(grad_alpha_pre, dim=1, keepdim=True)
            
            grad_scale_pre = pypto.mul(grad_alpha_fp32, n_levels)
            grad_scale_fp32 = pypto.mul(grad_scale_pre, scale_mask_f32)
            
            # --- 5. 数据流出 (Cast & Assemble) ---
            grad_w_bf16 = pypto.cast(grad_weight_fp32, pypto.DT_BF16)
            grad_s_bf16 = pypto.cast(grad_scale_fp32, pypto.DT_BF16)
            grad_o_bf16 = pypto.cast(grad_offset_fp32, pypto.DT_BF16)
            
            pypto.assemble(grad_w_bf16, [g_offset, 0], grad_weight_out)
            pypto.assemble(grad_s_bf16, [g_offset, 0], grad_scale_out)
            pypto.assemble(grad_o_bf16, [g_offset, 0], grad_offset_out)

        return grad_weight_out, grad_scale_out, grad_offset_out

    def asymmetric_qat_kernel_back(grad_output, weight_pto, scale_pto, offset_pto):
        grad_output_grouped = grad_output.view(-1, group_size)
        weight_pto_grouped = weight_pto.view(-1, group_size)
        grad_weight_pto_grouped, grad_scale_pto, grad_offset_pto = asymmetric_qat_backward_kernel(grad_output_grouped, weight_pto_grouped, scale_pto, offset_pto)
        return grad_weight_pto_grouped.view(grad_output.shape), grad_scale_pto, grad_offset_pto
 
    return asymmetric_qat_kernel_back
 
# ===== Performance Test Cases =====
# Test configurations for different model sizes
PERFORMANCE_TEST_CONFIGS = {
    "3B": [
        {"weight_shape": (1024, 2048), "description": "3B model config 1"},
        {"weight_shape": (768, 2048), "description": "3B model config 2"},
        {"weight_shape": (2048, 768), "description": "3B model config 3"},
        {"weight_shape": (4096, 2048), "description": "3B model config 4"},
    ],
    "7B": [
        {"weight_shape": (3072, 768), "description": "7B model config 1"},
        {"weight_shape": (768, 3072), "description": "7B model config 2"},
        {"weight_shape": (1024, 3072), "description": "7B model config 3"},
        {"weight_shape": (5632, 3072), "description": "7B model config 4"},
        {"weight_shape": (3072, 2816), "description": "7B model config 5"},
    ],
    "30B": [
        {"weight_shape": (2560, 1536), "description": "30B model config 1"},
        {"weight_shape": (2480, 2560), "description": "30B model config 2"},
        {"weight_shape": (6144, 2560), "description": "30B model config 3"},
        {"weight_shape": (2560, 3072), "description": "30B model config 4"},
    ]
}
 
def run_single_test(weight_shape, group_size, bit, device_id, run_mode, test_mode="both", test_name=""):
    """Run a single test case for given weight shape and mode."""
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    n, m = weight_shape
    groups_per_row = m // group_size
    num_groups = n * groups_per_row
    
    # Common Inputs
    weight_pto = (torch.randn(weight_shape, dtype=torch.float32, device=device)).to(torch.bfloat16)
    scale_pto = (torch.rand(num_groups, 1, dtype=torch.float32, device=device) * 0.1 + 0.01).to(torch.bfloat16)
    offset_pto = (torch.randn(num_groups, 1, dtype=torch.float32, device=device) * 0.1).to(torch.bfloat16)
    
    # --------------------------------------------------------------------------
    # FORWARD PASS TEST
    # --------------------------------------------------------------------------
    if test_mode in ["forward", "both"]:
        print("    [Forward]  Running...", end=" ")
        output_pto = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
        
        # Golden reference
        expected_out = asymmetric_qat_golden(weight_pto, scale_pto, offset_pto, group_size, bit)
        
        # Pypto Kernel
        kernel = create_asymmetric_qat_kernel(group_size, bit, run_mode=run_mode)
        kernel(weight_pto, scale_pto, offset_pto, output_pto)
        
        # Validation
        max_diff = (output_pto - expected_out).abs().max().item()
        assert_allclose(output_pto.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=1e-2, atol=1e-2)
        print(f"✓ Passed (Max diff: {max_diff:.6f})")
 
    # --------------------------------------------------------------------------
    # BACKWARD PASS TEST
    # --------------------------------------------------------------------------
    if test_mode in ["backward", "both"]:
        print("    [Backward] Running...", end=" ")
        # Dummy upstream gradient
        grad_output = torch.randn(weight_shape, dtype=torch.bfloat16, device=device)
        
        weight_pto = weight_pto.requires_grad_(True)
        scale_pto = scale_pto.requires_grad_(True)
        offset_pto = offset_pto.requires_grad_(True)
        # Golden reference
        # --- PyTorch AutoGrad 计算 ---
        output = asymmetric_qat_golden(weight_pto, scale_pto, offset_pto, group_size, bit)
        output.backward(grad_output)
        
        # 获取 PyTorch 算出的标准梯度
        exp_grad_w = weight_pto.grad.clone()
        exp_grad_s = scale_pto.grad.clone()
        exp_grad_o = offset_pto.grad.clone()
        
        # 清空 PyTorch 梯度
        weight_pto.grad = None
        scale_pto.grad = None
        offset_pto.grad = None
                
        # Pypto Kernel
        bw_kernel = create_asymmetric_qat_backward_kernel(group_size, bit, run_mode=run_mode)
        grad_weight_pto, grad_scale_pto, grad_offset_pto = bw_kernel(
            grad_output, weight_pto, scale_pto, offset_pto
        )
        
        # Validation
        diff_w = (grad_weight_pto - exp_grad_w).abs().max().item()
        diff_s = (grad_scale_pto - exp_grad_s).abs().max().item()
        diff_o = (grad_offset_pto - exp_grad_o).abs().max().item()
        
        assert_allclose(grad_weight_pto.float().cpu().numpy(), exp_grad_w.float().cpu().numpy(), rtol=1e-2, atol=1e-2)
        assert_allclose(grad_scale_pto.float().cpu().numpy(), exp_grad_s.float().cpu().numpy(), rtol=1e-2, atol=1e-2)
        assert_allclose(grad_offset_pto.float().cpu().numpy(), exp_grad_o.float().cpu().numpy(), rtol=1e-2, atol=1e-2)
        print(f"✓ Passed (Max diff W: {diff_w:.6f}, S: {diff_s:.6f}, O: {diff_o:.6f})")
 
 
def test_asymmetric_qat_3b(device_id=None, run_mode="npu", test_mode="both"):
    """Test 3B model configurations."""
    print("Test: 3B Model Configurations")
    group_size = 128
    bit = 4
    for idx, config in enumerate(PERFORMANCE_TEST_CONFIGS["3B"], 1):
        weight_shape = config["weight_shape"]
        print(f"  [{idx}/{len(PERFORMANCE_TEST_CONFIGS['3B'])}] Shape: {weight_shape}")
        run_single_test(weight_shape, group_size, bit, device_id, run_mode, test_mode, f"3B-config{idx}")
 
 
def test_asymmetric_qat_7b(device_id=None, run_mode="npu", test_mode="both"):
    """Test 7B model configurations."""
    print("Test: 7B Model Configurations")
    group_size = 128
    bit = 4
    for idx, config in enumerate(PERFORMANCE_TEST_CONFIGS["7B"], 1):
        weight_shape = config["weight_shape"]
        print(f"  [{idx}/{len(PERFORMANCE_TEST_CONFIGS['7B'])}] Shape: {weight_shape}")
        run_single_test(weight_shape, group_size, bit, device_id, run_mode, test_mode, f"7B-config{idx}")
 
 
def test_asymmetric_qat_30b(device_id=None, run_mode="npu", test_mode="both"):
    """Test 30B model configurations."""
    print("Test: 30B Model Configurations")
    group_size = 128
    bit = 4
    for idx, config in enumerate(PERFORMANCE_TEST_CONFIGS["30B"], 1):
        weight_shape = config["weight_shape"]
        print(f"  [{idx}/{len(PERFORMANCE_TEST_CONFIGS['30B'])}] Shape: {weight_shape}")
        run_single_test(weight_shape, group_size, bit, device_id, run_mode, test_mode, f"30B-config{idx}")
 
 
def main():
    import argparse
 
    parser = argparse.ArgumentParser(description="QAT Asymmetric Quantization Operator Tests")
    parser.add_argument('--run_mode', type=str, default="npu", choices=["npu", "sim"])
    parser.add_argument('--model', type=str, default="all", choices=["3b", "7b", "30b", "all"])
    # 新增测试模式参数
    parser.add_argument('--test_mode', type=str, default="backward", choices=["forward", "backward", "both"],
                        help="Choose to test 'forward', 'backward', or 'both' passes.")
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
 
    print("=" * 70)
    print(f"QAT Asymmetric Quantization Operator - Performance Tests [{args.test_mode.upper()}]")
    print("group_size: 128, bit: 4")
    print("=" * 70)
    print()
 
    if args.model in ["all", "3b"]:
        test_asymmetric_qat_3b(device_id, args.run_mode, args.test_mode)
        print()
 
    if args.model in ["all", "7b"]:
        test_asymmetric_qat_7b(device_id, args.run_mode, args.test_mode)
        print()
 
    if args.model in ["all", "30b"]:
        test_asymmetric_qat_30b(device_id, args.run_mode, args.test_mode)
        print()
 
    print("=" * 70)
    print(f"All selected performance tests ({args.test_mode}) passed successfully!")
    print("=" * 70)
 
if __name__ == "__main__":
    main()