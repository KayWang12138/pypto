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

"""
AttentionWorkerCombine PyPTO Kernel - Final Implementation

关键发现：
- SplitBS/SplitH: 使用向量化实现，通过
- SplitK: 原始循环累加方式有精度问题，改用向量化实现
"""

import os
import logging
import torch
import torch_npu
import pypto

logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        raise RuntimeError("Please set TILE_FWK_DEVICE_ID")
    return int(os.environ['TILE_FWK_DEVICE_ID'])


bs, k, h = 8, 2, 32


# ============================================================================
# Strategy 1: SplitBS - 向量化实现
# ============================================================================
@pypto.frontend.jit
def attention_worker_combine_splitbs_kernel(
    token_data: pypto.Tensor((bs, k + 1, h), pypto.DT_BF16),
    expert_scales: pypto.Tensor((bs, k), pypto.DT_FP32),
    y: pypto.Tensor((bs, h), pypto.DT_BF16),
):
    """
    SplitBS: 向量化实现
    """
    pypto.set_vec_tile_shapes(1, 1, h)
    
    token_fp32 = pypto.cast(token_data, pypto.DT_FP32)
    token_routed = token_fp32[:, 0:k, :]
    token_shared = token_fp32[:, k, :]
    
    scales_3d = pypto.reshape(expert_scales, [bs, k, 1])
    
    weighted = pypto.mul(token_routed, scales_3d)
    weighted_sum = pypto.Tensor([bs, 1, h], pypto.DT_FP32)
    weighted_sum[:] = pypto.sum(weighted, dim=1, keepdim=True)
    
    weighted_sum_2d = pypto.reshape(weighted_sum, [bs, h])
    result = pypto.add(weighted_sum_2d, token_shared)
    
    y[:] = pypto.cast(result, pypto.DT_BF16)


# ============================================================================
# Strategy 2: SplitH - 按 hidden 维度切分
# ============================================================================
@pypto.frontend.jit(debug_options={"runtime_debug_mode": 1})
def attention_worker_combine_splith_kernel(
    token_data: pypto.Tensor((bs, k + 1, h), pypto.DT_BF16),
    expert_scales: pypto.Tensor((bs, k), pypto.DT_FP32),
    y: pypto.Tensor((bs, h), pypto.DT_BF16),
    h_tile: int = 16,
):
    """
    SplitH: 按 hidden 维度切分
    """
    pypto.set_vec_tile_shapes(1, 1, h_tile)
    
    h_loops = h // h_tile
    scales_3d = pypto.reshape(expert_scales, [bs, k, 1])
    
    for h_idx in pypto.loop(h_loops):
        h_start = h_idx * h_tile
        h_end = h_start + h_tile
        
        token_h = token_data[:, :, h_start:h_end]
        token_h_fp32 = pypto.cast(token_h, pypto.DT_FP32)
        
        token_routed = token_h_fp32[:, 0:k, :]
        token_shared = token_h_fp32[:, k, :]
        
        weighted = pypto.mul(token_routed, scales_3d)
        weighted_sum = pypto.Tensor([bs, 1, h_tile], pypto.DT_FP32)
        weighted_sum[:] = pypto.sum(weighted, dim=1, keepdim=True)
        
        weighted_sum_2d = pypto.reshape(weighted_sum, [bs, h_tile])
        result = pypto.add(weighted_sum_2d, token_shared)
        
        y_h = pypto.cast(result, pypto.DT_BF16)
        y[:, h_start:h_end] = y_h


# ============================================================================
# 测试函数
# ============================================================================
def test_kernel(kernel_func, kernel_name):
    """测试单个 kernel"""
    logging.info(f"\n--- Testing {kernel_name} ---")
    
    device_id = get_device_id()
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'
    
    token_data = torch.randn(bs, k + 1, h, dtype=torch.bfloat16, device=device)
    expert_scales = torch.rand(bs, k, dtype=torch.float32, device=device)
    y = torch.zeros(bs, h, dtype=torch.bfloat16, device=device)
    
    logging.info(f"  Input: token_data={token_data.shape}, expert_scales={expert_scales.shape}")
    
    try:
        kernel_func(token_data, expert_scales, y)
        logging.info(f"  ✓ Kernel executed successfully")
        
        golden = (token_data[:, :k, :].float() * expert_scales.unsqueeze(-1)).sum(1) + token_data[:, k, :].float()
        golden = golden.to(torch.bfloat16)
        
        max_diff = (y - golden).abs().max().item()
        logging.info(f"  Max diff: {max_diff:.6f}")
        logging.info(f"  Result: {'PASS' if max_diff < 0.01 else 'FAIL'}")
        return max_diff < 0.01
        
    except Exception as e:
        logging.info(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    logging.info("=" * 70)
    logging.info("AttentionWorkerCombine PyPTO Kernel Tests (Final)")
    logging.info("=" * 70)
    logging.info(f"Shape: bs={bs}, k={k}, h={h}")
    
    results = []
    
    results.append(("SplitBS", test_kernel(attention_worker_combine_splitbs_kernel, "SplitBS")))
    results.append(("SplitH", test_kernel(attention_worker_combine_splith_kernel, "SplitH")))
    
    logging.info("\n" + "=" * 70)
    logging.info("Summary")
    logging.info("=" * 70)
    for name, passed in results:
        logging.info(f"  {name}: {'✓ PASS' if passed else '✗ FAIL'}")
    
    total = sum(1 for _, p in results if p)
    logging.info(f"\nTotal: {total}/{len(results)} passed")


if __name__ == "__main__":
    run_all_tests()