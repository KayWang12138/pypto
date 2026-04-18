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

"""PyPTO GroupedMatmulFinalizeRoutingV3 golden reference implementation.

纯 PyTorch 实现，禁止引入 pypto。
导出函数 grouped_matmul_finalize_routing_v3_golden() 供测试调用。

算子功能：
  GroupedMatmulFinalizeRoutingV3 是 MoE 场景下的融合算子，包含三个步骤：
  1. 分组矩阵乘法（GMM）：对每个专家组执行 MXFP8 scaled_matmul
  2. 路由分配（Routing）：按照 rowIndex 将每个 token 的 GMM 结果分配到输出位置
  3. 共享专家融合：将共享专家输出与 MoE 专家结果加权融合

参考实现: ../gmm_mxfp8.py (compute_golden_result 和 gen_golden 函数)
"""

import math
import torch
from typing import Optional, List


def _compute_scaled_matmul_golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    scaled_x: torch.Tensor,
    scaled_weight: torch.Tensor,
    transpose_x: bool = False,
    transpose_weight: bool = False,
) -> torch.Tensor:
    """
    Compute golden result for MXFP8 scaled matrix multiplication.
    
    参考 gmm_mxfp8.py 的 compute_golden_result 函数，处理：
    - K 维度对齐（Ceil(k/32) % 2 != 0 时裁剪）
    - scale 广播（repeat_interleave 32倍）
    - padding 处理
    
    Args:
        x: Input tensor [M, K] (MXFP8 format, FP8 dtype)
        weight: Weight tensor [K, N] or [N, K] if transposed (MXFP8 format)
        scaled_x: Scale factors for input [M, Ceil(K/64), 2] (FLOAT8_E8M0)
        scaled_weight: Scale factors for weight (FLOAT8_E8M0)
        transpose_x: Whether input is transposed (only False supported per SPEC)
        transpose_weight: Whether weight is transposed
        
    Returns:
        torch.Tensor: Result tensor [M, N] in FP32
    """
    if transpose_x:
        x = torch.swapaxes(x, -1, -2)
        scaled_x = torch.swapaxes(scaled_x, -1, -2)
        if len(scaled_x.shape) == 3:
            scaled_x = scaled_x.reshape(
                scaled_x.shape[0] * scaled_x.shape[1], scaled_x.shape[2]
            )
        scaled_x = torch.swapaxes(scaled_x, -1, -2)
    else:
        if len(scaled_x.shape) == 3:
            scaled_x = scaled_x.reshape(
                scaled_x.shape[0], scaled_x.shape[1] * scaled_x.shape[2]
            )
    
    if transpose_weight:
        weight = torch.swapaxes(weight, -1, -2)
        scaled_weight = torch.swapaxes(scaled_weight, 0, 1)
        scaled_weight = torch.swapaxes(scaled_weight, -1, -2)
        if len(scaled_weight.shape) == 3:
            scaled_weight = scaled_weight.reshape(
                scaled_weight.shape[0] * scaled_weight.shape[1], scaled_weight.shape[2]
            )
    else:
        scaled_weight = torch.swapaxes(scaled_weight, -1, -2)
        if len(scaled_weight.shape) == 3:
            scaled_weight = scaled_weight.reshape(
                scaled_weight.shape[0] * scaled_weight.shape[1], scaled_weight.shape[2]
            )
    
    k_dim = x.shape[-1]
    if math.ceil(k_dim / 32) % 2 != 0:
        scaled_x = scaled_x[:, :-1]
        scaled_weight = scaled_weight[:-1, :]
    
    scaled_x_broadcast = torch.repeat_interleave(scaled_x, repeats=32, dim=-1)
    scaled_weight_broadcast = torch.repeat_interleave(scaled_weight, repeats=32, dim=-2)
    
    x1_dims = len(x.shape)
    x2_dims = len(weight.shape)
    x1_pad_len = scaled_x_broadcast.shape[-1] - x.shape[-1]
    x2_pad_len = scaled_weight_broadcast.shape[-2] - weight.shape[-2]
    
    x1_pad = [0, x1_pad_len]
    for _ in range(x1_dims - 1):
        x1_pad += [0, 0]
    x_golden = torch.nn.functional.pad(x, x1_pad, mode='constant', value=0)
    
    weight_pad = [0, 0]
    weight_pad += [0, x2_pad_len]
    for _ in range(x2_dims - 2):
        weight_pad += [0, 0]
    weight_golden = torch.nn.functional.pad(weight, weight_pad, mode='constant', value=0)
    
    x_fp32 = x_golden.to(torch.float32)
    scaled_x_broadcast_fp32 = scaled_x_broadcast.to(torch.float32)
    x1_golden = x_fp32 * scaled_x_broadcast_fp32
    
    weight_fp32 = weight_golden.to(torch.float32)
    scaled_weight_broadcast_fp32 = scaled_weight_broadcast.to(torch.float32)
    weight_golden = weight_fp32 * scaled_weight_broadcast_fp32
    
    golden = torch.matmul(x1_golden, weight_golden)
    
    return golden


def grouped_matmul_finalize_routing_v3_golden(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    group_list: List[int],
    row_index: torch.Tensor,
    logit: torch.Tensor,
    batch: int,
    n: int,
    bias: Optional[torch.Tensor] = None,
    shared_input: Optional[torch.Tensor] = None,
    shared_input_weight: float = 1.0,
    shared_input_offset: int = 0,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    group_list_type: int = 1,
) -> torch.Tensor:
    """
    Golden reference for GroupedMatmulFinalizeRoutingV3 operator.
    
    Args:
        x1: Input tensor [M, K], MXFP8 format (FLOAT8_E4M3FN or FLOAT8_E5M2)
        x2: Weight tensor [E, K, N] or [E, N, K], MXFP8 format
        scale: Scale factors [E, Ceil(K/64), N, 2] or [E, N, Ceil(K/64), 2], FLOAT8_E8M0
        pertoken_scale: Token-level scale [M, Ceil(K/64), 2], FLOAT8_E8M0
        group_list: List of group sizes [E]
        row_index: Routing indices [M], INT64
        logit: MoE expert logits [M], FLOAT32
        batch: Output batch dimension
        n: Output feature dimension
        bias: Optional bias [E, N], BFLOAT16
        shared_input: Optional shared expert output [bsdp, N], BFLOAT16
        shared_input_weight: Weight for shared expert fusion
        shared_input_offset: Offset for shared expert in output
        transpose_x1: Whether x1 is transposed (only False supported)
        transpose_x2: Whether x2 is transposed
        group_list_type: 0 for cumsum mode, 1 for count mode
    
    Returns:
        torch.Tensor: Output [batch, N], FLOAT32
    """
    if group_list_type == 0:
        group_counts = []
        prev = 0
        for val in group_list:
            group_counts.append(val - prev)
            prev = val
        group_list = group_counts
    
    num_experts = x2.shape[0]
    m_total = x1.shape[0]
    
    intermediate = torch.zeros((m_total, n), dtype=torch.float32)
    
    begin = 0
    end = 0
    
    for i in range(num_experts):
        begin = end
        end = end + group_list[i]
        
        if group_list[i] <= 0:
            continue
        
        if transpose_x1:
            x_i = x1[:, begin:end]
            scaled_x_i = pertoken_scale[:, begin:end, :]
        else:
            x_i = x1[begin:end, :]
            scaled_x_i = pertoken_scale[begin:end, :, :]
        
        weight_i = x2[i]
        scale_i = scale[i]
        
        gmm_result = _compute_scaled_matmul_golden(
            x=x_i,
            weight=weight_i,
            scaled_x=scaled_x_i,
            scaled_weight=scale_i,
            transpose_x=transpose_x1,
            transpose_weight=transpose_x2,
        )
        
        if bias is not None:
            bias_i = bias[i].to(torch.float32)
            gmm_result = gmm_result + bias_i
        
        intermediate[begin:end, :] = gmm_result
    
    out = torch.zeros((batch, n), dtype=torch.float32)
    
    if m_total > 0:
        row_index_long = row_index.long()
        out = out.index_add_(0, row_index_long, intermediate, alpha=1.0)
    
    if shared_input is not None:
        shared_fp32 = shared_input.to(torch.float32)
        weighted_shared = shared_fp32 * shared_input_weight
        
        bsdp = shared_input.shape[0]
        for j in range(bsdp):
            target_row = shared_input_offset + j
            if target_row < batch:
                out[target_row, :] += weighted_shared[j, :]
    
    return out