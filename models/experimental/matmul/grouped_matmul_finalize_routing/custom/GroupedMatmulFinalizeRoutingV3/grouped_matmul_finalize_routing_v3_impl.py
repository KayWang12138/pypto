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

"""PyPTO GroupedMatmulFinalizeRoutingV3 kernel implementation.

实现 MoE 场景下的融合算子，包含三个核心步骤：
1. 分组矩阵乘法（GMM）- 使用 pypto.scaled_mm 处理 MXFP8 量化
2. 路由分配（Routing）- 使用 pypto.index_add_ 实现 scatter add
3. 共享专家融合 - 加权融合共享专家输出

参考实现: ../gmm_mxfp8.py (scaled_matmul_kernel)
设计文档: DESIGN.md
"""

from dataclasses import dataclass
from typing import Optional, List

import pypto
import torch
import torch_npu


@dataclass
class TileConfig:
    """Tile configuration for GroupedMatmulFinalizeRoutingV3.
    
    Attributes:
        m_tile_shape: Tile shape for M dimension in cube operation [mL0, mL1]
        k_tile_shape: Tile shape for K dimension in cube operation [kL0, kL1]
        n_tile_shape: Tile shape for N dimension in cube operation [nL0, nL1]
        vector_tile_shape: Tile shapes for vector operations (scale processing)
    """
    m_tile_shape: List[int]
    k_tile_shape: List[int]
    n_tile_shape: List[int]
    vector_tile_shape: List[int]


# Default tile configuration (from DESIGN.md and gmm_mxfp8.py)
DEFAULT_TILE_CONFIG = TileConfig(
    m_tile_shape=[9, 9],
    k_tile_shape=[256, 256],
    n_tile_shape=[256, 256],
    vector_tile_shape=[1, 8, 256, 32],
)


@pypto.frontend.jit
def grouped_matmul_finalize_routing_v3_kernel(
    x1: pypto.Tensor,
    x2: pypto.Tensor,
    scale: pypto.Tensor,
    pertoken_scale: pypto.Tensor,
    intermediate: pypto.Tensor,
    row_index: pypto.Tensor,
    out: pypto.Tensor,
    group_list: List[int],
    tile_config: TileConfig,
    transpose_x2: bool = False,
) -> None:
    """
    PyPTO JIT kernel for GroupedMatmulFinalizeRoutingV3.
    
    实现三个核心步骤：
    1. 分组矩阵乘法（GMM）- 遍历专家组执行 scaled_mm
    2. 路由分配 - 使用 index_add_ 实现 scatter add
    3. 共享专家融合（在 wrapper 中处理）
    
    Args:
        x1: Input tensor [M, K], MXFP8 format
        x2: Weight tensor [E, K, N] or [E, N, K] if transposed, MXFP8 format
        scale: Scale factors [E, Ceil(K/64), N, 2] or [E, N, Ceil(K/64), 2], FLOAT8_E8M0
        pertoken_scale: Token-level scale [M, Ceil(K/64), 2], FLOAT8_E8M0
        intermediate: Intermediate result tensor [M, N], FP32
        row_index: Routing indices [M], INT64
        out: Output tensor [batch, N], FP32
        group_list: List of group sizes [E]
        tile_config: Tile configuration
        transpose_x2: Whether x2 is transposed
    """
    num_experts = x2.shape[0]
    begin = 0
    end = 0
    
    # Step 1: 分组矩阵乘法（遍历专家组）
    for i in range(num_experts):
        begin = end
        end = end + group_list[i]
        
        if group_list[i] <= 0:
            continue
        
        # 提取当前专家的输入和权重
        x = x1[begin:end, :]
        weight = x2[i]
        
        # 提取 scale factors
        # pertoken_scale shape: [M, Ceil(K/64), 2] -> 切片 [begin:end, :, :]
        scaled_x = pertoken_scale[begin:end, :, :]
        scaled_weight = scale[i]
        
        # 设置 vector tile shapes（用于 scale 处理）
        pypto.set_vec_tile_shapes(
            tile_config.vector_tile_shape[0],
            tile_config.vector_tile_shape[1],
            tile_config.vector_tile_shape[2],
            tile_config.vector_tile_shape[3]
        )
        
        # 设置 cube tile shapes（scaled_mm 前置条件）
        pypto.set_cube_tile_shapes(
            tile_config.m_tile_shape,
            tile_config.k_tile_shape,
            tile_config.n_tile_shape
        )
        
        # 执行 MXFP8 scaled_matmul
        intermediate[begin:end, :] = pypto.scaled_mm(
            x, weight, pypto.DT_FP32, scaled_x, scaled_weight
        )
    
    # Step 2: 路由分配（Scatter Add）
    # 使用 index_add_ 将 intermediate 按 row_index 分配到 out
    pypto.set_vec_tile_shapes(1, tile_config.n_tile_shape[0])
    pypto.index_add_(out, 0, row_index, intermediate, alpha=1.0)


def grouped_matmul_finalize_routing_v3_wrapper(
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
    tile_config: Optional[TileConfig] = None,
) -> torch.Tensor:
    """
    Wrapper function for GroupedMatmulFinalizeRoutingV3.
    
    负责：
    1. 转换 group_list_type（cumsum -> count）
    2. 初始化输出 tensor
    3. 调用 JIT kernel
    4. 处理可选参数（bias, shared_input）
    
    Args:
        x1: Input tensor [M, K], MXFP8 format (FLOAT8_E4M3FN or FLOAT8_E5M2)
        x2: Weight tensor [E, K, N] or [E, N, K], MXFP8 format
        scale: Scale factors [E, Ceil(K/64), N, 2] or [E, N, Ceil(K/64), 2], FLOAT8_E8M0
        pertoken_scale: Token-level scale [M, Ceil(K/64), 2], FLOAT8_E8M0
        group_list: List of group sizes [E]
        row_index: Routing indices [M], INT64
        logit: MoE expert logits [M], FLOAT32 (required per SPEC)
        batch: Output batch dimension
        n: Output feature dimension
        bias: Optional bias [E, N], BFLOAT16
        shared_input: Optional shared expert output [bsdp, N], BFLOAT16
        shared_input_weight: Weight for shared expert fusion
        shared_input_offset: Offset for shared expert in output
        transpose_x1: Whether x1 is transposed (only False supported)
        transpose_x2: Whether x2 is transposed
        group_list_type: 0 for cumsum mode, 1 for count mode
        tile_config: Tile configuration (default: DEFAULT_TILE_CONFIG)
    
    Returns:
        torch.Tensor: Output [batch, N], FLOAT32
    """
    if tile_config is None:
        tile_config = DEFAULT_TILE_CONFIG
    
    # 转换 group_list_type
    if group_list_type == 0:  # cumsum mode
        group_counts = []
        prev = 0
        for val in group_list:
            group_counts.append(val - prev)
            prev = val
        group_list = group_counts
    
    # Move tensors to NPU
    x1 = x1.npu()
    x2 = x2.npu()
    scale = scale.npu()
    pertoken_scale = pertoken_scale.npu()
    row_index = row_index.npu()
    
    # 计算维度
    m = x1.shape[0]
    
    # 初始化 intermediate tensor [M, N]
    intermediate = torch.zeros((m, n), dtype=torch.float32).npu()
    
    # 初始化输出 tensor [batch, N]
    out = torch.zeros((batch, n), dtype=torch.float32).npu()
    
    # 准备 PyPTO tensors（按 kernel 参数顺序）
    # kernel 参数顺序：x1, x2, scale, pertoken_scale, intermediate, row_index, out
    tensors_and_axes = [
        (x1, []),      # x1: [M, K]
        (x2, []),      # x2: [E, K, N]
        (scale, []),   # scale: [E, Ceil(K/64), N, 2]
        (pertoken_scale, []),  # pertoken_scale: [M, Ceil(K/64), 2]
        (intermediate, []),    # intermediate: [M, N]
        (row_index, []),       # row_index: [M]
        (out, []),             # out: [batch, N]
    ]
    
    pto_tensors = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in tensors_and_axes]
    
    # 调用 JIT kernel
    grouped_matmul_finalize_routing_v3_kernel(
        *pto_tensors,
        group_list,
        tile_config,
        transpose_x2,
    )
    
    # Step 3: 处理 bias（可选）
    if bias is not None:
        bias = bias.npu()
        # 按 group_list 分配 bias 到 intermediate
        begin = 0
        end = 0
        for i in range(len(group_list)):
            begin = end
            end = end + group_list[i]
            if group_list[i] > 0:
                bias_i = bias[i].to(torch.float32)
                intermediate[begin:end, :] = intermediate[begin:end, :] + bias_i
    
    # Step 4: 共享专家融合（可选）
    if shared_input is not None:
        shared_input = shared_input.npu()
        shared_fp32 = shared_input.to(torch.float32)
        weighted_shared = shared_fp32 * shared_input_weight
        bsdp = shared_input.shape[0]
        for j in range(bsdp):
            target_row = shared_input_offset + j
            if target_row < batch:
                out[target_row, :] = out[target_row, :] + weighted_shared[j, :]
    
    return out.to(torch.float32)