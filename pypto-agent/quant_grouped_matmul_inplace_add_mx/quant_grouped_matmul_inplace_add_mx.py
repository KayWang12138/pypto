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

"""
Grouped Matrix Multiplication with MXFP8 Quantization (New Frontend)

This module implements grouped matrix multiplication with MXFP8 quantization using PyPTO new frontend.
Supports grouped GEMM operations with different weight groups and MXFP8 quantization format.
"""

import math
from dataclasses import dataclass

import numpy as np
import pypto
import torch
import torch_npu
from numpy.testing import assert_allclose


@dataclass
class GmmGoldenInputs:
    """
    Input parameters for generating golden result in grouped matrix multiplication.

    Attributes:
        a: Input tensor of shape [M, K]
        b: Weight tensor of shape [num_groups, K, N] or [num_groups, N, K]
        scaled_a: Scale factors for input tensor
        scaled_b:: Scale factors for weight tensor
        y: Output tensor of shape [num_groups, M, N] (初始值)
        group_list: List of group sizes for each weight group
        a_trans: Whether input tensor is transposed
        b_trans: Whether weight tensor is transposed
    """
    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    y: torch.Tensor
    group_list: list
    a_trans: bool
    b_trans: bool


@dataclass
class GmmMxfp8Inputs:
    """
    Input parameters for generating MXFP8 output.

    Attributes:
        a: Input tensor of shape [M, K]
        b: Weight tensor of shape [num_groups, K, N] or [num_groups, N, K]
        scaled_a: Scale factors for input tensor
        scaled_b: Scale factors for weight tensor
        y: Output tensor of shape [num_groups, M, N] (需要在外部初始化)
        group_list: List of group sizes for each weight group
        tile_config: Tile configuration for computation
    """
    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    y: torch.Tensor
    group_list: list
    tile_config: 'ShapeConfig'


@dataclass
class ShapeConfig:
    """
    Configuration parameters for grouped matrix multiplication with MXFP8 quantization.

    Attributes:
        ori_shape: Original shape [M, K, N]
        group_list: List of group sizes for each weight group
        tile_size: Tile size for computation
        m_tile_shape: Tile shape for M dimension in cube operation
        k_tile_shape: Tile shape for K dimension in cube operation
        n_tile_shape: Tile shape for N dimension in cube operation
        vector_tile_shape: Tile shapes for vector operations
        a_trans: Whether input tensor is transposed (default: False)
        b_trans: Whether weight tensor is transposed (default: False)
        a_format_nz: Whether input uses NZ format (default: False)
        b_format_nz: Whether weight uses NZ format (default: False)
        c_format_nz: Whether output uses NZ format (default: False)
        description: Description of the test case
    """
    ori_shape: list
    group_list: list
    tile_size: int
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    vector_tile_shape: list
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    description: str = ""


def compute_golden_result_simple(
    x: torch.Tensor,
    weight: torch.Tensor,
    scaled_x: torch.Tensor,
    scaled_weight: torch.Tensor
) -> torch.Tensor:
    """
    Compute golden (reference) result for MXFP8 scaled matrix multiplication.
    
    Simple implementation matching scaled_mm behavior:
    - x: [M, K_block] FP8 tensor
    - weight: [K_block, N] FP8 tensor  
    - scaled_x: [M, K_block // 64, 2] scale factors
    - scaled_weight: [K_block // 64, N, 2] scale factors
    
    MXFP8: every 64 elements share one scale factor
    """
    m, k_block = x.shape
    _, n = weight.shape
    
    # Convert to FP32
    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    
    # Process scale factors
    # scaled_x: [M, K_block // 64, 2] -> take first column [M, K_block // 64]
    scale_x = scaled_x[:, :, 0].to(torch.float32)  # [M, K_block // 64]
    
    # scaled_weight: [K_block // 64, N, 2] -> take first column [K_block // 64, N]
    # NO transpose needed, scale broadcasts along K dimension
    scale_weight = scaled_weight[:, :, 0].to(torch.float32)  # [K_block // 64, N]
    
    # Broadcast scales: repeat_interleave along K dimension (64 elements per block)
    # scale_x: [M, K_block // 64] -> [M, K_block]
    scale_x_broadcast = torch.repeat_interleave(scale_x, repeats=64, dim=-1)
    
    # scale_weight: [K_block // 64, N] -> [K_block, N]
    scale_weight_broadcast = torch.repeat_interleave(scale_weight, repeats=64, dim=0)
    
    # Handle case where K_block is not exactly divisible by 64
    if scale_x_broadcast.shape[-1] > k_block:
        scale_x_broadcast = scale_x_broadcast[:, :k_block]
    if scale_weight_broadcast.shape[0] > k_block:
        scale_weight_broadcast = scale_weight_broadcast[:k_block, :]
    
    # Apply scales
    x_scaled = x_fp32 * scale_x_broadcast  # [M, K_block]
    weight_scaled = weight_fp32 * scale_weight_broadcast  # [K_block, N]
    
    # Matrix multiplication
    result = torch.matmul(x_scaled, weight_scaled)  # [M, N]
    
    return result


def gen_golden(inputs: GmmGoldenInputs) -> torch.Tensor:
    """
    Generate golden (reference) output for grouped matrix multiplication using PyTorch.

    Args:
        inputs: Input parameters including tensors, scales, y, and transposition flags

    Returns:
        torch.Tensor: Golden output tensor of shape [num_groups, M, N]
    """
    a = inputs.a
    b = inputs.b
    scaled_a = inputs.scaled_a
    scaled_b = inputs.scaled_b
    y = inputs.y
    group_list = inputs.group_list

    round_num = b.shape[0]
    golden_result = y.clone()  # 从 y 的初始值开始
    begin = 0
    end = 0

    for i in range(round_num):
        begin = end
        end = end + group_list[i]

        # Extract input and weight for current group (切分 K 轴)
        x = a[:, begin:end]
        weight = b[i]
        scaled_x_golden = scaled_a[:, begin // 64 : end // 64, :]
        scaled_weight_golden = scaled_b[i]

        # Compute golden result for this group and inplace add
        golden_temp = compute_golden_result_simple(
            x=x,
            weight=weight,
            scaled_x=scaled_x_golden,
            scaled_weight=scaled_weight_golden
        )
        golden_result[i] = golden_result[i] + golden_temp

    return golden_result


@pypto.frontend.jit
def scaled_matmul_kernel(
    a: pypto.Tensor(),
    b: pypto.Tensor(),
    scaled_a: pypto.Tensor(),
    scaled_b: pypto.Tensor(),
    y: pypto.Tensor(),
    group_list: list,
    tile_config: ShapeConfig
):
    """
    Scaled matrix multiplication kernel for grouped GEMM with MXFP8 quantization.

    This kernel performs grouped matrix multiplication where each group uses
    a different K-axis block from input tensor, with MXFP8 quantization.

    Args:
        a: Input tensor [M, K]
        b: Weight tensor containing multiple weight groups [num_groups, K_block, N]
        scaled_a: Scale factors for input tensor
        scaled_b: Scale factors for weight tensor
        y: Output tensor [num_groups, M, N]
        group_list: List of group sizes for K-axis splitting
        tile_config: Tile configuration for computation
    """
    round_num = b.shape[0]
    begin = 0
    end = 0

    for i in range(round_num):
        begin = end
        end = end + group_list[i]

        # Extract input for current group (切分 K 轴)
        x = a[:, begin:end]
        weight = b[i]
        scaled_x = scaled_a[:, begin // 64 : end // 64, :]

        # Set vector tile shapes for scale processing
        pypto.set_vec_tile_shapes(
            tile_config.vector_tile_shape[0],
            tile_config.vector_tile_shape[1],
            tile_config.vector_tile_shape[2],
            tile_config.vector_tile_shape[3]
        )
        scaled_weight = scaled_b[i]

        # Set cube tile shapes and perform scaled matrix multiplication
        pypto.set_cube_tile_shapes(
            tile_config.m_tile_shape,
            tile_config.k_tile_shape,
            tile_config.n_tile_shape
        )
        mm_result = pypto.scaled_mm(x, weight, pypto.DT_FP32, scaled_x, scaled_weight)
        
        # 原地累加：使用 slice 切片 y[i:i+1] 获取 [1, M, N] 的 view
        mm_result_3d = pypto.unsqueeze(mm_result, 0)  # [M, N] -> [1, M, N]
        y_slice = y[i:i+1]  # 获取 [1, M, N] 的 view
        y_slice = pypto.add(y_slice, mm_result_3d)  # 累加
        y[i:i+1] = y_slice  # 写回


def gen_mxfp8(inputs: GmmMxfp8Inputs) -> torch.Tensor:
    """
    Generate MXFP8 output using PyPTO scaled matrix multiplication with new frontend.

    Args:
        inputs: Input parameters including tensors, scales, y, group list and tile config

    Returns:
        torch.Tensor: Output tensor of shape [num_groups, M, N] in FP32
    """
    a = inputs.a
    b = inputs.b
    scaled_a = inputs.scaled_a
    scaled_b = inputs.scaled_b
    y = inputs.y
    group_list = inputs.group_list
    tile_config = inputs.tile_config

    # Move tensors to NPU
    a = a.npu()
    b = b.npu()
    scaled_a = scaled_a.npu()
    scaled_b = scaled_b.npu()
    y = y.npu()

    # Execute scaled matrix multiplication kernel with new frontend
    scaled_matmul_kernel(a, b, scaled_a, scaled_b, y, group_list, tile_config)

    y = y.to(torch.float32)
    return y


def test_gmm_mxfp8(tile_config: ShapeConfig):
    """
    Test the grouped matrix multiplication with MXFP8 quantization.

    This function runs a complete test for a given configuration:
    1. Generate test data with MXFP8 format
    2. Compute golden (reference) output using PyTorch
    3. Compute output using PyPTO
    4. Compare results

    Args:
        tile_config: Configuration parameters for the test case
    """
    # Extract configuration parameters
    m = tile_config.ori_shape[0]
    k = tile_config.ori_shape[1]
    n = tile_config.ori_shape[2]
    a_trans = tile_config.a_trans
    b_trans = tile_config.b_trans
    group_list = tile_config.group_list
    num_groups = len(group_list)

    # Generate input tensor in MXFP8 format
    a = torch.randn((m, k), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
    scaled_a = torch.randn((m, k // 64, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    # Generate weight tensor in MXFP8 format (按 K 轴切分，每个 group 的 K 维度为 group_list[i])
    b_list = []
    scaled_b_list = []
    for i in range(num_groups):
        k_block = group_list[i]
        if b_trans:
            b_i = torch.randn((n, k_block), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
            scaled_b_i = torch.randn((n, k_block // 64, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
        else:
            b_i = torch.randn((k_block, n), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
            scaled_b_i = torch.randn((k_block // 64, n, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
        b_list.append(b_i)
        scaled_b_list.append(scaled_b_i)
    
    b = torch.stack(b_list, dim=0)
    scaled_b = torch.stack(scaled_b_list, dim=0)

    # Initialize y tensor with random values (not zeros, for inplace add)
    y_init = torch.randn((num_groups, m, n), dtype=torch.float32)
    y_init_npu = y_init.clone().npu()

    # Compute golden and PyPTO results
    golden = gen_golden(GmmGoldenInputs(
        a=a,
        b=b,
        scaled_a=scaled_a,
        scaled_b=scaled_b,
        y=y_init,
        group_list=group_list,
        a_trans=a_trans,
        b_trans=b_trans,
    ))
    result = gen_mxfp8(GmmMxfp8Inputs(
        a=a,
        b=b,
        scaled_a=scaled_a,
        scaled_b=scaled_b,
        y=y_init_npu,
        group_list=group_list,
        tile_config=tile_config,
    ))

    # Verify results
    assert_allclose(golden.cpu().numpy(), result.cpu().numpy(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_gmm_mxfp8(
        ShapeConfig(
            [16, 512, 7168],
            [256, 256],
            256,
            [16, 16],
            [256, 256],
            [256, 256],
            [1, 8, 256, 32],
            False,
            False,
            False,
            False,
            False,
            "Test with K-axis splitting: K=512 split into [256, 256]"
        )
    )
