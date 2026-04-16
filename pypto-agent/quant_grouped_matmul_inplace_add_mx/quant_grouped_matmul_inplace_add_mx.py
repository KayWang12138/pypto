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
        a: Input tensor of shape [K, M] (transposed format)
        b: Weight tensor of shape [K, N]
        scaled_a: Scale factors for input tensor [K//64, M, 2] (transposed format)
        scaled_b: Scale factors for weight tensor [K//64, N, 2]
        y: Output tensor of shape [num_groups, M, N] (初始值)
        group_list: List of group sizes for K-axis splitting
        a_trans: Whether input tensor is transposed (default: True)
        b_trans: Whether weight tensor is transposed (default: False)
    """
    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    y: torch.Tensor
    group_list: list
    a_trans: bool = True
    b_trans: bool = False


@dataclass
class GmmMxfp8Inputs:
    """
    Input parameters for generating MXFP8 output.

    Attributes:
        a: Input tensor of shape [K, M] (transposed format)
        b: Weight tensor of shape [K, N]
        scaled_a: Scale factors for input tensor [K//64, M, 2] (transposed format)
        scaled_b: Scale factors for weight tensor [K//64, N, 2]
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
        a_trans: Whether input tensor is transposed (default: True, x1 is [K, M])
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
    a_trans: bool = True
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    description: str = ""


@dataclass
class GoldenComputeInputs:
    """
    Input parameters for computing golden result in matrix multiplication.

    Attributes:
        x: Input tensor of shape [M, K] or [K, M] if transposed
        weight: Weight tensor of shape [K, N] or [N, K] if transposed
        scaled_x: Scale factors for input tensor
        scaled_weight: Scale factors for weight tensor
        a_trans: Whether input tensor is transposed
        b_trans: Whether weight tensor is transposed
    """
    x: torch.Tensor
    weight: torch.Tensor
    scaled_x: torch.Tensor
    scaled_weight: torch.Tensor
    a_trans: bool
    b_trans: bool


def compute_golden_result(inputs: GoldenComputeInputs) -> torch.Tensor:
    """
    Compute golden (reference) result for a single group's matrix multiplication.

    Args:
        inputs: Input parameters including tensors and transposition flags

    Returns:
        torch.Tensor: Golden output tensor
    """
    x = inputs.x
    weight = inputs.weight
    scaled_x_golden = inputs.scaled_x
    scaled_weight_golden = inputs.scaled_weight
    a_trans = inputs.a_trans
    b_trans = inputs.b_trans

    # Handle input transposition
    if a_trans:
        x = torch.swapaxes(x, -1, -2)
        scaled_x_golden = torch.swapaxes(scaled_x_golden, -1, -2)
        if len(scaled_x_golden.shape) == 3:
            scaled_x_golden = scaled_x_golden.reshape(
                scaled_x_golden.shape[0] * scaled_x_golden.shape[1], scaled_x_golden.shape[2]
            )
        scaled_x_golden = torch.swapaxes(scaled_x_golden, -1, -2)
    else:
        if len(scaled_x_golden.shape) == 3:
            scaled_x_golden = scaled_x_golden.reshape(
                scaled_x_golden.shape[0], scaled_x_golden.shape[1] * scaled_x_golden.shape[2]
            )

    # Handle weight transposition
    if b_trans:
        weight = torch.swapaxes(weight, -1, -2)
        if len(scaled_weight_golden.shape) == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(
                scaled_weight_golden.shape[0] * scaled_weight_golden.shape[1],
                scaled_weight_golden.shape[2]
            )
        scaled_weight_golden = torch.swapaxes(scaled_weight_golden, -1, -2)
    else:
        scaled_weight_golden = torch.swapaxes(scaled_weight_golden, -1, -2)
        if len(scaled_weight_golden.shape) == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(
                scaled_weight_golden.shape[0] * scaled_weight_golden.shape[1],
                scaled_weight_golden.shape[2]
            )

    # Adjust scales for K dimension alignment
    k_dim = x.shape[-1]
    if math.ceil(k_dim / 32) % 2 != 0:
        scaled_x_golden = scaled_x_golden[:, :-1]
        scaled_weight_golden = scaled_weight_golden[:-1, :]

    # Broadcast scale factors
    scaled_x_golden_broadcast = torch.repeat_interleave(scaled_x_golden, repeats=32, dim=-1)
    scaled_weight_golden_broadcast = torch.repeat_interleave(scaled_weight_golden, repeats=32, dim=-2)

    # Calculate padding lengths
    x1_dims = len(x.shape)
    x2_dims = len(weight.shape)
    x1_pad_len = scaled_x_golden_broadcast.shape[-1] - x.shape[-1]
    x2_pad_len = scaled_weight_golden_broadcast.shape[-2] - weight.shape[-2]

    # Pad input tensor
    x1_pad = [0, x1_pad_len]
    for _ in range(x1_dims - 1):
        x1_pad += [0, 0]
    x1_golden = torch.nn.functional.pad(x, x1_pad, mode='constant', value=0)

    # Pad weight tensor
    weight_pad = [0, 0]
    weight_pad += [0, x2_pad_len]
    for _ in range(x2_dims - 2):
        weight_pad += [0, 0]
    weight_golden = torch.nn.functional.pad(weight, weight_pad, mode='constant', value=0)

    # Apply scaling factors
    x_fp32 = x.to(torch.float32)
    scaled_x_golden_broadcast_fp32 = scaled_x_golden_broadcast.to(torch.float32)
    x1_golden = x_fp32 * scaled_x_golden_broadcast_fp32

    weight_fp32 = weight.to(torch.float32)
    scaled_weight_golden_broadcast_fp32 = scaled_weight_golden_broadcast.to(torch.float32)
    weight_golden = weight_fp32 * scaled_weight_golden_broadcast_fp32

    # Compute matrix multiplication
    golden = torch.matmul(x1_golden, weight_golden)

    return golden


def gen_golden(inputs: GmmGoldenInputs) -> torch.Tensor:
    """
    Generate golden (reference) output for grouped matrix multiplication using PyTorch.
    a is in [K, M] format (transposed), b is in [K, N] format.

    Args:
        inputs: Input parameters including tensors, scales, y, and group_list

    Returns:
        torch.Tensor: Golden output tensor of shape [num_groups, M, N]
    """
    a = inputs.a  # [K, M]
    b = inputs.b  # [K, N]
    scaled_a = inputs.scaled_a  # [K//64, M, 2]
    scaled_b = inputs.scaled_b  # [K//64, N, 2]
    y = inputs.y
    group_list = inputs.group_list
    a_trans = inputs.a_trans
    b_trans = inputs.b_trans

    round_num = len(group_list)
    golden_result = y.clone()
    begin = 0
    end = 0

    for i in range(round_num):
        begin = end
        end = end + group_list[i]

        # Extract input and weight for current group (切分 K 轴)
        # a: [K, M] (transposed) -> x: [K_block, M] or [M, K_block] depends on a_trans
        # b: [K, N] -> weight: [K_block, N]
        if a_trans:
            x = a[:, begin:end]  # [K, M] 切分 K 轴 -> [K_block, M] after transpose will be [M, K_block]
            scaled_x_golden = scaled_a[:, begin // 64 : end // 64, :]  # [K//64, M, 2] 切分后 [K_block//64, M, 2]
        else:
            x = a[begin:end, :]  # [M, K] 切分 K 轴 -> [M, K_block]
            scaled_x_golden = scaled_a[begin // 64 : end // 64, :, :]  # [M, K//64, 2] 切分后 [M, K_block//64, 2]
        
        weight = b[begin:end, :]  # [K_block, N]
        scaled_weight_golden = scaled_b[begin // 64 : end // 64, :, :]  # [K_block//64, N, 2]

        # Compute golden result for this group and inplace add
        golden_temp = compute_golden_result(
            GoldenComputeInputs(
                x=x,
                weight=weight,
                scaled_x=scaled_x_golden,
                scaled_weight=scaled_weight_golden,
                a_trans=a_trans,
                b_trans=b_trans,
            )
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
    
    默认处理转置格式：
    - a: [K, M] (transposed), scaled_a: [K//64, M, 2] (transposed)
    - b: [K, N] (not transposed), scaled_b: [K//64, N, 2] (not transposed)

    This kernel performs grouped matrix multiplication where each group uses
    a different K-axis block from input tensors, with MXFP8 quantization.

    Args:
        a: Input tensor [K, M] (transposed format)
        b: Weight tensor [K, N]
        scaled_a: Scale factors for input tensor [K//64, M, 2] (transposed format)
        scaled_b: Scale factors for weight tensor [K//64, N, 2]
        y: Output tensor [num_groups, M, N]
        group_list: List of group sizes for K-axis splitting
        tile_config: Tile configuration for computation
    """
    g = y.shape[0]
    m = y.shape[1]
    n = y.shape[2]
    mm_result_tensor = pypto.tensor([g, m, n], pypto.DT_FP32)

    begin = 0
    end = 0
    for i in range(g):
        begin = end
        end = end + group_list[i]

        # Extract input for current group (切分 K 轴)
        # a is [K, M] (transposed), slice along first dim -> x = a[:, begin:end]
        # But since a_trans=True, we need to slice the K dimension which is the first dim
        # Actually for transposed format [K, M], slicing K means a[begin:end, :]
        x = a[begin:end, :]  # [K_block, M]
        weight = b[begin:end, :]  # [K_block, N]
        scaled_x = scaled_a[begin // 64 : end // 64, :, :]  # [K_block//64, M, 2]
        scaled_weight = scaled_b[begin // 64 : end // 64, :, :]  # [K_block//64, N, 2]

        # Set vector tile shapes for scale processing
        pypto.set_vec_tile_shapes(
            tile_config.vector_tile_shape[0],
            tile_config.vector_tile_shape[1],
            tile_config.vector_tile_shape[2],
            tile_config.vector_tile_shape[3]
        )

        # Set cube tile shapes and perform scaled matrix multiplication
        pypto.set_cube_tile_shapes(
            tile_config.m_tile_shape,
            tile_config.k_tile_shape,
            tile_config.n_tile_shape
        )
        # x is [K, M] (transposed), scale_x is [K//64, M, 2] (transposed)
        # weight is [K, N] (not transposed), scale_weight is [K//64, N, 2] (not transposed)
        mm_result_tensor[i] = pypto.scaled_mm(
            x, weight, pypto.DT_FP32, scaled_x, scaled_weight,
            a_trans=True, scale_a_trans=True
        )
    y[:,:,:] = pypto.add(y, mm_result_tensor)


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
    
    Kernel 固定处理 a_trans=True 格式：
    - a: [K, M] (transposed), scaled_a: [K//64, M, 2] (transposed)
    - b: [K, N] (not transposed), scaled_b: [K//64, N, 2] (not transposed)

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
    group_list = tile_config.group_list
    num_groups = len(group_list)
    # Kernel 固定使用 a_trans=True
    a_trans = True
    b_trans = False

    # Generate input tensor in MXFP8 format - [K, M] (transposed format)
    a = torch.randn((k, m), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
    scaled_a = torch.randn((k // 64, m, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    # Generate weight tensor in MXFP8 format - 2D [K, N]
    b = torch.randn((k, n), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
    scaled_b = torch.randn((k // 64, n, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

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
            True,  # a_trans=True: x1 is [K, M]
            False,  # b_trans=False
            False,
            False,
            False,
            "Test with K-axis splitting: K=512 split into [256, 256], x1 transposed [K, M]"
        )
    )
