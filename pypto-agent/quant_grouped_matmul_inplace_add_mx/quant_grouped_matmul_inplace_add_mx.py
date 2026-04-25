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
        scaled_a: Scale factors for input tensor, shape ((K//64)+g, M, 2)
        scaled_b: Scale factors for weight tensor, shape ((K//64)+g, N, 2)
        y: Output tensor of shape [num_groups, M, N] (初始值)
        group_list: List of group sizes for K-axis splitting
        group_type: Type of group_list interpretation (default: 0)
            - 0: group_list elements are cumulative K values, last element equals K
            - 1: group_list elements are individual group sizes, sum equals K
        a_trans: Whether input tensor is transposed (default: True)
        b_trans: Whether weight tensor is transposed (default: False)
    
    MX量化scale存储格式说明:
        - scaled_a/scaled_b 形状: ((K//64)+g, M/N, 2)
        - 第 i 个 group 的 scale 偏移量 = sum(K_j/64 for j<i) + i
        - 第 i 个 group 的 scale 长度 = K_i/64
        - 示例: K=512, g=2, group_list=[256,256]
          - scaled_a shape: (10, M, 2)
          - group 0: offset=0, length=4, range [0:4,:,:]
          - group 1: offset=5, length=4, range [5:9,:,:]
    """
    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    y: torch.Tensor
    group_list: list
    group_type: int = 0
    a_trans: bool = True
    b_trans: bool = False


@dataclass
class GmmMxfp8Inputs:
    """
    Input parameters for generating MXFP8 output.

    Attributes:
        a: Input tensor of shape [K, M] (transposed format)
        b: Weight tensor of shape [K, N]
        scaled_a: Scale factors for input tensor, shape ((K//64)+g, M, 2)
        scaled_b: Scale factors for weight tensor, shape ((K//64)+g, N, 2)
        y: Output tensor of shape [num_groups, M, N] (需要在外部初始化)
        tile_config: Tile configuration for computation (包含 group_list 和 group_type)
    
    MX量化scale存储格式说明:
        - scaled_a/scaled_b 形状: ((K//64)+g, M/N, 2)
        - 第 i 个 group 的 scale 偏移量 = sum(K_j/64 for j<i) + i = begin/64 + i
        - 示例: K=512, g=2, group_list=[256,256]
          - scaled_a shape: ((512/64)+2, M, 2) = (10, M, 2)
          - group 0: scale_offset=0, scale_length=4, range [0:4,:,:]
          - group 1: scale_offset=5, scale_length=4, range [5:9,:,:]
    """
    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    y: torch.Tensor
    tile_config: 'ShapeConfig'


@dataclass
class ShapeConfig:
    """
    Configuration parameters for grouped matrix multiplication with MXFP8 quantization.

    Attributes:
        ori_shape: Original shape [M, K, N]
        group_list: List of group sizes for each weight group
        m_tile_shape: Tile shape for M dimension in cube operation
        k_tile_shape: Tile shape for K dimension in cube operation
        n_tile_shape: Tile shape for N dimension in cube operation
        vector_tile_shape: Tile shapes for vector operations
        group_type: Type of group_list interpretation (default: 0)
            - 0: group_list elements are cumulative K values, last element equals K
                  Example: [256, 512] means first group K=[0,256], second group K=[256,512]
            - 1: group_list elements are individual group sizes, sum equals K
                  Example: [256, 256] means K=256+256=512
        in_dtype: Input tensor data type (default: DT_FP8E4M3)
            - DT_FP8E4M3: FP8 E4M3FN format (torch.float8_e4m3fn)
            - DT_FP8E5M2: FP8 E5M2 format (torch.float8_e5m2)
        a_trans: Whether input tensor is transposed (default: True, x1 is [K, M])
        b_trans: Whether weight tensor is transposed (default: False)
        a_format_nz: Whether input uses NZ format (default: False)
        b_format_nz: Whether weight uses NZ format (default: False)
        c_format_nz: Whether output uses NZ format (default: False)
        description: Description of the test case
    
    约束说明：
        - K 轴必须 64 元素对齐（MX 量化要求）
        - 当 a_trans=True 时，M 维度（内轴）需要 32 字节对齐
        - 当 b_trans=False 时，N 维度（内轴）需要 32 字节对齐
    """
    ori_shape: list
    group_list: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    vector_tile_shape: list
    group_type: int = 0
    in_dtype: pypto.DataType = pypto.DT_FP8E4M3
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
    
    根据 a_trans 参数决定数据格式：
    - a_trans=True: a is [K, M], scaled_a is ((K//64)+g, M, 2)
    - a_trans=False: a is [M, K], scaled_a is ((K//64)+g, M, 2)
    - b is always [K, N], scaled_b is ((K//64)+g, N, 2)

    Args:
        inputs: Input parameters including tensors, scales, y, group_list and group_type

    Returns:
        torch.Tensor: Golden output tensor of shape [num_groups, M, N]
    """
    a = inputs.a
    b = inputs.b
    scaled_a = inputs.scaled_a  # ((K//64)+g, M, 2)
    scaled_b = inputs.scaled_b  # ((K//64)+g, N, 2)
    y = inputs.y
    group_list = inputs.group_list
    group_type = inputs.group_type
    a_trans = inputs.a_trans
    b_trans = inputs.b_trans

    round_num = len(group_list)
    golden_result = y.clone()
    
    # 根据 group_type 计算 begin 和 end (K轴切分)
    # group_type=0: group_list 各元素为累计值，最后一个元素等于 K
    # group_type=1: group_list 各元素为单独的 group size，累加得到 K
    begin = 0
    end = 0
    for i in range(round_num):
        if group_type == 1:
            # 累加方式: begin 指向当前组起始，end 指向当前组结束
            begin = end
            end = end + group_list[i]
        else:
            # group_list 直接给出累计值
            begin = 0 if i == 0 else group_list[i - 1]
            end = group_list[i]

        # 计算 scale 的偏移量
        # MX量化存储格式: scale_i 偏移 = sum(K_j/64 for j<i) + i = begin/64 + i
        scale_offset = begin // 64 + i
        scale_length = (end - begin) // 64

        # Extract input and weight for current group (切分 K 轴)
        # a_trans=True: a is [K, M], 切分 K 轴 = 切分第一维 -> a[begin:end, :] -> [K_block, M]
        # a_trans=False: a is [M, K], 切分 K 轴 = 切分第二维 -> a[:, begin:end] -> [M, K_block]
        if a_trans:
            x = a[begin:end, :]  # [K, M] 切分 K 轴 -> [K_block, M]
            scaled_x_golden = scaled_a[scale_offset : scale_offset + scale_length, :, :]  # [K_block//64, M, 2]
        else:
            x = a[:, begin:end]  # [M, K] 切分 K 轴 -> [M, K_block]
            scaled_x_golden = scaled_a[:, scale_offset : scale_offset + scale_length, :]  # [M, K_block//64, 2]
        
        weight = b[begin:end, :]  # [K_block, N]
        scaled_weight_golden = scaled_b[scale_offset : scale_offset + scale_length, :, :]  # [K_block//64, N, 2]

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


@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1},
)
def scaled_matmul_kernel(
    a: pypto.Tensor(),
    b: pypto.Tensor(),
    scaled_a: pypto.Tensor(),
    scaled_b: pypto.Tensor(),
    y: pypto.Tensor(),
    tile_config: ShapeConfig
):
    g = y.shape[0]
    m = y.shape[1]
    n = y.shape[2]
    mm_result_tensor = pypto.tensor([g, m, n], pypto.DT_FP32)

    group_list = tile_config.group_list
    group_type = tile_config.group_type
    a_trans = tile_config.a_trans
    b_trans = tile_config.b_trans
    
    begin = 0
    end = 0
    for i in range(g):
        if group_type == 1:
            # 累加方式: begin 指向当前组起始，end 指向当前组结束
            begin = end
            end = end + group_list[i]
        else:
            # group_list 直接给出累计值
            begin = 0 if i == 0 else group_list[i - 1]
            end = group_list[i]

        # 计算 scale 的偏移量 (MX量化存储格式)
        # scale_i 偏移 = sum(K_j/64 for j<i) + i = begin/64 + i
        scale_offset = begin // 64 + i
        scale_length = (end - begin) // 64

        # Extract input for current group (切分 K 轴)
        # 根据 a_trans 决定切分方式：
        # - a_trans=True: a=[K, M], 切分 K 轴 = 切分第一维 -> a[begin:end, :] -> [K_block, M]
        # - a_trans=False: a=[M, K], 切分 K 轴 = 切分第二维 -> a[:, begin:end] -> [M, K_block]
        if a_trans:
            x = a[begin:end, :]  # [K_block, M]
        else:
            x = a[:, begin:end]  # [M, K_block]
        
        # b 固定为 [K, N] (b_trans=False)，切分 K 轴 = 切分第一维
        weight = b[begin:end, :]  # [K_block, N]
        
        # 切分 scale tensor
        # 根据 a_trans 和 scale_a_trans 决定切分方式：
        # - a_trans=True: scaled_a[((K//64)+g, M, 2)], 切分第一维
        # - a_trans=False: scaled_a[(M, (K//64)+g, 2)], 切分第二维
        if a_trans:
            scaled_x = scaled_a[scale_offset : scale_offset + scale_length, :, :]  # [K_block//64, M, 2]
        else:
            scaled_x = scaled_a[:, scale_offset : scale_offset + scale_length, :]  # [M, K_block//64, 2]
        
        # scaled_b 固定为 ((K//64)+g, N, 2) 格式，切分第一维
        scaled_weight = scaled_b[scale_offset : scale_offset + scale_length, :, :]  # [K_block//64, N, 2]

        # Set vector tile shapes for scale processing
        pypto.set_vec_tile_shapes(
            tile_config.vector_tile_shape[0],
            tile_config.vector_tile_shape[1],
            tile_config.vector_tile_shape[2],
            tile_config.vector_tile_shape[3]
        )

        # Set cube tile shapes for scaled_mm
        # 重要: tile shape 需满足内轴 32 字节对齐约束
        pypto.set_cube_tile_shapes(
            tile_config.m_tile_shape,  # M 维度 tile
            tile_config.k_tile_shape,  # K 维度 tile
            tile_config.n_tile_shape   # N 维度 tile
        )
        
        # Cube 操作: scaled_mm (矩阵乘法)
        # 根据 a_trans 决定 scale_a_trans:
        # - a_trans=True: scale_a_trans=True (scale 形状 [K//64, M, 2])
        # - a_trans=False: scale_a_trans=False (scale 形状 [M, K//64, 2])
        scale_a_trans = a_trans
        mm_result_tensor[i] = pypto.scaled_mm(
            x, weight, pypto.DT_FP32, scaled_x, scaled_weight,
            a_trans=a_trans, scale_a_trans=scale_a_trans, b_trans=b_trans
        )
        
        y[:, :, :] = pypto.add(y, mm_result_tensor)


def gen_mxfp8(inputs: GmmMxfp8Inputs) -> torch.Tensor:
    """
    Generate MXFP8 output using PyPTO scaled matrix multiplication with new frontend.

    Args:
        inputs: Input parameters including tensors, scales, y and tile config

    Returns:
        torch.Tensor: Output tensor of shape [num_groups, M, N] in FP32
    """
    a = inputs.a
    b = inputs.b
    scaled_a = inputs.scaled_a
    scaled_b = inputs.scaled_b
    y = inputs.y
    tile_config = inputs.tile_config
    
    # Move tensors to NPU
    a = a.npu()
    b = b.npu()
    scaled_a = scaled_a.npu()
    scaled_b = scaled_b.npu()
    y = y.npu()

    # Execute scaled matrix multiplication kernel with new frontend
    # tile_config 包含 group_list、group_type 和 tile shapes
    scaled_matmul_kernel(a, b, scaled_a, scaled_b, y, tile_config)

    y = y.to(torch.float32)
    return y


def test_gmm_mxfp8(tile_config: ShapeConfig):
    """
    Test the grouped matrix multiplication with MXFP8 quantization.
    
    Kernel 固定处理 a_trans=True 格式：
    - a: [K, M] (transposed), scaled_a: ((K//64)+g, M, 2)
    - b: [K, N] (not transposed), scaled_b: ((K//64)+g, N, 2)

    This function runs a complete test for a given configuration:
    1. Generate test data with MXFP8 format
    2. Compute golden (reference) output using PyTorch
    3. Compute output using PyPTO
    4. Compare results

    Args:
        tile_config: Configuration parameters for the test case
    
    MX量化scale存储格式说明:
    ==========================
    - scaled_a/scaled_b 形状: ((K//64)+g, M/N, 2)
    - 第 i 个 group 的 scale 偏移量 = sum(K_j/64 for j<i) + i = begin/64 + i
    - 示例: K=512, g=2, group_list=[256,256]
      - scaled_a shape: ((512/64)+2, M, 2) = (10, M, 2)
      - group 0: K=[0,256], scale_offset=0, scale_length=4, range [0:4,:,:]
      - group 1: K=[256,512], scale_offset=5, scale_length=4, range [5:9,:,:]
    
    测试用例选择约束:
    ==================
    1. M 维度约束 (a_trans=True 时内轴):
       - M >= 32 (FP8 格式下 32 字节对齐)
       - m_tile_shape 的内轴分量 >= 32
       - 示例: M=32, m_tile_shape=[32, 32] ✓
       - 错误示例: M=16 或 m_tile_shape=[16, 16] ✗ (报错: ml0 memory not aligned)
    
    2. K 维度约束:
       - K >= 64 且 64 对齐 (MX 量化要求)
       - group_list 各元素之和 = K
       - 示例: K=512, group_list=[256, 256] ✓
    
    3. N 维度约束 (b_trans=False 时内轴):
       - N >= 32 (FP8 格式下 32 字节对齐)
       - n_tile_shape 的内轴分量 >= 32
       - 示例: N=7168 ✓
    
    4. Tile Shape 选择建议:
       - m_tile_shape: 推荐 [32, 32] 或 [64, 64]，保证 M 内轴 32 字节对齐
       - k_tile_shape: 推荐 [64, 64] 或更大，满足 MX 量化 64 对齐
       - n_tile_shape: 推荐 [32, 32] 或更大，保证 N 内轴 32 字节对齐
       - vector_tile_shape: 推荐 [1, 8, 256, 32]
    """
    # Extract configuration parameters
    m = tile_config.ori_shape[0]
    k = tile_config.ori_shape[1]
    n = tile_config.ori_shape[2]
    group_list = tile_config.group_list
    group_type = tile_config.group_type
    in_dtype = tile_config.in_dtype
    a_trans = tile_config.a_trans
    b_trans = tile_config.b_trans
    num_groups = len(group_list)

    # Map pypto dtype to torch dtype
    torch_dtype_map = {
        pypto.DT_FP8E4M3: torch.float8_e4m3fn,
        pypto.DT_FP8E5M2: torch.float8_e5m2,
    }
    torch_dtype = torch_dtype_map.get(in_dtype, torch.float8_e4m3fn)

    # Generate input tensor in MXFP8 format
    # 根据 a_trans 决定数据格式：
    # - a_trans=True: a=[K, M] (transposed format)
    # - a_trans=False: a=[M, K] (normal format)
    if a_trans:
        a = torch.randn((k, m), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
    else:
        a = torch.randn((m, k), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
    
    # Generate scaled_a in MXFP8 format
    # 根据 a_trans 决定 scale 格式：
    # - a_trans=True: scaled_a=((K//64)+g, M, 2) - scale_a_trans=True
    # - a_trans=False: scaled_a=(M, (K//64)+g, 2) - scale_a_trans=False
    # MX量化存储格式: 所有group的scale存储在一个连续tensor中
    if a_trans:
        scaled_a = torch.randn((k // 64 + num_groups, m, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    else:
        scaled_a = torch.randn((m, k // 64 + num_groups, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    # Generate weight tensor in MXFP8 format
    # 当前固定使用 b_trans=False: b=[K, N]
    # TODO: 如果需要支持 b_trans=True，需要添加对应逻辑
    if b_trans:
        b = torch.randn((n, k), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scaled_b = torch.randn((n, k // 64 + num_groups, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    else:
        b = torch.randn((k, n), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scaled_b = torch.randn((k // 64 + num_groups, n, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

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
        group_type=group_type,
        a_trans=a_trans,
        b_trans=b_trans,
    ))
    result = gen_mxfp8(GmmMxfp8Inputs(
        a=a,
        b=b,
        scaled_a=scaled_a,
        scaled_b=scaled_b,
        y=y_init_npu,
        tile_config=tile_config,
    ))

    # Verify results
    assert_allclose(golden.cpu().numpy(), result.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(golden)
    print(result)
    print(tile_config.description," PASSED")

# pypto.jit不支持
# scaled_mm接口k需要64对齐（pypto API限制）
# tile配置影响精度
# 代码初始tile配置固化（agent默认）

# MDE审视文档

# 0.147asendc

if __name__ == "__main__":
    # 测试用例1: 基础用例 (FP8E4M3)
    # - M=32, K=512, N=7168 (N较大)
    # - group_list=[256, 256], g=2
    # 性能分析：
    #   - M=32较小，算数强度受限
    #   - N=7168很大，切分次数需减少
    #   - 当前nL1=1024，切分7次 → 增大nL1减少切分
    # 优化配置：
    #   - mL1=M=32，消除A矩阵重复载入
    #   - kL1=512=K_group，使能大包搬运
    #   - nL1=2048，切分次数从7减到3.5
    #   - nL0=256增大算数强度
    #   - L0B: 256×64=16KB ✓
    test_gmm_mxfp8(
        ShapeConfig(
            ori_shape=[32, 512, 7168],
            group_list=[256, 256],
            m_tile_shape=[32, 32],
            k_tile_shape=[64, 256],
            n_tile_shape=[512, 2048],
            vector_tile_shape=[1, 8, 2048, 2],
            group_type=1,
            in_dtype=pypto.DT_FP8E4M3,
            a_trans=True,
            b_trans=False,
            a_format_nz=False,
            b_format_nz=False,
            c_format_nz=False,
            description="Case1: FP8E4M3, K=512, g=2, group_list=[256,256]"
        )
    )
    
    # 测试用例2: 更大M维度 (FP8E4M3)
    # - M=64, K=1024, N=4096
    # - group_list=[512, 512], g=2, K_block=512
    # cube_tile_shapes优化（参考Case1最优配置）：
    #   - mL1=64=M，消除A矩阵切分
    #   - kL1=512=K_block，使能大包搬运
    #   - nL1=2048，切分次数=4096/2048=2
    #   - nL0=512，增大算数强度
    #   - L0B约束: 512×64=32KB ✓ (<64KB)
    # vector_tile_shape约束：
    #   - dim0=1 (batch维度)
    #   - dim1=8 (K_block//64=512//64=8，对应scale tensor第一维)
    #   - dim2=2048 (对应nL1切分粒度)
    #   - dim3=2 (对应scale tensor第三维，固定值)
    test_gmm_mxfp8(
        ShapeConfig(
            ori_shape=[64, 1024, 4096],
            group_list=[512, 512],
            m_tile_shape=[64, 64],     # mL1=M，消除切分
            k_tile_shape=[64, 512],    # kL0=64, kL1=512=K_block，大包搬运
            n_tile_shape=[512, 2048],  # nL0=512, nL1=2048，切分2次
            vector_tile_shape=[1, 8, 2048, 2],  # dim1=8 (512//64), dim3=2 匹配scale tensor第三维
            group_type=1,
            in_dtype=pypto.DT_FP8E4M3,
            a_trans=True,
            b_trans=False,
            a_format_nz=False,
            b_format_nz=False,
            c_format_nz=False,
            description="Case2: FP8E4M3, M=64, K=1024, g=2, group_list=[512,512]"
        )
    )
    
    # 测试用例3: 3个分组 (FP8E4M3)
    # - M=32, K=768, N=2048
    # - group_list=[256, 256, 256], g=3, K_block=256
    # cube_tile_shapes优化（参考Case1最优配置）：
    #   - mL1=32=M，消除A矩阵切分
    #   - kL1=256=K_block，使能大包搬运
    #   - nL1=1024，切分次数=2048/1024=2
    #   - nL0=512，增大算数强度
    #   - L0B约束: 512×64=32KB ✓ (<64KB)
    # vector_tile_shape约束：
    #   - dim0=1 (batch维度)
    #   - dim1=4 (K_block//64=256//64=4，对应scale tensor第一维)
    #   - dim2=1024 (对应nL1切分粒度)
    #   - dim3=2 (对应scale tensor第三维，固定值)
    test_gmm_mxfp8(
        ShapeConfig(
            ori_shape=[32, 768, 2048],
            group_list=[256, 256, 256],
            m_tile_shape=[32, 32],     # mL1=M
            k_tile_shape=[64, 256],    # kL0=64, kL1=256=K_block，大包搬运
            n_tile_shape=[512, 1024],  # nL0=512, nL1=1024，切分2次
            vector_tile_shape=[1, 4, 1024, 2],  # dim1=4 (256//64), dim3=2 匹配scale tensor第三维
            group_type=1,
            in_dtype=pypto.DT_FP8E4M3,
            a_trans=True,
            b_trans=False,
            a_format_nz=False,
            b_format_nz=False,
            c_format_nz=False,
            description="Case3: FP8E4M3, K=768, g=3, group_list=[256,256,256]"
        )
    )
    
    # 测试用例4: group_type=0 (累计值模式) (FP8E4M3)
    # - M=32, K=512, N=1024
    # - group_list=[256, 512] (累计值，表示group0 K=[0,256], group1 K=[256,512])
    # - K_block: group0=256, group1=256
    # cube_tile_shapes优化（参考Case1最优配置）：
    #   - mL1=32=M，消除A矩阵切分
    #   - kL1=256=K_block，使能大包搬运
    #   - nL1=1024=N，消除B矩阵切分
    #   - nL0=512，增大算数强度
    #   - L0B约束: 512×64=32KB ✓ (<64KB)
    # vector_tile_shape约束：
    #   - dim0=1 (batch维度)
    #   - dim1=4 (K_block//64=256//64=4，对应scale tensor第一维)
    #   - dim2=1024 (对应nL1切分粒度)
    #   - dim3=2 (对应scale tensor第三维，固定值)
    test_gmm_mxfp8(
        ShapeConfig(
            ori_shape=[32, 512, 1024],
            group_list=[256, 512],
            m_tile_shape=[32, 32],     # mL1=M
            k_tile_shape=[64, 256],    # kL0=64, kL1=256=K_block，大包搬运
            n_tile_shape=[512, 1024],  # nL0=512, nL1=1024=N，消除切分
            vector_tile_shape=[1, 4, 1024, 2],  # dim1=4 (256//64), dim3=2 匹配scale tensor第三维
            group_type=0,
            in_dtype=pypto.DT_FP8E4M3,
            a_trans=True,
            b_trans=False,
            a_format_nz=False,
            b_format_nz=False,
            c_format_nz=False,
            description="Case4: FP8E4M3, group_type=0, group_list=[256,512] cumulative"
        )
    )

    # 测试用例5: FP8E5M2 数据类型
    # - M=32, K=512, N=1024
    # - group_list=[128, 384], g=2, K_block: 128, 384
    # cube_tile_shapes优化（参考Case1最优配置）：
    #   - mL1=32=M，消除A矩阵切分
    #   - kL1=384=max(K_block)，使能大包搬运
    #   - nL1=1024=N，消除B矩阵切分
    #   - nL0=512，增大算数强度
    #   - L0B约束: 512×64=32KB ✓ (<64KB)
    # vector_tile_shape约束：
    #   - dim0=1 (batch维度)
    #   - dim1=6 (max(K_block)//64=384//64=6，对应scale tensor第一维)
    #   - dim2=1024 (对应nL1切分粒度)
    #   - dim3=2 (对应scale tensor第三维，固定值)
    test_gmm_mxfp8(
        ShapeConfig(
            ori_shape=[32, 512, 1024],
            group_list=[128, 384],
            m_tile_shape=[32, 32],     # mL1=M
            k_tile_shape=[64, 384],    # kL0=64, kL1=384=max(K_block)，大包搬运
            n_tile_shape=[512, 1024],  # nL0=512, nL1=1024=N，消除切分
            vector_tile_shape=[1, 6, 1024, 2],  # dim1=6 (384//64), dim3=2 匹配scale tensor第三维
            group_type=1,
            in_dtype=pypto.DT_FP8E5M2,
            a_trans=True,
            b_trans=False,
            a_format_nz=False,
            b_format_nz=False,
            c_format_nz=False,
            description="Case5: FP8E5M2, K=512, g=2, group_list=[128, 384]"
        )
    )
