#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
QuantGroupedMatmulInplaceAdd with MX Quantization

This module implements quantized grouped matrix multiplication with inplace addition using MX quantization.
Designed for micro-batch training scenarios where gradient accumulation is needed.
"""

import math
from dataclasses import dataclass

import numpy as np
import pypto
import torch
import torch_npu
from numpy.testing import assert_allclose


@dataclass
class QuantGroupedMatmulInplaceAddConfig:
    """
    Configuration parameters for quantized grouped matmul inplace add operation.

    Attributes:
        ori_shape: Original shape [M, K, N]
        num_groups: Number of groups for grouped computation
        m_tile_shape: Tile shape for M dimension in cube operation
        k_tile_shape: Tile shape for K dimension in cube operation
        n_tile_shape: Tile shape for N dimension in cube operation
        vec_tile_shape: Tile shapes for vector operations
        x1_dtype: Data type for x1 input (default: DT_FP8E5M2)
        x2_dtype: Data type for x2 input (default: DT_FP8E5M2)
        scale_dtype: Data type for scale tensors (default: DT_FP8E8M0)
        out_dtype: Data type for output (default: DT_FP32)
        description: Description of test case
    """
    ori_shape: list
    num_groups: int
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    vec_tile_shape: list
    x1_dtype: pypto.DataType = pypto.DT_FP8E5M2
    x2_dtype: pypto.DataType = pypto.DT_FP8E5M2
    scale_dtype: pypto.DataType = pypto.DT_FP8E8M0
    out_dtype: pypto.DataType = pypto.DT_FP32
    description: str = ""


def quant_grouped_matmul_inplace_add_pypto(config: QuantGroupedMatmulInplaceAddConfig):
    """
    Create quantized grouped matmul inplace add kernel using PyPTO.

    Args:
        config: Configuration parameters for operation

    Returns:
        function: JIT-compiled kernel function
    """
    m, k, n = config.ori_shape
    num_groups = config.num_groups
    
    x1_shape = [k, m]
    x2_shape = [k, n]
    scale1_shape = [(k // 64) + num_groups, m, 2]
    scale2_shape = [(k // 64) + num_groups, n, 2]
    y_shape = [num_groups, m, n]

    @pypto.frontend.jit()
    def quant_grouped_matmul_inplace_add_impl(
        x1: pypto.Tensor(x1_shape, config.x1_dtype),
        x2: pypto.Tensor(x2_shape, config.x2_dtype),
        scale1: pypto.Tensor(scale1_shape, config.scale_dtype),
        scale2: pypto.Tensor(scale2_shape, config.scale_dtype),
        y: pypto.Tensor(y_shape, config.out_dtype)
    ) -> pypto.Tensor(y_shape, config.out_dtype):
        pypto.set_cube_tile_shapes(config.m_tile_shape, config.k_tile_shape, config.n_tile_shape)
        pypto.set_vec_tile_shapes(
            config.vec_tile_shape[0],
            config.vec_tile_shape[1],
            config.vec_tile_shape[2],
            config.vec_tile_shape[3]
        )

        # Perform scaled matrix multiplication
        matmul_result = pypto.scaled_mm(x1, x2, config.out_dtype, scale1, scale2)
        
        # Inplace addition: y = y + matmul_result
        # Broadcast matmul_result to match y's shape [num_groups, M, N]
        for i in range(num_groups):
            y[i, :, :] = y[i, :, :] + matmul_result
        
        return y

    return quant_grouped_matmul_inplace_add_impl


def gen_golden_output(x1: torch.Tensor, x2: torch.Tensor, scale1: torch.Tensor, 
                     scale2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute golden (reference) output for quantized grouped matmul inplace add.

    This function implements the MX quantization formula:
        y_i[m,n] = Σ((Σ(x1Slice_i * x2Slice_i)) * (scale1_i[m, j] * scale2_i[j, n])) + y_i[m,n]

    Args:
        x1: Input tensor of shape [K, M]
        x2: Input tensor of shape [K, N]
        scale1: Scale factors for x1, shape [(K/64) + num_groups, M, 2]
        scale2: Scale factors for x2, shape [(K/64) + num_groups, N, 2]
        y: Input tensor for inplace addition, shape [num_groups, M, N]

    Returns:
        torch.Tensor: Golden output tensor of shape [num_groups, M, N]
    """
    K, M = x1.shape
    _, N = x2.shape
    num_groups = y.shape[0]
    gsK = 32  # Quantization block size for K axis
    kLoops = math.ceil(K / gsK)
    
    result = y.clone().float()
    
    for g in range(num_groups):
        for m_idx in range(M):
            for n_idx in range(N):
                acc = 0.0
                for j in range(kLoops):
                    k_start = j * gsK
                    k_end = min(k_start + gsK, K)
                    
                    # Extract slices
                    x1_slice = x1[k_start:k_end, m_idx].float()
                    x2_slice = x2[k_start:k_end, n_idx].float()
                    
                    # Compute dot product
                    dot_product = torch.sum(x1_slice * x2_slice)
                    
                    # Get scale factors
                    scale1_idx = min(j, scale1.shape[0] - 1)
                    scale2_idx = min(j, scale2.shape[0] - 1)
                    
                    scale1_val = scale1[scale1_idx, m_idx, 0].float()
                    scale2_val = scale2[scale2_idx, n_idx, 0].float()
                    
                    # Accumulate with scaling
                    acc += dot_product * scale1_val * scale2_val
                
                result[g, m_idx, n_idx] += acc
    
    return result


def run_quant_grouped_matmul_inplace_add_case(config: QuantGroupedMatmulInplaceAddConfig):
    """
    Test quantized grouped matmul inplace add implementation.

    This function runs a complete test for a given configuration:
    1. Generate test data with MX quantization format
    2. Compute golden (reference) output using PyTorch
    3. Compute output using PyPTO
    4. Compare results

    Args:
        config: Configuration parameters for test case
    """
    m, k, n = config.ori_shape
    num_groups = config.num_groups

    torch.manual_seed(42)
    
    # Generate input tensors in FP8 format
    x1 = torch.randn((k, m), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e5m2).npu()
    x2 = torch.randn((k, n), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e5m2).npu()
    
    # Generate scale tensors for MX quantization
    # scale1 shape: [(K/64) + num_groups, M, 2]
    # scale2 shape: [(K/64) + num_groups, N, 2]
    scale1 = torch.randn(
        (k // 64 + num_groups, m, 2), dtype=torch.float32
    ).uniform_(0, 1).to(torch.float8_e8m0fnu).npu()
    scale2 = torch.randn(
        (k // 64 + num_groups, n, 2), dtype=torch.float32
    ).uniform_(0, 1).to(torch.float8_e8m0fnu).npu()
    
    # Generate input/output tensor for inplace addition
    y = torch.randn((num_groups, m, n), dtype=torch.float32).uniform_(0, 1).npu()
    
    # Compute PyPTO output
    pypto_kernel = quant_grouped_matmul_inplace_add_pypto(config)
    pypto_out = pypto_kernel(x1, x2, scale1, scale2, y)
    
    # Compute golden output
    golden_out = gen_golden_output(
        x1.cpu(), x2.cpu(), scale1.cpu(), y.cpu()
    )
    
    # Verify results
    assert_allclose(pypto_out.cpu().float(), golden_out.cpu().float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    run_quant_grouped_matmul_inplace_add_case(
        QuantGroupedMatmulInplaceAddConfig(
            ori_shape=[16, 64, 16],
            num_groups=2,
            m_tile_shape=[16, 16],
            k_tile_shape=[64, 64],
            n_tile_shape=[16, 16],
            vec_tile_shape=[1, 8, 256, 32],
            description="Basic test with 2 groups"
        )
    )
