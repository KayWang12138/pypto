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
QuantGroupedMatmulInplaceAdd with MX Quantization (New Frontend)

This module implements grouped matrix multiplication with inplace add and MXFP8 quantization using PyPTO new frontend.
Designed for micro-batch training scenarios for efficient gradient accumulation.
"""

from dataclasses import dataclass

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
        num_groups: Number of groups for computation
        m_tile_shape: Tile shape for M dimension in cube operation
        k_tile_shape: Tile shape for K dimension in cube operation
        n_tile_shape: Tile shape for N dimension in cube operation
        vec_tile_shape: Tile shapes for vector operations
        x1_dtype: Data type for x1 (default: DT_FP8E5M2)
        x2_dtype: Data type for x2 (default: DT_FP8E5M2)
        scale_dtype: Data type for scale (default: DT_FP8E8M0)
        out_dtype: Output data type (default: DT_FP32)
        description: Description of the test case
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
        config: Configuration parameters for the operation

    Returns:
        function: JIT-compiled kernel function
    """
    m, k, n = config.ori_shape
    num_groups = config.num_groups
    k_blocks = k // 64

    x1_shape = [k, m]
    x2_shape = [k, n]
    scale1_shape = [k_blocks + num_groups, m, 2]
    scale2_shape = [k_blocks + num_groups, n, 2]
    y_shape = [num_groups, m, n]

    @pypto.frontend.jit()
    def quant_grouped_matmul_inplace_add_impl(
        x1: pypto.Tensor(x1_shape, config.x1_dtype),
        x2: pypto.Tensor(x2_shape, config.x2_dtype),
        scale1: pypto.Tensor(scale1_shape, config.scale_dtype),
        scale2: pypto.Tensor(scale2_shape, config.scale_dtype),
        y: pypto.Tensor(y_shape, config.out_dtype)
    ) -> pypto.Tensor(y_shape, config.out_dtype):
        pypto.set_cube_tile_shapes(
            config.m_tile_shape,
            config.k_tile_shape,
            config.n_tile_shape
        )
        pypto.set_vec_tile_shapes(
            config.vec_tile_shape[0],
            config.vec_tile_shape[1],
            config.vec_tile_shape[2],
            config.vec_tile_shape[3]
        )

        for i in range(num_groups):
            scale1_slice = scale1[i * k_blocks : (i + 1) * k_blocks]
            scale2_slice = scale2[i * k_blocks : (i + 1) * k_blocks]
            mm_result = pypto.scaled_mm(
                x1, x2, config.out_dtype,
                scale1_slice, scale2_slice,
                a_trans=True, b_trans=False
            )
            y[i] = pypto.add(y[i], mm_result)

        return y

    return quant_grouped_matmul_inplace_add_impl


def gen_golden_output(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
    y: torch.Tensor,
    config: QuantGroupedMatmulInplaceAddConfig
) -> torch.Tensor:
    """
    Generate golden (reference) output using PyTorch.

    Args:
        x1: Input matrix 1 of shape [K, M]
        x2: Input matrix 2 of shape [K, N]
        scale1: Scale factors for x1 of shape [(K/64) + num_groups, M, 2]
        scale2: Scale factors for x2 of shape [(K/64) + num_groups, N, 2]
        y: Output matrix of shape [num_groups, M, N]
        config: Configuration parameters

    Returns:
        torch.Tensor: Golden output tensor of shape [num_groups, M, N]
    """
    m, k, n = config.ori_shape
    num_groups = config.num_groups
    k_blocks = k // 64
    gsK = 64

    x1_fp32 = x1.to(torch.float32)
    x2_fp32 = x2.to(torch.float32)
    golden_result = y.clone()

    for i in range(num_groups):
        scale1_group = scale1[i * k_blocks : (i + 1) * k_blocks]
        scale2_group = scale2[i * k_blocks : (i + 1) * k_blocks]

        scale1_fp32 = scale1_group.to(torch.float32)
        scale2_fp32 = scale2_group.to(torch.float32)

        if k_blocks > 0:
            scale1_broadcast = torch.repeat_interleave(scale1_fp32[:, :, 0], repeats=gsK, dim=0)
            scale2_broadcast = torch.repeat_interleave(scale2_fp32[:, :, 0], repeats=gsK, dim=0)

            if scale1_broadcast.shape[0] > k:
                scale1_broadcast = scale1_broadcast[:k]
            if scale2_broadcast.shape[0] > k:
                scale2_broadcast = scale2_broadcast[:k]

            x1_scaled = x1_fp32 * scale1_broadcast.unsqueeze(1)
            x2_scaled = x2_fp32 * scale2_broadcast.unsqueeze(1)
        else:
            x1_scaled = x1_fp32
            x2_scaled = x2_fp32

        mm_result = torch.matmul(x1_scaled.T, x2_scaled)
        golden_result[i] = golden_result[i] + mm_result

    return golden_result


def run_quant_grouped_matmul_inplace_add_case(config: QuantGroupedMatmulInplaceAddConfig):
    """
    Test the quantized grouped matmul inplace add implementation.

    Args:
        config: Configuration parameters for the test case
    """
    m, k, n = config.ori_shape
    num_groups = config.num_groups
    k_blocks = k // 64

    torch.manual_seed(42)

    x1 = torch.randn((k, m), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e5m2).npu()
    x2 = torch.randn((k, n), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e5m2).npu()
    y = torch.randn((num_groups, m, n), dtype=torch.float32).npu()
    y_golden = y.clone()

    scale1 = torch.ones(
        (k_blocks + num_groups, m, 2),
        dtype=torch.float8_e8m0fnu
    ).npu()
    scale2 = torch.ones(
        (k_blocks + num_groups, n, 2),
        dtype=torch.float8_e8m0fnu
    ).npu()

    kernel = quant_grouped_matmul_inplace_add_pypto(config)
    pypto_out = kernel(x1, x2, scale1, scale2, y)

    golden_out = gen_golden_output(
        x1.cpu(), x2.cpu(), scale1.cpu(), y_golden.cpu(), config
    )

    pypto_out_cpu = pypto_out.cpu().float()
    golden_out_cpu = golden_out.cpu().float()

    assert_allclose(pypto_out_cpu, golden_out_cpu, rtol=1e-3, atol=1e-3)
    print(f"Test passed: {config.description}")


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