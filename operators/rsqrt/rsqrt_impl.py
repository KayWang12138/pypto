#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO rsqrt kernel implementation.

公式: y = 1/sqrt(x)
"""

import pypto
import torch


# ─────────────────────────────────────────────
# 1. JIT Kernels (使用隐式 shape 推断)
# ─────────────────────────────────────────────

@pypto.frontend.jit
def rsqrt_kernel_2d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """rsqrt kernel for 2D input."""
    pypto.set_vec_tile_shapes(64, 128)
    out[:] = pypto.rsqrt(x)


@pypto.frontend.jit
def rsqrt_kernel_1d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """rsqrt kernel for 1D input."""
    pypto.set_vec_tile_shapes(128)
    out[:] = pypto.rsqrt(x)


# ─────────────────────────────────────────────
# 2. Wrapper 函数
# ─────────────────────────────────────────────

def rsqrt_wrapper(input: torch.Tensor) -> torch.Tensor:
    """rsqrt 算子 wrapper。

    逐元素计算输入张量的倒数平方根。

    Args:
        input: 输入 torch.Tensor，支持 1-4 维

    Returns:
        输出 tensor，shape 与输入相同
    """
    original_dtype = input.dtype
    original_shape = list(input.shape)
    ndim = input.dim()

    # 类型转换
    need_cast = original_dtype != torch.float32
    if need_cast:
        input = input.float()

    # 确保输入连续
    if not input.is_contiguous():
        input = input.contiguous()

    # 创建输出 tensor
    output = torch.empty(original_shape, dtype=torch.float32, device=input.device)

    # 根据维度选择 kernel，4D 输入 reshape 到 2D 处理
    if ndim == 1:
        rsqrt_kernel_1d(input, output)
    else:
        # 2D, 3D, 4D 都 reshape 到 2D 处理
        input_2d = input.reshape(-1, input.shape[-1])
        output_2d = torch.empty(input_2d.shape, dtype=torch.float32, device=input.device)
        rsqrt_kernel_2d(input_2d, output_2d)
        output = output_2d.reshape(original_shape)

    # 类型转换回原始 dtype
    if need_cast:
        output = output.to(original_dtype)

    return output
