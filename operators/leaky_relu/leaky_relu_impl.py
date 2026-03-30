#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO LeakyReLU kernel implementation.

公式: y = x if x >= 0 else alpha * x
等价于: y = max(x, alpha * x)

实现策略:
1. 使用 pypto.maximum(x, alpha*x) 实现
2. 1D/3D+/4D 输入统一 reshape 到 2D 处理
3. 使用 pypto.set_vec_tile_shapes(64, 128)
"""

import pypto
import torch
from typing import Optional


# ─────────────────────────────────────────────
# 1. JIT Kernels (使用隐式 shape 推断)
# ─────────────────────────────────────────────

@pypto.frontend.jit
def leaky_relu_kernel_2d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
    alpha: float = 0.01,
):
    """LeakyReLU kernel for 2D input.

    公式: y = max(x, alpha * x)

    Args:
        x: 输入 tensor，2D shape
        out: 输出 tensor，与输入 shape 相同
        alpha: 负区间斜率
    """
    pypto.set_vec_tile_shapes(64, 128)

    # 负区间: x * alpha
    neg_val = pypto.mul(x, alpha)

    # 取最大值: max(x, alpha*x)
    result = pypto.maximum(x, neg_val)

    # 写回输出
    out[:] = result


@pypto.frontend.jit
def leaky_relu_kernel_1d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
    alpha: float = 0.01,
):
    """LeakyReLU kernel for 1D input."""
    pypto.set_vec_tile_shapes(128)

    neg_val = pypto.mul(x, alpha)
    result = pypto.maximum(x, neg_val)
    out[:] = result


# ─────────────────────────────────────────────
# 2. Wrapper 函数
# ─────────────────────────────────────────────

def leaky_relu_wrapper(
    input: torch.Tensor,
    alpha: float = 0.01,
) -> torch.Tensor:
    """LeakyReLU 算子 wrapper。

    逐元素计算 LeakyReLU 激活函数。

    Args:
        input: 输入 torch.Tensor，支持 1-4 维
        alpha: 负区间斜率，默认 0.01

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

    # 根据维度选择 kernel
    if ndim == 1:
        output = torch.empty(original_shape, dtype=torch.float32, device=input.device)
        leaky_relu_kernel_1d(input, output, alpha)
    elif ndim == 2:
        output = torch.empty(original_shape, dtype=torch.float32, device=input.device)
        leaky_relu_kernel_2d(input, output, alpha)
    else:
        # 3D, 4D 都 reshape 到 2D 处理
        input_2d = input.reshape(-1, input.shape[-1])
        output_2d = torch.empty(input_2d.shape, dtype=torch.float32, device=input.device)
        leaky_relu_kernel_2d(input_2d, output_2d, alpha)
        output = output_2d.reshape(original_shape)

    # 类型转换回原始 dtype
    if need_cast:
        output = output.to(original_dtype)

    return output
