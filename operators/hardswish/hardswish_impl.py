#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO Hardswish kernel implementation.

公式: hardswish(x) = x * relu6(x + 3) / 6
等价于: hardswish(x) = x * min(max(x + 3, 0), 6) / 6

实现策略:
1. 计算 x + 3
2. 计算 relu6(x + 3) = min(max(x + 3, 0), 6)
3. 除以 6
4. 乘以 x

动态轴支持: 使用 pypto.Tensor([], pypto.DT_FP32) 隐式推断
"""

import pypto
import torch
from typing import Optional


# ─────────────────────────────────────────────
# 1. JIT Kernels (使用隐式 shape 推断)
# ─────────────────────────────────────────────

@pypto.frontend.jit
def hardswish_kernel_2d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """Hardswish kernel for 2D input.

    公式: hardswish(x) = x * relu6(x + 3) / 6

    Args:
        x: 输入 tensor，2D shape
        out: 输出 tensor，与输入 shape 相同
    """
    pypto.set_vec_tile_shapes(64, 128)

    # Step 1: x + 3
    x_plus_3 = pypto.add(x, 3.0)

    # Step 2: relu6(x + 3) = min(max(x + 3, 0), 6)
    # max(x + 3, 0)
    relu_part = pypto.maximum(x_plus_3, 0.0)
    # min(max(x + 3, 0), 6)
    relu6_result = pypto.minimum(relu_part, 6.0)

    # Step 3: relu6(x + 3) / 6
    hardsigmoid = pypto.div(relu6_result, 6.0)

    # Step 4: x * hardsigmoid(x)
    result = pypto.mul(x, hardsigmoid)

    # 写回输出
    out[:] = result


@pypto.frontend.jit
def hardswish_kernel_1d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """Hardswish kernel for 1D input."""
    pypto.set_vec_tile_shapes(128)

    x_plus_3 = pypto.add(x, 3.0)
    relu_part = pypto.maximum(x_plus_3, 0.0)
    relu6_result = pypto.minimum(relu_part, 6.0)
    hardsigmoid = pypto.div(relu6_result, 6.0)
    result = pypto.mul(x, hardsigmoid)
    out[:] = result


# ─────────────────────────────────────────────
# 2. Wrapper 函数
# ─────────────────────────────────────────────

def hardswish_wrapper(
    input: torch.Tensor,
) -> torch.Tensor:
    """Hardswish 算子 wrapper。

    逐元素计算 Hardswish 激活函数。

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

    # 根据维度选择 kernel
    if ndim == 1:
        output = torch.empty(original_shape, dtype=torch.float32, device=input.device)
        hardswish_kernel_1d(input, output)
    elif ndim == 2:
        output = torch.empty(original_shape, dtype=torch.float32, device=input.device)
        hardswish_kernel_2d(input, output)
    else:
        # 3D, 4D 都 reshape 到 2D 处理
        input_2d = input.reshape(-1, input.shape[-1])
        output_2d = torch.empty(input_2d.shape, dtype=torch.float32, device=input.device)
        hardswish_kernel_2d(input_2d, output_2d)
        output = output_2d.reshape(original_shape)

    # 类型转换回原始 dtype
    if need_cast:
        output = output.to(original_dtype)

    return output
