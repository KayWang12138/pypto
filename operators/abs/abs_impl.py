#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO abs kernel implementation.

abs 算子逐元素计算输入张量的绝对值。

公式: y = |x|

实现策略:
使用 pypto.abs() 直接计算绝对值。

参考: docs/api/operation/pypto-abs.md
"""

import pypto
import torch
from typing import Optional


# ─────────────────────────────────────────────
# 1. JIT Kernels (使用隐式 shape 推断)
# ─────────────────────────────────────────────

@pypto.frontend.jit
def abs_kernel_2d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """abs kernel for 2D input.

    使用 pypto.abs() 计算绝对值。

    Args:
        x: 输入 tensor，2D shape
        out: 输出 tensor，与输入 shape 相同
    """
    pypto.set_vec_tile_shapes(64, 128)
    out[:] = pypto.abs(x)


@pypto.frontend.jit
def abs_kernel_3d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """abs kernel for 3D input.

    使用 pypto.abs() 计算绝对值。

    Args:
        x: 输入 tensor，3D shape
        out: 输出 tensor，与输入 shape 相同
    """
    pypto.set_vec_tile_shapes(64, 64, 128)
    out[:] = pypto.abs(x)


@pypto.frontend.jit
def abs_kernel_4d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """abs kernel for 4D input (reshaped to 2D).

    使用 pypto.abs() 计算绝对值。
    4D 输入会先 reshape 到 2D 处理。

    Args:
        x: 输入 tensor，4D shape
        out: 输出 tensor，与输入 shape 相同
    """
    pypto.set_vec_tile_shapes(64, 128)
    out[:] = pypto.abs(x)


@pypto.frontend.jit
def abs_kernel_1d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """abs kernel for 1D input.

    使用 pypto.abs() 计算绝对值。

    Args:
        x: 输入 tensor，1D shape
        out: 输出 tensor，与输入 shape 相同
    """
    pypto.set_vec_tile_shapes(128)
    out[:] = pypto.abs(x)


# ─────────────────────────────────────────────
# 2. Wrapper 函数
# ─────────────────────────────────────────────

def abs_wrapper(
    input: torch.Tensor,
) -> torch.Tensor:
    """abs 算子 wrapper。

    逐元素计算输入张量的绝对值。

    Args:
        input: 输入 torch.Tensor，支持 1-4 维

    Returns:
        输出 tensor，shape 与输入相同，每个元素为输入对应元素的绝对值
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

    # 根据维度选择 kernel
    if ndim == 1:
        abs_kernel_1d(input, output)
    elif ndim == 2:
        abs_kernel_2d(input, output)
    elif ndim == 3:
        abs_kernel_3d(input, output)
    elif ndim == 4:
        # 4D 输入 reshape 到 2D 处理，避免 4D tiling 编译问题
        input_2d = input.reshape(-1, input.shape[-1])
        output_2d = torch.empty(input_2d.shape, dtype=torch.float32, device=input.device)
        abs_kernel_4d(input_2d, output_2d)
        output = output_2d.reshape(original_shape)
    else:
        # 对于其他维度，reshape 到 2D 处理
        input_2d = input.reshape(-1, input.shape[-1] if input.shape[-1] > 1 else 1)
        output_2d = torch.empty(input_2d.shape, dtype=torch.float32, device=input.device)
        abs_kernel_2d(input_2d, output_2d)
        output = output_2d.reshape(original_shape)

    # 类型转换回原始 dtype
    if need_cast:
        output = output.to(original_dtype)

    return output
