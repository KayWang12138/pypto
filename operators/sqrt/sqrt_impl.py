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
PyPTO sqrt kernel implementation.

sqrt 算子逐元素计算输入张量的平方根。

公式: y = sqrt(x)

实现策略:
使用 pypto.sqrt() 直接计算平方根。
"""

import pypto
import torch


# ─────────────────────────────────────────────
# 1. JIT Kernels (使用隐式 shape 推断)
# ─────────────────────────────────────────────

@pypto.frontend.jit
def sqrt_kernel_2d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """sqrt kernel for 2D input."""
    pypto.set_vec_tile_shapes(64, 128)
    out[:] = pypto.sqrt(x)


@pypto.frontend.jit
def sqrt_kernel_1d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """sqrt kernel for 1D input."""
    pypto.set_vec_tile_shapes(128)
    out[:] = pypto.sqrt(x)


# ─────────────────────────────────────────────
# 2. Wrapper 函数
# ─────────────────────────────────────────────

def sqrt_wrapper(input: torch.Tensor) -> torch.Tensor:
    """sqrt 算子 wrapper。

    逐元素计算输入张量的平方根。

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
        sqrt_kernel_1d(input, output)
    else:
        # 2D, 3D, 4D 都 reshape 到 2D 处理
        input_2d = input.reshape(-1, input.shape[-1])
        output_2d = torch.empty(input_2d.shape, dtype=torch.float32, device=input.device)
        sqrt_kernel_2d(input_2d, output_2d)
        output = output_2d.reshape(original_shape)

    # 类型转换回原始 dtype
    if need_cast:
        output = output.to(original_dtype)

    return output
