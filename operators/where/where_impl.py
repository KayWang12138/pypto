#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
"""PyPTO where kernel implementation.

实现说明：
  - 本文件是 where 算子的 PyPTO kernel 实现。
  - 导出函数 where_wrapper() 供 test_where.py 调用。
  - kernel 使用 @pypto.frontend.jit 装饰，内部使用 pypto.where API。
  - 基于 design.md 中的设计方案实现。
"""

import pypto
import torch
from typing import Union


# ─────────────────────────────────────────────
# 1. JIT Kernel - Tensor y 版本
# ─────────────────────────────────────────────

@pypto.frontend.jit
def where_kernel(
    condition: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BOOL),
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    """PyPTO where jit kernel - Tensor y 版本。

    Args:
        condition: 条件张量，DT_BOOL 类型
        x: condition 为 True 时选择的值
        y: condition 为 False 时选择的值
        output: 输出张量
    """
    # 设置 TileShape
    pypto.set_vec_tile_shapes(1, 128, 8, 64)

    # 调用 pypto.where
    result = pypto.where(condition, x, y)

    # 写回输出
    output[:] = result


# ─────────────────────────────────────────────
# 2. JIT Kernel - 标量 y 版本
# ─────────────────────────────────────────────

@pypto.frontend.jit
def where_scalar_kernel(
    condition: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BOOL),
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    y_scalar: pypto.Element,
):
    """PyPTO where jit kernel - 标量 y 版本。

    Args:
        condition: 条件张量，DT_BOOL 类型
        x: condition 为 True 时选择的值
        output: 输出张量
        y_scalar: condition 为 False 时选择的标量值
    """
    # 设置 TileShape
    pypto.set_vec_tile_shapes(1, 128, 8, 64)

    # 调用 pypto.where
    result = pypto.where(condition, x, y_scalar)

    # 写回输出
    output[:] = result


# ─────────────────────────────────────────────
# 3. Wrapper 函数（导出接口）
# ─────────────────────────────────────────────

def where_wrapper(
    condition: torch.Tensor,
    x: torch.Tensor,
    y: Union[torch.Tensor, float],
) -> torch.Tensor:
    """where 算子 wrapper，供 test_where.py 调用。

    负责：
    1. 构造输出 torch.Tensor
    2. 调用 JIT kernel
    3. 返回结果 torch.Tensor

    Args:
        condition: 条件张量，dtype 为 bool
        x: condition 为 True 时选择的值
        y: condition 为 False 时选择的值，支持 Tensor 或标量

    Returns:
        根据 condition 从 x 和 y 选择的元素
    """
    # 确保输入是连续的
    if not condition.is_contiguous():
        condition = condition.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    if isinstance(y, torch.Tensor) and not y.is_contiguous():
        y = y.contiguous()

    # 处理维度扩展到 4D
    if condition.dim() < 4:
        condition = condition.view([1] * (4 - condition.dim()) + list(condition.shape))
    if x.dim() < 4:
        x = x.view([1] * (4 - x.dim()) + list(x.shape))
    if isinstance(y, torch.Tensor) and y.dim() < 4:
        y = y.view([1] * (4 - y.dim()) + list(y.shape))

    # 广播到相同 shape
    target_shape = torch.broadcast_shapes(condition.shape, x.shape)
    if isinstance(y, torch.Tensor):
        target_shape = torch.broadcast_shapes(target_shape, y.shape)

    condition = condition.expand(target_shape)
    x = x.expand(target_shape)
    if isinstance(y, torch.Tensor):
        y = y.expand(target_shape)

    # 构造输出 tensor
    output = torch.empty(target_shape, dtype=x.dtype)

    # 转换为 pypto tensor
    condition_pto = pypto.from_torch(condition, dynamic_axis=[0, 1])
    x_pto = pypto.from_torch(x, dynamic_axis=[0, 1])
    output_pto = pypto.from_torch(output, dynamic_axis=[0, 1])

    # 根据是否为标量选择不同 kernel
    if isinstance(y, torch.Tensor):
        y_pto = pypto.from_torch(y, dynamic_axis=[0, 1])
        where_kernel(condition_pto, x_pto, y_pto, output_pto)
    else:
        # 使用 Element 类型传入标量
        y_element = pypto.Element(pypto.DT_FP32, float(y))
        where_scalar_kernel(condition_pto, x_pto, output_pto, y_element)

    return output
