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
"""PyPTO clip kernel implementation.

clip 算子将输入张量的每个元素限制在 [min_val, max_val] 范围内。

公式: y = min(max(x, min_val), max_val)

参考: models/experimental/vector/ 中的实现模式
"""

import pypto
import torch


# ─────────────────────────────────────────────
# 1. JIT Kernel (2D only, all inputs reshaped to 2D)
# ─────────────────────────────────────────────

@pypto.frontend.jit
def clip_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
    min_val: float,
    max_val: float,
):
    """PyPTO jit kernel for clip operation.

    使用隐式 shape 推断，避免 DYNAMIC 维度错误。
    所有输入都被 reshape 为 2D 进行处理。
    """
    pypto.set_vec_tile_shapes(64, 128)
    result = pypto.clip(x, min_val, max_val)
    y[:] = result


# ─────────────────────────────────────────────
# 2. Wrapper 函数
# ─────────────────────────────────────────────

def clip_wrapper(
    x: torch.Tensor,
    min_val: float = None,
    max_val: float = None,
) -> torch.Tensor:
    """clip 算子 wrapper。

    将输入张量的每个元素限制在 [min_val, max_val] 范围内。

    Args:
        x: 输入 torch.Tensor，支持任意维度
        min_val: 最小值边界，默认 None（不限制下界）
        max_val: 最大值边界，默认 None（不限制上界）

    Returns:
        输出 torch.Tensor，shape 与输入相同
    """
    # 默认值处理
    if min_val is None:
        min_val = float('-inf')
    if max_val is None:
        max_val = float('inf')

    original_shape = x.shape
    original_dtype = x.dtype
    ndim = x.dim()

    # pypto.clip 只支持 FP32，需要类型转换
    need_cast = original_dtype != torch.float32
    if need_cast:
        x = x.float()

    # 统一 reshape 为 2D 进行处理
    # 1D -> 2D, 3D -> 2D, 4D -> 2D, >4D -> 2D
    if ndim != 2:
        x = x.reshape(-1, x.shape[-1] if x.shape[-1] > 1 else x.shape[-2])

    # 创建输出 tensor
    output = torch.empty_like(x)

    # 调用 kernel
    clip_kernel(x, output, min_val, max_val)

    # 恢复原始 shape
    if ndim != 2:
        output = output.reshape(original_shape)

    # 恢复原始 dtype
    if need_cast:
        output = output.to(original_dtype)

    return output
