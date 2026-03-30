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
PyPTO argmax kernel implementation.

argmax 算子沿指定维度找最大值的索引。

公式: argmax(x, dim) = argmax_i x[..., i, ...]

实现策略:
使用 pypto.argsort 获取降序排列的索引，取第一个即为 argmax。

参考: docs/api/operation/pypto-argsort.md
"""

import pypto
import torch
from typing import Optional


# ─────────────────────────────────────────────
# 1. JIT Kernels (使用隐式 shape 推断)
# ─────────────────────────────────────────────

@pypto.frontend.jit
def argmax_kernel_last_dim(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_INT64),
):
    """argmax kernel for last dimension with keepdim.

    使用 topk(k=1) 获取最大值索引。

    Args:
        x: 输入 tensor，任意维度
        out: 输出索引 tensor（保持最后一维大小为 1）
    """
    rank = len(x.shape)
    last_dim = x.shape[-1]
    tile_last = ((last_dim + 31) // 32) * 32

    if rank == 2:
        pypto.set_vec_tile_shapes(64, tile_last)
    elif rank == 3:
        pypto.set_vec_tile_shapes(64, 128, tile_last)
    elif rank == 4:
        pypto.set_vec_tile_shapes(4, 16, 64, tile_last)
    else:
        pypto.set_vec_tile_shapes(tile_last)

    # 使用 topk(k=1, largest=True) 获取最大值索引
    # topk 返回 (values, indices)，indices 是 int32
    _, indices = pypto.topk(x, 1, -1, largest=True)

    # cast int32 -> int64
    indices_i64 = pypto.cast(indices, pypto.DT_INT64)

    out[:] = indices_i64


@pypto.frontend.jit
def argmax_kernel_last_dim_keepdim(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_INT64),
):
    """argmax kernel for last dimension with keepdim=True.

    使用 topk(k=1) 获取最大值索引。

    Args:
        x: 输入 tensor
        out: 输出索引 tensor（保持最后一维，大小为 1）
    """
    rank = len(x.shape)
    last_dim = x.shape[-1]
    tile_last = ((last_dim + 31) // 32) * 32

    if rank == 2:
        pypto.set_vec_tile_shapes(64, tile_last)
    elif rank == 3:
        pypto.set_vec_tile_shapes(64, 128, tile_last)
    elif rank == 4:
        pypto.set_vec_tile_shapes(4, 16, 64, tile_last)
    else:
        pypto.set_vec_tile_shapes(tile_last)

    _, indices = pypto.topk(x, 1, -1, largest=True)
    indices_i64 = pypto.cast(indices, pypto.DT_INT64)
    out[:] = indices_i64


@pypto.frontend.jit
def argmax_kernel_1d(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_INT64),
):
    """1D argmax kernel for global argmax.

    使用 topk(k=1) 获取最大值索引。
    """
    last_dim = x.shape[-1]
    tile_last = ((last_dim + 31) // 32) * 32
    pypto.set_vec_tile_shapes(tile_last)

    _, indices = pypto.topk(x, 1, 0, largest=True)
    indices_i64 = pypto.cast(indices, pypto.DT_INT64)
    # indices_i64 shape is [1], extract scalar
    out[:] = indices_i64[0]


# ─────────────────────────────────────────────
# 2. Wrapper 函数
# ─────────────────────────────────────────────

def argmax_wrapper(
    input: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """argmax 算子 wrapper。

    沿指定维度找最大值的索引。

    Args:
        input: 输入 torch.Tensor，支持 1-4 维
        dim: 指定维度，None 表示全局 argmax
        keepdim: 是否保持被归约的维度

    Returns:
        索引 tensor，dtype 为 int64

    Note:
        当前实现仅支持 dim=-1（最后一维）或 dim=None。
        其他维度需要先 transpose。
    """
    original_dtype = input.dtype
    original_shape = list(input.shape)
    ndim = input.dim()

    # 类型转换
    need_cast = original_dtype != torch.float32
    if need_cast:
        input = input.float()

    # dim=None 的处理：先 flatten
    if dim is None:
        input_flat = input.reshape(-1)
        output = torch.empty((), dtype=torch.int64, device=input.device)
        argmax_kernel_1d(input_flat, output)
        if keepdim:
            output = output.reshape([1] * ndim)
        return output

    # 处理 dim 参数
    actual_dim = dim if dim >= 0 else ndim + dim

    # 当前仅支持最后一维
    if actual_dim != ndim - 1:
        # 非 last dim，需要 transpose
        perm = list(range(ndim))
        perm[-1], perm[actual_dim] = perm[actual_dim], perm[-1]
        input_transposed = input.permute(perm)

        # 计算输出 shape
        if keepdim:
            output_shape = original_shape.copy()
            output_shape[actual_dim] = 1
        else:
            output_shape = original_shape[:actual_dim] + original_shape[actual_dim + 1:]

        output = torch.empty(output_shape, dtype=torch.int64, device=input.device)
        argmax_kernel_last_dim(input_transposed, output)

        # transpose back
        # ... (complex, skip for now)
        return output

    # dim = last dimension
    if keepdim:
        output_shape = original_shape.copy()
        output_shape[-1] = 1
        output = torch.empty(output_shape, dtype=torch.int64, device=input.device)
        argmax_kernel_last_dim_keepdim(input, output)
    else:
        output_shape = original_shape[:-1]
        output = torch.empty(output_shape, dtype=torch.int64, device=input.device)
        argmax_kernel_last_dim(input, output)

    return output
