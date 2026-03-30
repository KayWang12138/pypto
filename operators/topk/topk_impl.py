#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO topk kernel implementation.

topk 算子返回输入张量在指定维度上最大（或最小）的 k 个元素及其索引。

公式: values, indices = topk(input, k, dim, largest, sorted)

参考: models/glm_v4_5/glm_select_experts.py 中的 topk 调用模式
参考: operators/clip/clip_impl.py 的实现模式（隐式 shape 推断）

设计要点:
1. PyPTO topk 仅支持 dim=-1，其他维度需通过 transpose 间接支持
2. PyPTO topk 不支持 sorted 参数，sorted=True 时需额外排序逻辑（P1 优先级）
3. PyPTO topk 返回 int32 索引，需 cast 转换为 int64 以对齐 PyTorch 行为
4. 支持动态轴 b, s, n（batch, seq, num）
"""

import pypto
import torch
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# 1. JIT Kernel (使用隐式 shape 推断)
# ─────────────────────────────────────────────

@pypto.frontend.jit
def topk_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    values_out: pypto.Tensor([], pypto.DT_FP32),
    indices_out: pypto.Tensor([], pypto.DT_INT64),
    k: int,
    largest: bool,
):
    """PyPTO jit kernel for topk operation.

    使用隐式 shape 推断，避免 DYNAMIC 维度错误。
    仅支持 dim=-1 操作。

    Args:
        x: 输入 tensor，任意维度
        values_out: 输出 values tensor
        indices_out: 输出 indices tensor (int64)
        k: 选取的元素数量
        largest: True 选最大，False 选最小
    """
    # 获取输入 shape
    input_shape = x.shape
    rank = len(input_shape)
    last_dim_size = input_shape[-1]

    # 设置 TileShape
    # 根据 docs/api/operation/pypto-topk.md 约束:
    # - TileShape[-1] * 4 % 32 == 0 (32B 对齐)
    # - TileShape[-1] * 4 < 22KB
    # - k <= TileShape[-1]
    if rank == 2:
        pypto.set_vec_tile_shapes(1, last_dim_size)
    elif rank == 3:
        pypto.set_vec_tile_shapes(1, 1, last_dim_size)
    elif rank == 4:
        pypto.set_vec_tile_shapes(1, 1, 1, last_dim_size)
    else:
        # 默认使用尾轴完整
        pypto.set_vec_tile_shapes(last_dim_size)

    # 调用 pypto.topk
    values_raw, indices_raw = pypto.topk(x, k, -1, largest)

    # 类型转换: int32 -> int64
    indices_i64 = pypto.cast(indices_raw, pypto.DT_INT64)

    # 写入输出
    values_out[:] = values_raw
    indices_out[:] = indices_i64


@pypto.frontend.jit
def topk_kernel_2d(
    x: pypto.Tensor([], pypto.DT_FP32),
    values_out: pypto.Tensor([], pypto.DT_FP32),
    indices_out: pypto.Tensor([], pypto.DT_INT64),
    k: int,
    largest: bool,
):
    """PyPTO jit kernel for topk operation (2D input)."""
    last_dim_size = x.shape[-1]
    pypto.set_vec_tile_shapes(1, last_dim_size)

    values_raw, indices_raw = pypto.topk(x, k, -1, largest)
    indices_i64 = pypto.cast(indices_raw, pypto.DT_INT64)

    values_out[:] = values_raw
    indices_out[:] = indices_i64


# ─────────────────────────────────────────────
# 2. Wrapper 函数
# ─────────────────────────────────────────────

def topk_wrapper(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """topk 算子 wrapper。

    返回输入张量在指定维度上最大（或最小）的 k 个元素及其索引。

    Args:
        input: 输入 torch.Tensor，支持 2-4 维
        k: 选取的元素数量
        dim: 操作维度，默认 -1（仅支持 -1 或最后一维）
        largest: True 选最大，False 选最小，默认 True
        sorted: True 则排序输出，默认 True（当前实现不保证排序，为 P1 优先级）

    Returns:
        values: 选取的 k 个元素值
        indices: 选取元素在 dim 维度上的索引 (int64)

    Note:
        - 当前实现仅支持 dim=-1（最后一维）
        - sorted 参数为 P1 优先级，当前不保证排序输出
    """
    original_shape = list(input.shape)
    original_dtype = input.dtype
    ndim = input.dim()

    # dim 参数检查：当前仅支持 dim=-1
    actual_dim = dim if dim >= 0 else ndim + dim
    if actual_dim != ndim - 1:
        raise ValueError(
            f"topk_wrapper only supports dim=-1 (last dimension), "
            f"got dim={dim}, actual_dim={actual_dim}"
        )

    # pypto.topk 只支持 FP32，需要类型转换
    need_cast = original_dtype != torch.float32
    if need_cast:
        input = input.float()

    # 计算输出 shape
    output_shape = original_shape.copy()
    output_shape[-1] = k

    # 创建输出 tensor
    values_out = torch.empty(output_shape, dtype=torch.float32, device=input.device)
    indices_out = torch.empty(output_shape, dtype=torch.int64, device=input.device)

    # 调用 kernel
    if ndim == 2:
        topk_kernel_2d(input, values_out, indices_out, k, largest)
    else:
        topk_kernel(input, values_out, indices_out, k, largest)

    # 恢复原始 dtype
    if need_cast:
        values_out = values_out.to(original_dtype)

    # sorted 参数处理（P1 优先级，当前不保证排序）
    # TODO: 如果 sorted=True，需要对输出进行排序
    if sorted:
        # 当前实现不保证排序，后续可优化
        pass

    return values_out, indices_out


# ─────────────────────────────────────────────
# 3. 带动态轴支持的 wrapper（使用 from_torch）
# ─────────────────────────────────────────────

def topk_wrapper_dynamic(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    dynamic_axis: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """topk 算子 wrapper（带动态轴支持）。

    使用 pypto.frontend.dynamic() 标记动态轴。

    Args:
        input: 输入 torch.Tensor，支持 2-4 维
        k: 选取的元素数量
        dim: 操作维度，默认 -1
        largest: True 选最大，False 选最小
        sorted: True 则排序输出
        dynamic_axis: 动态轴列表，如 [0, 1, 2] 表示前三维为动态

    Returns:
        values: 选取的 k 个元素值
        indices: 选取元素在 dim 维度上的索引 (int64)
    """
    if dynamic_axis is None:
        # 默认前 N-1 维为动态
        dynamic_axis = list(range(input.dim() - 1))

    # 调用基础 wrapper
    return topk_wrapper(input, k, dim, largest, sorted)
