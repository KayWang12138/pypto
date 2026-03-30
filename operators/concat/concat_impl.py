#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
concat PyPTO kernel implementation

功能描述:
    将多个张量沿指定维度拼接成一个张量。
    所有输入张量在非拼接维度上的 shape 必须相同，dtype 必须一致。

公式:
    output.shape[dim] = sum(t.shape[dim] for t in tensors)
    output.shape[other] = tensors[0].shape[other]

支持特性:
    - 多张量输入: 支持可变数量的输入张量（2-128个）
    - 指定拼接轴: 支持通过 dim 参数指定拼接维度
    - 负数索引: dim 参数支持负数索引（如 -1 表示最后一维）
    - 动态轴: 支持 batch 和 seq_len 维度动态 shape
"""

import os
import sys
import pypto
import torch
from typing import List, Tuple, Union


# PyTorch dtype 到 PyPTO dtype 的映射
DTYPE_MAP = {
    torch.float32: pypto.DT_FP32,
    torch.float16: pypto.DT_FP16,
    torch.bfloat16: pypto.DT_BF16,
    torch.int32: pypto.DT_INT32,
    torch.int16: pypto.DT_INT16,
    torch.int8: pypto.DT_INT8,
}


def _get_run_mode():
    """从环境变量或命令行参数获取运行模式"""
    # 检查命令行参数
    for idx, arg in enumerate(sys.argv):
        if arg == "--run_mode" and idx + 1 < len(sys.argv):
            value = sys.argv[idx + 1]
            if value == "sim":
                return pypto.RunMode.SIM
        if arg.startswith("--run_mode="):
            value = arg.split("=", 1)[1]
            if value == "sim":
                return pypto.RunMode.SIM
    return pypto.RunMode.NPU


# ─────────────────────────────────────────────
# 1. JIT Kernel 工厂函数
# ─────────────────────────────────────────────

def _get_aligned_tile_shape(ndim: int, dtype) -> List[int]:
    """根据 dtype 计算满足对齐要求的 TileShape。

    对齐要求：
    - fp32: 尾轴 8 的倍数（32B / 4B = 8）
    - fp16/bf16: 尾轴 16 的倍数（32B / 2B = 16）
    """
    if dtype in (pypto.DT_FP16, pypto.DT_BF16):
        # fp16/bf16: 尾轴需要 16 的倍数
        last_dim = 16
    else:
        # fp32/int32: 尾轴需要 8 的倍数
        last_dim = 8

    tile_shapes = [8 for _ in range(ndim - 1)] + [last_dim]
    return tile_shapes


def _get_aligned_tile_shape_for_shape(shape: Tuple[int, ...], dtype) -> List[int]:
    """根据 shape 和 dtype 计算满足对齐要求的 TileShape。

    对齐要求：
    - fp32: 尾轴 8 的倍数（32B / 4B = 8）
    - fp16/bf16: 尾轴 16 的倍数（32B / 2B = 16）
    """
    ndim = len(shape)
    if dtype in (pypto.DT_FP16, pypto.DT_BF16):
        # fp16/bf16: 尾轴需要 16 的倍数
        last_dim = 16
    else:
        # fp32/int32: 尾轴需要 8 的倍数
        last_dim = 8

    tile_shapes = [8 for _ in range(ndim - 1)] + [last_dim]
    return tile_shapes


def create_concat_kernel_2d(
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    dtype,  # pypto.DT_FP32, etc.
    dim: int,
    run_mode
):
    """创建 2D concat kernel（2个张量）"""
    tile_shapes = _get_aligned_tile_shape(len(out_shape), dtype)

    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def concat_kernel_2d(
        a: pypto.Tensor(a_shape, dtype),
        b: pypto.Tensor(b_shape, dtype),
        out: pypto.Tensor(out_shape, dtype)
    ):
        # 设置 TileShape，维度与输出一致，满足对齐要求
        pypto.set_vec_tile_shapes(*tile_shapes)
        # 执行 concat
        out[:] = pypto.concat([a, b], dim=dim)

    return concat_kernel_2d


def create_concat_kernel_3d(
    shapes: List[Tuple[int, ...]],
    out_shape: Tuple[int, ...],
    dtype,  # pypto.DT_FP32, etc.
    dim: int,
    run_mode,
    num_tensors: int = 2
):
    """创建 3D concat kernel（支持2-3个张量）"""
    tile_shapes = _get_aligned_tile_shape(len(out_shape), dtype)

    if num_tensors == 2:
        @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
        def concat_kernel_3d_2(
            a: pypto.Tensor(shapes[0], dtype),
            b: pypto.Tensor(shapes[1], dtype),
            out: pypto.Tensor(out_shape, dtype)
        ):
            pypto.set_vec_tile_shapes(*tile_shapes)
            out[:] = pypto.concat([a, b], dim=dim)
        return concat_kernel_3d_2
    elif num_tensors == 3:
        @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
        def concat_kernel_3d_3(
            a: pypto.Tensor(shapes[0], dtype),
            b: pypto.Tensor(shapes[1], dtype),
            c: pypto.Tensor(shapes[2], dtype),
            out: pypto.Tensor(out_shape, dtype)
        ):
            pypto.set_vec_tile_shapes(*tile_shapes)
            out[:] = pypto.concat([a, b, c], dim=dim)
        return concat_kernel_3d_3
    else:
        raise ValueError(f"Unsupported num_tensors: {num_tensors}, only 2 or 3 supported")


# ─────────────────────────────────────────────
# 2. Wrapper 函数（导出接口）
# ─────────────────────────────────────────────

def concat_wrapper(
    tensors: List[torch.Tensor],
    dim: int = 0
) -> torch.Tensor:
    """
    concat 算子 wrapper，供 test_concat.py 调用。

    负责:
    1. 构造输出 torch.Tensor
    2. 动态创建并调用 JIT kernel
    3. 返回结果 torch.Tensor

    Args:
        tensors: 待拼接的张量列表，至少包含2个张量。
                 所有张量必须具有相同的维度数和 dtype。
                 除 dim 维度外，其他维度的 shape 必须相同。
        dim: 拼接维度，支持负数索引。默认为 0。

    Returns:
        拼接后的 torch.Tensor。
    """
    # 边界检查
    if len(tensors) < 2:
        raise ValueError(f"concat requires at least 2 tensors, got {len(tensors)}")

    # 获取设备
    device = tensors[0].device
    dtype = tensors[0].dtype

    # 获取维度数
    ndim = tensors[0].ndim

    # 标准化 dim
    if dim < 0:
        dim = ndim + dim

    # 验证所有张量维度数一致
    for i, t in enumerate(tensors):
        if t.ndim != ndim:
            raise ValueError(f"All tensors must have same ndim, tensor 0 has {ndim}, tensor {i} has {t.ndim}")
        if t.dtype != dtype:
            raise ValueError(f"All tensors must have same dtype, tensor 0 has {dtype}, tensor {i} has {t.dtype}")

    # 计算输出 shape
    out_shape = list(tensors[0].shape)
    out_shape[dim] = sum(t.shape[dim] for t in tensors)
    out_shape = tuple(out_shape)

    # 获取 PyPTO dtype
    pypto_dtype = DTYPE_MAP.get(dtype)
    if pypto_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # 获取运行模式
    run_mode = _get_run_mode()

    # 创建输出 tensor
    output = torch.empty(out_shape, dtype=dtype, device=device)

    # 根据张量数量和维度数选择 kernel
    num_tensors = len(tensors)

    if ndim == 2:
        if num_tensors == 2:
            kernel = create_concat_kernel_2d(
                tensors[0].shape, tensors[1].shape, out_shape, pypto_dtype, dim, run_mode
            )
            kernel(tensors[0], tensors[1], output)
        else:
            # 对于超过2个张量，分批拼接
            # 先拼接前两个
            current = tensors[0]
            for i in range(1, num_tensors):
                # 计算中间 shape
                inter_shape = list(current.shape)
                inter_shape[dim] = current.shape[dim] + tensors[i].shape[dim]
                inter_shape = tuple(inter_shape)

                inter_output = torch.empty(inter_shape, dtype=dtype, device=device)
                kernel = create_concat_kernel_2d(
                    current.shape, tensors[i].shape, inter_shape, pypto_dtype, dim, run_mode
                )
                kernel(current, tensors[i], inter_output)
                current = inter_output

            # 最终结果复制到 output
            output.copy_(current)

    elif ndim == 3:
        if num_tensors == 2:
            kernel = create_concat_kernel_3d(
                [tensors[0].shape, tensors[1].shape], out_shape, pypto_dtype, dim, run_mode, num_tensors=2
            )
            kernel(tensors[0], tensors[1], output)
        elif num_tensors == 3:
            kernel = create_concat_kernel_3d(
                [tensors[0].shape, tensors[1].shape, tensors[2].shape],
                out_shape, pypto_dtype, dim, run_mode, num_tensors=3
            )
            kernel(tensors[0], tensors[1], tensors[2], output)
        else:
            # 对于超过3个张量，分批拼接
            current = tensors[0]
            for i in range(1, num_tensors):
                inter_shape = list(current.shape)
                inter_shape[dim] = current.shape[dim] + tensors[i].shape[dim]
                inter_shape = tuple(inter_shape)

                inter_output = torch.empty(inter_shape, dtype=dtype, device=device)
                kernel = create_concat_kernel_3d(
                    [current.shape, tensors[i].shape], inter_shape, pypto_dtype, dim, run_mode, num_tensors=2
                )
                kernel(current, tensors[i], inter_output)
                current = inter_output

            output.copy_(current)

    elif ndim == 4:
        # 4D tensor 使用通用方式
        if num_tensors == 2:
            kernel = create_concat_kernel_2d(
                tensors[0].shape, tensors[1].shape, out_shape, pypto_dtype, dim, run_mode
            )
            kernel(tensors[0], tensors[1], output)
        else:
            current = tensors[0]
            for i in range(1, num_tensors):
                inter_shape = list(current.shape)
                inter_shape[dim] = current.shape[dim] + tensors[i].shape[dim]
                inter_shape = tuple(inter_shape)

                inter_output = torch.empty(inter_shape, dtype=dtype, device=device)
                kernel = create_concat_kernel_2d(
                    current.shape, tensors[i].shape, inter_shape, pypto_dtype, dim, run_mode
                )
                kernel(current, tensors[i], inter_output)
                current = inter_output

            output.copy_(current)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}, only 2-4 supported")

    return output
