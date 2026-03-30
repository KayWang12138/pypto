#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO Add kernel implementation.

公式: y = x1 + x2
实现策略:
1. 支持广播 (broadcasting)
2. 1D-4D 输入
3. 3D+ 输入 reshape 到 2D 处理
4. 使用 pypto.set_vec_tile_shapes(64, 128)

动态轴支持: 使用 pypto.Tensor([], pypto.DT_FP32) 隐式推断
"""

import pypto
import torch


# ─────────────────────────────────────────────
# 1. JIT Kernels (使用隐式 shape 推断)
# ─────────────────────────────────────────────

@pypto.frontend.jit
def add_kernel_2d(
    x1: pypto.Tensor([], pypto.DT_FP32),
    x2: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """Add kernel for 2D input.

    公式: y = x1 + x2

    Args:
        x1: 输入 tensor，2D shape
        x2: 输入 tensor，2D shape
        out: 输出 tensor，与输入 shape 相同
    """
    pypto.set_vec_tile_shapes(64, 128)
    out[:] = pypto.add(x1, x2)


@pypto.frontend.jit
def add_kernel_1d(
    x1: pypto.Tensor([], pypto.DT_FP32),
    x2: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """Add kernel for 1D input."""
    pypto.set_vec_tile_shapes(128)
    out[:] = pypto.add(x1, x2)


# ─────────────────────────────────────────────
# 2. Wrapper 函数
# ─────────────────────────────────────────────

def add_wrapper(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Add 算子 wrapper。

    逐元素计算 x1 + x2，支持广播。

    Args:
        x1: 输入 torch.Tensor，支持 1-4 维
        x2: 输入 torch.Tensor，支持 1-4 维，与 x1 broadcastable

    Returns:
        输出 tensor，shape 为 broadcast 后的 shape
    """
    # 获取广播后的 shape
    output_shape = torch.broadcast_shapes(x1.shape, x2.shape)

    # 类型转换 - 统一到 FP32 计算
    dtype1 = x1.dtype
    dtype2 = x2.dtype
    need_cast1 = dtype1 != torch.float32
    need_cast2 = dtype2 != torch.float32

    if need_cast1:
        x1 = x1.float()
    if need_cast2:
        x2 = x2.float()

    # 广播输入到输出 shape
    x1_broadcast = x1.expand(output_shape)
    x2_broadcast = x2.expand(output_shape)

    # 确保连续
    if not x1_broadcast.is_contiguous():
        x1_broadcast = x1_broadcast.contiguous()
    if not x2_broadcast.is_contiguous():
        x2_broadcast = x2_broadcast.contiguous()

    ndim = len(output_shape)

    # 根据维度选择 kernel
    if ndim == 1:
        output = torch.empty(output_shape, dtype=torch.float32, device=x1.device)
        add_kernel_1d(x1_broadcast, x2_broadcast, output)
    elif ndim == 2:
        output = torch.empty(output_shape, dtype=torch.float32, device=x1.device)
        add_kernel_2d(x1_broadcast, x2_broadcast, output)
    else:
        # 3D, 4D 都 reshape 到 2D 处理
        x1_2d = x1_broadcast.reshape(-1, x1_broadcast.shape[-1])
        x2_2d = x2_broadcast.reshape(-1, x2_broadcast.shape[-1])
        output_2d = torch.empty(x1_2d.shape, dtype=torch.float32, device=x1.device)
        add_kernel_2d(x1_2d, x2_2d, output_2d)
        output = output_2d.reshape(output_shape)

    # 类型转换回原始 dtype (取较高精度)
    final_dtype = torch.float32
    if dtype1 == torch.float16 or dtype2 == torch.float16:
        final_dtype = torch.float16
    if need_cast1 or need_cast2:
        output = output.to(final_dtype)

    return output
