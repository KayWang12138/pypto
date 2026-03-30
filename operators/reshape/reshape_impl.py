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
"""
reshape PyPTO kernel implementation.

公式: y = reshape(x, shape)
功能: 将输入张量变换为指定的形状，保持数据不变，仅改变维度视图。
      支持动态轴（batch、seq_len）和负维度自动推断。

实现说明:
  - reshape 是纯 shape 变换操作，不涉及数据搬运或计算
  - 需要设置 TileShape（PyPTO 框架要求，即使 reshape 不需要计算）
  - 不需要 Loop 结构（单步元数据操作）
  - 支持动态轴，需在 Tensor 描述中标注 pypto.DYNAMIC

注意：由于 PyPTO JIT 装饰器在模块加载时解析函数签名中的类型注解，
    类型注解中使用的变量必须在函数定义时可直接解析。
    因此，工厂函数内部的 JIT kernel 只能使用传入的参数（如 input_shape, output_shape），
    而不能使用工厂函数内定义的局部变量（如 pypto_dtype）。
"""

import os
import pypto
import torch
from typing import List, Optional, Union

# dtype 映射表: torch.dtype -> pypto dtype 常量
DTYPE_MAP = {
    torch.float32: pypto.DT_FP32,
    torch.float16: pypto.DT_FP16,
    torch.bfloat16: pypto.DT_BF16,
}


# ─────────────────────────────────────────────
# JIT Kernel 工厂函数 - Float32
# ─────────────────────────────────────────────

def create_reshape_kernel_fp32(input_shape: tuple, output_shape: tuple):
    """Create FP32 reshape kernel."""
    @pypto.frontend.jit
    def reshape_kernel(
        input_tensor: pypto.Tensor(input_shape, pypto.DT_FP32),
        output_tensor: pypto.Tensor(output_shape, pypto.DT_FP32),
    ):
        ndim = len(input_shape)
        if ndim <= 4:
            tile_shapes = [8] * ndim
            pypto.set_vec_tile_shapes(*tile_shapes)
        else:
            pypto.set_vec_tile_shapes(64, 64, 64, 64)
        result = pypto.reshape(input_tensor, list(output_shape))
        output_tensor[:] = result
    return reshape_kernel


def create_reshape_kernel_fp16(input_shape: tuple, output_shape: tuple):
    """Create FP16 reshape kernel."""
    @pypto.frontend.jit
    def reshape_kernel(
        input_tensor: pypto.Tensor(input_shape, pypto.DT_FP16),
        output_tensor: pypto.Tensor(output_shape, pypto.DT_FP16),
    ):
        ndim = len(input_shape)
        if ndim <= 4:
            tile_shapes = [8] * ndim
            pypto.set_vec_tile_shapes(*tile_shapes)
        else:
            pypto.set_vec_tile_shapes(64, 64, 64, 64)
        result = pypto.reshape(input_tensor, list(output_shape))
        output_tensor[:] = result
    return reshape_kernel


def create_reshape_kernel_bf16(input_shape: tuple, output_shape: tuple):
    """Create BF16 reshape kernel."""
    @pypto.frontend.jit
    def reshape_kernel(
        input_tensor: pypto.Tensor(input_shape, pypto.DT_BF16),
        output_tensor: pypto.Tensor(output_shape, pypto.DT_BF16),
    ):
        ndim = len(input_shape)
        if ndim <= 4:
            tile_shapes = [8] * ndim
            pypto.set_vec_tile_shapes(*tile_shapes)
        else:
            pypto.set_vec_tile_shapes(64, 64, 64, 64)
        result = pypto.reshape(input_tensor, list(output_shape))
        output_tensor[:] = result
    return reshape_kernel


# ─────────────────────────────────────────────
# Wrapper 函数（导出接口）
# ─────────────────────────────────────────────

def reshape_wrapper(
    x: torch.Tensor,
    shape: List[int],
) -> torch.Tensor:
    """reshape 算子 wrapper，供 test_reshape.py 调用。

    负责：
    1. 验证输入约束（元素总数匹配、输入 contiguous）
    2. 构造输出 torch.Tensor
    3. 调用 JIT kernel
    4. 返回结果 torch.Tensor

    Args:
        x: 输入 torch.Tensor，支持 float32/bfloat16/float16。
        shape: 目标形状，支持一个维度为 -1 表示自动推断。

    Returns:
        输出 torch.Tensor，形状为目标 shape，dtype 与输入相同。
    """
    # 1. 验证输入 contiguous
    if not x.is_contiguous():
        x = x.contiguous()

    # 2. 处理 -1 维度（自动推断）
    input_numel = x.numel()
    target_shape = list(shape)

    neg_idx = None
    known_size = 1
    for i, dim in enumerate(target_shape):
        if dim == -1:
            if neg_idx is not None:
                raise ValueError("shape 中最多只能有一个维度为 -1")
            neg_idx = i
        else:
            known_size *= dim

    if neg_idx is not None:
        if known_size == 0:
            raise ValueError("无法推断 -1 维度的大小")
        inferred_dim = input_numel // known_size
        if inferred_dim * known_size != input_numel:
            raise ValueError(f"元素总数不匹配: input={input_numel}, target={target_shape}")
        target_shape[neg_idx] = inferred_dim

    # 3. 验证元素总数匹配
    output_numel = 1
    for dim in target_shape:
        output_numel *= dim

    if input_numel != output_numel:
        raise ValueError(
            f"元素总数不匹配: input shape {list(x.shape)} has {input_numel} elements, "
            f"but target shape {target_shape} has {output_numel} elements"
        )

    # 4. 构造输出 tensor（同一设备）
    output = torch.empty(target_shape, dtype=x.dtype, device=x.device)

    # 5. 根据 dtype 选择 kernel 工厂函数并执行
    input_shape_tuple = tuple(x.shape)
    output_shape_tuple = tuple(target_shape)

    if x.dtype == torch.float32:
        kernel = create_reshape_kernel_fp32(input_shape_tuple, output_shape_tuple)
    elif x.dtype == torch.float16:
        kernel = create_reshape_kernel_fp16(input_shape_tuple, output_shape_tuple)
    elif x.dtype == torch.bfloat16:
        kernel = create_reshape_kernel_bf16(input_shape_tuple, output_shape_tuple)
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")

    kernel(x, output)

    return output


def reshape_dynamic_wrapper(
    x: torch.Tensor,
    shape: List[int],
    dynamic_axes: Optional[List[int]] = None,
) -> torch.Tensor:
    """reshape 算子动态版本 wrapper。

    支持动态轴的 reshape 操作，适用于 batch 和 seq_len 等动态维度。

    Args:
        x: 输入 torch.Tensor，支持 float32/bfloat16/float16。
        shape: 目标形状，支持一个维度为 -1 表示自动推断。
        dynamic_axes: 动态轴索引列表（从 0 开始）。

    Returns:
        输出 torch.Tensor，形状为目标 shape，dtype 与输入相同。

    Note:
        动态版本暂不实现，因为 PyPTO 的动态轴支持需要在 JIT kernel 中使用 pypto.DYNAMIC。
        目前 reshape_wrapper 已足够满足大多数场景。
    """
    # 暂时使用静态版本，动态轴功能待后续实现
    return reshape_wrapper(x, shape)
