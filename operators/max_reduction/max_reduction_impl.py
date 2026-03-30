#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY kind, either express or implied,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO max_reduction kernel implementation.

实现说明:
  - 本文件实现 max_reduction 算子的 PyPTO kernel。
  - 导出函数 max_reduction_wrapper() 供 test_max_reduction.py 调用。
  - 使用直接方案: pypto.max 实现 max 操作。
  - 支持动态轴 (b, s) 和 keepdim 参数.

设计依据:
  - 来源: design.md
  - API 映射: pypto.max(input, dim, keepdim)

约束:
  - 张量参数在前，非张量参数在后 (PyPTO 要求)
"""

import pypto
import torch


# ─────────────────────────────────────────────
# 1. 辅助函数
# ─────────────────────────────────────────────

def _get_output_shape(input_shape: tuple, dim: int, keepdim: bool) -> tuple:
    """计算输出 shape"""
    shape_list = list(input_shape)
    # 处理负索引
    if dim < 0:
        dim = len(shape_list) + dim

    if keepdim:
        shape_list[dim] = 1
    else:
        shape_list.pop(dim)

    return tuple(shape_list)


# ─────────────────────────────────────────────
# 2. JIT Kernel
# ─────────────────────────────────────────────

# 注意: 使用模块级 run_mode，支持命令行切换
import sys
_run_mode = pypto.RunMode.NPU
for idx, arg in enumerate(sys.argv):
    if arg == "--run_mode" and idx + 1 < len(sys.argv):
        if sys.argv[idx + 1] == "sim":
            _run_mode = pypto.RunMode.SIM
            break
    if arg.startswith("--run_mode="):
        if arg.split("=", 1)[1] == "sim":
            _run_mode = pypto.RunMode.SIM
            break


@pypto.frontend.jit(runtime_options={"run_mode": _run_mode})
def max_reduction_kernel(
    input_tensor: pypto.Tensor([], pypto.DT_FP32),
    output_tensor: pypto.Tensor([], pypto.DT_FP32),
    dim: int,
    keepdim: bool,
):
    """PyPTO jit kernel for max_reduction.

    实现 max(x, dim) = max(x_i for i in dim_axis)

    注意: 张量参数 (input_tensor, output_tensor) 在前，
          非张量参数 (dim, keepdim) 在后。

    Args:
        input_tensor: 输入 pypto.Tensor
        output_tensor: 输出 pypto.Tensor
        dim: 归约轴索引
        keepdim: 是否保持归约后的维度
    """
    # 设置 TileShape
    # 根据 design.md: 对于 3D 输入，使用 [8, 8, 8]
    # 尾轴 8 满足 float32 的 32B 对齐要求（8 * 4 = 32 bytes）
    # 次尾轴 8 < 255 满足约束
    ndim = len(input_tensor.shape)
    if ndim == 2:
        pypto.set_vec_tile_shapes(8, 8)
    elif ndim == 3:
        pypto.set_vec_tile_shapes(8, 8, 8)
    elif ndim == 4:
        pypto.set_vec_tile_shapes(8, 8, 8, 8)
    else:
        # 默认使用 8 填充每个维度
        tile_shapes = [8] * ndim
        pypto.set_vec_tile_shapes(*tile_shapes)

    # 步骤 1: 沿 dim 轴求最大值
    # pypto.amax 返回归约后的 tensor
    result = pypto.amax(input_tensor, dim, keepdim)

    # 输出写回
    output_tensor[:] = result


# ─────────────────────────────────────────────
# 3. Wrapper 函数（导出接口)
# ─────────────────────────────────────────────

def max_reduction_wrapper(
    x: torch.Tensor,
    dim: int,
    keepdim: bool = False,
) -> torch.Tensor:
    """max_reduction wrapper，供 test_max_reduction.py 调用。

    负责:
    1. 计算输出 shape
    2. 构造输出 torch.Tensor
    3. 调用 JIT kernel
    4. 返回结果 torch.Tensor

    Args:
        x: 输入 torch.Tensor
        dim: 归约轴索引
        keepdim: 是否保持归约后的维度，默认 False

    Returns:
        输出 torch.Tensor，归约后的最大值
    """
    # 确保输入是连续的
    if not x.is_contiguous():
        x = x.contiguous()

    # 计算输出 shape
    out_shape = _get_output_shape(x.shape, dim, keepdim)

    # 构造输出 tensor
    output = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    # 调用 kernel - 注意参数顺序: 张量在前，非张量在后
    max_reduction_kernel(x, output, dim, keepdim)

    return output
