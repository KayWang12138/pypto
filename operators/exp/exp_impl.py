#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO exp kernel implementation.

实现说明：
  - 导出函数 exp_wrapper() 供 test_exp.py 调用
  - kernel 使用 @pypto.frontend.jit 装饰，内部使用 pypto API
  - 支持 FP32/FP16 两种 dtype
  - 使用 pypto.exp 直接 API
  - 4D 输入使用 reshape(-1, last_dim) 策略，避免 4D tiling 编译问题
  - 公式: y = exp(x) = e^x
"""

import pypto
import torch


def configure_tiling(x):
    """根据输入 shape 动态设置 TileShape。

    统一使用 2D tiling: pypto.set_vec_tile_shapes(64, 128)
    4D 输入会先 reshape 到 2D 再调用 kernel。

    Args:
        x: 输入 tensor
    """
    # 统一使用 2D tiling 配置
    # 尾轴 128 满足 FP16/FP32 的 32B 对齐要求
    pypto.set_vec_tile_shapes(64, 128)


@pypto.frontend.jit
def exp_kernel_2d(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
):
    """PyPTO exp kernel (2D 版本)。

    指数函数: y = exp(x) = e^x

    逐元素计算 e 的 x 次方。

    Args:
        x: 输入 tensor，支持 FP32/FP16，必须是 2D
        out: 输出 tensor，shape 和 dtype 与 x 相同
    """
    configure_tiling(x)

    # 使用 PyPTO exp 直接 API
    result = pypto.exp(x)
    out[:] = result


def exp_wrapper(x: torch.Tensor) -> torch.Tensor:
    """exp wrapper，供 test_exp.py 调用。

    负责：
    1. 检查输入约束
    2. 处理 4D 输入的 reshape 策略
    3. 构造输出 torch.Tensor
    4. 调用 JIT kernel
    5. 返回结果 torch.Tensor

    Args:
        x: 输入 torch.Tensor，支持 float16/float32
           shape: [m, n] (2D) 或 [b, s, n, d] (4D)

    Returns:
        输出 torch.Tensor，shape 和 dtype 与输入相同
        y = exp(x)，逐元素计算 e 的 x 次方
    """
    # 约束检查
    if not x.is_contiguous():
        x = x.contiguous()

    if x.numel() == 0:
        raise ValueError("Input tensor must not be empty")

    # 检查维度
    ndim = x.dim()
    if ndim not in [2, 4]:
        raise ValueError(f"exp only supports 2D or 4D input, got {ndim}D")

    # 4D 输入使用 reshape 策略
    if ndim == 4:
        original_shape = x.shape
        last_dim = original_shape[-1]
        # reshape 到 2D
        x_2d = x.reshape(-1, last_dim)
        # 构造 2D 输出
        output_2d = torch.empty_like(x_2d)
        # 调用 2D kernel
        exp_kernel_2d(x_2d, output_2d)
        # reshape 回原始 shape
        output = output_2d.reshape(original_shape)
    else:
        # 2D 输入直接处理
        output = torch.empty_like(x)
        exp_kernel_2d(x, output)

    return output
