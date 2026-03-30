#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO tanh kernel implementation.

实现说明：
  - 导出函数 tanh_wrapper() 供 test_tanh.py 调用
  - kernel 使用 @pypto.frontend.jit 装饰，内部使用 pypto API
  - 支持 FP32/FP16/BF16 三种 dtype
  - PyPTO 无直接 tanh API，使用 exp/mul/sub/add/div 组合实现
  - 公式: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
"""

import pypto
import torch


def configure_tiling(x):
    """根据输入 shape 动态设置 TileShape。

    Args:
        x: 输入 tensor
    """
    ndim = len(x.shape)
    if ndim == 2:
        pypto.set_vec_tile_shapes(32, 128)
    elif ndim == 3:
        pypto.set_vec_tile_shapes(1, 32, 128)
    elif ndim == 4:
        pypto.set_vec_tile_shapes(1, 1, 32, 128)
    else:
        # 默认配置
        tile_list = [32 for _ in range(ndim)]
        pypto.set_vec_tile_shapes(*tile_list[:4])


@pypto.frontend.jit
def tanh_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
):
    """PyPTO tanh kernel。

    Tanh (Hyperbolic Tangent) 激活函数: y = (e^x - e^(-x)) / (e^x + e^(-x))

    由于 PyPTO 无直接 tanh API，使用组合实现：
    1. exp_x = pypto.exp(x)                    # e^x
    2. neg_x = pypto.mul(x, -1.0)              # -x
    3. exp_neg_x = pypto.exp(neg_x)            # e^(-x)
    4. numerator = pypto.sub(exp_x, exp_neg_x) # e^x - e^(-x)
    5. denominator = pypto.add(exp_x, exp_neg_x) # e^x + e^(-x)
    6. out[:] = pypto.div(numerator, denominator) # (e^x - e^(-x)) / (e^x + e^(-x))

    Args:
        x: 输入 tensor，支持 FP32/FP16/BF16
        out: 输出 tensor，shape 和 dtype 与 x 相同，值域 [-1, 1]
    """
    configure_tiling(x)

    # Step 1: 计算 e^x
    exp_x = pypto.exp(x)

    # Step 2: 计算 -x
    neg_x = pypto.mul(x, -1.0)

    # Step 3: 计算 e^(-x)
    exp_neg_x = pypto.exp(neg_x)

    # Step 4: 计算分子 e^x - e^(-x)
    numerator = pypto.sub(exp_x, exp_neg_x)

    # Step 5: 计算分母 e^x + e^(-x)
    denominator = pypto.add(exp_x, exp_neg_x)

    # Step 6: 计算最终结果 (e^x - e^(-x)) / (e^x + e^(-x))
    out[:] = pypto.div(numerator, denominator)


def tanh_wrapper(x: torch.Tensor) -> torch.Tensor:
    """tanh wrapper，供 test_tanh.py 调用。

    负责：
    1. 检查输入约束
    2. 构造输出 torch.Tensor
    3. 调用 JIT kernel
    4. 返回结果 torch.Tensor

    Args:
        x: 输入 torch.Tensor，支持 float16/float32/bfloat16

    Returns:
        输出 torch.Tensor，shape 和 dtype 与输入相同
        y = tanh(x)，值域 [-1, 1]
    """
    # 约束检查
    if not x.is_contiguous():
        x = x.contiguous()

    if x.numel() == 0:
        raise ValueError("Input tensor must not be empty")

    # 构造输出 tensor
    output = torch.empty_like(x)

    # 调用 kernel
    tanh_kernel(x, output)

    return output
