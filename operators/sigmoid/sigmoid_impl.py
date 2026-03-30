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
"""PyPTO sigmoid kernel implementation.

实现说明：
  - 导出函数 sigmoid_wrapper() 供 test_sigmoid.py 调用
  - kernel 使用 @pypto.frontend.jit 装饰，内部使用 pypto API
  - 支持 FP32/FP16/BF16 三种 dtype
  - 使用 pypto.sigmoid 直接 API
  - 公式: sigmoid(x) = 1 / (1 + exp(-x))
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
def sigmoid_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
):
    """PyPTO sigmoid kernel。

    Sigmoid 激活函数: y = 1 / (1 + exp(-x))

    将输入映射到 (0, 1) 区间。

    Args:
        x: 输入 tensor，支持 FP32/FP16/BF16
        out: 输出 tensor，shape 和 dtype 与 x 相同，值域 (0, 1)
    """
    configure_tiling(x)

    # 使用 PyPTO sigmoid 直接 API
    out[:] = pypto.sigmoid(x)


def sigmoid_wrapper(x: torch.Tensor) -> torch.Tensor:
    """sigmoid wrapper，供 test_sigmoid.py 调用。

    负责：
    1. 检查输入约束
    2. 构造输出 torch.Tensor
    3. 调用 JIT kernel
    4. 返回结果 torch.Tensor

    Args:
        x: 输入 torch.Tensor，支持 float16/float32/bfloat16

    Returns:
        输出 torch.Tensor，shape 和 dtype 与输入相同
        y = sigmoid(x)，值域 (0, 1)
    """
    # 约束检查
    if not x.is_contiguous():
        x = x.contiguous()

    if x.numel() == 0:
        raise ValueError("Input tensor must not be empty")

    # 构造输出 tensor
    output = torch.empty_like(x)

    # 调用 kernel
    sigmoid_kernel(x, output)

    return output
