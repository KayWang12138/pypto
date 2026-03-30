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
"""PyPTO silu kernel implementation.

实现说明：
  - 导出函数 silu_wrapper() 供 test_silu.py 调用
  - kernel 使用 @pypto.frontend.jit 装饰，内部使用 pypto API
  - 支持 FP32/FP16/BF16 三种 dtype
  - FP32 路径使用 pypto.sigmoid API
  - FP16/BF16 路径手动展开 sigmoid
  - 参考 examples/02_intermediate/operators/activation/activation.py
"""

import pypto
import torch


def configure_tiling(x):
    """根据输入 shape 动态设置 TileShape。

    Args:
        x: 输入 tensor
    """
    if len(x.shape) >= 2:
        # 多维输入：每个维度设置 tile size 为 32
        tile_list = [32 for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_list)
    else:
        # 1D 输入：使用默认 tile
        pypto.set_vec_tile_shapes(32, 128)


@pypto.frontend.jit
def silu_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
):
    """PyPTO silu kernel。

    SiLU (Sigmoid Linear Unit) 激活函数: y = x * sigmoid(x)

    根据 dtype 选择不同的实现路径：
    - FP32: 直接使用 pypto.sigmoid API
    - FP16/BF16: 手动展开 sigmoid

    Args:
        x: 输入 tensor，支持 FP32/FP16/BF16
        out: 输出 tensor，shape 和 dtype 与 x 相同
    """
    configure_tiling(x)

    # 根据 dtype 选择实现路径
    # 注意：pypto.sigmoid 仅支持 FP32
    # FP16/BF16 需要手动展开 sigmoid
    if x.dtype == pypto.DataType.DT_FP32:
        # 方案 A：FP32 路径 - 直接使用 sigmoid API
        sigmoid_x = pypto.sigmoid(x)
        out[:] = pypto.mul(x, sigmoid_x)
    else:
        # 方案 B：FP16/BF16 路径 - 手动展开 sigmoid
        # sigmoid(x) = 1 / (1 + exp(-x))
        neg_x = pypto.mul(x, -1.0)
        exp_neg_x = pypto.exp(neg_x)
        one_plus_exp = pypto.add(exp_neg_x, 1.0)
        sigmoid_x = pypto.reciprocal(one_plus_exp)
        out[:] = pypto.mul(x, sigmoid_x)


def silu_wrapper(x: torch.Tensor) -> torch.Tensor:
    """silu wrapper，供 test_silu.py 调用。

    负责：
    1. 构造输出 torch.Tensor
    2. 调用 JIT kernel
    3. 返回结果 torch.Tensor

    Args:
        x: 输入 torch.Tensor，支持 float16/float32/bfloat16

    Returns:
        输出 torch.Tensor，shape 和 dtype 与输入相同
        y = x * sigmoid(x)
    """
    # 构造输出 tensor
    output = torch.empty_like(x)

    # 调用 kernel
    silu_kernel(x, output)

    return output
