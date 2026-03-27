#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY kind, either express or implied,
# including but not limited to non-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Softmax 算子实现

基于 PyPTO 实现 softmax 归一化算子。
支持动态批次和序列长度。
"""

import os
import sys
import pypto
import torch


def softmax_core(x: pypto.Tensor, dim: int = -1) -> pypto.Tensor:
    """
    Softmax 核心计算：手动实现（更灵活，支持不同 dim）

    公式: softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

    Args:
        x: 输入 tensor
        dim: 归一化轴（默认 -1）

    Returns:
        归一化后的 tensor
    """
    # 数值稳定实现：减去最大值
    row_max = pypto.amax(x, dim=dim, keepdim=True)
    sub = x - row_max
    exp = pypto.exp(sub)
    esum = pypto.sum(exp, dim=dim, keepdim=True)
    return exp / esum


@pypto.frontend.jit
def softmax_kernel(
    input_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    dim: int = -1,
):
    """
    Softmax JIT kernel

    支持 4D 动态 shape [batch, seq, head, dim]
    使用 Loop 处理动态批次
    """
    bs, seqlen, head, dim_size = input_tensor.shape
    tile_b = 1
    b_loop = bs // tile_b

    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
        b_offset = idx * tile_b

        # 使用切片获取当前批次
        input_view = input_tensor[b_offset:b_offset + tile_b, ...]
        softmax_out = softmax_core(input_view, dim)
        # 写回输出（从 b_offset 开始）
        output_tensor[b_offset:, ...] = softmax_out


def softmax_wrapper(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax wrapper

    Args:
        x: 输入 torch.Tensor [batch, seq, head, dim]
        dim: 归一化轴（默认 -1）

    Returns:
        归一化后的 torch.Tensor
    """
    output = torch.empty_like(x)
    softmax_kernel(x, output, dim)
    return output
