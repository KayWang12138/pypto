#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY kind, either express or implied,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Mish PyPTO kernel implementation.

公式: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

实现要点:
1. 使用 tanh 近似: tanh(x) = 2 * sigmoid(2x) - 1
2. 1D/4D 输入特殊处理
3. 4D 输入需 reshape 为 2D
4. 使用 pypto.set_vec_tile_shapes(64, 128)
"""

import pypto
import torch
import math
from typing import Tuple, Optional


def get_device_id():
    """Get device ID from environment."""
    import os
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', '0')
    if device_id is None:
        raise RuntimeError("Please set TILE_FWK_DEVICE_ID environment variable")
    return int(device_id)


def mish_core(x: pypto.Tensor) -> pypto.Tensor:
    """Mish 核心计算逻辑。

    公式: mish(x) = x * tanh(softplus(x))
    使用 tanh 近似: tanh(x) = 2 * sigmoid(2x) - 1

    Args:
        x: pypto.Tensor 输入张量

    Returns:
        pypto.Tensor Mish 计算结果
    """
    # Step 1: softplus(x) = ln(1 + exp(x))
    exp_x = pypto.exp(x)
    one_plus_exp = pypto.add(exp_x, 1.0)
    softplus = pypto.log(one_plus_exp)

    # Step 2: tanh(softplus) = 2 * sigmoid(2 * softplus) - 1
    sp_2x = pypto.mul(softplus, 2.0)
    sigmoid_2x = pypto.sigmoid(sp_2x)
    tanh_out = pypto.add(pypto.mul(sigmoid_2x, 2.0), -1.0)

    # Step 3: mish(x) = x * tanh(softplus(x))
    result = pypto.mul(x, tanh_out)

    return result


@pypto.frontend.jit
def mish_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    """Mish JIT kernel for FP32.

    Args:
        x: 输入张量，支持任意 shape (1D-4D)
        out: 输出张量， shape 与输入相同
    """
    # 设置 Tiling 配置
    pypto.set_vec_tile_shapes(64, 128)

    # 计算并写回
    result = mish_core(x)
    out[:] = result


def mish_wrapper(x: torch.Tensor) -> torch.Tensor:
    """Mish wrapper，供 test_mish.py 调用。

    负责:
    1. 检查输入 tensor 是否连续
    2. 处理 1D/4D 输入的 reshape
    3. 调用 kernel
    4. 返回结果 tensor

    Args:
        x: 输入 torch.Tensor，支持 1D-4D，dtype=float32

    Returns:
        输出 torch.Tensor，shape 和 dtype 与输入相同
    """
    # 检查连续性
    if not x.is_contiguous():
        x = x.contiguous()

    # 记录原始 shape
    original_shape = x.shape
    original_ndim = x.ndim

    # 特殊处理 1D 输入: reshape 为 2D
    if original_ndim == 1:
        x = x.unsqueeze(0)  # [N] -> [1, N]

    # 3D/4D 输入统一 reshape 为 2D 避免 tiling 编译问题
    if original_ndim >= 3:
        x = x.reshape(-1, x.shape[-1])  # [B, ... , C] -> [B*..., C]

    # 创建输出 tensor
    out = torch.empty_like(x)

    # 调用 kernel
    mish_kernel(x, out)

    # 恢复原始 shape
    if original_ndim == 1:
        out = out.squeeze(0)  # [1, N] -> [N]
    elif original_ndim >= 3:
        out = out.reshape(original_shape)  # 恢复 [B, ..., C]

    return out
