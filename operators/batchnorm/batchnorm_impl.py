#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO BatchNorm kernel implementation.

实现说明:
  - 导出函数 batchnorm_wrapper() 供 test_batchnorm.py 调用
  - kernel 使用 @pypto.frontend.jit 装饰，内部使用 pypto API
  - 支持 FP32/FP16/BF16
  - 使用简化方案：3D 输入 [batch, seq_len, channels]，归一化沿 channels 维度
  - 由于 pypto.reshape 不支持动态轴，使用静态 shape kernel

数学公式:
    y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

其中 E[x] 和 Var[x] 沿 channels 维度计算
"""

import os
import sys
import pypto
import torch

# 运行模式配置
def _peek_run_mode_from_argv(default: str = "npu") -> str:
    """Read run_mode early so module-level decorators can use it."""
    for idx, arg in enumerate(sys.argv):
        if arg == "--run_mode" and idx + 1 < len(sys.argv):
            value = sys.argv[idx + 1]
            if value in ("npu", "sim"):
                return value
        if arg.startswith("--run_mode="):
            value = arg.split("=", 1)[1]
            if value in ("npu", "sim"):
                return value
    return default


global_run_mode = pypto.RunMode.NPU
if _peek_run_mode_from_argv("npu") == "sim":
    global_run_mode = pypto.RunMode.SIM


# -------------------------------------------------------------------------
# 1. 核心计算函数
# -------------------------------------------------------------------------

def batchnorm_core(
    x: pypto.Tensor,
    gamma: pypto.Tensor,
    beta: pypto.Tensor,
    eps: float,
    N: int,
    channels: int
) -> pypto.Tensor:
    """BatchNorm 核心计算逻辑

    对 2D tensor [N, channels] 进行归一化处理
    归一化沿 dim=0 (即 N 维度) 进行，对每个 channel 独立归一化

    Args:
        x: 2D pypto.Tensor [N, channels]
        gamma: 1D pypto.Tensor [channels]
        beta: 1D pypto.Tensor [channels]
        eps: 数值稳定性常数
        N: 第一个维度的大小 (batch * seq_len 或 batch * seq_len * H * W)
        channels: channels 维度大小

    Returns:
        2D pypto.Tensor [N, channels]
    """
    # Step 1: Compute mean along dim=0
    # sum along N dimension -> [1, channels]
    sum_x = pypto.sum(x, dim=0, keepdim=True)
    mean = pypto.mul(sum_x, pypto.Element(pypto.DT_FP32, 1.0 / N))

    # Step 2: Center
    centered = pypto.sub(x, mean)

    # Step 3: Compute variance
    squared = pypto.mul(centered, centered)
    sum_sq = pypto.sum(squared, dim=0, keepdim=True)
    var = pypto.mul(sum_sq, pypto.Element(pypto.DT_FP32, 1.0 / N))

    # Step 4: Normalize
    var_eps = pypto.add(var, pypto.Element(pypto.DT_FP32, eps))
    std = pypto.sqrt(var_eps)
    normalized = pypto.div(centered, std)

    # Step 5: Affine transform
    scaled = pypto.mul(normalized, gamma)
    output = pypto.add(scaled, beta)

    return output


# -------------------------------------------------------------------------
# 2. JIT Kernel (2D 输入，用于 3D case)
# -------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def batchnorm_kernel_2d(
    x: pypto.Tensor(),
    gamma: pypto.Tensor(),
    beta: pypto.Tensor(),
    output: pypto.Tensor(),
    eps: float,
    N: int,
    channels: int
):
    """BatchNorm kernel for 2D input [N, channels]

    对于 2D 输入，归一化沿 dim=0 进行
    """
    # 设置 Tiling
    pypto.set_vec_tile_shapes(64, channels)

    # 核心计算
    result = batchnorm_core(x, gamma, beta, eps, N, channels)

    # 输出写回
    pypto.assemble(result, [0, 0], output)


# -------------------------------------------------------------------------
# 3. Wrapper 函数
# -------------------------------------------------------------------------

def batchnorm_wrapper(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """BatchNorm wrapper，供 test_batchnorm.py 调用

    负责:
    1. 约束检查
    2. 构造输出 torch.Tensor
    3. 调用 JIT kernel
    4. 返回结果 torch.Tensor

    Args:
        x: 输入 torch.Tensor，支持 3D [batch, seq_len, channels] 或 5D [batch, seq_len, channels, H, W]
        gamma: 缩放参数 [channels]
        beta: 偏移参数 [channels]
        eps: 数值稳定性常数，默认 1e-5

    Returns:
        输出 torch.Tensor，shape 和 dtype 与输入相同
        y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
    """
    # 约束检查
    if not x.is_contiguous():
        x = x.contiguous()

    if x.numel() == 0:
        raise ValueError("Input tensor must not be empty")

    ndim = x.dim()
    if ndim not in (3, 5):
        raise ValueError(f"Unsupported input dimension: {ndim}, expected 3 or 5")

    # 确保输入类型一致，转换到 FP32 进行计算
    input_dtype = x.dtype
    if x.dtype == torch.float16:
        x = x.float()
        gamma = gamma.float() if gamma.dtype == torch.float16 else gamma
        beta = beta.float() if beta.dtype == torch.float16 else beta
    elif x.dtype == torch.bfloat16:
        x = x.float()
        gamma = gamma.float() if gamma.dtype == torch.bfloat16 else gamma
        beta = beta.float() if beta.dtype == torch.bfloat16 else beta

    # 获取 shape 参数
    shape = x.shape
    channels_val = shape[2]

    # 计算 N 和 reshape
    if ndim == 3:
        # 3D: [batch, seq_len, channels]
        batch_val, seq_len_val = shape[0], shape[1]
        N_val = batch_val * seq_len_val
        x_2d = x.view(N_val, channels_val)
    else:
        # 5D: [batch, seq_len, channels, H, W]
        batch_val, seq_len_val, H_val, W_val = shape[0], shape[1], shape[3], shape[4]
        N_val = batch_val * seq_len_val * H_val * W_val
        x_2d = x.view(N_val, channels_val)

    # 构造输出 tensor (2D)
    output_2d = torch.empty(N_val, channels_val, dtype=torch.float32, device=x.device)

    # 调用 kernel
    batchnorm_kernel_2d(x_2d, gamma, beta, output_2d, eps, N_val, channels_val)

    # Reshape 回原始 shape
    output = output_2d.view(shape)

    # 转换回原类型
    if input_dtype != torch.float32:
        output = output.to(input_dtype)

    return output
