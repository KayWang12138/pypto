#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO GroupNorm kernel implementation.

公式: y = γ * (x - μ) / sqrt(σ² + ε) + β
其中 μ 和 σ² 沿 (C//G, H, W) 维度计算

实现策略:
1. 4D 输入 [N, C, H, W] reshape 为 [N*G, C//G*H*W]
2. 沿 dim=1 计算均值和方差
3. 归一化
4. 应用 affine 变换
5. Reshape 回 [N, C, H, W]

动态轴支持: 使用 pypto.Tensor([], pypto.DT_FP32) 隐式推断
"""

import pypto
import torch
from typing import Optional
import os
import sys


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


# ─────────────────────────────────────────────
# 1. JIT Kernel
# ─────────────────────────────────────────────

@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def groupnorm_kernel_2d(
    x: pypto.Tensor(),
    gamma: pypto.Tensor(),
    beta: pypto.Tensor(),
    output: pypto.Tensor(),
    eps: float,
    M: int,  # N * G
    K: int,  # C//G * H * W
):
    """GroupNorm kernel for 2D input [M, K]

    M = N * G (batch * groups)
    K = C//G * H * W (channels_per_group * spatial)

    沿 dim=1 (K) 计算均值和方差
    """
    # 设置 Tiling
    pypto.set_vec_tile_shapes(64, K)

    # Step 1: Compute mean along dim=1
    sum_x = pypto.sum(x, dim=1, keepdim=True)  # [M, 1]
    inv_K = 1.0 / K
    mean = pypto.mul(sum_x, inv_K)

    # Step 2: Center
    centered = pypto.sub(x, mean)  # [M, K]

    # Step 3: Compute variance
    squared = pypto.mul(centered, centered)
    sum_sq = pypto.sum(squared, dim=1, keepdim=True)  # [M, 1]
    var = pypto.mul(sum_sq, inv_K)

    # Step 4: Normalize
    var_eps = pypto.add(var, eps)
    std = pypto.sqrt(var_eps)
    normalized = pypto.div(centered, std)  # [M, K]

    # Step 5: Apply affine (gamma, beta are per-channel, will broadcast)
    # Note: affine transform is done in wrapper after reshape

    # 输出写回
    pypto.assemble(normalized, [0, 0], output)


# ─────────────────────────────────────────────
# 2. Wrapper 函数
# ─────────────────────────────────────────────

def groupnorm_wrapper(
    x: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """GroupNorm wrapper，供 test_groupnorm.py 调用。

    Args:
        x: 输入 torch.Tensor，支持 4D [N, C, H, W]
        num_groups: 分组数 G，要求 C % G == 0
        weight: 缩放参数 [C]，可选
        bias: 偏移参数 [C]，可选
        eps: 数值稳定性常数，默认 1e-5

    Returns:
        输出 tensor，shape 与输入相同
    """
    # 约束检查
    if not x.is_contiguous():
        x = x.contiguous()

    if x.numel() == 0:
        raise ValueError("Input tensor must not be empty")

    ndim = x.dim()
    if ndim != 4:
        raise ValueError(f"Unsupported input dimension: {ndim}, expected 4")

    N, C, H, W = x.shape

    if C % num_groups != 0:
        raise ValueError(f"C ({C}) must be divisible by num_groups ({num_groups})")

    # 类型转换
    input_dtype = x.dtype
    if x.dtype == torch.float16:
        x = x.float()
        if weight is not None and weight.dtype == torch.float16:
            weight = weight.float()
        if bias is not None and bias.dtype == torch.float16:
            bias = bias.float()
    elif x.dtype == torch.bfloat16:
        x = x.float()
        if weight is not None and weight.dtype == torch.bfloat16:
            weight = weight.float()
        if bias is not None and bias.dtype == torch.bfloat16:
            bias = bias.float()

    # Reshape: [N, C, H, W] -> [N, G, C//G, H, W] -> [N*G, C//G*H*W]
    G = num_groups
    C_per_G = C // G

    # Reshape to [N*G, C//G*H*W]
    x_2d = x.view(N * G, C_per_G * H * W)

    M = N * G
    K = C_per_G * H * W

    # 构造输出 tensor
    output_2d = torch.empty(M, K, dtype=torch.float32, device=x.device)

    # 调用 kernel
    groupnorm_kernel_2d(x_2d, weight if weight is not None else torch.ones(C, dtype=torch.float32, device=x.device),
                         bias if bias is not None else torch.zeros(C, dtype=torch.float32, device=x.device),
                         output_2d, eps, M, K)

    # Reshape 回 [N, G, C//G, H, W] -> [N, C, H, W]
    output = output_2d.view(N, G, C_per_G, H, W)
    output = output.permute(0, 1, 2, 3, 4).contiguous()  # Keep same layout
    output = output.view(N, C, H, W)

    # Apply affine transformation (gamma * x + beta)
    if weight is not None:
        # weight shape [C], reshape to [1, C, 1, 1] for broadcasting
        weight = weight.view(1, C, 1, 1)
        output = output * weight

    if bias is not None:
        # bias shape [C], reshape to [1, C, 1, 1] for broadcasting
        bias = bias.view(1, C, 1, 1)
        output = output + bias

    # 转换回原类型
    if input_dtype != torch.float32:
        output = output.to(input_dtype)

    return output
