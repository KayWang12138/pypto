#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""
GroupNorm Golden 参考实现

公式: y = γ * (x - μ) / sqrt(σ² + ε) + β
其中 μ 和 σ² 沿 (C//G, H, W) 维度计算

置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API)
"""

import torch
from typing import Optional


def groupnorm_golden(
    x: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """GroupNorm 参考实现 (PyTorch)

    Args:
        x: 输入 tensor，shape [N, C, H, W] 或 [N, C, L]
        num_groups: 分组数 G，要求 C % G == 0
        weight: 缩放参数，shape [C]，可选
        bias: 偏移参数，shape [C]，可选
        eps: 数值稳定性常数

    Returns:
        输出 tensor，shape 与输入相同
    """
    return torch.nn.functional.group_norm(
        x, num_groups, weight=weight, bias=bias, eps=eps
    )
