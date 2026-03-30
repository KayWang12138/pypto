#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""
LeakyReLU Golden 参考实现

公式: y = x if x >= 0 else alpha * x
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API)
"""

import torch
from typing import Optional


def leaky_relu_golden(
    x: torch.Tensor,
    alpha: float = 0.01,
) -> torch.Tensor:
    """LeakyReLU 参考实现 (PyTorch)

    Args:
        x: 输入 tensor，支持 1D-4D
        alpha: 负区间斜率，默认 0.01

    Returns:
        输出 tensor，shape 与输入相同
    """
    return torch.nn.functional.leaky_relu(x, negative_slope=alpha)
