#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""
Mul Golden 参考实现

公式: y = x1 * x2
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API)
"""

import torch


def mul_golden(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Mul 参考实现 (PyTorch)

    Args:
        x1: 输入 tensor
        x2: 输入 tensor，与 x1 broadcastable

    Returns:
        输出 tensor，x1 * x2
    """
    return torch.mul(x1, x2)
