#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""
Eq Golden 参考实现

公式: y = (x1 == x2)
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API)
"""

import torch


def eq_golden(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Eq 参考实现 (PyTorch)

    Args:
        x1: 输入 tensor
        x2: 输入 tensor，与 x1 broadcastable

    Returns:
        输出 tensor，bool 类型，x1 == x2
    """
    return torch.eq(x1, x2)
