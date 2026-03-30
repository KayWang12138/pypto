#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""
sqrt Golden 参考实现

公式: y = sqrt(x) (逐元素计算平方根)
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API)
"""

import torch


def sqrt_golden(x: torch.Tensor) -> torch.Tensor:
    """sqrt 参考实现 (PyTorch)

    逐元素计算输入张量的平方根。

    Args:
        x: 输入 tensor，支持任意 shape，dtype 为 float32。

    Returns:
        输出 tensor，shape 与输入相同，每个元素为输入对应元素的平方根。
    """
    return torch.sqrt(x)
