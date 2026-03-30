#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""rsqrt Golden 参考实现

公式: y = 1/sqrt(x)
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API)
"""

import torch


def rsqrt_golden(x: torch.Tensor) -> torch.Tensor:
    """rsqrt 参考实现 (PyTorch)

    逐元素计算输入张量的倒数平方根。

    Args:
        x: 输入 tensor

    Returns:
        输出 tensor，每个元素为 1/sqrt(x)
    """
    return torch.rsqrt(x)
