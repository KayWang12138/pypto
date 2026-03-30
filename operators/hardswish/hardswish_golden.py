#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
"""
Hardswish Golden 参考实现

公式: hardswish(x) = x * relu6(x + 3) / 6
等价于: hardswish(x) = x * min(max(x + 3, 0), 6) / 6
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API)
"""

import torch


def hardswish_golden(x: torch.Tensor) -> torch.Tensor:
    """Hardswish 参考实现 (PyTorch)

    Args:
        x: 输入 tensor，支持 1D-4D

    Returns:
        输出 tensor，shape 与输入相同
    """
    return torch.nn.functional.hardswish(x)
