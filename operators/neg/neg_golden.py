#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
"""neg Golden 参考实现

公式: y = -x
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API)
"""

import torch


def neg_golden(x: torch.Tensor) -> torch.Tensor:
    """neg 参考实现 (PyTorch)"""
    return torch.neg(x)
