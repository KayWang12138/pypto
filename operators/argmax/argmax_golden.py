#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""argmax Golden 参考实现

公式:
    argmax(x, dim) = argmax_i x[..., i, ...]
    当 dim=None 时: argmax(x) = argmax_i x.flatten()[i]

置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API)
"""

import torch
from typing import Optional


def argmax_golden(
    input: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """argmax 参考实现 (PyTorch)

    沿指定维度找最大值的索引。

    Args:
        input: 输入张量，支持 1-4 维
        dim: 指定维度，None 表示全局 argmax
        keepdim: 是否保持被归约的维度

    Returns:
        索引张量，dtype 为 int64
    """
    return torch.argmax(input, dim=dim, keepdim=keepdim)
