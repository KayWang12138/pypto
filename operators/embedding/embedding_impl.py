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
"""
PyPTO embedding kernel implementation.

embedding 算子实现查表操作，将索引张量映射到对应的嵌入向量。

公式: output[i, j, :] = weight[indices[i, j], :]

实现策略:
由于 pypto.gather 要求 index.dim = input.dim，而 embedding 的 indices 和 weight 维度不匹配，
因此采用 flatten + index_select + reshape 的方式实现。

参考: design.md 中的 API 映射设计
"""

import pypto
import torch
from typing import Optional, Tuple


# ─────────────────────────────────────────────
# 1. JIT Kernel (使用隐式 shape 推断)
# ─────────────────────────────────────────────

@pypto.frontend.jit
def embedding_kernel(
    weight: pypto.Tensor([], pypto.DT_FP32),
    indices_flat: pypto.Tensor([], pypto.DT_INT64),
    output_flat: pypto.Tensor([], pypto.DT_FP32),
):
    """PyPTO jit kernel for embedding operation.

    使用 flatten 后的 indices 进行查表，然后 reshape 回 3D。

    Args:
        weight: Embedding 表 [vocab_size, embed_dim]
        indices_flat: 展平后的索引 [batch*seq]
        output_flat: 展平后的输出 [batch*seq, embed_dim]
    """
    # 设置 tiling 配置
    pypto.set_vec_tile_shapes(64, 128)

    # 使用 index_select 沿 dim=0 查表
    result = pypto.index_select(weight, 0, indices_flat)

    # 输出写回
    output_flat[:] = result


@pypto.frontend.jit
def embedding_kernel_with_padding(
    weight: pypto.Tensor([], pypto.DT_FP32),
    indices_flat: pypto.Tensor([], pypto.DT_INT64),
    output_flat: pypto.Tensor([], pypto.DT_FP32),
    padding_idx: float,
):
    """PyPTO jit kernel for embedding operation with padding_idx.

    注意：由于 pypto.where 的广播限制，padding 处理在 wrapper 中通过 PyTorch 完成。
    此 kernel 仅执行查表操作。

    Args:
        weight: Embedding 表 [vocab_size, embed_dim]
        indices_flat: 展平后的索引 [batch*seq]
        output_flat: 展平后的输出 [batch*seq, embed_dim]
        padding_idx: 填充索引值（未使用）
    """
    pypto.set_vec_tile_shapes(64, 128)

    # 核心查表操作
    result = pypto.index_select(weight, 0, indices_flat)

    # 输出写回
    output_flat[:] = result


# ─────────────────────────────────────────────
# 2. Wrapper 函数
# ─────────────────────────────────────────────

def embedding_wrapper(
    indices: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: Optional[int] = None,
) -> torch.Tensor:
    """embedding 算子 wrapper。

    查表操作，将索引张量映射到对应的嵌入向量。

    Args:
        indices: 索引张量，shape [batch, seq]，dtype int64
        weight: Embedding 查找表，shape [vocab_size, embed_dim]，dtype float32
        padding_idx: 可选，指定填充索引（当前未使用，保留以兼容 PyTorch API）

    Returns:
        output: 嵌入向量输出，shape [batch, seq, embed_dim]

    Note:
        根据 PyTorch 行为，padding_idx 在 forward 时不会将输出置零，
        仅在 backward 时影响梯度。PyPTO 作为推理框架，行为与 PyTorch forward 一致。
    """
    original_indices_dtype = indices.dtype
    original_weight_dtype = weight.dtype
    original_shape = indices.shape

    # 确保输入类型正确
    if original_indices_dtype != torch.int64:
        indices = indices.long()
    if original_weight_dtype != torch.float32:
        weight = weight.float()

    batch, seq = indices.shape
    vocab_size, embed_dim = weight.shape

    # 展平 indices: [batch, seq] -> [batch*seq]
    indices_flat = indices.reshape(-1)

    # 创建展平后的输出 tensor: [batch*seq, embed_dim]
    output_flat = torch.empty(
        (batch * seq, embed_dim),
        dtype=torch.float32,
        device=weight.device,
    )

    # 调用 kernel
    embedding_kernel(weight, indices_flat, output_flat)

    # reshape 回 3D: [batch*seq, embed_dim] -> [batch, seq, embed_dim]
    output = output_flat.reshape(batch, seq, embed_dim)

    # 恢复原始 dtype
    if original_weight_dtype != torch.float32:
        output = output.to(original_weight_dtype)

    return output
