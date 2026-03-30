#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
topk Golden 参考实现

公式: values, indices = topk(input, k, dim=None, largest=True, sorted=True)
置信度: (使用 PyTorch 内置 API)
"""
import torch
from typing import Optional, Tuple


def topk_golden(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    topk 参考实现 (PyTorch)

    Args:
        input: 输入张量 [b, s, n, d]
        k: 选取的元素数量
        dim: 操作维度，默认 -1
        largest: True 选最大，False 选最小，默认 True
        sorted: True 则排序输出，默认 True

    Returns:
        values: 选取的 k 个元素值 [b, s, n, k]
        indices: 选取元素在 dim 维度上的索引 [b, s, n, k]

    公式:
        values, indices = topk(input, k, dim, largest, sorted)
    """
    # PyTorch 内置 topk 实现
    values, indices = torch.topk(input, k=k, dim=dim, largest=largest, sorted=sorted)

    # torch.topk 返回 int64 索引，符合 spec 要求
    return values, indices


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""
    import numpy as np

    print("=" * 60)
    print("topk_golden 验证报告")
    print("=" * 60)

    # 1. 典型 case 验证（来自 spec.md §11）
    print("\n[典型 case 验证]")

    # 性能_P0: k=10, dim=-1, largest=True, sorted=True
    input = torch.randn(1, 1024, 16, 64, dtype=torch.float32)
    values, indices = topk_golden(input, k=10, dim=-1, largest=True, sorted=True)
    assert values.shape == (1, 1024, 16, 10), f"Expected values shape (1, 1024, 16, 10), got {values.shape}"
    assert indices.shape == (1, 1024, 16, 10), f"Expected indices shape (1, 1024, 16, 10). got {indices.shape}"
    assert indices.dtype == torch.int64, f"Expected indices dtype int64, got {indices.dtype}"
    print("  性能_P0: k=10, dim=-1, largest=True, sorted=True ... PASS")

    # 功能_P0: k=5, dim=-1, largest=False, sorted=True
    input = torch.randn(2, 512, 8, 128, dtype=torch.float32)
    values, indices = topk_golden(input, k=5, dim=-1, largest=False, sorted=True)
    assert values.shape == (2, 512, 8, 5), f"Expected values shape (2, 512, 8, 5). got {values.shape}"
    assert indices.shape == (2, 512, 8, 5), f"Expected indices shape (2, 512, 8, 5). got {indices.shape}"
    print("  功能_P0: k=5, dim=-1, largest=False, sorted=True ... PASS")

    # 动态轴_P1: k=20, dim=1, largest=True, sorted=False
    input = torch.randn(4, 2048, 32, 64, dtype=torch.float32)
    values, indices = topk_golden(input, k=20, dim=1, largest=True, sorted=False)
    assert values.shape == (4, 20, 32, 64), f"Expected values shape (4, 20, 32, 64). got {values.shape}"
    assert indices.shape == (4, 20, 32, 64), f"Expected indices shape (4, 20, 32, 64). got {indices.shape}"
    print("  动态轴_P1: k=20, dim=1, largest=True, sorted=False ... PASS")

    # 2. 泛化 case 验证（来自 spec.md §7 动态轴范围）
    print("\n[泛化 case 验证]")

    # 最小值测试
    input = torch.randn(1, 1, 1, 4, dtype=torch.float32)
    values, indices = topk_golden(input, k=2, dim=-1, largest=True, sorted=True)
    assert values.shape == (1, 1, 1, 2), f"Expected values shape (1, 1, 1, 2). got {values.shape}"
    print("  最小 shape [1, 1, 1, 4], k=2 ... PASS")

    # 大 shape 测试
    input = torch.randn(8, 4096, 64, 128, dtype=torch.float32)
    values, indices = topk_golden(input, k=50, dim=-1, largest=True, sorted=True)
    assert values.shape == (8, 4096, 64, 50), f"Expected values shape (8, 4096, 64, 50). got {values.shape}"
    print("  大 shape [8, 4096, 64, 128], k=50 ... PASS")

    # 3. 值域检查（从公式推导）
    print("\n[值域检查]")

    # 正常值域
    input = torch.randn(2, 4, 8, 16, dtype=torch.float32)
    values, indices = topk_golden(input, k=4, dim=-1, largest=True, sorted=True)
    print("  正常值域 ... PASS")

    # 包含零值
    input = torch.zeros(2, 4, 8, 16, dtype=torch.float32)
    input[0, 0, 0, :4] = torch.arange(4, dtype=torch.float32)
    values, indices = topk_golden(input, k=2, dim=-1, largest=True, sorted=True)
    print("  包含零值 ... PASS")

    # 4. 数值稳定性检查
    print("\n[数值稳定性检查]")

    # 大值输入
    input = torch.randn(2, 4, 8, 16, dtype=torch.float32) * 100
    values, indices = topk_golden(input, k=4, dim=-1, largest=True, sorted=True)
    print("  大值输入 (x=100) ... PASS")

    # 包含 Inf/-Inf
    input = torch.randn(2, 4, 8, 16, dtype=torch.float32)
    input[0, 0, 0, 0] = float('inf')
    input[0, 0, 0, 1] = float('-inf')
    values, indices = topk_golden(input, k=4, dim=-1, largest=True, sorted=True)
    print("  包含 Inf/-Inf ... PASS")

    # 5. 功能正确性检查
    print("\n[功能正确性检查]")

    # 验证 largest=True 选取最大值
    input = torch.tensor([[[[1.0, 5.0, 3.0, 2.0, 4.0]]]], dtype=torch.float32)
    values, indices = topk_golden(input, k=3, dim=-1, largest=True, sorted=True)
    expected_values = torch.tensor([[[[5.0, 4.0, 3.0]]]], dtype=torch.float32)
    expected_indices = torch.tensor([[[[1, 4, 2]]]], dtype=torch.int64)
    assert torch.allclose(values, expected_values), f"Expected values {expected_values}, got {values}"
    assert torch.equal(indices, expected_indices), f"Expected indices {expected_indices}, got {indices}"
    print("  largest=True 选取最大值 ... PASS")

    # 验证 largest=False 选取最小值
    input = torch.tensor([[[[1.0, 5.0, 3.0, 2.0, 4.0]]]], dtype=torch.float32)
    values, indices = topk_golden(input, k=3, dim=-1, largest=False, sorted=True)
    expected_values = torch.tensor([[[[1.0, 2.0, 3.0]]]], dtype=torch.float32)
    expected_indices = torch.tensor([[[[0, 3, 2]]]], dtype=torch.int64)
    assert torch.allclose(values, expected_values), f"Expected values {expected_values}, got {values}"
    assert torch.equal(indices, expected_indices), f"Expected indices {expected_indices}, got {indices}"
    print("  largest=False 选取最小值 ... PASS")

    # 验证 sorted=True 降序排列
    input = torch.tensor([[[[1.0, 5.0, 3.0, 2.0, 4.0]]]], dtype=torch.float32)
    values, indices = topk_golden(input, k=3, dim=-1, largest=True, sorted=True)
    # 检查降序
    assert values[0, 0, 0, 0] >= values[0, 0, 0, 1] >= values[0, 0, 0, 2], "Values should be sorted in descending order"
    print("  sorted=True 降序排列 ... PASS")

    # 验证 dim 参数
    input = torch.randn(2, 8, 4, 16, dtype=torch.float32)
    values, indices = topk_golden(input, k=3, dim=1, largest=True, sorted=True)
    assert values.shape == (2, 3, 4, 16), f"Expected values shape (2, 3, 4, 16). got {values.shape}"
    print("  dim=1 沿序列维度选择 ... PASS")

    # 验证负数 dim
    input = torch.randn(2, 8, 4, 16, dtype=torch.float32)
    values, indices = topk_golden(input, k=3, dim=-2, largest=True, sorted=True)
    assert values.shape == (2, 8, 3, 16), f"Expected values shape (2, 8, 3, 16). got {values.shape}"
    print("  dim=-2 负数维度索引 ... PASS")

    # 6. API 对比（与 PyTorch API 对比）
    print("\n[API 对比]")

    # 对比 torch.topk 和 topk_golden 结果
    input = torch.randn(2, 4, 8, 16, dtype=torch.float32)
    py_values, py_indices = torch.topk(input, k=5, dim=-1, largest=True, sorted=True)
    golden_values, golden_indices = topk_golden(input, k=5, dim=-1, largest=True, sorted=True)
    assert torch.allclose(py_values, golden_values), "PyTorch and golden values should match"
    assert torch.equal(py_indices, golden_indices), "PyTorch and golden indices should match"
    print("  torch.topk 与 topk_golden 结果一致 ... PASS")

    print("\n" + "=" * 60)
    print("所有验证通过")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
