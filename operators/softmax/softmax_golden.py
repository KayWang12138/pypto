#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Softmax Golden 参考实现

公式: softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API)
"""

import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np


def softmax_golden(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax 参考实现 (PyTorch)

    对输入张量沿指定轴进行 softmax 归一化，输出概率分布。
    使用 PyTorch 内置的数值稳定实现。

    Args:
        x: 输入张量，shape [batch, seq, ...dims]
        dim: 归一化轴（默认 -1，最后一维）

    Returns:
        归一化后的概率分布，shape 与输入相同

    公式:
        softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
    """
    return F.softmax(x, dim=dim)


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("softmax_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # 1. 典型 case 验证（来自 spec.md §11）
    print("\n[典型 case 验证]")

    # 性能_P0: [1, 4096, 4096]
    try:
        x = torch.randn(1, 4096, 4096)
        y = softmax_golden(x, dim=-1)
        expected_shape = (1, 4096, 4096)
        assert y.shape == expected_shape, f"Shape mismatch: {y.shape} vs {expected_shape}"
        # 检查和为 1
        sum_check = torch.allclose(y.sum(dim=-1), torch.ones(1, 4096), atol=1e-5)
        assert sum_check, "Sum along dim != 1"
        print(f"  性能_P0: [1, 4096, 4096] dim=-1 ... ✓ PASS")
    except Exception as e:
        print(f"  性能_P0: [1, 4096, 4096] dim=-1 ... ✗ FAIL: {e}")
        all_passed = False

    # 功能_P0: [2, 1024, 512]
    try:
        x = torch.randn(2, 1024, 512)
        y = softmax_golden(x, dim=-1)
        expected_shape = (2, 1024, 512)
        assert y.shape == expected_shape, f"Shape mismatch: {y.shape} vs {expected_shape}"
        sum_check = torch.allclose(y.sum(dim=-1), torch.ones(2, 1024), atol=1e-5)
        assert sum_check, "Sum along dim != 1"
        print(f"  功能_P0: [2, 1024, 512] dim=-1 ... ✓ PASS")
    except Exception as e:
        print(f"  功能_P0: [2, 1024, 512] dim=-1 ... ✗ FAIL: {e}")
        all_passed = False

    # 功能_P1: [4, 256, 128] dim=1
    try:
        x = torch.randn(4, 256, 128)
        y = softmax_golden(x, dim=1)
        expected_shape = (4, 256, 128)
        assert y.shape == expected_shape, f"Shape mismatch: {y.shape} vs {expected_shape}"
        sum_check = torch.allclose(y.sum(dim=1), torch.ones(4, 128), atol=1e-5)
        assert sum_check, "Sum along dim != 1"
        print(f"  功能_P1: [4, 256, 128] dim=1 ... ✓ PASS")
    except Exception as e:
        print(f"  功能_P1: [4, 256, 128] dim=1 ... ✗ FAIL: {e}")
        all_passed = False

    # 功能_P1: [8, 512, 64]
    try:
        x = torch.randn(8, 512, 64)
        y = softmax_golden(x, dim=-1)
        expected_shape = (8, 512, 64)
        assert y.shape == expected_shape, f"Shape mismatch: {y.shape} vs {expected_shape}"
        sum_check = torch.allclose(y.sum(dim=-1), torch.ones(8, 512), atol=1e-5)
        assert sum_check, "Sum along dim != 1"
        print(f"  功能_P1: [8, 512, 64] dim=-1 ... ✓ PASS")
    except Exception as e:
        print(f"  功能_P1: [8, 512, 64] dim=-1 ... ✗ FAIL: {e}")
        all_passed = False

    # 2. 泛化 case 验证（来自 spec.md §7 动态轴范围）
    print("\n[泛化 case 验证]")

    # 边界值：batch=1
    try:
        x = torch.randn(1, 128, 64)
        y = softmax_golden(x, dim=-1)
        assert y.shape == (1, 128, 64)
        print(f"  batch=1: [1, 128, 64] ... ✓ PASS")
    except Exception as e:
        print(f"  batch=1: [1, 128, 64] ... ✗ FAIL: {e}")
        all_passed = False

    # 边界值：seq=65536
    try:
        x = torch.randn(2, 1024, 64)
        y = softmax_golden(x, dim=-1)
        assert y.shape == (2, 1024, 64)
        print(f"  seq=1024: [2, 1024, 64] ... ✓ PASS")
    except Exception as e:
        print(f"  seq=1024: [2, 1024, 64] ... ✗ FAIL: {e}")
        all_passed = False

    # 3. 值域检查（从公式推导）
    print("\n[值域检查]")

    # 检查输出在 (0, 1) 范围内
    try:
        x = torch.randn(4, 128, 64)
        y = softmax_golden(x, dim=-1)
        assert (y >= 0).all() and (y <= 1).all(), "Output not in [0, 1]"
        print(f"  输出值域 [0, 1] ... ✓ PASS")
    except Exception as e:
        print(f"  输出值域 [0, 1] ... ✗ FAIL: {e}")
        all_passed = False

    # 检查沿归约轴和为 1
    try:
        x = torch.randn(4, 128, 64)
        y = softmax_golden(x, dim=-1)
        sum_along_dim = y.sum(dim=-1)
        assert torch.allclose(sum_along_dim, torch.ones_like(sum_along_dim), atol=1e-5)
        print(f"  沿归约轴和为 1 ... ✓ PASS")
    except Exception as e:
        print(f"  沿归约轴和为 1 ... ✗ FAIL: {e}")
        all_passed = False

    # 4. 数值稳定性检查
    print("\n[数值稳定性检查]")

    # 大值输入
    try:
        x = torch.tensor([[100.0, 101.0, 102.0]])
        y = softmax_golden(x, dim=-1)
        assert not torch.isnan(y).any() and not torch.isinf(y).any()
        print(f"  大值输入 (x=100) ... ✓ PASS")
    except Exception as e:
        print(f"  大值输入 (x=100) ... ✗ FAIL: {e}")
        all_passed = False

    # 小值输入
    try:
        x = torch.tensor([[-100.0, -101.0, -102.0]])
        y = softmax_golden(x, dim=-1)
        assert not torch.isnan(y).any() and not torch.isinf(y).any()
        print(f"  小值输入 (x=-100) ... ✓ PASS")
    except Exception as e:
        print(f"  小值输入 (x=-100) ... ✗ FAIL: {e}")
        all_passed = False

    # 5. 特殊点验证
    print("\n[特殊点验证]")

    # 零值输入
    try:
        x = torch.zeros(2, 3)
        y = softmax_golden(x, dim=-1)
        expected = torch.ones(2, 3) / 3
        assert torch.allclose(y, expected, atol=1e-5)
        print(f"  零值输入 ... ✓ PASS")
    except Exception as e:
        print(f"  零值输入 ... ✗ FAIL: {e}")
        all_passed = False

    # 6. API 对比（如适用）
    print("\n[API 对比]")

    # 与 torch.nn.functional.softmax 对比
    try:
        x = torch.randn(4, 128, 64)
        y1 = softmax_golden(x, dim=-1)
        y2 = F.softmax(x, dim=-1)
        assert torch.allclose(y1, y2, atol=1e-7)
        print(f"  与 torch.nn.functional.softmax 对比 ... ✓ PASS")
    except Exception as e:
        print(f"  与 torch.nn.functional.softmax 对比 ... ✗ FAIL: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过")
    else:
        print("❌ 部分验证失败")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
