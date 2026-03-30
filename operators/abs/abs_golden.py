#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
abs Golden 参考实现

公式: y = |x|（逐元素取绝对值）
置信度: (使用 PyTorch 内置 API)
"""

import torch
from typing import Optional, List, Tuple


# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────

def abs_golden(x: torch.Tensor) -> torch.Tensor:
    """
    abs 参考实现 (PyTorch)

    逐元素计算输入张量的绝对值。

    Args:
        x: 输入 tensor，支持任意 shape，dtype 为 float32。

    Returns:
        输出 tensor，shape 与输入相同，每个元素为输入对应元素的绝对值。

    公式:
        y = |x|
    """
    return torch.abs(x)


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("abs_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # 1. 典型 case 验证（来自 spec.md)
    print("\n[典型 case 验证]")

    # 配置 1: 功能_P0 [128, 1024]
    try:
        x = torch.randn(128, 1024, dtype=torch.float32)
        y = abs_golden(x)
        expected = torch.abs(x)
        if y.shape == expected.shape and torch.allclose(y, expected, rtol=1e-5, atol=1e-5):
            print(f"  功能_P0: [128, 1024] ... PASS")
        else:
            print(f"  功能_P0: [128, 1024] ... FAIL (shape or value mismatch)")
            all_passed = False
    except Exception as e:
        print(f"  功能_P0: [128, 1024] ... FAIL ({e})")
        all_passed = False

    # 配置 2: 动态轴_P0 [64, 1024]（模拟动态轴）
    try:
        x = torch.randn(64, 1024, dtype=torch.float32)
        y = abs_golden(x)
        expected = torch.abs(x)
        if y.shape == expected.shape and torch.allclose(y, expected, rtol=1e-5, atol=1e-5):
            print(f"  动态轴_P0: [64, 1024] ... PASS")
        else:
            print(f"  动态轴_P0: [64, 1024] ... FAIL (shape or value mismatch)")
            all_passed = False
    except Exception as e:
        print(f"  动态轴_P0: [64, 1024] ... FAIL ({e})")
        all_passed = False

    # 配置 3: 性能_P0 [4096, 4096]
    try:
        x = torch.randn(4096, 4096, dtype=torch.float32)
        y = abs_golden(x)
        expected = torch.abs(x)
        if y.shape == expected.shape and torch.allclose(y, expected, rtol=1e-5, atol=1e-5):
            print(f"  性能_P0: [4096, 4096] ... PASS")
        else:
            print(f"  性能_P0: [4096, 4096] ... FAIL (shape or value mismatch)")
            all_passed = False
    except Exception as e:
        print(f"  性能_P0: [4096, 4096] ... FAIL ({e})")
        all_passed = False

    # 2. 泛化 case 验证（动态轴范围采样）
    print("\n[泛化 case 验证]")

    # 小 shape [1, 32]
    try:
        x = torch.randn(1, 32, dtype=torch.float32)
        y = abs_golden(x)
        expected = torch.abs(x)
        if y.shape == expected.shape and torch.allclose(y, expected, rtol=1e-5, atol=1e-5):
            print(f"  小 shape [1, 32] ... PASS")
        else:
            print(f"  小 shape [1, 32] ... FAIL")
            all_passed = False
    except Exception as e:
        print(f"  小 shape [1, 32] ... FAIL ({e})")
        all_passed = False

    # 中 shape [512, 512]
    try:
        x = torch.randn(512, 512, dtype=torch.float32)
        y = abs_golden(x)
        expected = torch.abs(x)
        if y.shape == expected.shape and torch.allclose(y, expected, rtol=1e-5, atol=1e-5):
            print(f"  中 shape [512, 512] ... PASS")
        else:
            print(f"  中 shape [512, 512] ... FAIL")
            all_passed = False
    except Exception as e:
        print(f"  中 shape [512, 512] ... FAIL ({e})")
        all_passed = False

    # 4D shape [2, 4, 8, 16]
    try:
        x = torch.randn(2, 4, 8, 16, dtype=torch.float32)
        y = abs_golden(x)
        expected = torch.abs(x)
        if y.shape == expected.shape and torch.allclose(y, expected, rtol=1e-5, atol=1e-5):
            print(f"  4D shape [2, 4, 8, 16] ... PASS")
        else:
            print(f"  4D shape [2, 4, 8, 16] ... FAIL")
            all_passed = False
    except Exception as e:
        print(f"  4D shape [2, 4, 8, 16] ... FAIL ({e})")
        all_passed = False

    # 3. 值域检查（从公式推导）
    print("\n[值域检查]")

    # 检查输出非负
    try:
        x = torch.randn(100, 100, dtype=torch.float32)
        y = abs_golden(x)
        if (y >= 0).all():
            print(f"  检查输出非负 ... PASS")
        else:
            print(f"  检查输出非负 ... FAIL (存在负值)")
            all_passed = False
    except Exception as e:
        print(f"  检查输出非负 ... FAIL ({e})")
        all_passed = False

    # 4. 数值稳定性检查
    print("\n[数值稳定性检查]")

    # 大值输入
    try:
        x = torch.tensor([1e10, -1e10, 1e20, -1e20], dtype=torch.float32)
        y = abs_golden(x)
        expected = torch.abs(x)
        if torch.allclose(y, expected, rtol=1e-5, atol=1e-5):
            print(f"  大值输入 ... PASS")
        else:
            print(f"  大值输入 ... FAIL")
            all_passed = False
    except Exception as e:
        print(f"  大值输入 ... FAIL ({e})")
        all_passed = False

    # 零值输入
    try:
        x = torch.zeros(10, 10, dtype=torch.float32)
        y = abs_golden(x)
        if (y == 0).all():
            print(f"  零值输入 ... PASS")
        else:
            print(f"  零值输入 ... FAIL")
            all_passed = False
    except Exception as e:
        print(f"  零值输入 ... FAIL ({e})")
        all_passed = False

    # 5. 特殊值检查
    print("\n[特殊值检查]")

    # 负值输入
    try:
        x = torch.tensor([-1.0, -2.0, -3.0, -100.0], dtype=torch.float32)
        y = abs_golden(x)
        expected = torch.tensor([1.0, 2.0, 3.0, 100.0], dtype=torch.float32)
        if torch.allclose(y, expected, rtol=1e-5, atol=1e-5):
            print(f"  负值输入 ... PASS")
        else:
            print(f"  负值输入 ... FAIL")
            all_passed = False
    except Exception as e:
        print(f"  负值输入 ... FAIL ({e})")
        all_passed = False

    # 正值输入
    try:
        x = torch.tensor([1.0, 2.0, 3.0, 100.0], dtype=torch.float32)
        y = abs_golden(x)
        expected = torch.tensor([1.0, 2.0, 3.0, 100.0], dtype=torch.float32)
        if torch.allclose(y, expected, rtol=1e-5, atol=1e-5):
            print(f"  正值输入 ... PASS")
        else:
            print(f"  正值输入 ... FAIL")
            all_passed = False
    except Exception as e:
        print(f"  正值输入 ... FAIL ({e})")
        all_passed = False

    # 6. API 对比
    print("\n[API 对比]")

    try:
        x = torch.randn(100, 100, dtype=torch.float32)
        y = abs_golden(x)
        expected = torch.abs(x)
        if torch.allclose(y, expected, rtol=1e-5, atol=1e-5):
            print(f"  与 torch.abs 对比 ... PASS")
        else:
            print(f"  与 torch.abs 对比 ... FAIL")
            all_passed = False
    except Exception as e:
        print(f"  与 torch.abs 对比 ... FAIL ({e})")
        all_passed = False

    # 7. 函数签名检查
    print("\n[函数签名检查]")

    import inspect
    sig = inspect.signature(abs_golden)
    params = list(sig.parameters.keys())
    if params == ['x']:
        print(f"  签名正确: abs_golden(x: torch.Tensor) -> torch.Tensor ... PASS")
    else:
        print(f"  签名错误: 期望 ['x'], 实际 {params} ... FAIL")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("所有验证通过")
    else:
        print("部分验证失败，请检查上述 FAIL 项")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _validate()
