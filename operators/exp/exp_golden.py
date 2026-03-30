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
"""exp Golden 参考实现

公式: y = exp(x) = e^x
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API torch.exp)

边界条件:
  - exp(0) = 1.0
  - exp(-inf) = 0.0
  - exp(inf) = inf
  - exp(nan) = nan
"""

import torch
from typing import Tuple


def exp_golden(x: torch.Tensor) -> torch.Tensor:
    """exp 参考实现 (PyTorch)

    逐元素计算指数函数: y = e^x

    Args:
        x: 输入 tensor，支持 float32/float16
           shape: [m, n] (2D) 或 [b, s, n, d] (4D)

    Returns:
        输出 tensor，shape 和 dtype 与输入相同
        y = exp(x)，逐元素计算 e 的 x 次方

    公式:
        y = exp(x) = e^x

    Examples:
        >>> x = torch.tensor([0.0, 1.0, 2.0])
        >>> exp_golden(x)
        tensor([1.0000, 2.7183, 7.3891])
    """
    return torch.exp(x)


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("exp_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # 1. 典型 case 验证（来自 spec.md §12）
    print("\n[典型 case 验证]")

    test_cases = [
        ("功能_2D_FP32", [1024, 1024], torch.float32),
        ("功能_4D_FP32", [2, 4096, 16, 128], torch.float32),
        ("功能_2D_FP16", [1024, 1024], torch.float16),
        ("性能_FP32", [4, 8192, 32, 128], torch.float32),
        ("性能_FP16", [4, 8192, 32, 128], torch.float16),
    ]

    for name, shape, dtype in test_cases:
        try:
            x = torch.randn(shape, dtype=dtype)
            y = exp_golden(x)

            # 形状一致性检查
            if y.shape != x.shape:
                print(f"  {name}: {shape} ... ✗ FAIL (shape mismatch: {y.shape} vs {x.shape})")
                all_passed = False
                continue

            # dtype 一致性检查
            if y.dtype != x.dtype:
                print(f"  {name}: {shape} ... ✗ FAIL (dtype mismatch: {y.dtype} vs {x.dtype})")
                all_passed = False
                continue

            # 与 torch.exp 对比
            expected = torch.exp(x)
            if dtype == torch.float32:
                rtol, atol = 1e-3, 1e-5
            else:
                rtol, atol = 1e-2, 1e-3

            if torch.allclose(y, expected, rtol=rtol, atol=atol):
                print(f"  {name}: {shape} ... ✓ PASS")
            else:
                max_diff = (y - expected).abs().max().item()
                print(f"  {name}: {shape} ... ✗ FAIL (max_diff={max_diff:.6e})")
                all_passed = False
        except Exception as e:
            print(f"  {name}: {shape} ... ✗ FAIL ({e})")
            all_passed = False

    # 2. 值域检查（从公式推导）
    print("\n[值域检查]")

    # exp(0) = 1
    x_zero = torch.tensor([0.0])
    y_zero = exp_golden(x_zero)
    if torch.allclose(y_zero, torch.tensor([1.0])):
        print(f"  exp(0) = 1.0 ... ✓ PASS")
    else:
        print(f"  exp(0) = 1.0 ... ✗ FAIL (got {y_zero.item()})")
        all_passed = False

    # exp(1) = e ≈ 2.718
    x_one = torch.tensor([1.0])
    y_one = exp_golden(x_one)
    expected_e = torch.tensor([2.718281828])
    if torch.allclose(y_one, expected_e, rtol=1e-5):
        print(f"  exp(1) = e ≈ 2.718 ... ✓ PASS")
    else:
        print(f"  exp(1) = e ≈ 2.718 ... ✗ FAIL (got {y_one.item()})")
        all_passed = False

    # 3. 边界条件检查
    print("\n[边界条件检查]")

    # exp(-inf) = 0
    x_neg_inf = torch.tensor([float('-inf')])
    y_neg_inf = exp_golden(x_neg_inf)
    if y_neg_inf.item() == 0.0:
        print(f"  exp(-inf) = 0.0 ... ✓ PASS")
    else:
        print(f"  exp(-inf) = 0.0 ... ✗ FAIL (got {y_neg_inf.item()})")
        all_passed = False

    # exp(inf) = inf
    x_pos_inf = torch.tensor([float('inf')])
    y_pos_inf = exp_golden(x_pos_inf)
    if y_pos_inf.item() == float('inf'):
        print(f"  exp(inf) = inf ... ✓ PASS")
    else:
        print(f"  exp(inf) = inf ... ✗ FAIL (got {y_pos_inf.item()})")
        all_passed = False

    # exp(nan) = nan
    x_nan = torch.tensor([float('nan')])
    y_nan = exp_golden(x_nan)
    if torch.isnan(y_nan).all():
        print(f"  exp(nan) = nan ... ✓ PASS")
    else:
        print(f"  exp(nan) = nan ... ✗ FAIL")
        all_passed = False

    # 4. 数值稳定性检查
    print("\n[数值稳定性检查]")

    # 大正值输入
    x_large = torch.tensor([100.0])
    y_large = exp_golden(x_large)
    if torch.isfinite(y_large):
        print(f"  exp(100) = {y_large.item():.2e} ... ✓ PASS (finite)")
    else:
        print(f"  exp(100) = inf ... ⚠ WARNING (overflow expected)")

    # 大负值输入
    x_neg_large = torch.tensor([-100.0])
    y_neg_large = exp_golden(x_neg_large)
    if torch.isfinite(y_neg_large) and y_neg_large.item() >= 0:
        print(f"  exp(-100) = {y_neg_large.item():.2e} ... ✓ PASS (underflow to 0)")
    else:
        print(f"  exp(-100) ... ✗ FAIL")
        all_passed = False

    # 5. 数学属性检查
    print("\n[数学属性检查]")

    # 单调性: exp 是单调递增函数
    x_mono = torch.tensor([-1.0, 0.0, 1.0])
    y_mono = exp_golden(x_mono)
    if torch.all(y_mono[1:] > y_mono[:-1]):
        print(f"  单调递增性 ... ✓ PASS")
    else:
        print(f"  单调递增性 ... ✗ FAIL")
        all_passed = False

    # 正定性: exp(x) > 0 对所有 x
    x_range = torch.linspace(-10, 10, 100)
    y_range = exp_golden(x_range)
    if torch.all(y_range > 0):
        print(f"  正定性 (exp(x) > 0) ... ✓ PASS")
    else:
        print(f"  正定性 (exp(x) > 0) ... ✗ FAIL")
        all_passed = False

    # 6. API 对比
    print("\n[API 对比]")

    x_api = torch.randn(100, 100)
    y_golden = exp_golden(x_api)
    y_torch = torch.exp(x_api)

    if torch.allclose(y_golden, y_torch):
        print(f"  与 torch.exp 对比 ... ✓ PASS")
    else:
        print(f"  与 torch.exp 对比 ... ✗ FAIL")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过")
    else:
        print("❌ 部分验证失败")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _validate()
