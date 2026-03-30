#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO Mish Golden Reference Implementation.

公式: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

参考:
- PyTorch: torch.nn.functional.mish (PyTorch 1.9+)
- 论文: Mish: A Self Regularized Non-Monotonic Activation Function (Diganta Misra, 2019)
"""

import torch
import torch.nn.functional as F
from typing import Union, Tuple

# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────

def mish_golden(x: torch.Tensor) -> torch.Tensor:
    """Mish 激活函数的 PyTorch 参考实现。

    公式: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    特性:
    - 平滑的非单调激活函数
    - 输出范围: 负无穷到正无穷（无界）
    - mish(0) = 0
    - x -> +inf: mish(x) -> x
    - x -> -inf: mish(x) -> 0

    Args:
        x: 输入 tensor，支持 1D-4D，dtype=float32

    Returns:
        输出 tensor，shape 和 dtype 与输入相同
    """
    # 方法 1: 使用 PyTorch 内置 F.mish (PyTorch 1.9+)
    # return F.mish(x)

    # 方法 2: 手动实现，更通用
    # softplus(x) = ln(1 + exp(x))
    softplus_x = F.softplus(x)
    # mish(x) = x * tanh(softplus(x))
    return x * torch.tanh(softplus_x)


def mish_golden_manual(x: torch.Tensor) -> torch.Tensor:
    """Mish 激活函数的手动实现（无 torch.nn.functional.mish）。

    完全展开公式，用于验证和理解计算过程。
    """
    # softplus(x) = ln(1 + exp(x))
    exp_x = torch.exp(x)
    one_plus_exp = 1.0 + exp_x
    softplus_x = torch.log(one_plus_exp)

    # tanh(softplus) = (exp(2*softplus) - 1) / (exp(2*softplus) + 1)
    # 但直接使用 torch.tanh 更稳定
    tanh_softplus = torch.tanh(softplus_x)

    # mish(x) = x * tanh(softplus(x))
    return x * tanh_softplus


# ─────────────────────────────────────────────
# 验证辅助函数
# ─────────────────────────────────────────────

def verify_shape_consistency():
    """验证不同 shape 输入的 shape 一致性。"""
    test_cases = [
        ("1D", torch.randn(1024)),
        ("2D", torch.randn(128, 1024)),
        ("3D", torch.randn(2, 128, 1024)),
        ("4D", torch.randn(2, 4, 128, 1024)),
    ]

    print("[Shape 一致性验证]")
    for name, x in test_cases:
        y = mish_golden(x)
        status = "PASS" if y.shape == x.shape else "FAIL"
        print(f"  {name}: input={tuple(x.shape)}, output={tuple(y.shape)} ... {status}")
        assert y.shape == x.shape, f"Shape mismatch: {x.shape} vs {y.shape}"


def verify_numerical_properties():
    """验证数值属性。"""
    print("\n[数值属性验证]")

    # 特殊点验证
    print("  特殊点:")
    x_zero = torch.tensor([0.0])
    y_zero = mish_golden(x_zero)
    print(f"    mish(0) = {y_zero.item():.6f} (期望: 0.0)")
    assert abs(y_zero.item()) < 1e-6, "mish(0) should be 0"

    # 正极大值
    x_large = torch.tensor([100.0])
    y_large = mish_golden(x_large)
    print(f"    mish(100) = {y_large.item():.6f} (期望: ≈100)")
    assert abs(y_large.item() - 100.0) < 1.0, "mish(100) should be close to 100"

    # 负极大值
    x_neg_large = torch.tensor([-100.0])
    y_neg_large = mish_golden(x_neg_large)
    print(f"    mish(-100) = {y_neg_large.item():.6f} (期望: ≈0)")
    assert abs(y_neg_large.item()) < 1e-6, "mish(-100) should be close to 0"

    # 单调性检查（mish 在大多数情况下单调递增，但在极小负值区域有轻微非单调性）
    # 参考: https://github.com/digantamisra98/Mish/issues/24
    # mish 不是完全单调的，跳过单调性检查
    print("  单调性:")
    x_mono = torch.linspace(-5, 5, 100)
    y_mono = mish_golden(x_mono)
    diff = y_mono[1:] - y_mono[:-1]
    min_diff = diff.min().item()
    max_diff = diff.max().item()
    print(f"    mish 在 [-5, 5] 上的 diff 范围: [{min_diff:.6f}, {max_diff:.6f}]")
    # mish 在正区间是单调的
    x_pos = torch.linspace(0.1, 10, 100)
    y_pos = mish_golden(x_pos)
    diff_pos = y_pos[1:] - y_pos[:-1]
    is_monotonic_pos = (diff_pos > 0).all().item()
    print(f"    mish 在 [0.1, 10] 上单调递增: {is_monotonic_pos}")
    assert is_monotonic_pos, "mish should be monotonically increasing for positive inputs"

    # 非负输出（对于正输入）
    x_pos = torch.rand(100) * 10  # [0, 10]
    y_pos = mish_golden(x_pos)
    all_non_neg = (y_pos >= 0).all().item()
    print(f"    mish(x) >= 0 for x >= 0: {all_non_neg}")
    assert all_non_neg, "mish(x) should be non-negative for x >= 0"


def verify_numerical_stability():
    """验证数值稳定性。"""
    print("\n[数值稳定性验证]")

    # 大值输入
    x_large = torch.tensor([50.0, 100.0, 500.0])
    y_large = mish_golden(x_large)
    has_nan_large = torch.isnan(y_large).any().item()
    has_inf_large = torch.isinf(y_large).any().item()
    print(f"  大值输入 (50, 100, 500): NaN={has_nan_large}, Inf={has_inf_large}")
    assert not has_nan_large, "Should not produce NaN for large inputs"

    # 负大值输入
    x_neg_large = torch.tensor([-50.0, -100.0, -500.0])
    y_neg_large = mish_golden(x_neg_large)
    has_nan_neg = torch.isnan(y_neg_large).any().item()
    print(f"  负大值输入 (-50, -100, -500): NaN={has_nan_neg}")
    assert not has_nan_neg, "Should not produce NaN for negative large inputs"


def verify_against_pytorch_builtin():
    """与 PyTorch 内置 mish 对比验证。"""
    print("\n[PyTorch 内置 API 对比]")

    x = torch.randn(100, 100)

    # 使用我们的实现
    y_ours = mish_golden(x)

    # 使用 PyTorch 内置 (1.9+)
    try:
        y_torch = F.mish(x)
        max_diff = (y_ours - y_torch).abs().max().item()
        print(f"  与 torch.nn.functional.mish 最大差异: {max_diff:.2e}")
        assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance"
    except AttributeError:
        print("  PyTorch 版本不支持 F.mish，跳过对比")


def verify_typical_cases():
    """验证 spec.md 中定义的典型配置。"""
    print("\n[典型配置验证]")

    cases = [
        ("功能_1D", (1024,)),
        ("功能_2D", (128, 1024)),
        ("功能_3D", (2, 128, 1024)),
        ("功能_4D", (2, 4, 128, 1024)),
        ("性能_2D", (4096, 4096)),
    ]

    for name, shape in cases:
        x = torch.randn(shape, dtype=torch.float32)
        y = mish_golden(x)
        status = "PASS" if y.shape == x.shape and not torch.isnan(y).any() else "FAIL"
        print(f"  {name}: shape={shape}, output_shape={tuple(y.shape)} ... {status}")
        assert y.shape == x.shape, f"Shape mismatch for {name}"
        assert not torch.isnan(y).any(), f"NaN in output for {name}"


def verify_boundary_conditions():
    """验证边界条件。"""
    print("\n[边界条件验证]")

    # 零值
    x = torch.zeros(10, 10)
    y = mish_golden(x)
    print(f"  零值输入: mish(0) = {y[0, 0].item():.6f}")
    assert abs(y[0, 0].item()) < 1e-6, "mish(0) should be 0"

    # 小值
    x_small = torch.randn(10, 10) * 0.001
    y_small = mish_golden(x_small)
    has_nan = torch.isnan(y_small).any().item()
    print(f"  小值输入 (x~0.001): NaN={has_nan}")
    assert not has_nan

    # 混合正负值
    x_mixed = torch.randn(100, 100)
    y_mixed = mish_golden(x_mixed)
    has_nan_mixed = torch.isnan(y_mixed).any().item()
    print(f"  混合正负值: NaN={has_nan_mixed}")
    assert not has_nan_mixed


def _validate():
    """运行所有验证并生成报告。"""
    print("=" * 60)
    print("mish_golden 验证报告")
    print("=" * 60)

    try:
        verify_shape_consistency()
        verify_numerical_properties()
        verify_numerical_stability()
        verify_against_pytorch_builtin()
        verify_typical_cases()
        verify_boundary_conditions()

        print("\n" + "=" * 60)
        print("所有验证通过")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n验证失败: {e}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = _validate()
    exit(0 if success else 1)
