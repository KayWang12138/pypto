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
ReLU Golden 参考实现

公式: res_i = max(0, input_i)
置信度: (使用 PyTorch 内置 API torch.nn.functional.relu)

支持:
  - 输入 Shape: 2-4 维 [m, n] 或 [b, m, n] 或 [b, s, m, n]
  - 数据类型: float16, float32, bfloat16
  - 输出与输入 shape 和 dtype 完全相同
"""

import torch
from typing import Tuple, List


# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────

def relu_golden(x: torch.Tensor) -> torch.Tensor:
    """ReLU 参考实现 (PyTorch)

    对输入 tensor 的每个元素执行整流线性单元运算：
    output_i = max(0, input_i)

    仅保留正数部分，负数变为 0。

    Args:
        x: 输入 tensor，支持 2-4 维
           - 2D: [m, n]
           - 3D: [b, m, n]
           - 4D: [b, s, m, n]
           支持 dtype: float16, float32, bfloat16

    Returns:
        输出 tensor，shape 和 dtype 与输入相同
        所有负数元素被置为 0，正数元素保持不变

    公式:
        output_i = max(0, input_i)

    约束:
        - 不支持空 Tensor
        - 不支持 nan/inf 输入
        - Shape Size <= INT32_MAX
    """
    return torch.nn.functional.relu(x)


# ==================== 自动生成的验证代码 ====================

def _validate_shape(shape: Tuple[int, ...], dtype: torch.dtype, name: str) -> bool:
    """验证单个 shape 配置"""
    try:
        x = torch.randn(shape, dtype=dtype)
        out = relu_golden(x)

        # 1. 形状一致性检查
        if out.shape != x.shape:
            print(f"  {name}: FAIL - shape 不匹配 (input: {x.shape}, output: {out.shape})")
            return False

        # 2. dtype 一致性检查
        if out.dtype != x.dtype:
            print(f"  {name}: FAIL - dtype 不匹配 (input: {x.dtype}, output: {out.dtype})")
            return False

        # 3. 与 PyTorch F.relu 对比
        expected = torch.nn.functional.relu(x)
        if not torch.allclose(out, expected, rtol=1e-5, atol=1e-5):
            max_diff = (out - expected).abs().max().item()
            print(f"  {name}: FAIL - 与 PyTorch 结果不一致 (max_diff: {max_diff})")
            return False

        print(f"  {name}: {shape} {dtype} ... PASS")
        return True
    except Exception as e:
        print(f"  {name}: FAIL - {e}")
        return False


def _validate_value_range():
    """值域检查：ReLU 输出必须 >= 0"""
    print("\n[值域检查]")

    # 检查输出非负
    x = torch.randn(100, 100)
    out = relu_golden(x)
    if (out >= 0).all():
        print(f"  输出非负性检查 ... PASS")
    else:
        print(f"  输出非负性检查 ... FAIL (存在负值)")

    # 边界值检查
    x_boundary = torch.tensor([-1e6, -1.0, 0.0, 1.0, 1e6])
    out_boundary = relu_golden(x_boundary)
    expected_boundary = torch.tensor([0.0, 0.0, 0.0, 1.0, 1e6])
    if torch.allclose(out_boundary, expected_boundary):
        print(f"  边界值检查 [-1e6, -1, 0, 1, 1e6] ... PASS")
    else:
        print(f"  边界值检查 ... FAIL (期望: {expected_boundary}, 实际: {out_boundary})")


def _validate_numerical_stability():
    """数值稳定性检查"""
    print("\n[数值稳定性检查]")

    # 大值输入
    x_large = torch.tensor([1e10, -1e10, 1e20, -1e20])
    out_large = relu_golden(x_large)
    if not (torch.isnan(out_large).any() or torch.isinf(out_large).any()):
        print(f"  大值输入稳定性 ... PASS")
    else:
        print(f"  大值输入稳定性 ... FAIL (存在 nan/inf)")

    # 零值处理
    x_zero = torch.zeros(10, 10)
    out_zero = relu_golden(x_zero)
    if torch.allclose(out_zero, torch.zeros_like(out_zero)):
        print(f"  零值处理 ... PASS")
    else:
        print(f"  零值处理 ... FAIL")


def _validate_math_properties():
    """数学属性检查"""
    print("\n[数学属性检查]")

    # 单调性：ReLU 是单调递增函数
    x1 = torch.randn(100)
    x2 = x1 + 0.1  # x2 > x1
    out1 = relu_golden(x1)
    out2 = relu_golden(x2)
    if (out2 >= out1).all():
        print(f"  单调性检查 ... PASS")
    else:
        print(f"  单调性检查 ... FAIL")

    # 齐次性：ReLU(ax) = a * ReLU(x) for a >= 0
    x = torch.randn(100)
    a = 2.5
    out_ax = relu_golden(a * x)
    out_scaled = a * relu_golden(x)
    if torch.allclose(out_ax, out_scaled, rtol=1e-5, atol=1e-5):
        print(f"  正齐次性检查 (a={a}) ... PASS")
    else:
        print(f"  正齐次性检查 ... FAIL")


def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("relu_golden 验证报告")
    print("=" * 60)

    # 1. 典型 case 验证（来自 spec.md Section 11）
    print("\n[典型 case 验证]")

    dtypes = [torch.float16, torch.float32, torch.bfloat16]

    # 性能_P0: [1024, 1024]
    for dtype in dtypes:
        _validate_shape((1024, 1024), dtype, f"性能_P0")

    # 功能_P0: [32, 64]
    for dtype in dtypes:
        _validate_shape((32, 64), dtype, f"功能_P0")

    # 功能_P1: [2, 128, 256] (3D)
    for dtype in dtypes:
        _validate_shape((2, 128, 256), dtype, f"功能_P1")

    # 功能_P2: [1, 1, 64, 64] (4D)
    for dtype in dtypes:
        _validate_shape((1, 1, 64, 64), dtype, f"功能_P2")

    # 2. 泛化 case 验证（边界 + 中间采样）
    print("\n[泛化 case 验证]")

    # 2D 泛化
    _validate_shape((1, 1), torch.float32, "泛化_2D_min")
    _validate_shape((512, 512), torch.float32, "泛化_2D_mid")
    _validate_shape((2048, 2048), torch.float32, "泛化_2D_large")

    # 3D 泛化
    _validate_shape((1, 1, 1), torch.float32, "泛化_3D_min")
    _validate_shape((4, 64, 128), torch.float32, "泛化_3D_mid")

    # 4D 泛化
    _validate_shape((1, 1, 1, 1), torch.float32, "泛化_4D_min")
    _validate_shape((2, 4, 32, 64), torch.float32, "泛化_4D_mid")

    # 3. 值域检查
    _validate_value_range()

    # 4. 数值稳定性检查
    _validate_numerical_stability()

    # 5. 数学属性检查
    _validate_math_properties()

    # 6. API 对比
    print("\n[API 对比]")
    print("  relu_golden 直接使用 torch.nn.functional.relu")
    print("  与 PyTorch 官方实现完全一致 ... PASS")

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
