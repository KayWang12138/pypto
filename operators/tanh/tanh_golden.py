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
Tanh Golden 参考实现

公式: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
置信度: (使用 PyTorch 内置 API torch.tanh)

支持:
  - 输入 Shape: 2-4 维 [m, n] 或 [b, m, n] 或 [b, s, m, n]
  - 数据类型: float16, float32, bfloat16
  - 输出与输入 shape 和 dtype 完全相同
  - 输出值域: [-1, 1]
"""

import torch
from typing import Tuple, List


# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────

def tanh_golden(x: torch.Tensor) -> torch.Tensor:
    """Tanh 参考实现 (PyTorch)

    对输入 tensor 的每个元素执行双曲正切运算：
    output_i = tanh(input_i) = (e^x - e^(-x)) / (e^x + e^(-x))

    将输入映射到 [-1, 1] 区间。

    Args:
        x: 输入 tensor，支持 2-4 维
           - 2D: [m, n]
           - 3D: [b, m, n]
           - 4D: [b, s, m, n]
           支持 dtype: float16, float32, bfloat16

    Returns:
        输出 tensor，shape 和 dtype 与输入相同
        输出值域为 [-1, 1]

    公式:
        tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    边界条件:
        - x = 0: tanh(0) = 0
        - x -> +inf: tanh(x) -> 1
        - x -> -inf: tanh(x) -> -1
        - NaN/Inf: 按 IEEE 754 标准传播

    约束:
        - 不支持空 Tensor
        - 不支持 nan/inf 输入
        - Shape Size <= INT32_MAX
    """
    return torch.tanh(x)


# ==================== 自动生成的验证代码 ====================

def _validate_shape(shape: Tuple[int, ...], dtype: torch.dtype, name: str) -> bool:
    """验证单个 shape 配置"""
    try:
        x = torch.randn(shape, dtype=dtype)
        out = tanh_golden(x)

        # 1. 形状一致性检查
        if out.shape != x.shape:
            print(f"  {name}: FAIL - shape 不匹配 (input: {x.shape}, output: {out.shape})")
            return False

        # 2. dtype 一致性检查
        if out.dtype != x.dtype:
            print(f"  {name}: FAIL - dtype 不匹配 (input: {x.dtype}, output: {out.dtype})")
            return False

        # 3. 与 PyTorch torch.tanh 对比
        expected = torch.tanh(x)
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
    """值域检查：tanh 输出必须在 [-1, 1]"""
    print("\n[值域检查]")

    # 检查输出在 [-1, 1] 范围内
    x = torch.randn(100, 100)
    out = tanh_golden(x)
    if (out >= -1).all() and (out <= 1).all():
        print(f"  输出范围 [-1, 1] 检查 ... PASS")
    else:
        print(f"  输出范围 [-1, 1] 检查 ... FAIL (存在越界值)")

    # 边界值检查
    x_boundary = torch.tensor([-1e6, -10.0, -1.0, 0.0, 1.0, 10.0, 1e6])
    out_boundary = tanh_golden(x_boundary)
    # tanh(-inf) -> -1, tanh(+inf) -> 1, tanh(0) = 0
    expected_boundary = torch.tensor([-1.0, -1.0, -0.7616, 0.0, 0.7616, 1.0, 1.0])
    if torch.allclose(out_boundary, expected_boundary, atol=1e-3):
        print(f"  边界值检查 ... PASS")
    else:
        print(f"  边界值检查 ... FAIL (实际: {out_boundary})")


def _validate_numerical_stability():
    """数值稳定性检查"""
    print("\n[数值稳定性检查]")

    # 大值输入
    x_large = torch.tensor([1e10, -1e10, 1e20, -1e20])
    out_large = tanh_golden(x_large)
    if not torch.isnan(out_large).any():
        print(f"  大值输入稳定性 ... PASS (无 NaN)")
    else:
        print(f"  大值输入稳定性 ... FAIL (存在 NaN)")

    # 检查大值趋近于 +/-1
    if torch.allclose(out_large[0], torch.tensor(1.0)) and torch.allclose(out_large[1], torch.tensor(-1.0)):
        print(f"  大值趋近边界 (+/-1) ... PASS")
    else:
        print(f"  大值趋近边界 ... FAIL (实际: {out_large[:2]})")

    # 零值处理
    x_zero = torch.zeros(10, 10)
    out_zero = tanh_golden(x_zero)
    if torch.allclose(out_zero, torch.zeros_like(out_zero)):
        print(f"  零值处理 ... PASS")
    else:
        print(f"  零值处理 ... FAIL")


def _validate_math_properties():
    """数学属性检查"""
    print("\n[数学属性检查]")

    # 奇函数性质：tanh(-x) = -tanh(x)
    x = torch.randn(100)
    out_pos = tanh_golden(x)
    out_neg = tanh_golden(-x)
    if torch.allclose(out_neg, -out_pos, rtol=1e-5, atol=1e-5):
        print(f"  奇函数性质 tanh(-x) = -tanh(x) ... PASS")
    else:
        print(f"  奇函数性质 ... FAIL")

    # 单调性：tanh 是单调递增函数
    x1 = torch.randn(100)
    x2 = x1 + 0.1  # x2 > x1
    out1 = tanh_golden(x1)
    out2 = tanh_golden(x2)
    if (out2 >= out1).all():
        print(f"  单调性检查 ... PASS")
    else:
        print(f"  单调性检查 ... FAIL")

    # tanh(x) = 2*sigmoid(2x) - 1
    sigmoid_relation = 2 * torch.sigmoid(2 * x) - 1
    if torch.allclose(out_pos, sigmoid_relation, rtol=1e-4, atol=1e-4):
        print(f"  sigmoid 关系 tanh(x) = 2*sigmoid(2x)-1 ... PASS")
    else:
        max_diff = (out_pos - sigmoid_relation).abs().max().item()
        print(f"  sigmoid 关系 ... FAIL (max diff: {max_diff})")


def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("tanh_golden 验证报告")
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
    print("  tanh_golden 直接使用 torch.tanh")
    print("  与 PyTorch 官方实现完全一致 ... PASS")

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
