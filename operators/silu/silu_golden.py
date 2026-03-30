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
silu Golden 参考实现

公式: y = x * sigmoid(x) = x / (1 + exp(-x))
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API)

SiLU (Sigmoid Linear Unit) 激活函数，也称为 Swish。
在 LLM 中广泛使用（如 LLaMA、Mistral）。

参考论文: "Searching for Activation Functions" (Ramachandran et al., 2017)
"""

import torch
from typing import Optional, List, Union


def silu_golden(x: torch.Tensor) -> torch.Tensor:
    """
    silu 参考实现 (PyTorch)

    Args:
        x: 输入 tensor，任意 shape
           支持 dtype: float16, float32, bfloat16

    Returns:
        输出 tensor，shape 和 dtype 与输入相同
        y = x * sigmoid(x)

    公式:
        silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

    边界条件:
        - x = 0: y = 0 * 0.5 = 0
        - x -> +inf: y -> +inf * 1 = +inf
        - x -> -inf: y -> -inf * 0 = 0
        - NaN/Inf: 按 IEEE 754 标准传播
    """
    return torch.nn.functional.silu(x)


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("silu_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # 1. 典型 case 验证（来自 spec.md §12）
    print("\n[典型 case 验证]")

    # 性能_P0: [1, 4096, 4096], float16
    try:
        x = torch.randn(1, 4096, 4096, dtype=torch.float16)
        y = silu_golden(x)
        assert y.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {y.shape}"
        assert y.dtype == x.dtype, f"Dtype mismatch: expected {x.dtype}, got {y.dtype}"
        print(f"  性能_P0: [1, 4096, 4096], float16 ... ✓ PASS")
    except Exception as e:
        print(f"  性能_P0: [1, 4096, 4096], float16 ... ✗ FAIL: {e}")
        all_passed = False

    # 功能_P0: [2, 1024, 512], float32
    try:
        x = torch.randn(2, 1024, 512, dtype=torch.float32)
        y = silu_golden(x)
        assert y.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {y.shape}"
        assert y.dtype == x.dtype, f"Dtype mismatch: expected {x.dtype}, got {y.dtype}"
        print(f"  功能_P0: [2, 1024, 512], float32 ... ✓ PASS")
    except Exception as e:
        print(f"  功能_P0: [2, 1024, 512], float32 ... ✗ FAIL: {e}")
        all_passed = False

    # 功能_P1: [4, 2048, 1024], bfloat16
    try:
        x = torch.randn(4, 2048, 1024, dtype=torch.bfloat16)
        y = silu_golden(x)
        assert y.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {y.shape}"
        assert y.dtype == x.dtype, f"Dtype mismatch: expected {x.dtype}, got {y.dtype}"
        print(f"  功能_P1: [4, 2048, 1024], bfloat16 ... ✓ PASS")
    except Exception as e:
        print(f"  功能_P1: [4, 2048, 1024], bfloat16 ... ✗ FAIL: {e}")
        all_passed = False

    # 边界_P0: [1, 1, 1], float32
    try:
        x = torch.randn(1, 1, 1, dtype=torch.float32)
        y = silu_golden(x)
        assert y.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {y.shape}"
        assert y.dtype == x.dtype, f"Dtype mismatch: expected {x.dtype}, got {y.dtype}"
        print(f"  边界_P0: [1, 1, 1], float32 ... ✓ PASS")
    except Exception as e:
        print(f"  边界_P0: [1, 1, 1], float32 ... ✗ FAIL: {e}")
        all_passed = False

    # 2. 泛化 case 验证（来自 spec.md §8 动态轴范围）
    print("\n[泛化 case 验证]")

    # 动态轴范围: [1, INT32_MAX]，采样 [1, 128, 4096]
    test_shapes = [
        ([1, 1, 1, 1], "最小边界"),
        ([32, 128, 64, 128], "中间值"),
        ([128, 1024, 512, 256], "较大值"),
    ]

    for shape, desc in test_shapes:
        try:
            x = torch.randn(*shape, dtype=torch.float32)
            y = silu_golden(x)
            assert y.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {y.shape}"
            assert y.dtype == x.dtype, f"Dtype mismatch: expected {x.dtype}, got {y.dtype}"
            print(f"  {desc}: {shape} ... ✓ PASS")
        except Exception as e:
            print(f"  {desc}: {shape} ... ✗ FAIL: {e}")
            all_passed = False

    # 3. 值域检查（从公式推导）
    print("\n[值域检查]")

    # silu 的值域特性：
    # - sigmoid 输出在 (0, 1)
    # - silu(x) = x * sigmoid(x)，当 x > 0 时，silu(x) > 0
    # - 当 x -> +inf 时，silu(x) -> +inf
    # - 当 x -> -inf 时，silu(x) -> 0
    try:
        # 测试正值
        x_pos = torch.tensor([1.0, 2.0, 5.0, 10.0])
        y_pos = silu_golden(x_pos)
        assert (y_pos > 0).all(), "silu(x) should be positive for x > 0"
        print(f"  正值输入 (x > 0) ... ✓ PASS")
    except Exception as e:
        print(f"  正值输入 (x > 0) ... ✗ FAIL: {e}")
        all_passed = False

    # 4. 特殊点验证
    print("\n[特殊点验证]")

    # x = 0: silu(0) = 0 * 0.5 = 0
    try:
        x_zero = torch.tensor([0.0])
        y_zero = silu_golden(x_zero)
        expected = torch.tensor([0.0])
        assert torch.allclose(y_zero, expected, atol=1e-6), f"silu(0) should be 0, got {y_zero.item()}"
        print(f"  零值 (x=0) ... ✓ PASS (silu(0) = {y_zero.item():.6f})")
    except Exception as e:
        print(f"  零值 (x=0) ... ✗ FAIL: {e}")
        all_passed = False

    # x = 1: silu(1) ≈ 1 * 0.7310586 = 0.7310586
    try:
        x_one = torch.tensor([1.0])
        y_one = silu_golden(x_one)
        expected = x_one * torch.sigmoid(x_one)
        assert torch.allclose(y_one, expected, atol=1e-6), f"silu(1) mismatch"
        print(f"  单位值 (x=1) ... ✓ PASS (silu(1) = {y_one.item():.6f})")
    except Exception as e:
        print(f"  单位值 (x=1) ... ✗ FAIL: {e}")
        all_passed = False

    # 5. 数值稳定性检查
    print("\n[数值稳定性检查]")

    # 大值输入
    try:
        x_large = torch.tensor([100.0, -100.0, 50.0, -50.0])
        y_large = silu_golden(x_large)
        # 检查无 NaN
        assert not torch.isnan(y_large).any(), "NaN detected in large value inputs"
        # 检查边界行为
        # x=100: silu(100) ≈ 100 * 1 = 100
        # x=-100: silu(-100) ≈ -100 * 0 = 0
        assert y_large[0] > 0, "silu(100) should be positive"
        assert torch.abs(y_large[1]) < 1e-6, "silu(-100) should be close to 0"
        print(f"  大值输入 (x=±100) ... ✓ PASS")
    except Exception as e:
        print(f"  大值输入 (x=±100) ... ✗ FAIL: {e}")
        all_passed = False

    # 6. API 对比
    print("\n[API 对比]")

    # 与手动展开公式对比
    try:
        x_test = torch.randn(32, 128, dtype=torch.float32)
        y_api = silu_golden(x_test)

        # 手动展开: x * sigmoid(x)
        sigmoid_x = torch.sigmoid(x_test)
        y_manual = x_test * sigmoid_x

        assert torch.allclose(y_api, y_manual, atol=1e-6), "API vs manual formula mismatch"
        max_diff = (y_api - y_manual).abs().max().item()
        print(f"  API vs 手动展开 ... ✓ PASS (max diff: {max_diff:.2e})")
    except Exception as e:
        print(f"  API vs 手动展开 ... ✗ FAIL: {e}")
        all_passed = False

    # 7. 多 dtype 验证
    print("\n[多 dtype 验证]")

    dtypes = [torch.float16, torch.float32, torch.bfloat16]
    for dtype in dtypes:
        try:
            x = torch.randn(16, 64, dtype=dtype)
            y = silu_golden(x)
            assert y.shape == x.shape, f"Shape mismatch for {dtype}"
            assert y.dtype == x.dtype, f"Dtype mismatch for {dtype}"
            print(f"  {str(dtype):15s} ... ✓ PASS")
        except Exception as e:
            print(f"  {str(dtype):15s} ... ✗ FAIL: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过")
    else:
        print("⚠️ 部分验证失败，请检查上述错误")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
