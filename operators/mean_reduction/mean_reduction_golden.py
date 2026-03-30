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
mean_reduction Golden 参考实现

公式: y = mean(x, dim) = sum(x_i) / N
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API)

功能描述:
    沿指定轴计算张量均值的归约操作，支持动态轴和 keepdim 参数。

参数说明:
    x: 输入张量，任意 shape
    dim: 归约轴（整数）
    keepdim: 是否保持归约后的维度（默认 False）

返回值:
    归约后的均值张量
"""

import torch
from typing import Optional, Union


# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────

def mean_reduction_golden(
    x: torch.Tensor,
    dim: int,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    mean_reduction 参考实现 (PyTorch)

    沿指定轴计算张量均值。直接使用 PyTorch 内置 torch.mean API。

    Args:
        x: 输入 tensor，任意 shape。
           支持数据类型: float32, float16, bfloat16
        dim: 归约轴索引（整数）。
             支持负索引（如 -1 表示最后一维）。
        keepdim: 是否保持归约后的维度。
                 - False (默认): 移除归约轴
                 - True: 归约轴维度变为 1

    Returns:
        归约后的均值 tensor。
        - keepdim=False: shape 移除 dim 维度
        - keepdim=True: shape 在 dim 维度上为 1

    公式:
        y = mean(x, dim) = sum(x_i) / N
        其中 N 是 dim 轴上的元素数量

    Examples:
        >>> x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> mean_reduction_golden(x, dim=1, keepdim=False)
        tensor([2., 5.])
        >>> mean_reduction_golden(x, dim=1, keepdim=True)
        tensor([[2.],
                [5.]])
    """
    return torch.mean(x, dim=dim, keepdim=keepdim)


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def get_output_shape(input_shape: tuple, dim: int, keepdim: bool) -> tuple:
    """
    计算输出 shape

    Args:
        input_shape: 输入张量的 shape
        dim: 归约轴
        keepdim: 是否保持维度

    Returns:
        输出张量的 shape
    """
    shape_list = list(input_shape)
    # 处理负索引
    if dim < 0:
        dim = len(shape_list) + dim

    if keepdim:
        shape_list[dim] = 1
    else:
        shape_list.pop(dim)

    return tuple(shape_list)


# ─────────────────────────────────────────────
# 自动生成的验证代码
# ─────────────────────────────────────────────

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("mean_reduction_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # 1. 典型 case 验证（来自 spec.md §12 典型配置）
    print("\n[典型 case 验证]")

    # 性能_P0: dim=1, keepdim=False, [b, s, 4096] -> [b, 4096]
    print("  性能_P0: dim=1, keepdim=False, [2, 512, 4096] -> [2, 4096] ... ", end="")
    try:
        x = torch.randn(2, 512, 4096, dtype=torch.float32)
        out = mean_reduction_golden(x, dim=1, keepdim=False)
        expected_shape = (2, 4096)
        if out.shape == expected_shape:
            # 验证计算正确性
            ref = torch.mean(x, dim=1, keepdim=False)
            if torch.allclose(out, ref, atol=1e-6, rtol=1e-6):
                print("✓ PASS")
            else:
                print("✗ FAIL (值不匹配)")
                all_passed = False
        else:
            print(f"✗ FAIL (shape 不匹配: 期望 {expected_shape}, 实际 {out.shape})")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    # 功能_P0: dim=1, keepdim=True, [b, s, 4096] -> [b, 1, 4096]
    print("  功能_P0: dim=1, keepdim=True, [2, 512, 4096] -> [2, 1, 4096] ... ", end="")
    try:
        x = torch.randn(2, 512, 4096, dtype=torch.float32)
        out = mean_reduction_golden(x, dim=1, keepdim=True)
        expected_shape = (2, 1, 4096)
        if out.shape == expected_shape:
            ref = torch.mean(x, dim=1, keepdim=True)
            if torch.allclose(out, ref, atol=1e-6, rtol=1e-6):
                print("✓ PASS")
            else:
                print("✗ FAIL (值不匹配)")
                all_passed = False
        else:
            print(f"✗ FAIL (shape 不匹配: 期望 {expected_shape}, 实际 {out.shape})")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    # 功能_P1: dim=0, keepdim=False, [b, s, 4096] -> [s, 4096]
    print("  功能_P1: dim=0, keepdim=False, [4, 256, 4096] -> [256, 4096] ... ", end="")
    try:
        x = torch.randn(4, 256, 4096, dtype=torch.float32)
        out = mean_reduction_golden(x, dim=0, keepdim=False)
        expected_shape = (256, 4096)
        if out.shape == expected_shape:
            ref = torch.mean(x, dim=0, keepdim=False)
            if torch.allclose(out, ref, atol=1e-6, rtol=1e-6):
                print("✓ PASS")
            else:
                print("✗ FAIL (值不匹配)")
                all_passed = False
        else:
            print(f"✗ FAIL (shape 不匹配: 期望 {expected_shape}, 实际 {out.shape})")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    # 动态_P0: 动态 shape 测试
    print("  动态_P0: dim=1, keepdim=False, 动态 shape ... ", end="")
    try:
        # 测试多组动态 shape
        for b, s, d in [(1, 64, 128), (8, 256, 512), (16, 1024, 256)]:
            x = torch.randn(b, s, d, dtype=torch.float32)
            out = mean_reduction_golden(x, dim=1, keepdim=False)
            expected_shape = (b, d)
            if out.shape != expected_shape:
                print(f"✗ FAIL (shape 不匹配: [{b},{s},{d}] -> 期望 {expected_shape}, 实际 {out.shape})")
                all_passed = False
                break
        else:
            print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    # 2. 泛化 case 验证（来自 spec.md §8 动态轴范围）
    print("\n[泛化 case 验证]")

    # 不同维度数测试
    print("  2D tensor [m, n] ... ", end="")
    try:
        x = torch.randn(16, 32, dtype=torch.float32)
        out = mean_reduction_golden(x, dim=1, keepdim=False)
        if out.shape == (16,):
            print("✓ PASS")
        else:
            print(f"✗ FAIL (shape: {out.shape})")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    print("  3D tensor [b, s, d] ... ", end="")
    try:
        x = torch.randn(4, 64, 128, dtype=torch.float32)
        out = mean_reduction_golden(x, dim=1, keepdim=False)
        if out.shape == (4, 128):
            print("✓ PASS")
        else:
            print(f"✗ FAIL (shape: {out.shape})")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    print("  4D tensor [b, s, n, d] ... ", end="")
    try:
        x = torch.randn(2, 16, 32, 64, dtype=torch.float32)
        out = mean_reduction_golden(x, dim=2, keepdim=False)
        if out.shape == (2, 16, 64):
            print("✓ PASS")
        else:
            print(f"✗ FAIL (shape: {out.shape})")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    # 负索引测试
    print("  负索引 dim=-1 ... ", end="")
    try:
        x = torch.randn(4, 8, 16, dtype=torch.float32)
        out = mean_reduction_golden(x, dim=-1, keepdim=False)
        if out.shape == (4, 8):
            print("✓ PASS")
        else:
            print(f"✗ FAIL (shape: {out.shape})")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    # 3. 值域检查（从公式推导）
    print("\n[值域检查]")

    # 零值输入
    print("  零值输入 ... ", end="")
    try:
        x = torch.zeros(4, 8, 16, dtype=torch.float32)
        out = mean_reduction_golden(x, dim=1, keepdim=False)
        if torch.allclose(out, torch.zeros_like(out)):
            print("✓ PASS")
        else:
            print("✗ FAIL (非零输出)")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    # 常量输入
    print("  常量输入 (全为 5.0) ... ", end="")
    try:
        x = torch.full((4, 8, 16), 5.0, dtype=torch.float32)
        out = mean_reduction_golden(x, dim=1, keepdim=False)
        if torch.allclose(out, torch.full((4, 16), 5.0, dtype=torch.float32)):
            print("✓ PASS")
        else:
            print("✗ FAIL (输出非 5.0)")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    # 4. 数值稳定性检查
    print("\n[数值稳定性检查]")

    # 大值输入
    print("  大值输入 (x=1e6) ... ", end="")
    try:
        x = torch.full((4, 8, 16), 1e6, dtype=torch.float32)
        out = mean_reduction_golden(x, dim=1, keepdim=False)
        if torch.allclose(out, torch.full((4, 16), 1e6, dtype=torch.float32)):
            print("✓ PASS")
        else:
            print("✗ FAIL (精度损失)")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    # 小值输入
    print("  小值输入 (x=1e-6) ... ", end="")
    try:
        x = torch.full((4, 8, 16), 1e-6, dtype=torch.float32)
        out = mean_reduction_golden(x, dim=1, keepdim=False)
        if torch.allclose(out, torch.full((4, 16), 1e-6, dtype=torch.float32), atol=1e-10):
            print("✓ PASS")
        else:
            print("✗ FAIL (精度损失)")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    # 5. API 对比（与 PyTorch 直接对比）
    print("\n[API 对比]")

    print("  与 torch.mean 对比 ... ", end="")
    try:
        x = torch.randn(4, 64, 128, dtype=torch.float32)
        out_golden = mean_reduction_golden(x, dim=1, keepdim=False)
        out_torch = torch.mean(x, dim=1, keepdim=False)
        if torch.allclose(out_golden, out_torch, atol=1e-7, rtol=1e-7):
            print("✓ PASS (完全一致)")
        else:
            max_diff = torch.max(torch.abs(out_golden - out_torch)).item()
            print(f"✗ FAIL (最大差异: {max_diff})")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    # 6. dtype 支持检查
    print("\n[dtype 支持检查]")

    print("  float16 支持 ... ", end="")
    try:
        x = torch.randn(4, 8, 16, dtype=torch.float16)
        out = mean_reduction_golden(x, dim=1, keepdim=False)
        if out.shape == (4, 16) and out.dtype == torch.float16:
            print("✓ PASS")
        else:
            print(f"✗ FAIL (dtype: {out.dtype})")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    print("  bfloat16 支持 ... ", end="")
    try:
        x = torch.randn(4, 8, 16, dtype=torch.bfloat16)
        out = mean_reduction_golden(x, dim=1, keepdim=False)
        if out.shape == (4, 16) and out.dtype == torch.bfloat16:
            print("✓ PASS")
        else:
            print(f"✗ FAIL (dtype: {out.dtype})")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL (异常: {e})")
        all_passed = False

    # 结果汇总
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过")
    else:
        print("❌ 存在验证失败项")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _validate()
