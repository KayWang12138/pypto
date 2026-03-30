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
matmul Golden 参考实现

公式: C = A @ B, C_{ij} = sum_k(A_{ik} * B_{kj})
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API torch.matmul)

功能说明:
    - 支持 2D 矩阵乘法: [M, K] @ [K, N] -> [M, N]
    - 支持 3D/4D batch matmul: [B, M, K] @ [B, K, N] -> [B, M, N]
    - 支持 batch 维度广播: [B1, 1, M, K] @ [1, B2, K, N] -> [B1, B2, M, N]
    - 支持动态轴: batch, M, N, K 可动态变化
"""

import torch
from typing import Optional


def matmul_golden(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    matmul 参考实现 (PyTorch)

    实现批量矩阵乘法，支持 batch 维度广播。
    最后两个维度执行标准矩阵乘法，前面的维度作为 batch 维度并支持广播。

    Args:
        a: 左操作数，shape [..., M, K], dtype: float32/float16
           最后两维为 MxK 矩阵，前面的维度为 batch 维度
        b: 右操作数，shape [..., K, N], dtype: float32/float16
           最后两维为 KxN 矩阵，前面的维度为 batch 维度

    Returns:
        输出矩阵 C，shape [..., M, N], dtype 与输入一致
        最后两维为 MxN 矩阵，batch 维度由输入广播决定

    公式:
        C = A @ B
        C[..., i, j] = sum_k(A[..., i, k] * B[..., k, j])

    约束:
        - A 的 K 维度（倒数第一维）必须等于 B 的 K 维度（倒数第二维）
        - batch 维度支持广播（维度为 1 的轴可扩展）

    Examples:
        >>> # 2D 矩阵乘法
        >>> a = torch.randn(128, 512)
        >>> b = torch.randn(512, 256)
        >>> c = matmul_golden(a, b)  # [128, 256]

        >>> # 3D batch matmul
        >>> a = torch.randn(8, 128, 512)
        >>> b = torch.randn(8, 512, 256)
        >>> c = matmul_golden(a, b)  # [8, 128, 256]

        >>> # 4D batch matmul with broadcast
        >>> a = torch.randn(2, 1, 128, 512)
        >>> b = torch.randn(1, 4, 512, 256)
        >>> c = matmul_golden(a, b)  # [2, 4, 128, 256]
    """
    # 使用 PyTorch 的 matmul，自动处理广播和不同维度
    # torch.matmul 支持:
    #   - 2D @ 2D -> 2D
    #   - 3D+ @ 3D+ -> 3D+ (batch matmul)
    #   - batch 维度广播
    return torch.matmul(a, b)


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("matmul_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # 1. 典型 case 验证（来自 spec.md §12）
    print("\n[典型 case 验证]")

    # 性能_P0: 大规模 batch matmul
    try:
        a = torch.randn(1024, 4096, 4096, dtype=torch.float32)
        b = torch.randn(1024, 4096, 4096, dtype=torch.float32)
        c = matmul_golden(a, b)
        expected_shape = (1024, 4096, 4096)
        assert c.shape == expected_shape, f"Shape mismatch: {c.shape} != {expected_shape}"
        print(f"  性能_P0: A:[1024,4096,4096], B:[1024,4096,4096] ... ✓ PASS")
    except Exception as e:
        print(f"  性能_P0: A:[1024,4096,4096], B:[1024,4096,4096] ... ✗ FAIL: {e}")
        all_passed = False

    # 功能_P0: 典型 Attention QK^T 场景
    try:
        a = torch.randn(1, 128, 64, 512, dtype=torch.float32)
        b = torch.randn(1, 128, 512, 256, dtype=torch.float32)
        c = matmul_golden(a, b)
        expected_shape = (1, 128, 64, 256)
        assert c.shape == expected_shape, f"Shape mismatch: {c.shape} != {expected_shape}"
        # 验证数值正确性
        c_ref = torch.matmul(a, b)
        assert torch.allclose(c, c_ref, rtol=1e-3, atol=1e-3), "Numerical mismatch"
        print(f"  功能_P0: A:[1,128,64,512], B:[1,128,512,256] ... ✓ PASS")
    except Exception as e:
        print(f"  功能_P0: A:[1,128,64,512], B:[1,128,512,256] ... ✗ FAIL: {e}")
        all_passed = False

    # 广播_P1: batch 广播场景
    try:
        a = torch.randn(8, 1, 1024, 4096, dtype=torch.float32)
        b = torch.randn(1, 4, 4096, 1024, dtype=torch.float32)
        c = matmul_golden(a, b)
        expected_shape = (8, 4, 1024, 1024)
        assert c.shape == expected_shape, f"Shape mismatch: {c.shape} != {expected_shape}"
        c_ref = torch.matmul(a, b)
        assert torch.allclose(c, c_ref, rtol=1e-3, atol=1e-3), "Numerical mismatch"
        print(f"  广播_P1: A:[8,1,1024,4096], B:[1,4,4096,1024] ... ✓ PASS")
    except Exception as e:
        print(f"  广播_P1: A:[8,1,1024,4096], B:[1,4,4096,1024] ... ✗ FAIL: {e}")
        all_passed = False

    # 2D_P0: 标准 2D 矩阵乘法
    try:
        a = torch.randn(4096, 4096, dtype=torch.float32)
        b = torch.randn(4096, 4096, dtype=torch.float32)
        c = matmul_golden(a, b)
        expected_shape = (4096, 4096)
        assert c.shape == expected_shape, f"Shape mismatch: {c.shape} != {expected_shape}"
        c_ref = torch.matmul(a, b)
        assert torch.allclose(c, c_ref, rtol=1e-3, atol=1e-3), "Numerical mismatch"
        print(f"  2D_P0: A:[4096,4096], B:[4096,4096] ... ✓ PASS")
    except Exception as e:
        print(f"  2D_P0: A:[4096,4096], B:[4096,4096] ... ✗ FAIL: {e}")
        all_passed = False

    # 2. 泛化 case 验证（来自 spec.md §8 动态轴范围）
    print("\n[泛化 case 验证]")

    # 小规模 2D
    try:
        a = torch.randn(16, 32, dtype=torch.float32)
        b = torch.randn(32, 64, dtype=torch.float32)
        c = matmul_golden(a, b)
        assert c.shape == (16, 64)
        c_ref = torch.matmul(a, b)
        assert torch.allclose(c, c_ref, rtol=1e-3, atol=1e-3)
        print(f"  小规模 2D: A:[16,32], B:[32,64] ... ✓ PASS")
    except Exception as e:
        print(f"  小规模 2D: A:[16,32], B:[32,64] ... ✗ FAIL: {e}")
        all_passed = False

    # 中等规模 3D
    try:
        a = torch.randn(32, 256, 512, dtype=torch.float32)
        b = torch.randn(32, 512, 256, dtype=torch.float32)
        c = matmul_golden(a, b)
        assert c.shape == (32, 256, 256)
        c_ref = torch.matmul(a, b)
        assert torch.allclose(c, c_ref, rtol=1e-3, atol=1e-3)
        print(f"  中等规模 3D: A:[32,256,512], B:[32,512,256] ... ✓ PASS")
    except Exception as e:
        print(f"  中等规模 3D: A:[32,256,512], B:[32,512,256] ... ✗ FAIL: {e}")
        all_passed = False

    # 4D 广播
    try:
        a = torch.randn(2, 1, 64, 128, dtype=torch.float32)
        b = torch.randn(1, 4, 128, 64, dtype=torch.float32)
        c = matmul_golden(a, b)
        assert c.shape == (2, 4, 64, 64)
        c_ref = torch.matmul(a, b)
        assert torch.allclose(c, c_ref, rtol=1e-3, atol=1e-3)
        print(f"  4D 广播: A:[2,1,64,128], B:[1,4,128,64] ... ✓ PASS")
    except Exception as e:
        print(f"  4D 广播: A:[2,1,64,128], B:[1,4,128,64] ... ✗ FAIL: {e}")
        all_passed = False

    # 3. dtype 验证
    print("\n[dtype 验证]")

    # float16
    try:
        a = torch.randn(64, 128, dtype=torch.float16)
        b = torch.randn(128, 64, dtype=torch.float16)
        c = matmul_golden(a, b)
        assert c.dtype == torch.float16
        c_ref = torch.matmul(a, b)
        assert torch.allclose(c, c_ref, rtol=1e-2, atol=1e-2)
        print(f"  float16: A:[64,128], B:[128,64] ... ✓ PASS")
    except Exception as e:
        print(f"  float16: A:[64,128], B:[128,64] ... ✗ FAIL: {e}")
        all_passed = False

    # float32
    try:
        a = torch.randn(64, 128, dtype=torch.float32)
        b = torch.randn(128, 64, dtype=torch.float32)
        c = matmul_golden(a, b)
        assert c.dtype == torch.float32
        c_ref = torch.matmul(a, b)
        assert torch.allclose(c, c_ref, rtol=1e-3, atol=1e-3)
        print(f"  float32: A:[64,128], B:[128,64] ... ✓ PASS")
    except Exception as e:
        print(f"  float32: A:[64,128], B:[128,64] ... ✗ FAIL: {e}")
        all_passed = False

    # 4. 边界条件验证
    print("\n[边界条件验证]")

    # 零值输入
    try:
        a = torch.zeros(32, 64, dtype=torch.float32)
        b = torch.randn(64, 128, dtype=torch.float32)
        c = matmul_golden(a, b)
        assert torch.all(c == 0), "Zero input should produce zero output"
        print(f"  零值输入: A:[32,64] (全零) ... ✓ PASS")
    except Exception as e:
        print(f"  零值输入: A:[32,64] (全零) ... ✗ FAIL: {e}")
        all_passed = False

    # 包含负数
    try:
        a = torch.randn(32, 64, dtype=torch.float32) - 0.5
        b = torch.randn(64, 128, dtype=torch.float32) - 0.5
        c = matmul_golden(a, b)
        c_ref = torch.matmul(a, b)
        assert torch.allclose(c, c_ref, rtol=1e-3, atol=1e-3)
        print(f"  负数输入: A:[32,64], B:[64,128] ... ✓ PASS")
    except Exception as e:
        print(f"  负数输入: A:[32,64], B:[64,128] ... ✗ FAIL: {e}")
        all_passed = False

    # 5. 数值稳定性检查
    print("\n[数值稳定性检查]")

    # 大值输入
    try:
        a = torch.randn(32, 64, dtype=torch.float32) * 100
        b = torch.randn(64, 128, dtype=torch.float32) * 100
        c = matmul_golden(a, b)
        c_ref = torch.matmul(a, b)
        assert torch.allclose(c, c_ref, rtol=1e-2, atol=1e-2)
        has_nan_or_inf = torch.isnan(c).any() or torch.isinf(c).any()
        print(f"  大值输入: A:[32,64]*100, B:[64,128]*100 ... ✓ PASS (NaN/Inf: {has_nan_or_inf})")
    except Exception as e:
        print(f"  大值输入: A:[32,64]*100, B:[64,128]*100 ... ✗ FAIL: {e}")
        all_passed = False

    # 小值输入
    try:
        a = torch.randn(32, 64, dtype=torch.float32) * 1e-6
        b = torch.randn(64, 128, dtype=torch.float32) * 1e-6
        c = matmul_golden(a, b)
        c_ref = torch.matmul(a, b)
        assert torch.allclose(c, c_ref, rtol=1e-3, atol=1e-3)
        print(f"  小值输入: A:[32,64]*1e-6, B:[64,128]*1e-6 ... ✓ PASS")
    except Exception as e:
        print(f"  小值输入: A:[32,64]*1e-6, B:[64,128]*1e-6 ... ✗ FAIL: {e}")
        all_passed = False

    # 6. API 对比（与 torch.matmul 对比）
    print("\n[API 对比]")

    try:
        # 随机测试多组数据
        test_cases = [
            ((16, 32), (32, 64)),
            ((8, 64, 128), (8, 128, 64)),
            ((2, 1, 32, 64), (1, 4, 64, 32)),
        ]

        for i, (shape_a, shape_b) in enumerate(test_cases):
            a = torch.randn(*shape_a, dtype=torch.float32)
            b = torch.randn(*shape_b, dtype=torch.float32)
            c_golden = matmul_golden(a, b)
            c_torch = torch.matmul(a, b)
            assert torch.allclose(c_golden, c_torch, rtol=1e-5, atol=1e-5), f"Case {i} mismatch"
            print(f"  对比 torch.matmul - Case {i}: {shape_a} @ {shape_b} ... ✓ PASS")
    except Exception as e:
        print(f"  对比 torch.matmul ... ✗ FAIL: {e}")
        all_passed = False

    # 最终结果
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过")
    else:
        print("⚠️  部分验证失败，请检查")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _validate()
