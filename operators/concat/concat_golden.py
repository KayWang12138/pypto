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
concat Golden 参考实现

公式: 沿指定维度拼接多个张量，保持其他维度不变
置信度: ⭐⭐⭐⭐⭐ (使用 PyTorch 内置 API torch.cat)

功能描述:
    将多个张量沿指定维度拼接成一个张量。
    所有输入张量在非拼接维度上的 shape 必须相同，dtype 必须一致。

支持特性:
    - 多张量输入: 支持可变数量的输入张量（至少2个）
    - 指定拼接轴: 支持通过 dim 参数指定拼接维度
    - 负数索引: dim 参数支持负数索引（如 -1 表示最后一维）
"""

import torch
from typing import List, Union


def concat_golden(
    tensors: List[torch.Tensor],
    dim: int = 0
) -> torch.Tensor:
    """
    concat 参考实现 (PyTorch)

    沿指定维度拼接多个张量，保持其他维度不变。
    直接使用 PyTorch 内置的 torch.cat API。

    Args:
        tensors: 待拼接的张量列表，至少包含2个张量。
                 所有张量必须具有相同的维度数和 dtype。
                 除 dim 维度外，其他维度的 shape 必须相同。
        dim: 拼接维度，支持负数索引。默认为 0。
             例如: dim=-1 表示沿最后一维拼接。

    Returns:
        拼接后的张量。输出 shape 在 dim 维度上为各输入张量
        在该维度大小的和，其他维度保持不变。

    公式:
        output.shape[dim] = sum(t.shape[dim] for t in tensors)
        output.shape[other] = tensors[0].shape[other]

    Examples:
        >>> a = torch.tensor([[1, 1], [1, 1]])
        >>> b = torch.tensor([[0, 0], [0, 0]])
        >>> concat_golden([a, b], dim=0)
        tensor([[1, 1],
                [1, 1],
                [0, 0],
                [0, 0]])

        >>> concat_golden([a, b], dim=1)
        tensor([[1, 1, 0, 0],
                [1, 1, 0, 0]])
    """
    # 边界条件处理：空列表
    if len(tensors) == 0:
        raise ValueError("tensors list cannot be empty")

    # 边界条件处理：单张量输入，返回副本
    if len(tensors) == 1:
        return tensors[0].clone()

    # 过滤掉 dim 维度大小为 0 的张量（空张量）
    non_empty_tensors = []
    for t in tensors:
        if t.shape[dim] > 0:
            non_empty_tensors.append(t)

    # 如果所有张量都是空的，返回第一个张量
    if len(non_empty_tensors) == 0:
        return tensors[0].clone()

    # 如果只有一个非空张量，返回它
    if len(non_empty_tensors) == 1:
        return non_empty_tensors[0].clone()

    # 使用 PyTorch 内置 API 进行拼接
    return torch.cat(non_empty_tensors, dim=dim)


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("concat_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # 1. 典型 case 验证（来自 spec.md §12）
    print("\n[典型 case 验证]")

    # 性能_P0: 大规模张量拼接性能
    # dim=-1, N=2, [4096,1024], [4096,1024] -> [4096,2048]
    try:
        a = torch.randn(4096, 1024)
        b = torch.randn(4096, 1024)
        out = concat_golden([a, b], dim=-1)
        expected_shape = (4096, 2048)
        assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
        # 验证数值正确性
        expected = torch.cat([a, b], dim=-1)
        assert torch.allclose(out, expected), "Value mismatch"
        print(f"  性能_P0: [4096,1024], [4096,1024], dim=-1 ... ✓ PASS")
    except Exception as e:
        print(f"  性能_P0: [4096,1024], [4096,1024], dim=-1 ... ✗ FAIL: {e}")
        all_passed = False

    # 功能_P0: 基础功能验证，拼接两个张量
    # dim=-1, N=2, [b,s,1024], [b,s,512] -> [b,s,1536]
    try:
        b, s = 2, 128
        a = torch.randn(b, s, 1024)
        b_tensor = torch.randn(b, s, 512)
        out = concat_golden([a, b_tensor], dim=-1)
        expected_shape = (b, s, 1536)
        assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
        expected = torch.cat([a, b_tensor], dim=-1)
        assert torch.allclose(out, expected), "Value mismatch"
        print(f"  功能_P0: [2,128,1024], [2,128,512], dim=-1 ... ✓ PASS")
    except Exception as e:
        print(f"  功能_P0: [2,128,1024], [2,128,512], dim=-1 ... ✗ FAIL: {e}")
        all_passed = False

    # 功能_P0: 动态 seq_len 维度拼接
    # dim=1, N=2, [b,s1,d], [b,s2,d] -> [b,s1+s2,d]
    try:
        b, s1, s2, d = 2, 64, 128, 256
        a = torch.randn(b, s1, d)
        b_tensor = torch.randn(b, s2, d)
        out = concat_golden([a, b_tensor], dim=1)
        expected_shape = (b, s1 + s2, d)
        assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
        expected = torch.cat([a, b_tensor], dim=1)
        assert torch.allclose(out, expected), "Value mismatch"
        print(f"  功能_P0 (seq): [2,64,256], [2,128,256], dim=1 ... ✓ PASS")
    except Exception as e:
        print(f"  功能_P0 (seq): [2,64,256], [2,128,256], dim=1 ... ✗ FAIL: {e}")
        all_passed = False

    # 功能_P0: 动态 batch 维度拼接
    # dim=0, N=2, [b1,s,d], [b2,s,d] -> [b1+b2,s,d]
    try:
        b1, b2, s, d = 16, 32, 128, 256
        a = torch.randn(b1, s, d)
        b_tensor = torch.randn(b2, s, d)
        out = concat_golden([a, b_tensor], dim=0)
        expected_shape = (b1 + b2, s, d)
        assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
        expected = torch.cat([a, b_tensor], dim=0)
        assert torch.allclose(out, expected), "Value mismatch"
        print(f"  功能_P0 (batch): [16,128,256], [32,128,256], dim=0 ... ✓ PASS")
    except Exception as e:
        print(f"  功能_P0 (batch): [16,128,256], [32,128,256], dim=0 ... ✗ FAIL: {e}")
        all_passed = False

    # 功能_P1: 拼接三个张量，batch 维度
    try:
        b1, b2, b3, s, d = 4, 8, 12, 64, 128
        a = torch.randn(b1, s, d)
        b_tensor = torch.randn(b2, s, d)
        c = torch.randn(b3, s, d)
        out = concat_golden([a, b_tensor, c], dim=0)
        expected_shape = (b1 + b2 + b3, s, d)
        assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
        expected = torch.cat([a, b_tensor, c], dim=0)
        assert torch.allclose(out, expected), "Value mismatch"
        print(f"  功能_P1: 3 tensors, dim=0 ... ✓ PASS")
    except Exception as e:
        print(f"  功能_P1: 3 tensors, dim=0 ... ✗ FAIL: {e}")
        all_passed = False

    # 2. 泛化 case 验证（来自 spec.md §8 动态轴范围）
    print("\n[泛化 case 验证]")

    # batch 边界值测试: b=1
    try:
        a = torch.randn(1, 64, 128)
        b_tensor = torch.randn(1, 64, 128)
        out = concat_golden([a, b_tensor], dim=-1)
        assert out.shape == (1, 64, 256)
        print(f"  batch=1: [1,64,128] + [1,64,128], dim=-1 ... ✓ PASS")
    except Exception as e:
        print(f"  batch=1: [1,64,128] + [1,64,128], dim=-1 ... ✗ FAIL: {e}")
        all_passed = False

    # batch 边界值测试: b=1024
    try:
        a = torch.randn(1024, 64, 128)
        b_tensor = torch.randn(512, 64, 128)
        out = concat_golden([a, b_tensor], dim=0)
        assert out.shape == (1536, 64, 128)
        print(f"  batch=1024: [1024,64,128] + [512,64,128], dim=0 ... ✓ PASS")
    except Exception as e:
        print(f"  batch=1024: [1024,64,128] + [512,64,128], dim=0 ... ✗ FAIL: {e}")
        all_passed = False

    # seq_len 边界值测试: s=1
    try:
        a = torch.randn(16, 1, 128)
        b_tensor = torch.randn(16, 1, 128)
        out = concat_golden([a, b_tensor], dim=1)
        assert out.shape == (16, 2, 128)
        print(f"  seq=1: [16,1,128] + [16,1,128], dim=1 ... ✓ PASS")
    except Exception as e:
        print(f"  seq=1: [16,1,128] + [16,1,128], dim=1 ... ✗ FAIL: {e}")
        all_passed = False

    # 3. 负数索引测试
    print("\n[负数索引测试]")

    try:
        a = torch.randn(4, 8, 16)
        b_tensor = torch.randn(4, 8, 32)
        out = concat_golden([a, b_tensor], dim=-1)
        assert out.shape == (4, 8, 48)
        print(f"  dim=-1 (last dim): ... ✓ PASS")
    except Exception as e:
        print(f"  dim=-1 (last dim): ... ✗ FAIL: {e}")
        all_passed = False

    try:
        a = torch.randn(4, 8, 16)
        b_tensor = torch.randn(4, 8, 16)
        out = concat_golden([a, b_tensor], dim=-2)
        assert out.shape == (4, 16, 16)
        print(f"  dim=-2 (middle dim): ... ✓ PASS")
    except Exception as e:
        print(f"  dim=-2 (middle dim): ... ✗ FAIL: {e}")
        all_passed = False

    try:
        a = torch.randn(4, 8, 16)
        b_tensor = torch.randn(12, 8, 16)
        out = concat_golden([a, b_tensor], dim=-3)
        assert out.shape == (16, 8, 16)
        print(f"  dim=-3 (first dim): ... ✓ PASS")
    except Exception as e:
        print(f"  dim=-3 (first dim): ... ✗ FAIL: {e}")
        all_passed = False

    # 4. 边界条件测试
    print("\n[边界条件测试]")

    # 零值处理
    try:
        a = torch.zeros(4, 8, 16)
        b_tensor = torch.zeros(4, 8, 16)
        out = concat_golden([a, b_tensor], dim=0)
        assert out.shape == (8, 8, 16)
        assert torch.all(out == 0)
        print(f"  零值输入: ... ✓ PASS")
    except Exception as e:
        print(f"  零值输入: ... ✗ FAIL: {e}")
        all_passed = False

    # NaN/Inf 处理
    try:
        a = torch.tensor([[[float('nan'), 1.0], [2.0, 3.0]]])
        b_tensor = torch.tensor([[[float('inf'), 4.0], [5.0, 6.0]]])
        out = concat_golden([a, b_tensor], dim=0)
        assert out.shape == (2, 2, 2)
        assert torch.isnan(out[0, 0, 0])
        assert torch.isinf(out[1, 0, 0])
        print(f"  NaN/Inf 保持不变: ... ✓ PASS")
    except Exception as e:
        print(f"  NaN/Inf 保持不变: ... ✗ FAIL: {e}")
        all_passed = False

    # 单张量输入
    try:
        a = torch.randn(4, 8, 16)
        out = concat_golden([a], dim=0)
        assert out.shape == a.shape
        assert torch.allclose(out, a)
        assert out is not a  # 确保是副本
        print(f"  单张量输入返回副本: ... ✓ PASS")
    except Exception as e:
        print(f"  单张量输入返回副本: ... ✗ FAIL: {e}")
        all_passed = False

    # 5. 数值稳定性检查
    print("\n[数值稳定性检查]")

    # 大值输入
    try:
        a = torch.randn(64, 128) * 1e6
        b_tensor = torch.randn(64, 128) * 1e6
        out = concat_golden([a, b_tensor], dim=1)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))
        print(f"  大值输入 (1e6): ... ✓ PASS")
    except Exception as e:
        print(f"  大值输入 (1e6): ... ✗ FAIL: {e}")
        all_passed = False

    # 小值输入
    try:
        a = torch.randn(64, 128) * 1e-6
        b_tensor = torch.randn(64, 128) * 1e-6
        out = concat_golden([a, b_tensor], dim=1)
        assert not torch.any(torch.isnan(out))
        print(f"  小值输入 (1e-6): ... ✓ PASS")
    except Exception as e:
        print(f"  小值输入 (1e-6): ... ✗ FAIL: {e}")
        all_passed = False

    # 6. API 对比
    print("\n[API 对比]")

    try:
        a = torch.randn(8, 16, 32)
        b_tensor = torch.randn(8, 16, 64)
        golden_out = concat_golden([a, b_tensor], dim=-1)
        torch_out = torch.cat([a, b_tensor], dim=-1)
        assert torch.allclose(golden_out, torch_out)
        print(f"  与 torch.cat 对比: ... ✓ PASS")
    except Exception as e:
        print(f"  与 torch.cat 对比: ... ✗ FAIL: {e}")
        all_passed = False

    # 7. dtype 支持测试
    print("\n[dtype 支持测试]")

    for dtype, dtype_name in [(torch.float32, "float32"), (torch.float16, "float16"), (torch.bfloat16, "bfloat16")]:
        try:
            a = torch.randn(4, 8, 16, dtype=dtype)
            b_tensor = torch.randn(4, 8, 32, dtype=dtype)
            out = concat_golden([a, b_tensor], dim=-1)
            assert out.dtype == dtype
            assert out.shape == (4, 8, 48)
            print(f"  {dtype_name}: ... ✓ PASS")
        except Exception as e:
            print(f"  {dtype_name}: ... ✗ FAIL: {e}")
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
