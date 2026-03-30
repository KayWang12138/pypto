#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
clip Golden 参考实现

公式: y = min(max(x, min_val), max_val)
功能: 将输入张量的每个元素限制在 [min_val, max_val] 范围内

置信度: 5/5 (使用 PyTorch 内置 API torch.clamp)
"""

import torch
from typing import Optional, Union


# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────

def clip_golden(
    x: torch.Tensor,
    min_val: Optional[Union[float, int]] = None,
    max_val: Optional[Union[float, int]] = None
) -> torch.Tensor:
    """clip 参考实现 (PyTorch)

    将输入张量的每个元素限制在 [min_val, max_val] 范围内。
    等价于 PyTorch 的 torch.clamp 函数。

    Args:
        x: 输入张量，支持任意维度和动态 shape
        min_val: 最小值边界（标量），如果为 None 则不限制下界
        max_val: 最大值边界（标量），如果为 None 则不限制上界

    Returns:
        输出张量，shape 与输入 x 相同，每个元素被限制在 [min_val, max_val] 范围内

    公式:
        y = min(max(x, min_val), max_val)

    示例:
        >>> x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        >>> clip_golden(x, min_val=-1.0, max_val=1.0)
        tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    """
    return torch.clamp(x, min=min_val, max=max_val)


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("clip_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # 1. 典型 case 验证（来自 spec.md §12）
    print("\n[典型 case 验证]")

    # 性能_P0: x: [1024, 1024], min=-1.0, max=1.0
    try:
        x = torch.randn(1024, 1024)
        out = clip_golden(x, min_val=-1.0, max_val=1.0)
        expected = torch.clamp(x, min=-1.0, max=1.0)
        assert out.shape == expected.shape, f"Shape mismatch: {out.shape} vs {expected.shape}"
        assert torch.allclose(out, expected, atol=0.001, rtol=0.001), "Value mismatch"
        print("  性能_P0: [1024, 1024], min=-1.0, max=1.0 ... OK PASS")
    except Exception as e:
        print(f"  性能_P0: [1024, 1024], min=-1.0, max=1.0 ... X FAIL: {e}")
        all_passed = False

    # 功能_P0: x: [32, 64, 128], min=-1.0, max=1.0
    try:
        x = torch.randn(32, 64, 128)
        out = clip_golden(x, min_val=-1.0, max_val=1.0)
        expected = torch.clamp(x, min=-1.0, max=1.0)
        assert out.shape == expected.shape, f"Shape mismatch: {out.shape} vs {expected.shape}"
        assert torch.allclose(out, expected, atol=0.001, rtol=0.001), "Value mismatch"
        print("  功能_P0: [32, 64, 128], min=-1.0, max=1.0 ... OK PASS")
    except Exception as e:
        print(f"  功能_P0: [32, 64, 128], min=-1.0, max=1.0 ... X FAIL: {e}")
        all_passed = False

    # 动态_P0: x: [batch, seq, hidden] (模拟动态 shape)
    try:
        for batch, seq, hidden in [(1, 64, 128), (4, 256, 512), (16, 512, 1024)]:
            x = torch.randn(batch, seq, hidden)
            out = clip_golden(x, min_val=0.0, max_val=1.0)
            expected = torch.clamp(x, min=0.0, max=1.0)
            assert out.shape == expected.shape, f"Shape mismatch: {out.shape} vs {expected.shape}"
            assert torch.allclose(out, expected, atol=0.001, rtol=0.001), "Value mismatch"
        print("  动态_P0: [batch, seq, hidden] 动态 shape ... OK PASS")
    except Exception as e:
        print(f"  动态_P0: [batch, seq, hidden] 动态 shape ... X FAIL: {e}")
        all_passed = False

    # 2. 泛化 case 验证（来自 spec.md §7 动态轴范围）
    print("\n[泛化 case 验证]")

    # 2D case
    try:
        x = torch.randn(128, 256)
        out = clip_golden(x, min_val=-2.0, max_val=2.0)
        expected = torch.clamp(x, min=-2.0, max=2.0)
        assert torch.allclose(out, expected, atol=0.001, rtol=0.001)
        print("  2D: [128, 256] ... OK PASS")
    except Exception as e:
        print(f"  2D: [128, 256] ... X FAIL: {e}")
        all_passed = False

    # 3D case
    try:
        x = torch.randn(8, 64, 128)
        out = clip_golden(x, min_val=-0.5, max_val=0.5)
        expected = torch.clamp(x, min=-0.5, max=0.5)
        assert torch.allclose(out, expected, atol=0.001, rtol=0.001)
        print("  3D: [8, 64, 128] ... OK PASS")
    except Exception as e:
        print(f"  3D: [8, 64, 128] ... X FAIL: {e}")
        all_passed = False

    # 4D case
    try:
        x = torch.randn(2, 4, 16, 32)
        out = clip_golden(x, min_val=-1.0, max_val=1.0)
        expected = torch.clamp(x, min=-1.0, max=1.0)
        assert torch.allclose(out, expected, atol=0.001, rtol=0.001)
        print("  4D: [2, 4, 16, 32] ... OK PASS")
    except Exception as e:
        print(f"  4D: [2, 4, 16, 32] ... X FAIL: {e}")
        all_passed = False

    # 3. 值域检查（从公式推导）
    print("\n[值域检查]")

    # 输出应该在 [min_val, max_val] 范围内
    try:
        x = torch.randn(100, 100) * 10  # 大范围随机值
        out = clip_golden(x, min_val=-1.0, max_val=1.0)
        assert out.min() >= -1.0 - 1e-6, f"Min value {out.min()} < min_val"
        assert out.max() <= 1.0 + 1e-6, f"Max value {out.max()} > max_val"
        print("  输出范围检查 [min, max] ... OK PASS")
    except Exception as e:
        print(f"  输出范围检查 [min, max] ... X FAIL: {e}")
        all_passed = False

    # 4. 数值稳定性检查
    print("\n[数值稳定性检查]")

    # 大值输入
    try:
        x = torch.tensor([1e6, -1e6, 1e-6, -1e-6])
        out = clip_golden(x, min_val=-1.0, max_val=1.0)
        assert not torch.isnan(out).any(), "NaN detected"
        assert not torch.isinf(out).any(), "Inf detected"
        print("  大值/小值输入 ... OK PASS")
    except Exception as e:
        print(f"  大值/小值输入 ... X FAIL: {e}")
        all_passed = False

    # 边界值
    try:
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = clip_golden(x, min_val=-1.0, max_val=1.0)
        expected = torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0])
        assert torch.allclose(out, expected, atol=0.001, rtol=0.001)
        print("  边界值输入 ... OK PASS")
    except Exception as e:
        print(f"  边界值输入 ... X FAIL: {e}")
        all_passed = False

    # 5. 特殊情况检查
    print("\n[特殊情况检查]")

    # None min/max (note: torch.clamp requires at least one of min/max to be non-None)
    try:
        x = torch.randn(10, 10)
        out_min_only = clip_golden(x, min_val=0.0, max_val=None)
        out_max_only = clip_golden(x, min_val=None, max_val=1.0)
        assert torch.allclose(out_min_only, torch.clamp(x, min=0.0))
        assert torch.allclose(out_max_only, torch.clamp(x, max=1.0))
        print("  None min/max 处理 (单侧) ... OK PASS")
    except Exception as e:
        print(f"  None min/max 处理 ... X FAIL: {e}")
        all_passed = False

    # 6. API 对比
    print("\n[API 对比]")

    try:
        x = torch.randn(64, 64)
        out_golden = clip_golden(x, min_val=-0.5, max_val=0.5)
        out_torch = torch.clamp(x, min=-0.5, max=0.5)
        assert torch.allclose(out_golden, out_torch, atol=0.001, rtol=0.001)
        print("  与 torch.clamp 对比 ... OK PASS")
    except Exception as e:
        print(f"  与 torch.clamp 对比 ... X FAIL: {e}")
        all_passed = False

    # 7. dtype 支持
    print("\n[dtype 支持]")

    for dtype in [torch.float32, torch.float16]:
        try:
            x = torch.randn(32, 32, dtype=dtype)
            out = clip_golden(x, min_val=-1.0, max_val=1.0)
            assert out.dtype == dtype, f"dtype mismatch: {out.dtype} vs {dtype}"
            print(f"  {dtype} ... OK PASS")
        except Exception as e:
            print(f"  {dtype} ... X FAIL: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("OK 所有验证通过")
    else:
        print("X 部分验证失败")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _validate()
