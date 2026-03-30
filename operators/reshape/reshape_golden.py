#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""
reshape Golden 参考实现

公式: y = reshape(x, shape)
功能: 将输入张量变换为指定的形状，保持数据不变，仅改变维度视图。
      支持动态轴（batch、seq_len）和负维度自动推断。

置信度: (5/5) 使用 PyTorch 内置 API torch.reshape()
"""

import torch
from typing import List, Union, Tuple


def reshape_golden(
    x: torch.Tensor,
    shape: List[int],
) -> torch.Tensor:
    """
    reshape 参考实现 (PyTorch)

    将输入张量变换为指定的形状。支持负维度自动推断。

    Args:
        x: 输入张量，支持任意维度和 dtype (float32/bfloat16/float16)。
        shape: 目标形状，支持一个维度为 -1 表示自动推断。

    Returns:
        输出张量，形状为目标 shape，dtype 与输入相同。

    公式:
        y = reshape(x, shape)

    约束:
        - shape 中最多一个维度为 -1
        - 元素总数必须匹配: prod(input.shape) = prod(target_shape)

    Example:
        >>> x = torch.randn(2, 3, 4)
        >>> y = reshape_golden(x, [2, 12])
        >>> y.shape
        torch.Size([2, 12])
        >>> y = reshape_golden(x, [2, -1])
        >>> y.shape
        torch.Size([2, 12])
    """
    return torch.reshape(x, shape)


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("reshape_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # 1. 典型 case 验证（来自 spec.md §11）
    # 按优先级顺序：功能_P0 -> 功能_P1
    print("\n[典型 case 验证]")

    # 功能_P0: 基础 reshape
    # 输入 Shape: [2, 3, 4], 参数: shape=[2, 12], 输出 Shape: [2, 12]
    try:
        x = torch.randn(2, 3, 4)
        y = reshape_golden(x, [2, 12])
        assert y.shape == torch.Size([2, 12]), f"Shape mismatch: {y.shape} vs [2, 12]"
        assert torch.allclose(y.flatten(), x.flatten()), "Data mismatch after reshape"
        print("  功能_P0 (基础): [2,3,4] -> [2,12] ... PASS")
    except Exception as e:
        print(f"  功能_P0 (基础): [2,3,4] -> [2,12] ... FAIL: {e}")
        all_passed = False

    # 功能_P0: 动态轴 reshape (模拟动态 batch 和 seq)
    # 输入 Shape: [b, s1, 64], 参数: shape=[b, s2, 8], 输出 Shape: [b, s2, 8]
    try:
        b, s1, s2 = 4, 16, 128
        x = torch.randn(b, s1, 64)
        y = reshape_golden(x, [b, s2, 8])
        assert y.shape == torch.Size([b, s2, 8]), f"Shape mismatch: {y.shape} vs [{b}, {s2}, 8]"
        assert torch.allclose(y.flatten(), x.flatten()), "Data mismatch after reshape"
        print(f"  功能_P0 (动态轴): [{b},{s1},64] -> [{b},{s2},8] ... PASS")
    except Exception as e:
        print(f"  功能_P0 (动态轴): [4,16,64] -> [4,128,8] ... FAIL: {e}")
        all_passed = False

    # 功能_P0: 负维度自动推断
    # 输入 Shape: [2, 3, 4], 参数: shape=[2, -1], 输出 Shape: [2, 12]
    try:
        x = torch.randn(2, 3, 4)
        y = reshape_golden(x, [2, -1])
        assert y.shape == torch.Size([2, 12]), f"Shape mismatch: {y.shape} vs [2, 12]"
        assert torch.allclose(y.flatten(), x.flatten()), "Data mismatch after reshape"
        print("  功能_P0 (负维度): [2,3,4] -> [2,-1] => [2,12] ... PASS")
    except Exception as e:
        print(f"  功能_P0 (负维度): [2,3,4] -> [2,-1] => [2,12] ... FAIL: {e}")
        all_passed = False

    # 功能_P1: 展平为一维
    # 输入 Shape: [2, 3, 4], 参数: shape=[-1], 输出 Shape: [24]
    try:
        x = torch.randn(2, 3, 4)
        y = reshape_golden(x, [-1])
        assert y.shape == torch.Size([24]), f"Shape mismatch: {y.shape} vs [24]"
        assert torch.allclose(y, x.flatten()), "Data mismatch after flatten"
        print("  功能_P1 (展平): [2,3,4] -> [-1] => [24] ... PASS")
    except Exception as e:
        print(f"  功能_P1 (展平): [2,3,4] -> [-1] => [24] ... FAIL: {e}")
        all_passed = False

    # 功能_P1: 增加维度
    # 输入 Shape: [2, 12], 参数: shape=[2, 3, 4], 输出 Shape: [2, 3, 4]
    try:
        x = torch.randn(2, 12)
        y = reshape_golden(x, [2, 3, 4])
        assert y.shape == torch.Size([2, 3, 4]), f"Shape mismatch: {y.shape} vs [2, 3, 4]"
        assert torch.allclose(y.flatten(), x.flatten()), "Data mismatch after reshape"
        print("  功能_P1 (增维): [2,12] -> [2,3,4] ... PASS")
    except Exception as e:
        print(f"  功能_P1 (增维): [2,12] -> [2,3,4] ... FAIL: {e}")
        all_passed = False

    # 2. 泛化 case 验证（来自 spec.md §7 动态轴范围）
    print("\n[泛化 case 验证]")

    # 边界采样: batch 最小值 1, 中间值 64, 最大值 128
    # seq_len 最小值 1, 中间值 512, 最大值 1024
    test_cases = [
        (1, 1, 64),      # 最小 batch, 最小 seq
        (1, 512, 64),    # 最小 batch, 中间 seq
        (64, 256, 64),   # 中间 batch, 中间 seq
        (128, 1024, 64), # 最大 batch, 最大 seq
    ]

    for b, s, d in test_cases:
        try:
            x = torch.randn(b, s, d)
            new_s = s * d // 8
            y = reshape_golden(x, [b, new_s, 8])
            assert y.shape == torch.Size([b, new_s, 8]), f"Shape mismatch"
            assert torch.allclose(y.flatten(), x.flatten()), "Data mismatch"
            print(f"  泛化: [{b},{s},{d}] -> [{b},{new_s},8] ... PASS")
        except Exception as e:
            print(f"  泛化: [{b},{s},{d}] ... FAIL: {e}")
            all_passed = False

    # 3. 值域检查（从公式推导）
    print("\n[值域检查]")

    # reshape 不改变数值，只改变 shape，所以输出值域 = 输入值域
    # 检查数据是否完整保持
    try:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = reshape_golden(x, [2, 3])
        assert torch.allclose(y, x.reshape(2, 3)), "Data not preserved"
        print("  数据完整性检查 ... PASS")
    except Exception as e:
        print(f"  数据完整性检查 ... FAIL: {e}")
        all_passed = False

    # 4. 数值稳定性检查
    print("\n[数值稳定性检查]")

    # 检查 NaN/Inf 是否保持不变
    try:
        x = torch.tensor([1.0, float('nan'), float('inf'), -float('inf'), 0.0, 1.0])
        y = reshape_golden(x, [2, 3])
        assert torch.isnan(y[0, 1]), "NaN not preserved"
        assert torch.isinf(y[0, 2]) and y[0, 2] > 0, "Inf not preserved"
        assert torch.isinf(y[1, 0]) and y[1, 0] < 0, "-Inf not preserved"
        print("  NaN/Inf 保持 ... PASS")
    except Exception as e:
        print(f"  NaN/Inf 保持 ... FAIL: {e}")
        all_passed = False

    # 检查零值
    try:
        x = torch.zeros(3, 4)
        y = reshape_golden(x, [2, 6])
        assert torch.all(y == 0), "Zero values not preserved"
        print("  零值保持 ... PASS")
    except Exception as e:
        print(f"  零值保持 ... FAIL: {e}")
        all_passed = False

    # 5. API 对比（与 PyTorch API 对比）
    print("\n[API 对比]")

    try:
        x = torch.randn(2, 3, 4)
        our_result = reshape_golden(x, [2, 12])
        torch_result = torch.reshape(x, [2, 12])
        assert torch.allclose(our_result, torch_result), "API mismatch with torch.reshape"
        print("  与 torch.reshape 一致性 ... PASS")
    except Exception as e:
        print(f"  与 torch.reshape 一致性 ... FAIL: {e}")
        all_passed = False

    # 6. dtype 支持检查
    print("\n[dtype 支持检查]")

    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        try:
            x = torch.randn(2, 3, 4, dtype=dtype)
            y = reshape_golden(x, [2, 12])
            assert y.dtype == dtype, f"dtype changed: {y.dtype} vs {dtype}"
            assert y.shape == torch.Size([2, 12])
            print(f"  {str(dtype):12s} ... PASS")
        except Exception as e:
            print(f"  {str(dtype):12s} ... FAIL: {e}")
            all_passed = False

    # 7. 函数签名检查
    print("\n[函数签名检查]")

    import inspect
    sig = inspect.signature(reshape_golden)
    params = list(sig.parameters.keys())
    expected_params = ['x', 'shape']
    if params == expected_params:
        print(f"  参数列表: {params} ... PASS")
    else:
        print(f"  参数列表: {params} vs 期望 {expected_params} ... FAIL")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("PASS 所有验证通过")
    else:
        print("FAIL 部分验证失败")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _validate()
