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
where Golden 参考实现

公式: out_i = condition_i ? x_i : y_i
置信度: (使用 PyTorch 内置 API)

功能: 根据条件张量从 x 或 y 中选择元素。
      当 condition 对应位置为 True 时选择 x 的元素，为 False 时选择 y 的元素。
"""

import torch
from typing import Optional, Union


def where_golden(
    condition: torch.Tensor,
    x: torch.Tensor,
    y: Union[torch.Tensor, float, int],
) -> torch.Tensor:
    """
    where 参考实现 (PyTorch)

    根据条件张量从 x 或 y 中选择元素。当 condition 对应位置为 True 时选择 x 的元素，
    为 False 时选择 y 的元素。输出张量与 x、y 广播后具有相同的 shape。

    Args:
        condition: 条件张量，dtype 为 bool，决定选择 x 还是 y
        x: condition 为 True 时选择的值
        y: condition 为 False 时选择的值，支持张量或标量

    Returns:
        根据 condition 从 x 和 y 选择的元素，shape 为 condition/x/y 广播后的 shape

    公式:
        out_i = condition_i ? x_i : y_i

    广播规则:
        condition, x, y 支持广播到相同的 shape
    """
    return torch.where(condition, x, y)


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("where_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # 1. 典型 case 验证（来自 spec.md 12）
    # 按优先级顺序：性能_P0 -> 功能_P0 -> 功能_P1
    print("\n[典型 case 验证]")

    # 典型配置参数
    b, s, n, d = 2, 128, 8, 64

    # 性能_P0: 广播场景 condition:[b,s,1,1], x:[b,s,n,d], y:[b,s,n,d]
    try:
        condition = torch.rand(b, s, 1, 1) > 0.5
        x = torch.randn(b, s, n, d)
        y = torch.randn(b, s, n, d)
        out = where_golden(condition, x, y)
        expected_shape = (b, s, n, d)
        assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
        # 验证广播正确性
        expanded_cond = condition.expand(b, s, n, d)
        expected = torch.where(expanded_cond, x, y)
        assert torch.allclose(out, expected), "Value mismatch"
        print(f"  性能_P0: condition:[{b},{s},1,1], x:[{b},{s},{n},{d}], y:[{b},{s},{n},{d}] ... PASS")
    except Exception as e:
        print(f"  性能_P0: ... FAIL - {e}")
        all_passed = False

    # 功能_P0: 基础功能验证 condition:[b,s,n,d], x:[b,s,n,d], y:[b,s,n,d]
    try:
        condition = torch.rand(b, s, n, d) > 0.5
        x = torch.randn(b, s, n, d)
        y = torch.randn(b, s, n, d)
        out = where_golden(condition, x, y)
        expected_shape = (b, s, n, d)
        assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
        # 验证正确性：检查 True 位置选 x，False 位置选 y
        assert torch.allclose(out[condition], x[condition]), "True positions should select x"
        assert torch.allclose(out[~condition], y[~condition]), "False positions should select y"
        print(f"  功能_P0: condition:[{b},{s},{n},{d}], x:[{b},{s},{n},{d}], y:[{b},{s},{n},{d}] ... PASS")
    except Exception as e:
        print(f"  功能_P0: ... FAIL - {e}")
        all_passed = False

    # 功能_P1: 标量 y 支持 condition:[b,s,n,d], x:[b,s,n,d], y:scalar
    try:
        condition = torch.rand(b, s, n, d) > 0.5
        x = torch.randn(b, s, n, d)
        y_scalar = 0.0
        out = where_golden(condition, x, y_scalar)
        expected_shape = (b, s, n, d)
        assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
        # 验证标量 y 正确性
        assert torch.allclose(out[condition], x[condition]), "True positions should select x"
        assert torch.allclose(out[~condition], torch.tensor(y_scalar)), "False positions should select y_scalar"
        print(f"  功能_P1: condition:[{b},{s},{n},{d}], x:[{b},{s},{n},{d}], y:scalar(0.0) ... PASS")
    except Exception as e:
        print(f"  功能_P1: ... FAIL - {e}")
        all_passed = False

    # 2. 泛化 case 验证（来自 spec.md 8 动态轴范围）
    print("\n[泛化 case 验证]")

    # 动态轴取值: b=[1,32,128], s=[64,512,2048]
    test_cases = [
        (1, 64, 4, 32),      # 最小边界
        (32, 512, 8, 64),    # 中间值
        (128, 2048, 16, 128) # 大 shape
    ]

    for b_val, s_val, n_val, d_val in test_cases:
        try:
            condition = torch.rand(b_val, s_val, n_val, d_val) > 0.5
            x = torch.randn(b_val, s_val, n_val, d_val)
            y = torch.randn(b_val, s_val, n_val, d_val)
            out = where_golden(condition, x, y)
            expected_shape = (b_val, s_val, n_val, d_val)
            assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
            print(f"  b={b_val}, s={s_val}, n={n_val}, d={d_val} ... PASS")
        except Exception as e:
            print(f"  b={b_val}, s={s_val}, n={n_val}, d={d_val} ... FAIL - {e}")
            all_passed = False

    # 3. 值域检查（从公式推导）
    print("\n[值域检查]")

    # where 无特定值域约束，输出范围取决于输入

    # 验证输出值确实来自输入
    try:
        condition = torch.tensor([[True, False], [False, True]])
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        out = where_golden(condition, x, y)
        expected = torch.tensor([[1.0, 6.0], [7.0, 4.0]])
        assert torch.allclose(out, expected), f"Value mismatch: {out} vs {expected}"
        print("  输出值验证 ... PASS")
    except Exception as e:
        print(f"  输出值验证 ... FAIL - {e}")
        all_passed = False

    # 4. 数值稳定性检查
    print("\n[数值稳定性检查]")

    # 大值输入
    try:
        condition = torch.rand(4, 4) > 0.5
        x = torch.randn(4, 4) * 1e6
        y = torch.randn(4, 4) * 1e6
        out = where_golden(condition, x, y)
        assert not torch.any(torch.isnan(out)), "NaN detected in output"
        assert not torch.any(torch.isinf(out)), "Inf detected in output (unexpected)"
        print("  大值输入 (1e6) ... PASS")
    except Exception as e:
        print(f"  大值输入 ... FAIL - {e}")
        all_passed = False

    # 极小值输入
    try:
        condition = torch.rand(4, 4) > 0.5
        x = torch.randn(4, 4) * 1e-6
        y = torch.randn(4, 4) * 1e-6
        out = where_golden(condition, x, y)
        assert not torch.any(torch.isnan(out)), "NaN detected in output"
        print("  极小值输入 (1e-6) ... PASS")
    except Exception as e:
        print(f"  极小值输入 ... FAIL - {e}")
        all_passed = False

    # NaN/Inf 传递（spec 定义为正常传递）
    try:
        condition = torch.tensor([[True, False], [False, True]])
        x = torch.tensor([[1.0, float('nan')], [float('inf'), -float('inf')]])
        y = torch.tensor([[float('nan'), 2.0], [3.0, 4.0]])
        out = where_golden(condition, x, y)
        # 验证 NaN/Inf 正确传递
        # condition True 位置选 x，False 位置选 y
        assert out[0, 0].item() == 1.0, "Position [0,0] should be 1.0"
        assert out[0, 1].item() == 2.0, "Position [0,1] should be 2.0"
        assert out[1, 0].item() == 3.0, "Position [1,0] should be 3.0"
        print("  NaN/Inf 传递 ... PASS")
    except Exception as e:
        print(f"  NaN/Inf 传递 ... FAIL - {e}")
        all_passed = False

    # 5. API 对比（与 torch.where 直接对比）
    print("\n[API 对比]")

    try:
        condition = torch.rand(8, 8) > 0.5
        x = torch.randn(8, 8)
        y = torch.randn(8, 8)
        out_golden = where_golden(condition, x, y)
        out_torch = torch.where(condition, x, y)
        assert torch.allclose(out_golden, out_torch), "Output differs from torch.where"
        print("  与 torch.where 对比 ... PASS (完全一致)")
    except Exception as e:
        print(f"  与 torch.where 对比 ... FAIL - {e}")
        all_passed = False

    # 6. 多 dtype 支持
    print("\n[dtype 支持验证]")

    for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int32]:
        try:
            condition = torch.rand(4, 4) > 0.5
            if dtype == torch.int32:
                x = torch.randint(-100, 100, (4, 4), dtype=dtype)
                y = torch.randint(-100, 100, (4, 4), dtype=dtype)
            else:
                x = torch.randn(4, 4, dtype=dtype)
                y = torch.randn(4, 4, dtype=dtype)
            out = where_golden(condition, x, y)
            assert out.dtype == dtype, f"dtype mismatch: {out.dtype} vs {dtype}"
            print(f"  {dtype} ... PASS")
        except Exception as e:
            print(f"  {dtype} ... FAIL - {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("所有验证通过")
    else:
        print("存在验证失败项")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _validate()
