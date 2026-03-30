#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""
linear Golden 参考实现

公式: output = input @ weight^T + bias
置信度: (使用 PyTorch 内置 API torch.nn.functional.linear)

功能: 线性层（全连接层）算子，对输入张量进行线性变换。
      支持任意维度的输入张量，对最后一维进行线性变换，保持其他维度不变。
"""

import torch
from typing import Optional


def linear_golden(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    linear 参考实现 (PyTorch)

    对输入张量进行线性变换: output = input @ weight^T + bias

    Args:
        input: 输入张量，shape 为 [..., in_features]，最后一维为输入特征数。
               支持任意维度（2D、3D、4D 等），前 N-1 维可动态变化。
        weight: 权重矩阵，shape 为 [out_features, in_features]。
        bias: 偏置向量，shape 为 [out_features]，可选。

    Returns:
        输出张量，shape 为 [..., out_features]，最后一维为输出特征数。
        与输入张量的前 N-1 维保持一致。

    公式:
        output = input @ weight.T + bias

    Examples:
        >>> input = torch.randn(2, 3, 64)  # [batch, seq, in_features]
        >>> weight = torch.randn(128, 64)  # [out_features, in_features]
        >>> bias = torch.randn(128)        # [out_features]
        >>> output = linear_golden(input, weight, bias)
        >>> output.shape
        torch.Size([2, 3, 128])
    """
    # 使用 PyTorch 内置的 F.linear 实现
    # F.linear 自动处理:
    #   1. weight 的转置
    #   2. 任意维度输入的广播
    #   3. bias 的广播加法
    return torch.nn.functional.linear(input, weight, bias)


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""

    print("=" * 60)
    print("linear_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # ==================== 1. 典型 case 验证（来自 spec.md §12）====================
    print("\n[典型 case 验证]")

    # 性能_P0_2D: in_features=4096, out_features=4096, has_bias=True
    # input: [batch, 4096], weight: [4096, 4096], bias: [4096]
    try:
        input_2d = torch.randn(16, 4096, dtype=torch.float32)
        weight_2d = torch.randn(4096, 4096, dtype=torch.float32)
        bias_2d = torch.randn(4096, dtype=torch.float32)
        output_2d = linear_golden(input_2d, weight_2d, bias_2d)
        expected_shape = (16, 4096)
        assert output_2d.shape == expected_shape, f"Shape mismatch: {output_2d.shape} vs {expected_shape}"
        print(f"  性能_P0_2D: batch=16, in=4096, out=4096, has_bias=True ... PASS")
    except Exception as e:
        print(f"  性能_P0_2D: ... FAIL - {e}")
        all_passed = False

    # 性能_P0_3D: in_features=1024, out_features=4096, has_bias=True
    # input: [batch, seq_len, 1024], weight: [4096, 1024], bias: [4096]
    try:
        input_3d = torch.randn(8, 128, 1024, dtype=torch.float32)
        weight_3d = torch.randn(4096, 1024, dtype=torch.float32)
        bias_3d = torch.randn(4096, dtype=torch.float32)
        output_3d = linear_golden(input_3d, weight_3d, bias_3d)
        expected_shape = (8, 128, 4096)
        assert output_3d.shape == expected_shape, f"Shape mismatch: {output_3d.shape} vs {expected_shape}"
        print(f"  性能_P0_3D: batch=8, seq=128, in=1024, out=4096, has_bias=True ... PASS")
    except Exception as e:
        print(f"  性能_P0_3D: ... FAIL - {e}")
        all_passed = False

    # 功能_P0_no_bias: in_features=512, out_features=512, has_bias=False
    # input: [batch, 512], weight: [512, 512]
    try:
        input_no_bias = torch.randn(16, 512, dtype=torch.float32)
        weight_no_bias = torch.randn(512, 512, dtype=torch.float32)
        output_no_bias = linear_golden(input_no_bias, weight_no_bias, bias=None)
        expected_shape = (16, 512)
        assert output_no_bias.shape == expected_shape, f"Shape mismatch: {output_no_bias.shape} vs {expected_shape}"
        print(f"  功能_P0_no_bias: batch=16, in=512, out=512, has_bias=False ... PASS")
    except Exception as e:
        print(f"  功能_P0_no_bias: ... FAIL - {e}")
        all_passed = False

    # 功能_P1_4D: in_features=256, out_features=512, has_bias=True
    # input: [batch, heads, seq_len, 256], weight: [512, 256], bias: [512]
    try:
        input_4d = torch.randn(2, 8, 64, 256, dtype=torch.float32)
        weight_4d = torch.randn(512, 256, dtype=torch.float32)
        bias_4d = torch.randn(512, dtype=torch.float32)
        output_4d = linear_golden(input_4d, weight_4d, bias_4d)
        expected_shape = (2, 8, 64, 512)
        assert output_4d.shape == expected_shape, f"Shape mismatch: {output_4d.shape} vs {expected_shape}"
        print(f"  功能_P1_4D: batch=2, heads=8, seq=64, in=256, out=512, has_bias=True ... PASS")
    except Exception as e:
        print(f"  功能_P1_4D: ... FAIL - {e}")
        all_passed = False

    # ==================== 2. 泛化 case 验证（来自 spec.md §8 动态轴范围）====================
    print("\n[泛化 case 验证]")

    # 动态轴范围: [1, INT32_MAX]，测试边界值
    # batch 采样: 1, 32, 128
    # seq_len 采样: 1, 64, 256

    # 最小 batch (batch=1)
    try:
        input_min = torch.randn(1, 128, dtype=torch.float32)
        weight_min = torch.randn(64, 128, dtype=torch.float32)
        output_min = linear_golden(input_min, weight_min)
        assert output_min.shape == (1, 64)
        print(f"  最小 batch: batch=1, in=128, out=64 ... PASS")
    except Exception as e:
        print(f"  最小 batch: ... FAIL - {e}")
        all_passed = False

    # 最小 seq_len (seq_len=1)
    try:
        input_min_seq = torch.randn(4, 1, 256, dtype=torch.float32)
        weight_min_seq = torch.randn(128, 256, dtype=torch.float32)
        output_min_seq = linear_golden(input_min_seq, weight_min_seq)
        assert output_min_seq.shape == (4, 1, 128)
        print(f"  最小 seq_len: batch=4, seq=1, in=256, out=128 ... PASS")
    except Exception as e:
        print(f"  最小 seq_len: ... FAIL - {e}")
        all_passed = False

    # 中等规模
    try:
        input_mid = torch.randn(32, 64, 512, dtype=torch.float32)
        weight_mid = torch.randn(256, 512, dtype=torch.float32)
        bias_mid = torch.randn(256, dtype=torch.float32)
        output_mid = linear_golden(input_mid, weight_mid, bias_mid)
        assert output_mid.shape == (32, 64, 256)
        print(f"  中等规模: batch=32, seq=64, in=512, out=256 ... PASS")
    except Exception as e:
        print(f"  中等规模: ... FAIL - {e}")
        all_passed = False

    # 大规模
    try:
        input_large = torch.randn(128, 256, 1024, dtype=torch.float32)
        weight_large = torch.randn(2048, 1024, dtype=torch.float32)
        bias_large = torch.randn(2048, dtype=torch.float32)
        output_large = linear_golden(input_large, weight_large, bias_large)
        assert output_large.shape == (128, 256, 2048)
        print(f"  大规模: batch=128, seq=256, in=1024, out=2048 ... PASS")
    except Exception as e:
        print(f"  大规模: ... FAIL - {e}")
        all_passed = False

    # ==================== 3. 值域检查（从公式推导）====================
    print("\n[值域检查]")

    # 零值输入
    try:
        input_zero = torch.zeros(4, 128, dtype=torch.float32)
        weight_zero = torch.randn(64, 128, dtype=torch.float32)
        bias_zero = torch.randn(64, dtype=torch.float32)
        output_zero = linear_golden(input_zero, weight_zero, bias_zero)
        # 零值输入时，输出应该等于 bias
        expected = bias_zero.unsqueeze(0).expand(4, -1)
        assert torch.allclose(output_zero, expected, atol=1e-6)
        print(f"  零值输入: output == bias ... PASS")
    except Exception as e:
        print(f"  零值输入: ... FAIL - {e}")
        all_passed = False

    # 无 bias 时零值输入
    try:
        input_zero_nb = torch.zeros(4, 128, dtype=torch.float32)
        weight_zero_nb = torch.randn(64, 128, dtype=torch.float32)
        output_zero_nb = linear_golden(input_zero_nb, weight_zero_nb, bias=None)
        assert torch.allclose(output_zero_nb, torch.zeros_like(output_zero_nb), atol=1e-6)
        print(f"  无 bias 零值输入: output == 0 ... PASS")
    except Exception as e:
        print(f"  无 bias 零值输入: ... FAIL - {e}")
        all_passed = False

    # ==================== 4. 数值稳定性检查 ====================
    print("\n[数值稳定性检查]")

    # 大值输入
    try:
        input_large_val = torch.randn(4, 128, dtype=torch.float32) * 1000
        weight_large_val = torch.randn(64, 128, dtype=torch.float32)
        bias_large_val = torch.randn(64, dtype=torch.float32)
        output_large_val = linear_golden(input_large_val, weight_large_val, bias_large_val)
        assert not torch.isnan(output_large_val).any(), "NaN detected in output"
        assert not torch.isinf(output_large_val).any(), "Inf detected in output"
        print(f"  大值输入 (x1000): 无 NaN/Inf ... PASS")
    except Exception as e:
        print(f"  大值输入: ... FAIL - {e}")
        all_passed = False

    # 小值输入
    try:
        input_small_val = torch.randn(4, 128, dtype=torch.float32) * 1e-6
        weight_small_val = torch.randn(64, 128, dtype=torch.float32)
        output_small_val = linear_golden(input_small_val, weight_small_val)
        assert not torch.isnan(output_small_val).any(), "NaN detected in output"
        print(f"  小值输入 (x1e-6): 无 NaN ... PASS")
    except Exception as e:
        print(f"  小值输入: ... FAIL - {e}")
        all_passed = False

    # ==================== 5. API 对比（与 torch.nn.functional.linear 对比）====================
    print("\n[API 对比]")

    try:
        input_cmp = torch.randn(8, 64, 256, dtype=torch.float32)
        weight_cmp = torch.randn(512, 256, dtype=torch.float32)
        bias_cmp = torch.randn(512, dtype=torch.float32)

        # 我们的实现
        output_ours = linear_golden(input_cmp, weight_cmp, bias_cmp)
        # PyTorch 内置实现
        output_torch = torch.nn.functional.linear(input_cmp, weight_cmp, bias_cmp)

        assert torch.allclose(output_ours, output_torch, atol=1e-6, rtol=1e-6)
        print(f"  与 torch.nn.functional.linear 对比: 一致 ... PASS")
    except Exception as e:
        print(f"  API 对比: ... FAIL - {e}")
        all_passed = False

    # ==================== 6. 函数签名检查 ====================
    print("\n[函数签名检查]")

    try:
        import inspect
        sig = inspect.signature(linear_golden)
        params = list(sig.parameters.keys())
        assert 'input' in params, "Missing 'input' parameter"
        assert 'weight' in params, "Missing 'weight' parameter"
        assert 'bias' in params, "Missing 'bias' parameter"
        print(f"  函数签名: {sig} ... PASS")
    except Exception as e:
        print(f"  函数签名: ... FAIL - {e}")
        all_passed = False

    # ==================== 最终结果 ====================
    print("\n" + "=" * 60)
    if all_passed:
        print("PASS 所有验证通过")
    else:
        print("FAIL 部分验证失败，请检查上述报告")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _validate()
