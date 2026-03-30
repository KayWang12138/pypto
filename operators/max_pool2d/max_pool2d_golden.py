#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE; IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""max_pool2d Golden 参考实现

公式: output[n, c, h, w] = max_{i,j}(input[n, c, h*s_h + i*d_h - p_h, w*s_w + j*d_w - p_w])
置信度: (使用 PyTorch 内置 API)

支持参数:
  - kernel_size: 池化窗口大小 (int 或 tuple)
  - stride: 步长 (int 或 tuple)，默认等于 kernel_size
  - padding: 填充 (int 或 tuple)，默认 0
  - dilation: 空洞率 (int 或 tuple)，默认 1
  - ceil_mode: 是否使用 ceil 模式计算输出大小，默认 False
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union


def max_pool2d_golden(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    ceil_mode: bool = False,
) -> torch.Tensor:
    """max_pool2d 参考实现 (PyTorch)

    对输入 tensor 在空间维度（H 和 W）上进行最大池化操作。
    对于输入 tensor 的每个滑动窗口，输出窗口内的最大值。

    Args:
        input: 输入 tensor，shape 为 (N, C, H_in, W_in) 或 (C, H_in, W_in)
               支持 float16 和 float32 数据类型
        kernel_size: 池化窗口大小，可以是单个 int 或 (int, int) 元组
        stride: 池化步长，默认等于 kernel_size，可以是单个 int 或 (int, int) 元组
        padding: 填充大小，默认为 0，可以是单个 int 或 (int, int) 元组
        dilation: 空洞率，默认为 1，可以是单个 int 或 (int, int) 元组
        ceil_mode: 是否使用 ceil 模式计算输出大小，默认为 False

    Returns:
        output: 池化输出 tensor，shape 为 (N, C, H_out, W_out) 或 (C, H_out, W_out)
                dtype 与输入相同

    公式:
        output[n, c, h, w] = max_{i in [0, kH), j in [0, kW)}
                             input[n, c, h*stride_h + i*dilation_h - padding_h,
                                   w*stride_w + j*dilation_w - padding_w]

    输出大小计算:
        if ceil_mode:
            H_out = ceil((H_in + 2*padding_h - dilation_h*(kernel_h-1) - 1) / stride_h + 1)
            W_out = ceil((W_in + 2*padding_w - dilation_w*(kernel_w-1) - 1) / stride_w + 1)
        else:
            H_out = floor((H_in + 2*padding_h - dilation_h*(kernel_h-1) - 1) / stride_h + 1)
            W_out = floor((H_in + 2*padding_w - dilation_w*(kernel_w-1) - 1) / stride_w + 1)
    """
    # 记录原始输入维度，用于确定输出维度
    input_dim = input.dim()

    # 如果是 3D 输入 (C, H, W)，添加 batch 维度
    if input_dim == 3:
        input = input.unsqueeze(0)

    # 调用 PyTorch 内置 max_pool2d
    output = F.max_pool2d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    # 如果原始输入是 3D，移除 batch 维度
    if input_dim == 3:
        output = output.squeeze(0)

    return output


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""
    import math

    print("=" * 60)
    print("max_pool2d_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # ==================== 典型 case 验证 ====================
    print("\n[典型 case 验证]")

    # 性能_P0: resnet_pool
    print("  性能_P0 (resnet_pool): kernel=3, stride=2, padding=1 ...", end=" ")
    try:
        x = torch.randn(32, 64, 112, 112, dtype=torch.float32)
        y = max_pool2d_golden(x, kernel_size=3, stride=2, padding=1)
        expected_shape = (32, 64, 56, 56)
        assert y.shape == expected_shape, f"shape mismatch: {y.shape} vs {expected_shape}"
        # 验证与 PyTorch 直接调用一致
        y_torch = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        assert torch.allclose(y, y_torch, atol=1e-5, rtol=1e-5), "结果与 PyTorch 不一致"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    # 性能_P0: vgg_pool
    print("  性能_P0 (vgg_pool): kernel=2, stride=2, padding=0 ...", end=" ")
    try:
        x = torch.randn(32, 128, 56, 56, dtype=torch.float32)
        y = max_pool2d_golden(x, kernel_size=2, stride=2, padding=0)
        expected_shape = (32, 128, 28, 28)
        assert y.shape == expected_shape, f"shape mismatch: {y.shape} vs {expected_shape}"
        y_torch = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        assert torch.allclose(y, y_torch, atol=1e-5, rtol=1e-5), "结果与 PyTorch 不一致"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    # 功能_P0: dynamic_batch (测试不同 batch size)
    print("  功能_P0 (dynamic_batch): 不同 batch size ...", end=" ")
    try:
        for batch in [1, 16, 64]:
            x = torch.randn(batch, 64, 224, 224, dtype=torch.float32)
            y = max_pool2d_golden(x, kernel_size=3, stride=2, padding=1)
            expected_h = (224 + 2 - 3) // 2 + 1  # 112
            expected_w = (224 + 2 - 3) // 2 + 1  # 112
            assert y.shape == (batch, 64, expected_h, expected_w), f"shape mismatch for batch={batch}"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    # 功能_P0: dynamic_hw (测试不同 H/W)
    print("  功能_P0 (dynamic_hw): 不同 H/W ...", end=" ")
    try:
        for h, w in [(32, 32), (64, 64), (128, 256)]:
            x = torch.randn(16, 32, h, w, dtype=torch.float32)
            y = max_pool2d_golden(x, kernel_size=2, stride=2, padding=0)
            expected_h, expected_w = h // 2, w // 2
            assert y.shape == (16, 32, expected_h, expected_w), f"shape mismatch for h={h}, w={w}"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    # 功能_P1: float16_test
    print("  功能_P1 (float16_test): float16 dtype ...", end=" ")
    try:
        x = torch.randn(16, 64, 56, 56, dtype=torch.float16)
        y = max_pool2d_golden(x, kernel_size=3, stride=2, padding=1)
        assert y.dtype == torch.float16, f"dtype mismatch: {y.dtype}"
        expected_shape = (16, 64, 28, 28)
        assert y.shape == expected_shape, f"shape mismatch: {y.shape} vs {expected_shape}"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    # 功能_P1: dilation_test
    # 注意：PyTorch 要求 padding <= kernel_size // 2，对于 dilation 需要调整参数
    print("  功能_P1 (dilation_test): dilation=2 ...", end=" ")
    try:
        x = torch.randn(8, 32, 32, 32, dtype=torch.float32)
        # 使用 kernel=5, padding=2, dilation=2 来满足约束
        # 有效 kernel 大小 = (kernel-1)*dilation + 1 = 9
        y = max_pool2d_golden(x, kernel_size=5, stride=1, padding=2, dilation=2)
        expected_shape = (8, 32, 28, 28)
        assert y.shape == expected_shape, f"shape mismatch: {y.shape} vs {expected_shape}"
        y_torch = F.max_pool2d(x, kernel_size=5, stride=1, padding=2, dilation=2)
        assert torch.allclose(y, y_torch, atol=1e-5, rtol=1e-5), "结果与 PyTorch 不一致"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    # 功能_P1: ceil_mode_test
    print("  功能_P1 (ceil_mode_test): ceil_mode=True ...", end=" ")
    try:
        x = torch.randn(8, 32, 7, 7, dtype=torch.float32)
        y = max_pool2d_golden(x, kernel_size=2, stride=2, padding=0, ceil_mode=True)
        expected_shape = (8, 32, 4, 4)
        assert y.shape == expected_shape, f"shape mismatch: {y.shape} vs {expected_shape}"
        y_torch = F.max_pool2d(x, kernel_size=2, stride=2, padding=0, ceil_mode=True)
        assert torch.allclose(y, y_torch, atol=1e-5, rtol=1e-5), "结果与 PyTorch 不一致"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    # ==================== 3D 输入测试 ====================
    print("\n[3D 输入验证]")

    print("  3D input (C, H, W) ...", end=" ")
    try:
        x = torch.randn(64, 112, 112, dtype=torch.float32)
        y = max_pool2d_golden(x, kernel_size=3, stride=2, padding=1)
        expected_shape = (64, 56, 56)
        assert y.shape == expected_shape, f"shape mismatch: {y.shape} vs {expected_shape}"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    # ==================== 值域检查 ====================
    print("\n[值域检查]")

    print("  输出值应在输入值范围内 ...", end=" ")
    try:
        x = torch.randn(4, 16, 32, 32, dtype=torch.float32)
        y = max_pool2d_golden(x, kernel_size=3, stride=2, padding=1)
        assert y.min() >= x.min(), f"输出最小值 {y.min()} 小于输入最小值 {x.min()}"
        assert y.max() <= x.max(), f"输出最大值 {y.max()} 大于输入最大值 {x.max()}"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    # ==================== 数值稳定性检查 ====================
    print("\n[数值稳定性检查]")

    print("  大值输入 ...", end=" ")
    try:
        x = torch.randn(4, 16, 32, 32, dtype=torch.float32) * 100
        y = max_pool2d_golden(x, kernel_size=3, stride=2, padding=1)
        assert not torch.isnan(y).any(), "输出包含 NaN"
        assert not torch.isinf(y).any(), "输出包含 Inf"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    print("  小值输入 ...", end=" ")
    try:
        x = torch.randn(4, 16, 32, 32, dtype=torch.float32) * 1e-5
        y = max_pool2d_golden(x, kernel_size=3, stride=2, padding=1)
        assert not torch.isnan(y).any(), "输出包含 NaN"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    # ==================== API 对比 ====================
    print("\n[API 对比]")

    print("  与 torch.nn.functional.max_pool2d 一致性 ...", end=" ")
    try:
        test_cases = [
            ((8, 32, 64, 64), 3, 2, 1, 1, False),
            ((4, 16, 32, 32), 2, 2, 0, 1, False),
            ((2, 8, 16, 16), 3, 1, 1, 2, False),
            ((4, 16, 15, 15), 2, 2, 0, 1, True),
        ]
        for shape, k, s, p, d, ceil in test_cases:
            x = torch.randn(*shape, dtype=torch.float32)
            y_golden = max_pool2d_golden(x, kernel_size=k, stride=s, padding=p, dilation=d, ceil_mode=ceil)
            y_torch = F.max_pool2d(x, kernel_size=k, stride=s, padding=p, dilation=d, ceil_mode=ceil)
            assert torch.allclose(y_golden, y_torch, atol=1e-5, rtol=1e-5), \
                f"shape={shape}, k={k}, s={s}, p={p}, d={d}, ceil={ceil}"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    # ==================== 边界条件测试 ====================
    print("\n[边界条件测试]")

    print("  全零输入 ...", end=" ")
    try:
        x = torch.zeros(4, 16, 32, 32, dtype=torch.float32)
        y = max_pool2d_golden(x, kernel_size=3, stride=2, padding=1)
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-6), "全零输入应输出全零"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    print("  单元素 kernel ...", end=" ")
    try:
        x = torch.randn(4, 16, 32, 32, dtype=torch.float32)
        y = max_pool2d_golden(x, kernel_size=1, stride=1, padding=0)
        assert torch.allclose(y, x, atol=1e-6), "kernel_size=1, stride=1 应输出与输入相同"
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("所有验证通过")
    else:
        print("部分验证失败，请检查")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _validate()
