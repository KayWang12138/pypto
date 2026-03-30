#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY kind, either express or implied,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
avg_pool2d Golden 参考实现
公式: output[n, c, oh, ow] = (1 / (k_h * k_w)) * sum_{i=0}^{k_h-1} sum_{j=0}^{k_w-1} input[n, c, oh*s_h+i, ow*s_w+j]
置信度: (基于 PyTorch 内置 API)
"""

import torch
import numpy as np
from typing import Tuple, Optional
import math


def avg_pool2d_golden(
    x: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]] = None,
    padding_mode: str = 'SAME',
) -> torch.Tensor:
    """
    avg_pool2d 参考实现 (PyTorch)

    实现 2D 平均池化操作，支持 SAME 和 VALID 两种填充模式。

    Args:
        x: 输入张量, shape: [batch_size, channels, in_h, in_w]
        kernel_size: 池化窗口大小 (k_h, k_w)
        stride: 步长 (s_h, s_w), 默认等于 kernel_size
        padding_mode: 填充模式: 'SAME' 或 'VALID'
        eps: 数值稳定性参数,默认 1e-5

    Returns:
        输出张量, shape: [batch_size, channels, out_h, out_w]

    公式:
        output[n, c, oh, ow] = (1 / (k_h * k_w)) * sum_{i=0}^{k_h-1} sum_{j=0}^{k_w-1} input[n, c, oh*s_h+i, ow*s_w+j]
    """
    # 处理步长
    if stride is None:
        stride = kernel_size
    k_h, k_w = kernel_size
    s_h, s_w = stride

    batch_size, channels, in_h, in_w = x.shape

    # 计算输出尺寸和 padding
    if padding_mode.upper() == 'VALID':
        out_h = (in_h - k_h + s_h) // s_h
        out_w = (in_w - k_w + s_w) // s_w
        padding = 0
    elif padding_mode.upper() == 'SAME':
        out_h = math.ceil(in_h / s_h)
        out_w = math.ceil(in_w / s_w)
        # 计算 SAME padding (参考 TensorFlow 的实现)
        pad_h = max(0, (out_h - 1) * s_h + k_h - in_h)
        pad_w = max(0, (out_w - 1) * s_w + k_w - in_w)
        t_pad = pad_h // 2
        b_pad = pad_h - t_pad
        l_pad = pad_w // 2
        r_pad = pad_w - l_pad
        padding = (t_pad, l_pad)  # PyTorch avg_pool2d 的 padding 格式是 (pad_h, pad_w)
    else:
        raise ValueError(f"Invalid padding_mode: {padding_mode}. Must be 'SAME' or 'VALID'")

    # 使用 PyTorch 内置 API
    # 注意: PyTorch avg_pool2d 的 padding 参数格式是 (pad_h, pad_w) 或单个整数
    # SAME 模式需要 ceil_mode=True 来获得正确的输出尺寸 (ceil(in_h / stride))
    # VALID 模式使用 floor 模式
    ceil_mode = (padding_mode.upper() == 'SAME')
    return torch.nn.functional.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode
    )


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""
    from numpy.testing import assert_allclose
    import sys

    print("=" * 60)
    print("avg_pool2d_golden 验证报告")
    print("=" * 60)

    # 1. 典型 case 验证（来自 spec.md 典型配置）
    print("\n[典型 case 验证]")
    test_cases = [
        {
            "name": "SAME_P0",
            "shape": (2, 3, 6, 6),
            "kernel_size": (2, 2),
            "stride": (2, 2),
            "padding_mode": "SAME",
            "expected_out_shape": (2, 3, 3, 3),
        },
        {
            "name": "VALID_P0",
            "shape": (4, 8, 12, 12),
            "kernel_size": (3, 3),
            "stride": (2, 2),
            "padding_mode": "VALID",
            "expected_out_shape": (4, 8, 5, 5),
        },
        {
            "name": "SAME_large",
            "shape": (8, 64, 56, 56),
            "kernel_size": (3, 3),
            "stride": (2, 2),
            "padding_mode": "SAME",
            "expected_out_shape": (8, 64, 28, 28),
        },
        {
            "name": "VALID_stride1",
            "shape": (2, 16, 8, 8),
            "kernel_size": (2, 2),
            "stride": (1, 1),
            "padding_mode": "VALID",
            "expected_out_shape": (2, 16, 7, 7),
        },
    ]
    all_passed = True
    for case in test_cases:
        try:
            x = torch.randn(case["shape"], dtype=torch.float32)
            out = avg_pool2d_golden(
                x,
                case["kernel_size"],
                case["stride"],
                case["padding_mode"],
            )
            # 检查 shape
            expected_shape = case["expected_out_shape"]
            if out.shape != expected_shape:
                print(f"  {case['name']}: FAILED - shape mismatch. Expected {expected_shape}, got {out.shape}")
                all_passed = False
                continue
            print(f"  {case['name']}: PASSED - shape {out.shape}")
        except Exception as e:
            print(f"  {case['name']}: ERROR - {e}")
            all_passed = False
    # 2. 边界条件验证
    print("\n[边界条件验证]")
    # 零值输入
    try:
        x = torch.zeros((2, 3, 6, 6), dtype=torch.float32)
        out = avg_pool2d_golden(x, (2, 2), (2, 2), "SAME")
        if out.abs().max() < 1e-6:
            print("  零值输入: PASSED")
        else:
            print(f"  零值输入: WARNING - 输出非零: {out.abs().max().item():.6f}")
    except Exception as e:
        print(f"  零值输入: ERROR - {e}")
        all_passed = False
    # 3. 值域检查
    print("\n[值域检查]")
    try:
        x = torch.randn((2, 3, 6, 6), dtype=torch.float32)
        out = avg_pool2d_golden(x, (2, 2), (2, 2), "SAME")
        # 平均池化输出应在输入值域内
        if out.min() >= x.min() and out.max() <= x.max():
            print("  输出值域在输入值域内: PASSED")
        else:
            print(f"  输出值域: WARNING - 可能存在数值问题")
    except Exception as e:
        print(f"  值域检查: ERROR - {e}")
        all_passed = False
    # 4. 数值稳定性检查
    print("\n[数值稳定性检查]")
    try:
        x = torch.randn((2, 3, 6, 6), dtype=torch.float32) * 100
        out = avg_pool2d_golden(x, (2, 2), (2, 2), "SAME")
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("  数值稳定性: FAILED - 存在 NaN 或 Inf")
            all_passed = False
        else:
            print("  数值稳定性: PASSED")
    except Exception as e:
        print(f"  数值稳定性: ERROR - {e}")
        all_passed = False
    # 最终报告
    print("\n" + "=" * 60)
    if all_passed:
        print("所有验证通过")
    else:
        print("部分验证失败,请检查上述错误")
    print("=" * 60)
if __name__ == "__main__":
    _validate()
