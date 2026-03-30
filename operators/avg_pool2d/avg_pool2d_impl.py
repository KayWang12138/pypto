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
avg_pool2d PyPTO kernel implementation.

实现 2D 平均池化操作，支持 SAME 和 VALID 两种填充模式。
参考: models/experimental/vector/AvgPool2d/avg_pool2d.py
"""

import os
import pypto
import torch
from typing import Tuple, Optional


def get_run_mode():
    """获取运行模式。"""
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        return pypto.RunMode.SIM
    return pypto.RunMode.NPU


def calculate_pool2d_params(
    batch_size: int,
    channels: int,
    in_h: int,
    in_w: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding_mode: str
) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
    """
    计算池化参数。

    Returns:
        (t_pad, b_pad, l_pad, r_pad, out_h, out_w, k_h, k_w, s_h, s_w)
    """
    k_h, k_w = kernel_size
    s_h, s_w = stride

    if padding_mode.upper() == 'VALID':
        t_pad = b_pad = l_pad = r_pad = 0
        out_h = (in_h - k_h + s_h) // s_h
        out_w = (in_w - k_w + s_w) // s_w
    elif padding_mode.upper() == 'SAME':
        out_h = (in_h + s_h - 1) // s_h
        out_w = (in_w + s_w - 1) // s_w
        pad_h = max(0, (out_h - 1) * s_h + k_h - in_h)
        pad_w = max(0, (out_w - 1) * s_w + k_w - in_w)
        t_pad = pad_h // 2
        b_pad = pad_h - t_pad
        l_pad = pad_w // 2
        r_pad = pad_w - l_pad
    else:
        raise ValueError(f"Invalid padding_mode: {padding_mode}. Must be 'SAME' or 'VALID'")

    return t_pad, b_pad, l_pad, r_pad, out_h, out_w, k_h, k_w, s_h, s_w


def create_avg_pool2d_kernel(
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding_mode: str,
    batch_size: int,
    channels: int,
    in_h: int,
    in_w: int,
    dynamic: bool = True
):
    """
    创建 avg_pool2d kernel。

    Args:
        kernel_size: 池化窗口大小 (k_h, k_w)
        stride: 步长 (s_h, s_w)
        padding_mode: 填充模式 'SAME' 或 'VALID'
        batch_size: 批次大小
        channels: 通道数
        in_h: 输入高度
        in_w: 输入宽度
        dynamic: 是否使用动态轴
    """
    t_pad, b_pad, l_pad, r_pad, out_h, out_w, k_h, k_w, s_h, s_w = calculate_pool2d_params(
        batch_size, channels, in_h, in_w, kernel_size, stride, padding_mode
    )

    # 动态轴定义
    if dynamic:
        bs = pypto.frontend.dynamic("batch_size")
        cs = pypto.frontend.dynamic("channels")
    else:
        bs = batch_size
        cs = channels

    run_mode = get_run_mode()

    @pypto.frontend.jit(
        pass_options={"vec_nbuffer_setting": {-1: 2, 0: 8}},
        runtime_options={
            "run_mode": run_mode,
            "stitch_function_num_initial": 128,
            "stitch_function_outcast_memory": 1024,
            "stitch_function_inner_memory": 1024
        },
        debug_options=dict(runtime_debug_mode=1, compile_debug_mode=1)
    )
    def avg_pool2d_kernel(
        input_tensor: pypto.Tensor((bs, cs, in_h, in_w), pypto.DT_FP32),
        output_tensor: pypto.Tensor((bs, cs, out_h, out_w), pypto.DT_FP32),
    ):
        bc_total = bs * cs
        pypto.set_vec_tile_shapes(16, 16, 4, 128)
        input_reshaped = pypto.reshape(input_tensor, [bs * cs, in_h, in_w], inplace=True)
        output_tmp = pypto.tensor((bc_total, out_h, out_w), pypto.DT_FP32)

        for bc_idx, unroll_length in pypto.loop_unroll(
            0, bc_total, 1,
            name="LOOP_BC",
            idx_name="bc_idx",
            unroll_list=[8, 4, 2, 1]
        ):
            input_cur = input_reshaped[bc_idx: bc_idx + unroll_length, :, :]

            for oh in range(out_h):
                h_start = oh * s_h - t_pad
                h_end = h_start + k_h
                h_start_clamped = max(h_start, 0)
                h_end_clamped = min(h_end, in_h)
                cur_k_h = h_end_clamped - h_start_clamped

                pypto.set_vec_tile_shapes(16, 16, 128)
                if cur_k_h > 0:
                    input_single_row = input_cur[:, h_start_clamped:h_end_clamped, :]
                else:
                    input_single_row = None

                input_single_row_1 = pypto.sum(input_single_row, 1, keepdim=True)

                for ow in range(out_w):
                    w_start = ow * s_w - l_pad
                    w_end = w_start + k_w
                    w_start_clamped = max(w_start, 0)
                    w_end_clamped = min(w_end, in_w)
                    cur_k_w = w_end_clamped - w_start_clamped

                    if cur_k_h > 0 and cur_k_w > 0:
                        window = input_single_row_1[:, :, w_start_clamped:w_end_clamped]
                        sum_val = pypto.sum(window, dim=2, keepdim=True)
                        avg_val = sum_val / (k_h * k_w)
                        pypto.set_vec_tile_shapes(unroll_length, 4, 128)
                        pypto.assemble(avg_val, [bc_idx, oh, ow], output_tmp)
                    else:
                        zero_val = pypto.zeros([unroll_length, 1, 1], dtype=pypto.DT_FP32)
                        pypto.set_vec_tile_shapes(unroll_length, 4, 128)
                        pypto.assemble(zero_val, [bc_idx, oh, ow], output_tmp)

            pypto.set_vec_tile_shapes(unroll_length, 4, 128)
            output_tensor.move(pypto.reshape(output_tmp, [bs, cs, out_h, out_w], inplace=True))

    return avg_pool2d_kernel


def avg_pool2d_wrapper(
    x: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]] = None,
    padding_mode: str = 'SAME',
) -> torch.Tensor:
    """
    avg_pool2d wrapper，供 test_avg_pool2d.py 调用。

    负责：
    1. 计算输出 shape
    2. 构造输出 torch.Tensor
    3. 调用 JIT kernel
    4. 返回结果 torch.Tensor

    Args:
        x: 输入 torch.Tensor, shape: [batch_size, channels, in_h, in_w]
        kernel_size: 池化窗口大小 (k_h, k_w)
        stride: 步长 (s_h, s_w), 默认等于 kernel_size
        padding_mode: 填充模式: 'SAME' 或 'VALID'

    Returns:
        输出 torch.Tensor, shape: [batch_size, channels, out_h, out_w]
    """
    if stride is None:
        stride = kernel_size

    batch_size, channels, in_h, in_w = x.shape

    # 计算输出 shape
    t_pad, b_pad, l_pad, r_pad, out_h, out_w, _, _, _, _ = calculate_pool2d_params(
        batch_size, channels, in_h, in_w, kernel_size, stride, padding_mode
    )

    # 创建输出 tensor
    output = torch.empty(batch_size, channels, out_h, out_w, dtype=x.dtype, device=x.device)

    # 创建并调用 kernel
    kernel = create_avg_pool2d_kernel(
        kernel_size=kernel_size,
        stride=stride,
        padding_mode=padding_mode,
        batch_size=batch_size,
        channels=channels,
        in_h=in_h,
        in_w=in_w,
        dynamic=True
    )

    kernel(x, output)

    return output
