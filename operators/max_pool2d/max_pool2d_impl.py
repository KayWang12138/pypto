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

"""PyPTO max_pool2d kernel implementation.

max_pool2d 算子实现，支持：
  - kernel_size (int 或 tuple)
  - stride (int 或 tuple，默认等于 kernel_size)
  - padding (int 或 tuple)
  - dilation (int 或 tuple，默认 1)
  - ceil_mode (bool，默认 False)
  - float16 和 float32 数据类型
  - 3D (C, H, W) 和 4D (N, C, H, W) 输入
  - 动态轴 (batch, H, W)

注意: 当前版本不支持 dilation > 1 的情况，因为 PyPTO 切片不支持步长。
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import pypto
import torch


@dataclass(frozen=True)
class MaxPool2dParams:
    """max_pool2d 参数"""
    batch_size: int
    channels: int
    in_h: int
    in_w: int
    k_h: int
    k_w: int
    s_h: int
    s_w: int
    t_pad: int
    b_pad: int
    l_pad: int
    r_pad: int
    d_h: int
    d_w: int
    out_h: int
    out_w: int


def calculate_max_pool2d_params(
    shape: Tuple[int, ...],
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    ceil_mode: bool = False,
) -> MaxPool2dParams:
    """计算 max_pool2d 参数"""
    # 处理 shape (支持 3D 和 4D)
    if len(shape) == 3:
        batch_size, channels, in_h, in_w = 1, shape[0], shape[1], shape[2]
    else:
        batch_size, channels, in_h, in_w = shape

    # 处理 kernel_size
    if isinstance(kernel_size, int):
        k_h, k_w = kernel_size, kernel_size
    else:
        k_h, k_w = kernel_size

    # 处理 stride
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        s_h, s_w = stride, stride
    else:
        s_h, s_w = stride

    # 处理 padding
    if isinstance(padding, int):
        t_pad = b_pad = l_pad = r_pad = padding
    elif len(padding) == 2:
        l_pad, r_pad = padding[1], padding[1]
        t_pad, b_pad = padding[0], padding[0]
    else:  # 4-tuple
        l_pad, r_pad, t_pad, b_pad = padding

    # 处理 dilation
    if isinstance(dilation, int):
        d_h, d_w = dilation, dilation
    else:
        d_h, d_w = dilation

    # 计算输出大小
    effective_k_h = (k_h - 1) * d_h + 1
    effective_k_w = (k_w - 1) * d_w + 1

    if ceil_mode:
        out_h = math.ceil((in_h + t_pad + b_pad - effective_k_h) / s_h + 1)
        out_w = math.ceil((in_w + l_pad + r_pad - effective_k_w) / s_w + 1)
    else:
        out_h = math.floor((in_h + t_pad + b_pad - effective_k_h) / s_h + 1)
        out_w = math.floor((in_w + l_pad + r_pad - effective_k_w) / s_w + 1)

    return MaxPool2dParams(
        batch_size=batch_size,
        channels=channels,
        in_h=in_h,
        in_w=in_w,
        k_h=k_h,
        k_w=k_w,
        s_h=s_h,
        s_w=s_w,
        t_pad=t_pad,
        b_pad=b_pad,
        l_pad=l_pad,
        r_pad=r_pad,
        d_h=d_h,
        d_w=d_w,
        out_h=out_h,
        out_w=out_w,
    )


def max_pool2d(
    shape: Tuple[int, ...],
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    ceil_mode: bool = False,
    run_mode: str = "npu",
    dynamic: bool = True,
    dtype: pypto.DataType = pypto.DT_FP32,
):
    """创建 max_pool2d kernel

    注意: 当前仅支持 dilation=1 的情况
    """
    params = calculate_max_pool2d_params(shape, kernel_size, stride, padding, dilation, ceil_mode)

    batch_size = params.batch_size
    channels = params.channels
    in_h = params.in_h
    in_w = params.in_w
    k_h = params.k_h
    k_w = params.k_w
    s_h = params.s_h
    s_w = params.s_w
    t_pad = params.t_pad
    b_pad = params.b_pad
    l_pad = params.l_pad
    r_pad = params.r_pad
    d_h = params.d_h
    d_w = params.d_w
    out_h = params.out_h
    out_w = params.out_w

    # 检查 dilation 支持
    if d_h > 1 or d_w > 1:
        raise ValueError("max_pool2d with dilation > 1 is not supported in this version. "
                        "PyPTO tensor slicing does not support stride parameter.")

    if dynamic:
        batch_size = pypto.frontend.dynamic("batch_size")
        channels = pypto.frontend.dynamic("channels")

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(
        pass_options={"vec_nbuffer_setting": {-1: 2, 0: 8}},
        runtime_options={
            "run_mode": mode,
            "stitch_function_num_initial": 128,
            "stitch_function_outcast_memory": 1024,
            "stitch_function_inner_memory": 1024,
        },
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1},
    )
    def max_pool2d_kernel(
        input_tensor: pypto.Tensor((batch_size, channels, in_h, in_w), dtype),
        output_result: pypto.Tensor((batch_size, channels, out_h, out_w), dtype),
    ):
        bc_total = batch_size * channels
        pypto.set_vec_tile_shapes(16, 16, 4, 128)
        input_reshaped = pypto.reshape(input_tensor, [batch_size * channels, in_h, in_w], inplace=True)
        output_tmp = pypto.tensor((bc_total, out_h, out_w), dtype)

        for bc_idx, unroll_length in pypto.loop_unroll(
            0, bc_total, 1,
            name="LOOP_BC",
            idx_name="bc_idx",
            unroll_list=[8, 4, 2, 1]
        ):
            input_cur = input_reshaped[bc_idx: bc_idx + unroll_length, :, :]

            for oh in range(out_h):
                # 计算输入窗口范围
                h_start = oh * s_h - t_pad
                h_end = h_start + k_h

                # 边界 clamp
                h_start_clamped = max(h_start, 0)
                h_end_clamped = min(h_end, in_h)
                cur_k_h = h_end_clamped - h_start_clamped

                pypto.set_vec_tile_shapes(16, 16, 128)

                if cur_k_h > 0:
                    # 提取 H 方向窗口
                    input_h_window = input_cur[:, h_start_clamped:h_end_clamped, :]
                    # 对 H 维度做 max 归约
                    window_h_max = pypto.amax(input_h_window, dim=1, keepdim=True)
                else:
                    window_h_max = None

                for ow in range(out_w):
                    # 计算输入窗口范围
                    w_start = ow * s_w - l_pad
                    w_end = w_start + k_w

                    # 边界 clamp
                    w_start_clamped = max(w_start, 0)
                    w_end_clamped = min(w_end, in_w)
                    cur_k_w = w_end_clamped - w_start_clamped

                    if cur_k_h > 0 and cur_k_w > 0 and window_h_max is not None:
                        # 提取 W 方向窗口
                        window = window_h_max[:, :, w_start_clamped:w_end_clamped]
                        # 对 W 维度做 max 归约
                        max_val = pypto.amax(window, dim=2, keepdim=True)

                        pypto.set_vec_tile_shapes(unroll_length, 4, 128)
                        pypto.assemble(max_val, [bc_idx, oh, ow], output_tmp)
                    else:
                        # 窗口无效时，填充极小值
                        min_val = pypto.full([unroll_length, 1, 1], -1e38, dtype=dtype)
                        pypto.set_vec_tile_shapes(unroll_length, 4, 128)
                        pypto.assemble(min_val, [bc_idx, oh, ow], output_tmp)

            pypto.set_vec_tile_shapes(unroll_length, 4, 128)
            output_result.move(pypto.reshape(output_tmp, [batch_size, channels, out_h, out_w], inplace=True))

    return max_pool2d_kernel


def max_pool2d_wrapper(
    x: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    ceil_mode: bool = False,
    run_mode: str = "npu",
) -> torch.Tensor:
    """max_pool2d wrapper 函数

    Args:
        x: 输入 tensor，shape 为 (N, C, H_in, W_in) 或 (C, H_in, W_in)
        kernel_size: 池化窗口大小
        stride: 池化步长，默认等于 kernel_size
        padding: 填充大小
        dilation: 空洞率 (注意: 当前仅支持 dilation=1)
        ceil_mode: 是否使用 ceil 模式
        run_mode: 运行模式 ("npu" 或 "sim")

    Returns:
        输出 tensor，shape 为 (N, C, H_out, W_out) 或 (C, H_out, W_out)
    """
    # 确定输入维度
    input_dim = x.dim()
    if input_dim == 3:
        # 3D 输入 (C, H, W) -> 添加 batch 维度
        x_4d = x.unsqueeze(0)
        shape = (1, x.shape[0], x.shape[1], x.shape[2])
    else:
        x_4d = x
        shape = tuple(x.shape)

    # 确定 dtype
    if x.dtype == torch.float16:
        dtype = pypto.DT_FP16
    else:
        dtype = pypto.DT_FP32

    # 计算输出参数
    params = calculate_max_pool2d_params(shape, kernel_size, stride, padding, dilation, ceil_mode)

    # 创建输出 tensor
    if input_dim == 3:
        output_shape = (params.channels, params.out_h, params.out_w)
    else:
        output_shape = (params.batch_size, params.channels, params.out_h, params.out_w)

    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    # 创建并执行 kernel
    kernel = max_pool2d(
        shape=shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        run_mode=run_mode,
        dynamic=True,
        dtype=dtype,
    )

    kernel(x_4d if input_dim == 4 else x.unsqueeze(0), output if input_dim == 4 else output.unsqueeze(0))

    if input_dim == 3:
        output = output.squeeze(0)

    return output
