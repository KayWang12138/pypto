#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
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
adaptive_avg_pool2d PyPTO 实现

自适应平均池化，将任意尺寸的输入池化到指定的输出尺寸。
池化窗口的大小和位置根据输入和输出尺寸动态计算。

参考: models/experimental/vector/AvgPool2d/avg_pool2d.py
"""

import pypto
import torch
from typing import Union, Tuple


# ─────────────────────────────────────────────
# JIT Kernel 工厂函数
# ─────────────────────────────────────────────

def create_adaptive_avg_pool2d_kernel(
    batch_size: int,
    channels: int,
    in_h: int,
    in_w: int,
    out_h: int,
    out_w: int,
    dynamic: bool = True,
    run_mode: str = "npu",
):
    """创建 adaptive_avg_pool2d kernel。

    Args:
        batch_size: batch 维度大小
        channels: channel 维度大小
        in_h: 输入高度
        in_w: 输入宽度
        out_h: 输出高度
        out_w: 输出宽度
        dynamic: 是否启用动态轴
        run_mode: 运行模式 ("npu" 或 "sim")
    """
    # 动态轴声明
    if dynamic:
        batch_size = pypto.frontend.dynamic("batch_size")
        channels = pypto.frontend.dynamic("channels")

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")

    @pypto.frontend.jit(
        pass_options={"vec_nbuffer_setting": {-1: 2, 0: 8}},
        runtime_options={
            "run_mode": mode,
            "stitch_function_num_initial": 128,
            "stitch_function_outcast_memory": 1024,
            "stitch_function_inner_memory": 1024,
        },
        debug_options=dict(runtime_debug_mode=1, compile_debug_mode=1),
    )
    def adaptive_avg_pool2d_kernel(
        input_tensor: pypto.Tensor((batch_size, channels, in_h, in_w), pypto.DT_FP32),
        output_result: pypto.Tensor((batch_size, channels, out_h, out_w), pypto.DT_FP32),
    ):
        """PyPTO jit kernel for adaptive_avg_pool2d."""
        bc_total = batch_size * channels
        pypto.set_vec_tile_shapes(16, 16, 4, 128)

        # reshape input: [N, C, H, W] -> [N*C, H, W]
        input_reshaped = pypto.reshape(input_tensor, [batch_size * channels, in_h, in_w], inplace=True)

        # 创建中间输出 tensor
        output_tmp = pypto.tensor((bc_total, out_h, out_w), pypto.DT_FP32)

        # 遍历 batch*channel 维度
        for bc_idx, unroll_length in pypto.loop_unroll(
            0, bc_total, 1,
            name="LOOP_BC",
            idx_name="bc_idx",
            unroll_list=[8, 4, 2, 1]
        ):
            input_cur = input_reshaped[bc_idx: bc_idx + unroll_length, :, :]

            # 遍历输出 H 维度
            for oh in range(out_h):
                # 计算输入窗口的 H 范围
                # h_start = floor(oh * in_h / out_h)
                # h_end = ceil((oh + 1) * in_h / out_h)
                h_start = (oh * in_h) // out_h
                h_end = ((oh + 1) * in_h + out_h - 1) // out_h
                h_end = min(h_end, in_h)

                cur_k_h = h_end - h_start

                # 沿 H 维度求和
                pypto.set_vec_tile_shapes(16, 16, 128)
                if cur_k_h > 0:
                    input_single_row = input_cur[:, h_start:h_end, :]
                    input_single_row_1 = pypto.sum(input_single_row, 1, keepdim=True)
                else:
                    input_single_row_1 = None

                # 遍历输出 W 维度
                for ow in range(out_w):
                    # 计算输入窗口的 W 范围
                    w_start = (ow * in_w) // out_w
                    w_end = ((ow + 1) * in_w + out_w - 1) // out_w
                    w_end = min(w_end, in_w)

                    cur_k_w = w_end - w_start
                    win_size = cur_k_h * cur_k_w

                    if cur_k_h > 0 and cur_k_w > 0:
                        # 提取窗口并计算平均值
                        window = input_single_row_1[:, :, w_start:w_end]
                        sum_val = pypto.sum(window, 2, keepdim=True)
                        avg_val = sum_val / win_size

                        # 写入输出
                        pypto.set_vec_tile_shapes(unroll_length, 4, 128)
                        pypto.assemble(avg_val, [bc_idx, oh, ow], output_tmp)
                    else:
                        # 窗口为空，填充 0
                        zero_val = pypto.zeros([unroll_length, 1, 1], dtype=pypto.DT_FP32)
                        pypto.set_vec_tile_shapes(unroll_length, 4, 128)
                        pypto.assemble(zero_val, [bc_idx, oh, ow], output_tmp)

            # 写回输出
            pypto.set_vec_tile_shapes(unroll_length, 4, 128)
            output_result.move(pypto.reshape(output_tmp, [batch_size, channels, out_h, out_w], inplace=True))

    return adaptive_avg_pool2d_kernel


# ─────────────────────────────────────────────
# Wrapper 函数（导出接口）
# ─────────────────────────────────────────────

def adaptive_avg_pool2d_wrapper(
    x: torch.Tensor,
    output_size: Union[int, Tuple[int, int]],
) -> torch.Tensor:
    """adaptive_avg_pool2d wrapper，供 test 调用。

    Args:
        x: 输入 torch.Tensor，形状为 [N, C, H, W]
        output_size: 输出尺寸，可以是单个 int 或 (oH, oW) 元组

    Returns:
        输出 torch.Tensor，形状为 [N, C, oH, oW]
    """
    # 解析 output_size
    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size

    # 获取输入 shape
    batch_size, channels, in_h, in_w = x.shape

    # 创建输出 tensor
    output = torch.empty(
        (batch_size, channels, out_h, out_w),
        dtype=x.dtype,
        device=x.device,
    )

    # 创建 kernel
    kernel = create_adaptive_avg_pool2d_kernel(
        batch_size, channels,
        in_h, in_w, out_h, out_w,
        dynamic=True,
        run_mode="npu",
    )

    # 调用 kernel
    kernel(x, output)

    return output
