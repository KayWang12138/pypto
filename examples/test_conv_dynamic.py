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
Conv Dynamic Tiling Examples for PyPTO (Unified Implementation)

This file demonstrates unified dynamic tiling for Conv1D and Conv2D operations:

Conv2D (4D tensors: [batch, cin, hin, win]):
  - batch: 前端循环切分（TileShape.tileBatch 必须为 1）
  - cout: TileShape 动态切分
  - hout: TileShape 动态切分
  - wout: 前端循环切分

Conv1D (3D tensors: [batch, cin, win]):
  - batch: 前端循环切分（TileShape.tileBatch 必须为 1）
  - cout: TileShape 动态切分
  - wout: 前端循环切分（与 Conv2D 保持一致的动态模式）

重要约束说明：
- TileL1Info.tileBatch 必须为 1（硬件约束）
- batch 轴动态切分通过前端 pypto.loop 循环实现
- wout 轴动态切分通过前端 pypto.loop 循环实现（Conv1D/Conv2D 统一）
- 参考: pypto-set_conv_tile_shapes.md

动态轴切分模式统一：
- batch轴：前端循环（TileBatch=1 固定约束）
- cout轴：TileShape动态切分
- hout轴：Conv2D TileShape切分，Conv1D 固定为1
- wout轴：前端循环（两种Conv统一模式）

Usage:
     python test_conv_dynamic.py --test conv2d_bias    # Run conv2d with bias test
     python test_conv_dynamic.py --test conv2d_no_bias # Run conv2d without bias test
     python test_conv_dynamic.py --test conv1d_bias    # Run conv1d with bias test
     python test_conv_dynamic.py --test conv1d_no_bias # Run conv1d without bias test
     python test_conv_dynamic.py --test all            # Run all tests
     python test_conv_dynamic.py --device_id 0         # Specify NPU device
     python test_conv_dynamic.py --run_mode sim        # Use simulation mode
     python test_conv_dynamic.py --test conv3d_bias --case_name conv3d_small  # Run specific test case
"""

import argparse
import os
import sys
import pypto
import torch
import numpy as np
import time
import toolss
from conv_test_config import TEST_CASES, get_cases_by_conv_type, get_case_by_name

global_run_mode = pypto.RunMode.NPU


# ============================================================================
# Helper Functions for Conv Dynamic Kernel
# ============================================================================

def setup_tile_config(tile_cfg: dict):
    """
    Setup tile L1 and L0 config from tile_cfg dict.
    
    Returns:
        (tile_l1_config, tile_l0_config, vec_tile_config)
    """
    tile_l1_params = tile_cfg["tile_l1"]
    tile_l1_config = pypto.pypto_impl.TileL1Info(
        tileHin=tile_l1_params["tileHin"], tileHout=tile_l1_params["tileHout"],
        tileWin=tile_l1_params["tileWin"], tileWout=tile_l1_params["tileWout"],
        tileCinFmap=tile_l1_params["tileCinFmap"], tileCinWeight=tile_l1_params["tileCinWeight"],
        tileN=tile_l1_params["tileN"], tileBatch=tile_l1_params["tileBatch"]
    )
    
    tile_l0_params = tile_cfg["tile_l0"]
    tile_l0_config = pypto.pypto_impl.TileL0Info(
        tileH=tile_l0_params["tileH"], tileW=tile_l0_params["tileW"],
        tileK=tile_l0_params["tileK"], tileN=tile_l0_params["tileN"]
    )
    
    vec_tile_config = tile_cfg["vec_tile"]
    
    return tile_l1_config, tile_l0_config, vec_tile_config


def calculate_tile_input_size(tile_out, stride, dilation, kernel):
    """
    Calculate required input tile size from output tile size.
    
    Args:
        tile_out: Output tile size (symbolic_scalar)
        stride: Stride value
        dilation: Dilation value
        kernel: Kernel size
    
    Returns:
        Required input tile size
    """
    return (tile_out - 1) * stride + (kernel - 1) * dilation + 1


def calculate_offset_and_current(out_idx, tile_out, stride, in_size, tile_in):
    """
    Calculate input offset and current size for a tile.
    
    Since input is pre-padded by torch, offset is always >= 0.
    No need to handle left/top overflow (already handled by torch padding).
    
    Args:
        out_idx: Output tile index
        tile_out: Output tile size (symbolic_scalar)
        stride: Stride value
        in_size: Total input size (already padded)
        tile_in: Input tile size
    
    Returns:
        (offset, current_size) where:
        - offset: Input offset for this tile (always >= 0)
        - current_size: Valid input size for this tile
    """
    # Calculate input offset from output tile position
    offset = out_idx * tile_out * stride
    
    # Calculate valid input size (may be smaller at boundaries)
    current_size = (in_size - offset).min(tile_in)
    
    return offset, current_size


def create_conv_extend_params(use_bias: bool, bias_tensor, cout_offset: int, tile_cout):
    """
    Create extend_params dict for conv operation.
    
    Args:
        use_bias: Whether to use bias
        bias_tensor: Bias tensor
        cout_offset: Output channel offset
        tile_cout: Output channel tile size
    
    Returns:
        extend_params dict
    """
    if use_bias:
        return {"bias_tensor": bias_tensor[cout_offset:cout_offset + tile_cout]}
    return {}


# ============================================================================
# Conv1D Dynamic Kernel (3D tensors: [batch, cin, win])
# ============================================================================
# 
# Note: Input tensor is pre-padded by torch before calling this kernel.
#       params["padding"] should be [0, 0] since input is already padded.
# ============================================================================

@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode},
                    debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1})
def conv1d_dynamic_kernel(
    input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_c_tensor: pypto.Tensor([pypto.DYNAMIC]),
    output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    params: dict):
    """
    Conv1D kernel with dynamic tiling (3D tensors).
    Input is pre-padded by torch, so no padding needed in kernel.
    """
    use_bias = params.get("use_bias", True)
    tile_cfg = params["tile_config"]
    
    # Input shape is already padded (padded_fmap_shape)
    _, cin, win = params["shape"][0]
    _, _, kw = params["shape"][1]
    batch, cout, wo = params["shape"][2]
    
    stride_w = params.get("strides", [1])[0]
    dilation_w = params.get("dilations", [1])[0]

    # Setup tile config
    tile_l1_config, tile_l0_config, vec_tile_config = setup_tile_config(tile_cfg)
    
    tile_batch = pypto.symbolic_scalar(tile_cfg["tile_batch"])
    tile_cout = pypto.symbolic_scalar(tile_cfg["tile_cout"])
    tile_wout = pypto.symbolic_scalar(tile_cfg["tile_wout"])

    pypto.set_conv_tile_shapes(tile_l1_config, tile_l0_config)
    pypto.set_vec_tile_shapes(*vec_tile_config)

    batch_loop = (batch + tile_batch - 1) // tile_batch
    cout_loop = (cout + tile_cout - 1) // tile_cout
    wout_loop = (wo + tile_wout - 1) // tile_wout

    tile_win = calculate_tile_input_size(tile_wout, stride_w, dilation_w, kw)

    for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_L1_batchIdx", idx_name="batch_idx"):
        for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
            for wout_idx in pypto.loop(0, wout_loop, 1, name="LOOP_L1_woutIdx", idx_name="wout_idx"):
                batch_offset = batch_idx * tile_batch
                cout_offset = cout_idx * tile_cout
                wout_offset = wout_idx * tile_wout

                # Calculate input offset and current size (input is pre-padded)
                win_offset, win_current = calculate_offset_and_current(wout_idx, tile_wout, stride_w, win, tile_win)

                # Get input view - input tensor is already pre-padded by torch
                input_a_view = pypto.view(
                    input_a_tensor,
                    [tile_batch, cin, tile_win],
                    [batch_offset, 0, win_offset],
                    valid_shape=[tile_batch, cin, win_current]
                )
                input_b_view = input_b_tensor[cout_offset:cout_offset + tile_cout, 0:cin, 0:kw]

                # Call conv with padding=0 (input is pre-padded)
                extend_params = create_conv_extend_params(use_bias, input_c_tensor, cout_offset, tile_cout)

                output_view = pypto.conv(
                    input_a_view, input_b_view,
                    pypto.DT_FP16,
                    [stride_w],
                    (0, 0),  # No padding - input is pre-padded
                    [dilation_w],
                    extend_params=extend_params,
                    groups=1
                )

                pypto.assemble(output_view, [batch_offset, cout_offset, wout_offset], output_c_tensor)


# ============================================================================
# Conv2D Dynamic Kernel (4D tensors: [batch, cin, hin, win])
# ============================================================================
# 
# Note: Input tensor is pre-padded by torch before calling this kernel.
#       params["padding"] should be [0, 0, 0, 0] since input is already padded.
# ============================================================================

@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode},
                    debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1})
def conv2d_dynamic_kernel(
    input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_c_tensor: pypto.Tensor([pypto.DYNAMIC]),
    output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    params: dict):
    """
    Conv2D kernel with dynamic tiling (4D tensors).
    Input is pre-padded by torch, so no padding needed in kernel.
    """
    use_bias = params.get("use_bias", True)
    tile_cfg = params["tile_config"]
    
    # Input shape is already padded (padded_fmap_shape)
    _, _, hin, win = params["shape"][0]
    _, cin, kh, kw = params["shape"][1]
    batch, cout, ho, wo = params["shape"][2]
    
    stride_h = params.get("strides", [1, 1])[0]
    stride_w = params.get("strides", [1, 1])[1]
    dilation_h = params.get("dilations", [1, 1])[0]
    dilation_w = params.get("dilations", [1, 1])[1]

    # Setup tile config
    tile_l1_config, tile_l0_config, vec_tile_config = setup_tile_config(tile_cfg)
    
    tile_batch = pypto.symbolic_scalar(tile_cfg["tile_batch"])
    tile_cout = pypto.symbolic_scalar(tile_cfg["tile_cout"])
    tile_hout = pypto.symbolic_scalar(tile_cfg["tile_hout"])
    tile_wout = pypto.symbolic_scalar(tile_cfg["tile_wout"])

    pypto.set_conv_tile_shapes(tile_l1_config, tile_l0_config)
    pypto.set_vec_tile_shapes(*vec_tile_config)

    batch_loop = (batch + tile_batch - 1) // tile_batch
    cout_loop = (cout + tile_cout - 1) // tile_cout
    hout_loop = (ho + tile_hout - 1) // tile_hout
    wout_loop = (wo + tile_wout - 1) // tile_wout

    tile_hin = calculate_tile_input_size(tile_hout, stride_h, dilation_h, kh)
    tile_win = calculate_tile_input_size(tile_wout, stride_w, dilation_w, kw)

    for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_L1_batchIdx", idx_name="batch_idx"):
        for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
            for hout_idx in pypto.loop(0, hout_loop, 1, name="LOOP_L1_houtIdx", idx_name="hout_idx"):
                for wout_idx in pypto.loop(0, wout_loop, 1, name="LOOP_L1_woutIdx", idx_name="wout_idx"):
                    batch_offset = batch_idx * tile_batch
                    cout_offset = cout_idx * tile_cout
                    hout_offset = hout_idx * tile_hout
                    wout_offset = wout_idx * tile_wout

                    # Calculate input offset and current size (input is pre-padded)
                    hin_offset, hin_current = calculate_offset_and_current(hout_idx, tile_hout, stride_h, hin, tile_hin)
                    win_offset, win_current = calculate_offset_and_current(wout_idx, tile_wout, stride_w, win, tile_win)

                    # Get input view
                    input_a_view = pypto.view(
                        input_a_tensor,
                        [tile_batch, cin, tile_hin, tile_win],
                        [batch_offset, 0, hin_offset, win_offset],
                        valid_shape=[tile_batch, cin, hin_current, win_current]
                    )
                    input_b_view = input_b_tensor[cout_offset:cout_offset + tile_cout, 0:cin, 0:kh, 0:kw]

                    # Call conv with padding=0 (input is pre-padded)
                    extend_params = create_conv_extend_params(use_bias, input_c_tensor, cout_offset, tile_cout)

                    output_view = pypto.conv(
                        input_a_view, input_b_view,
                        pypto.DT_FP16,
                        [stride_h, stride_w],
                        (0, 0, 0, 0),  # No padding - input is pre-padded
                        [dilation_h, dilation_w],
                        extend_params=extend_params,
                        groups=1
                    )

                    pypto.assemble(output_view, [batch_offset, cout_offset, hout_offset, wout_offset], output_c_tensor)


# ============================================================================
# Conv3D Dynamic Kernel (5D tensors: [batch, cin, din, hin, win])
# ============================================================================
# 
# Note: Input tensor is pre-padded by torch before calling this kernel.
#       params["padding"] should be [0, 0, 0, 0, 0, 0] since input is already padded.
# ============================================================================

@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode},
                    debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1})
def conv3d_dynamic_kernel(
    input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_c_tensor: pypto.Tensor([pypto.DYNAMIC]),
    output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    params: dict):
    """
    Conv3D kernel with dynamic tiling (5D tensors).
    Input is pre-padded by torch, so no padding needed in kernel.
    """
    use_bias = params.get("use_bias", True)
    tile_cfg = params["tile_config"]
    
    # Input shape is already padded (padded_fmap_shape)
    _, _, din, hin, win = params["shape"][0]
    _, cin, kd, kh, kw = params["shape"][1]
    batch, cout, do, ho, wo = params["shape"][2]
    
    stride_d = params.get("strides", [1, 1, 1])[0]
    stride_h = params.get("strides", [1, 1, 1])[1]
    stride_w = params.get("strides", [1, 1, 1])[2]
    dilation_d = params.get("dilations", [1, 1, 1])[0]
    dilation_h = params.get("dilations", [1, 1, 1])[1]
    dilation_w = params.get("dilations", [1, 1, 1])[2]

    # Setup tile config
    tile_l1_config, tile_l0_config, vec_tile_config = setup_tile_config(tile_cfg)
    
    tile_batch = pypto.symbolic_scalar(tile_cfg["tile_batch"])
    tile_cout = pypto.symbolic_scalar(tile_cfg["tile_cout"])
    tile_dout = pypto.symbolic_scalar(tile_cfg["tile_dout"])
    tile_hout = pypto.symbolic_scalar(tile_cfg["tile_hout"])
    tile_wout = pypto.symbolic_scalar(tile_cfg["tile_wout"])

    pypto.set_conv_tile_shapes(tile_l1_config, tile_l0_config)
    pypto.set_vec_tile_shapes(*vec_tile_config)

    batch_loop = (batch + tile_batch - 1) // tile_batch
    cout_loop = (cout + tile_cout - 1) // tile_cout
    dout_loop = (do + tile_dout - 1) // tile_dout
    hout_loop = (ho + tile_hout - 1) // tile_hout
    wout_loop = (wo + tile_wout - 1) // tile_wout

    tile_din = calculate_tile_input_size(tile_dout, stride_d, dilation_d, kd)
    tile_hin = calculate_tile_input_size(tile_hout, stride_h, dilation_h, kh)
    tile_win = calculate_tile_input_size(tile_wout, stride_w, dilation_w, kw)

    for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_L1_batchIdx", idx_name="batch_idx"):
        for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
            for dout_idx in pypto.loop(0, dout_loop, 1, name="LOOP_L1_doutIdx", idx_name="dout_idx"):
                for hout_idx in pypto.loop(0, hout_loop, 1, name="LOOP_L1_houtIdx", idx_name="hout_idx"):
                    for wout_idx in pypto.loop(0, wout_loop, 1, name="LOOP_L1_woutIdx", idx_name="wout_idx"):
                        batch_offset = batch_idx * tile_batch
                        cout_offset = cout_idx * tile_cout
                        dout_offset = dout_idx * tile_dout
                        hout_offset = hout_idx * tile_hout
                        wout_offset = wout_idx * tile_wout

                        # Calculate input offset and current size (input is pre-padded)
                        din_offset, din_current = calculate_offset_and_current(dout_idx, tile_dout, stride_d, din, tile_din)
                        hin_offset, hin_current = calculate_offset_and_current(hout_idx, tile_hout, stride_h, hin, tile_hin)
                        win_offset, win_current = calculate_offset_and_current(wout_idx, tile_wout, stride_w, win, tile_win)

                        # Get input view
                        input_a_view = pypto.view(
                            input_a_tensor,
                            [tile_batch, cin, tile_din, tile_hin, tile_win],
                            [batch_offset, 0, din_offset, hin_offset, win_offset],
                            valid_shape=[tile_batch, cin, din_current, hin_current, win_current]
                        )
                        input_b_view = input_b_tensor[cout_offset:cout_offset + tile_cout, 0:cin, 0:kd, 0:kh, 0:kw]

                        # Call conv with padding=0 (input is pre-padded)
                        extend_params = create_conv_extend_params(use_bias, input_c_tensor, cout_offset, tile_cout)

                        output_view = pypto.conv(
                            input_a_view, input_b_view,
                            pypto.DT_FP16,
                            [stride_d, stride_h, stride_w],
                            (0, 0, 0, 0, 0, 0),  # No padding - input is pre-padded
                            [dilation_d, dilation_h, dilation_w],
                            extend_params=extend_params,
                            groups=1
                        )

                        pypto.assemble(output_view, [batch_offset, cout_offset, dout_offset, hout_offset, wout_offset], output_c_tensor)


# ============================================================================
# Test Functions
# ============================================================================

def run_conv_test(
    case_config: dict,
    use_bias: bool,
    device_id: int = None):
    """
    Run single test case with given configuration.
    
    Args:
        case_config: Test case configuration dict from TEST_CASES
        use_bias: True to use bias, False to skip bias
        device_id: NPU device ID
    """
    case_name = case_config["name"]
    conv_type = case_config["conv_type"]
    bias_type = "with bias" if use_bias else "without bias"
    
    print("=" * 60)
    print(f"Test Case: {case_name} - {conv_type.upper()} {bias_type}")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'
    dtype = torch.float16

    fmap_shape = case_config["fmap_shape"]
    weight_shape = case_config["weight_shape"]
    bias_shape = case_config["bias_shape"]
    out_shape = case_config["out_shape"]
    strides = case_config["strides"]
    padding = case_config["padding"]
    dilations = case_config["dilations"]

    # ============================================================================
    # Step 1: Pre-pad input tensor using torch (handle all left/top/right/bottom padding)
    # ============================================================================
    # padding format in config:
    #   - conv3d: [pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right]
    #   - conv2d: [pad_top, pad_bottom, pad_left, pad_right]
    #   - conv1d: [pad_left, pad_right]
    # 
    # torch.nn.functional.pad format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    # Note: torch pad is from last dim to first dim
    
    if conv_type == "conv3d":
        # conv3d: [batch, cin, din, hin, win] -> pad din, hin, win
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = padding
        torch_padding = (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        # torch conv3d expects padding=(pad_front, pad_top, pad_left) for symmetric padding
        torch_conv_padding = (pad_front, pad_top, pad_left)
    elif conv_type == "conv2d":
        # conv2d: [batch, cin, hin, win] -> pad hin, win
        pad_top, pad_bottom, pad_left, pad_right = padding
        torch_padding = (pad_left, pad_right, pad_top, pad_bottom)
        # torch conv2d expects padding=(pad_top, pad_left) for symmetric padding
        torch_conv_padding = (pad_top, pad_left)
    else:
        # conv1d: [batch, cin, win] -> pad win
        pad_left, pad_right = padding
        torch_padding = (pad_left, pad_right)
        torch_conv_padding = pad_left

    a = torch.randn(fmap_shape, dtype=dtype, device=device)
    b = torch.randn(weight_shape, dtype=dtype, device=device)
    c = torch.randn(bias_shape, dtype=dtype, device=device) if use_bias else torch.zeros(bias_shape, dtype=dtype, device=device)

    # Pre-pad input tensor using torch
    a_padded = torch.nn.functional.pad(a, torch_padding, mode='constant', value=0.0)
    
    # Calculate padded input shape
    if conv_type == "conv3d":
        padded_fmap_shape = (
            fmap_shape[0],  # batch
            fmap_shape[1],  # cin
            fmap_shape[2] + pad_front + pad_back,   # din + padding
            fmap_shape[3] + pad_top + pad_bottom,   # hin + padding
            fmap_shape[4] + pad_left + pad_right    # win + padding
        )
    elif conv_type == "conv2d":
        padded_fmap_shape = (
            fmap_shape[0],  # batch
            fmap_shape[1],  # cin
            fmap_shape[2] + pad_top + pad_bottom,   # hin + padding
            fmap_shape[3] + pad_left + pad_right    # win + padding
        )
    else:
        padded_fmap_shape = (
            fmap_shape[0],  # batch
            fmap_shape[1],  # cin
            fmap_shape[2] + pad_left + pad_right    # win + padding
        )

    # Compute expected output using torch conv with original padding
    bias_tensor = c if use_bias else None
    if conv_type == "conv3d":
        expected = torch.conv3d(a, b, bias=bias_tensor, padding=torch_conv_padding, stride=strides, dilation=dilations, groups=1)
    elif conv_type == "conv2d":
        expected = torch.conv2d(a, b, bias=bias_tensor, padding=torch_conv_padding, stride=strides, dilation=dilations, groups=1)
    else:
        expected = torch.conv1d(a, b, bias=bias_tensor, padding=torch_conv_padding, stride=strides, dilation=dilations, groups=1)

    out = torch.empty(out_shape, dtype=dtype, device=device)
    start_time = time.time()
    
    # ============================================================================
    # Step 2: Call kernel with padded input and padding=0
    # ============================================================================
    # Since input is already padded, kernel only handles tile boundary overflow (right/bottom pad)
    params = {
        "shape": [padded_fmap_shape, weight_shape, out_shape],  # Use padded input shape
        "strides": strides,
        "padding": [0, 0, 0, 0] if conv_type == "conv2d" else ([0, 0] if conv_type == "conv1d" else [0, 0, 0, 0, 0, 0]),  # No padding in kernel
        "dilations": dilations,
        "use_bias": use_bias,
        "tile_config": case_config["tile_config"]
    }
    
    if conv_type == "conv3d":
        conv3d_dynamic_kernel(a_padded, b, c, out, params)
    elif conv_type == "conv2d":
        conv2d_dynamic_kernel(a_padded, b, c, out, params)
    else:
        conv1d_dynamic_kernel(a_padded, b, c, out, params)
    
    end_time = time.time()
    print(f"Kernel execution time: {end_time - start_time:.4f} seconds")
    print(f"Original input shape: {fmap_shape}")
    print(f"Padded input shape: {padded_fmap_shape}")
    print(f"Weight shape: {weight_shape}")
    print(f"Strides: {strides}")
    print(f"Padding: {padding}")
    print(f"Dilations: {dilations}")
    if use_bias:
        print(f"Bias shape: {bias_shape}")
    print(f"Output shape: {out_shape}")
    
    if global_run_mode == pypto.RunMode.NPU:
        # all_pass = compare_precision(out, expected, rtol=1e-3, atol=1e-3, max_errors=100)
        golden = expected.cpu().to(torch.float32).numpy()
        pto_res = out.cpu().to(torch.float32).numpy()
        result = toolss.dataCompare(pto_res, golden, 0.001, 0.001)
        if result == "Pass":
            print(f"✓ Test case '{case_name}' passed - All points matched tolerance!")
        else:
            print(f"✗ Test case '{case_name}' failed - Some points did not match tolerance")


def run_all_conv_tests(conv_type: str, use_bias: bool, device_id: int = None):
    """
    Run all test cases for specified conv type.
    
    Args:
        conv_type: "conv3d", "conv2d", "conv1d", or "all"
        use_bias: True to use bias, False to skip bias
        device_id: NPU device ID
    """
    cases_to_run = []
    
    if conv_type == "all":
        cases_to_run = list(TEST_CASES.keys())
    else:
        cases_to_run = [name for name, cfg in TEST_CASES.items() if cfg["conv_type"] == conv_type]
    
    if not cases_to_run:
        print(f"No test cases found for conv_type: {conv_type}")
        return
    
    for case_name in cases_to_run:
        run_conv_test(TEST_CASES[case_name], use_bias, device_id)
        print("\n")


# ============================================================================
# Test Entry Functions (for command line compatibility)
# ============================================================================

def test_conv3d_with_bias(device_id: int = None, case_name: str = None):
    if case_name:
        run_conv_test(TEST_CASES[case_name], use_bias=True, device_id=device_id)
    else:
        run_all_conv_tests("conv3d", use_bias=True, device_id=device_id)


def test_conv3d_no_bias(device_id: int = None, case_name: str = None):
    if case_name:
        run_conv_test(TEST_CASES[case_name], use_bias=False, device_id=device_id)
    else:
        run_all_conv_tests("conv3d", use_bias=False, device_id=device_id)


def test_conv2d_with_bias(device_id: int = None, case_name: str = None):
    if case_name:
        run_conv_test(TEST_CASES[case_name], use_bias=True, device_id=device_id)
    else:
        run_all_conv_tests("conv2d", use_bias=True, device_id=device_id)


def test_conv2d_no_bias(device_id: int = None, case_name: str = None):
    if case_name:
        run_conv_test(TEST_CASES[case_name], use_bias=False, device_id=device_id)
    else:
        run_all_conv_tests("conv2d", use_bias=False, device_id=device_id)


def test_conv1d_with_bias(device_id: int = None, case_name: str = None):
    if case_name:
        run_conv_test(TEST_CASES[case_name], use_bias=True, device_id=device_id)
    else:
        run_all_conv_tests("conv1d", use_bias=True, device_id=device_id)


def test_conv1d_no_bias(device_id: int = None, case_name: str = None):
    if case_name:
        run_conv_test(TEST_CASES[case_name], use_bias=False, device_id=device_id)
    else:
        run_all_conv_tests("conv1d", use_bias=False, device_id=device_id)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Conv Dynamic Tiling Examples (Conv1D/Conv2D/Conv3D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test conv3d_bias     Run conv3d with bias test
  %(prog)s --test conv3d_no_bias  Run conv3d without bias test
  %(prog)s --test conv2d_bias     Run conv2d with bias test
  %(prog)s --test conv2d_no_bias  Run conv2d without bias test
  %(prog)s --test conv1d_bias     Run conv1d with bias test
  %(prog)s --test conv1d_no_bias  Run conv1d without bias test
  %(prog)s --test all             Run all tests
  %(prog)s --device_id 0          Use NPU device 0
  %(prog)s --run_mode sim         Use simulation mode

重要约束：
  - TileL1Info.tileBatch 必须为 1（硬件约束）
  - batch 轴动态切分通过前端循环实现
        """
    )
    
    parser.add_argument('--device_id', type=int, default=0, help='NPU device ID')
    parser.add_argument('--run_mode', type=str, default='npu', choices=['npu', 'sim'],
                       help='Run mode: npu or sim')
    parser.add_argument('--test', type=str, default='conv2d_bias', 
                       choices=['conv3d_bias', 'conv3d_no_bias', 'conv2d_bias', 'conv2d_no_bias', 
                                'conv1d_bias', 'conv1d_no_bias', 'all'],
                       help='Test case to run')
    parser.add_argument('--case_name', type=str, default=None,
                       help='Specific test case name from TEST_CASES (e.g., conv3d_small, conv2d_large)')
    
    args = parser.parse_args()
    
    global global_run_mode
    global_run_mode = pypto.RunMode.NPU if args.run_mode == 'npu' else pypto.RunMode.SIM
    
    if global_run_mode == pypto.RunMode.NPU:
        import torch_npu
        torch.npu.set_device(args.device_id)
        print(f"Running on NPU device {args.device_id}")
    else:
        print("Running in simulation mode")
    
    print()
    
    try:
        if args.test == 'conv3d_bias':
            test_conv3d_with_bias(args.device_id, args.case_name)
        elif args.test == 'conv3d_no_bias':
            test_conv3d_no_bias(args.device_id, args.case_name)
        elif args.test == 'conv2d_bias':
            test_conv2d_with_bias(args.device_id, args.case_name)
        elif args.test == 'conv2d_no_bias':
            test_conv2d_no_bias(args.device_id, args.case_name)
        elif args.test == 'conv1d_bias':
            test_conv1d_with_bias(args.device_id, args.case_name)
        elif args.test == 'conv1d_no_bias':
            test_conv1d_no_bias(args.device_id, args.case_name)
        elif args.test == 'all':
            test_conv3d_with_bias(args.device_id, args.case_name)
            print("\n" + "=" * 60 + "\n")
            test_conv3d_no_bias(args.device_id, args.case_name)
            print("\n" + "=" * 60 + "\n")
            test_conv2d_with_bias(args.device_id, args.case_name)
            print("\n" + "=" * 60 + "\n")
            test_conv2d_no_bias(args.device_id, args.case_name)
            print("\n" + "=" * 60 + "\n")
            test_conv1d_with_bias(args.device_id, args.case_name)
            print("\n" + "=" * 60 + "\n")
            test_conv1d_no_bias(args.device_id, args.case_name)
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()