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


def compare_precision(actual, expected, rtol=1e-3, atol=1e-3, max_errors=100):
    """
    Compare precision with detailed output including pass rate and error points.
    
    Args:
        actual: Actual output tensor (NPU result)
        expected: Expected output tensor (golden reference)
        rtol: Relative tolerance
        atol: Absolute tolerance
        max_errors: Maximum number of error points to print
    
    Returns:
        bool: True if all points pass, False otherwise
    """
    actual_np = actual.cpu().numpy() if hasattr(actual, 'cpu') else actual
    expected_np = expected.cpu().numpy() if hasattr(expected, 'cpu') else expected
    
    abs_diff = np.abs(actual_np - expected_np)
    rel_diff = abs_diff / (np.abs(expected_np) + atol)
    
    pass_mask = (abs_diff <= atol) | (rel_diff <= rtol)
    total_points = actual_np.size
    pass_count = np.sum(pass_mask)
    fail_count = total_points - pass_count
    pass_rate = pass_count / total_points * 100
    
    print(f"\n{'='*60}")
    print(f"Precision Comparison Results:")
    print(f"{'='*60}")
    print(f"Total points: {total_points}")
    print(f"Pass count: {pass_count}")
    print(f"Fail count: {fail_count}")
    print(f"Pass rate: {pass_rate:.4f}%")
    print(f"Max absolute diff: {np.max(abs_diff):.6f}")
    print(f"Max relative diff: {np.max(rel_diff):.6f}")
    print(f"Mean absolute diff: {np.mean(abs_diff):.6f}")
    print(f"Tolerance: rtol={rtol}, atol={atol}")
    
    if fail_count > 0:
        print(f"\n{'='*60}")
        print(f"Top {min(max_errors, fail_count)} Error Points:")
        print(f"{'='*60}")
        
        fail_indices = np.where(~pass_mask)
        fail_flat_indices = np.flatnonzero(~pass_mask)
        
        for i, flat_idx in enumerate(fail_flat_indices[:max_errors]):
            multi_idx = np.unravel_index(flat_idx, actual_np.shape)
            actual_val = actual_np[multi_idx]
            expected_val = expected_np[multi_idx]
            abs_err = abs_diff[multi_idx]
            rel_err = rel_diff[multi_idx]
            
            print(f"[{i+1}] Index: {multi_idx}")
            print(f"    Actual: {actual_val:.6f}, Expected: {expected_val:.6f}")
            print(f"    Abs diff: {abs_err:.6f}, Rel diff: {rel_err:.6f}")
        
        if fail_count > max_errors:
            print(f"\n... and {fail_count - max_errors} more error points")
    
    print(f"{'='*60}")
    
    return fail_count == 0


# ============================================================================
# Helper Functions for Calculating Input Offsets
# ============================================================================

def cal_hin_idxoffset(hout_idx, tile_hout, ho, hin, kh, strideh, dilationh, pad_top, pad_bottom):
    """
    Calculate hin offset and current size based on hout_idx (for Conv2D/Conv3D).
    
    Returns:
        (cal_flag, hin_offset, hin_current, (pad_top_current, pad_bottom_current))
    """
    pad_top_current = pypto.symbolic_scalar(0)
    pad_bottom_current = pypto.symbolic_scalar(0)
    hin_offset = hout_idx * tile_hout * strideh - pad_top
    need_hin = (tile_hout - 1) * strideh + (kh - 1) * dilationh + 1
    
    if pypto.cond(hin_offset + need_hin <= 0):
        return False, 0, 0, (pad_top_current, pad_bottom_current)
    if pypto.cond(hin - hin_offset <= 0):
        return False, 0, 0, (pad_top_current, pad_bottom_current)
    
    pad_top_current = (-hin_offset).max(0)
    hin_current = hin_offset + need_hin
    pad_bottom_current = (hin_current - hin).max(0)
    hin_offset = hin_offset.max(0)
    hin_current = (hin - hin_offset).min(need_hin)
    
    return True, hin_offset, hin_current, (pad_top_current, pad_bottom_current)


def cal_din_idxoffset(dout_idx, tile_dout, do, din, kd, strided, dilationd, pad_front, pad_back):
    """
    Calculate din offset and current size based on dout_idx (for Conv3D D axis).
    
    Returns:
        (cal_flag, din_offset, din_current, (pad_front_current, pad_back_current))
    """
    pad_front_current = pypto.symbolic_scalar(0)
    pad_back_current = pypto.symbolic_scalar(0)
    din_offset = dout_idx * tile_dout * strided - pad_front
    need_din = (tile_dout - 1) * strided + (kd - 1) * dilationd + 1
    
    if pypto.cond(din_offset + need_din <= 0):
        return False, 0, 0, (pad_front_current, pad_back_current)
    if pypto.cond(din - din_offset <= 0):
        return False, 0, 0, (pad_front_current, pad_back_current)
    
    pad_front_current = (-din_offset).max(0)
    din_current = din_offset + need_din
    pad_back_current = (din_current - din).max(0)
    din_offset = din_offset.max(0)
    din_current = (din - din_offset).min(need_din)
    
    return True, din_offset, din_current, (pad_front_current, pad_back_current)


def cal_win_idxoffset(wout_idx, tile_wout, wo, win, kw, stridew, dilationw, pad_left, pad_right):
    """
    Calculate win offset and current size based on wout_idx.
    Applicable to both Conv1D and Conv2D (w axis).
    
    Returns:
        (cal_flag, win_offset, win_current, (pad_left_current, pad_right_current))
    """
    pad_left_current = pypto.symbolic_scalar(0)
    pad_right_current = pypto.symbolic_scalar(0)
    win_offset = wout_idx * tile_wout * stridew - pad_left
    need_win = (tile_wout - 1) * stridew + (kw - 1) * dilationw + 1
    
    if pypto.cond(win_offset + need_win <= 0):
        return False, 0, 0, (pad_left_current, pad_right_current)
    if pypto.cond(win - win_offset <= 0):
        return False, 0, 0, (pad_left_current, pad_right_current)
    
    pad_left_current = (-win_offset).max(0)
    win_current = win_offset + need_win
    pad_right_current = (win_current - win).max(0)
    win_offset = win_offset.max(0)
    win_current = (win - win_offset).min(need_win)
    
    return True, win_offset, win_current, (pad_left_current, pad_right_current)


# ============================================================================
# Conv2D Dynamic Kernel (4D tensors: [batch, cin, hin, win])
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
    Tile config read from params["tile_config"].
    """
    use_bias = params.get("use_bias", True)
    tile_cfg = params["tile_config"]
    
    _, _, hin, win = params["shape"][0]
    _, cin, kh, kw = params["shape"][1]
    batch, cout, ho, wo = params["shape"][2]
    stride_h = params.get("strides", [1, 1])[0]
    stride_w = params.get("strides", [1, 1])[1]
    dilation_h = params.get("dilations", [1, 1])[0]
    dilation_w = params.get("dilations", [1, 1])[1]
    pad_h = params.get("padding", [0, 0, 0, 0])[:2]
    pad_w = params.get("padding", [0, 0, 0, 0])[-2:]

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

    tile_hin = (tile_hout - 1) * stride_h + (kh - 1) * dilation_h + 1
    tile_win = (tile_wout - 1) * stride_w + (kw - 1) * dilation_w + 1

    for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_L1_batchIdx", idx_name="batch_idx"):
        for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
            for hout_idx in pypto.loop(0, hout_loop, 1, name="LOOP_L1_houtIdx", idx_name="hout_idx"):
                for wout_idx in pypto.loop(0, wout_loop, 1, name="LOOP_L1_woutIdx", idx_name="wout_idx"):
                    batch_offset = batch_idx * tile_batch
                    cout_offset = cout_idx * tile_cout
                    hout_offset = hout_idx * tile_hout
                    wout_offset = wout_idx * tile_wout

                    cal_flag_w, win_offset, win_current, update_pad_w = \
                        cal_win_idxoffset(wout_idx, tile_wout, wo, win, kw,
                                         stride_w, dilation_w, pad_w[0], pad_w[1])

                    cal_flag_h, hin_offset, hin_current, update_pad_h = \
                        cal_hin_idxoffset(hout_idx, tile_hout, ho, hin, kh,
                                         stride_h, dilation_h, pad_h[0], pad_h[1])
                    cal_flag = cal_flag_h and cal_flag_w

                    if cal_flag:
                        input_a_view = pypto.view(
                            input_a_tensor,
                            [tile_batch, cin, tile_hin, tile_win],
                            [batch_offset, 0, hin_offset, win_offset],
                            valid_shape=[tile_batch, cin, hin_current, win_current]
                        )
                        input_b_view = input_b_tensor[cout_offset:cout_offset + tile_cout, 0:cin, 0:kh, 0:kw]
                        
                        padding = (update_pad_h[0], update_pad_h[1], update_pad_w[0], update_pad_w[1])
                        strides = [stride_h, stride_w]
                        dilations = [dilation_h, dilation_w]
                        
                        extend_params = {"bias_tensor": input_c_tensor[cout_offset:cout_offset + tile_cout]} if use_bias else {}

                        output_view = pypto.conv(
                            input_a_view, input_b_view,
                            pypto.DT_FP16,
                            strides,
                            padding,
                            dilations,
                            extend_params=extend_params,
                            groups=1
                        )

                        pypto.assemble(output_view, [batch_offset, cout_offset, hout_offset, wout_offset], output_c_tensor)


# ============================================================================
# Conv1D Dynamic Kernel (3D tensors: [batch, cin, win])
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
    Tile config read from params["tile_config"].
    """
    use_bias = params.get("use_bias", True)
    tile_cfg = params["tile_config"]
    
    _, cin, win = params["shape"][0]
    _, _, kw = params["shape"][1]
    batch, cout, wo = params["shape"][2]
    stride_w = params.get("strides", [1])[0]
    dilation_w = params.get("dilations", [1])[0]
    pad_w = params.get("padding", [1, 1])

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
    
    tile_batch = pypto.symbolic_scalar(tile_cfg["tile_batch"])
    tile_cout = pypto.symbolic_scalar(tile_cfg["tile_cout"])
    tile_wout = pypto.symbolic_scalar(tile_cfg["tile_wout"])

    pypto.set_conv_tile_shapes(tile_l1_config, tile_l0_config)
    pypto.set_vec_tile_shapes(*vec_tile_config)

    batch_loop = (batch + tile_batch - 1) // tile_batch
    cout_loop = (cout + tile_cout - 1) // tile_cout
    wout_loop = (wo + tile_wout - 1) // tile_wout

    tile_win = (tile_wout - 1) * stride_w + (kw - 1) * dilation_w + 1

    for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_L1_batchIdx", idx_name="batch_idx"):
        for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
            for wout_idx in pypto.loop(0, wout_loop, 1, name="LOOP_L1_woutIdx", idx_name="wout_idx"):
                batch_offset = batch_idx * tile_batch
                cout_offset = cout_idx * tile_cout
                wout_offset = wout_idx * tile_wout

                cal_flag, win_offset, win_current, update_pad_w = \
                    cal_win_idxoffset(wout_idx, tile_wout, wo, win, kw,
                                     stride_w, dilation_w, pad_w[0], pad_w[1])

                if cal_flag:
                    input_a_view = pypto.view(
                        input_a_tensor,
                        [tile_batch, cin, tile_win],
                        [batch_offset, 0, win_offset],
                        valid_shape=[tile_batch, cin, win_current]
                    )
                    input_b_view = input_b_tensor[cout_offset:cout_offset + tile_cout, 0:cin, 0:kw]
                    
                    padding = (update_pad_w[0], update_pad_w[1])
                    strides = [stride_w]
                    dilations = [dilation_w]
                    
                    extend_params = {"bias_tensor": input_c_tensor[cout_offset:cout_offset + tile_cout]} if use_bias else {}

                    output_view = pypto.conv(
                        input_a_view, input_b_view,
                        pypto.DT_FP16,
                        strides,
                        padding,
                        dilations,
                        extend_params=extend_params,
                        groups=1
                    )

                    pypto.assemble(output_view, [batch_offset, cout_offset, wout_offset], output_c_tensor)


# ============================================================================
# Conv3D Dynamic Kernel (5D tensors: [batch, cin, din, hin, win])
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
    
    Dynamic tiling axes:
      - batch: tile_batch = 1 (前端循环)
      - cout: tile_cout = 64 (TileShape动态)
      - dout: tile_dout = 8 (TileShape动态)
      - hout: tile_hout = 8 (TileShape动态)
      - wout: tile_wout = 16 (前端循环)
    """
    use_bias = params.get("use_bias", True)
    tile_cfg = params["tile_config"]
    
    _, _, din, hin, win = params["shape"][0]
    _, cin, kd, kh, kw = params["shape"][1]
    batch, cout, do, ho, wo = params["shape"][2]
    stride_d = params.get("strides", [1, 1, 1])[0]
    stride_h = params.get("strides", [1, 1, 1])[1]
    stride_w = params.get("strides", [1, 1, 1])[2]
    dilation_d = params.get("dilations", [1, 1, 1])[0]
    dilation_h = params.get("dilations", [1, 1, 1])[1]
    dilation_w = params.get("dilations", [1, 1, 1])[2]
    pad_d = params.get("padding", [0, 0, 0, 0, 0, 0])[:2]
    pad_h = params.get("padding", [0, 0, 0, 0, 0, 0])[2:4]
    pad_w = params.get("padding", [0, 0, 0, 0, 0, 0])[-2:]

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

    tile_din = (tile_dout - 1) * stride_d + (kd - 1) * dilation_d + 1
    tile_hin = (tile_hout - 1) * stride_h + (kh - 1) * dilation_h + 1
    tile_win = (tile_wout - 1) * stride_w + (kw - 1) * dilation_w + 1

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

                        cal_flag_w, win_offset, win_current, update_pad_w = \
                            cal_win_idxoffset(wout_idx, tile_wout, wo, win, kw,
                                             stride_w, dilation_w, pad_w[0], pad_w[1])

                        cal_flag_h, hin_offset, hin_current, update_pad_h = \
                            cal_hin_idxoffset(hout_idx, tile_hout, ho, hin, kh,
                                             stride_h, dilation_h, pad_h[0], pad_h[1])

                        cal_flag_d, din_offset, din_current, update_pad_d = \
                            cal_din_idxoffset(dout_idx, tile_dout, do, din, kd,
                                             stride_d, dilation_d, pad_d[0], pad_d[1])

                        cal_flag = cal_flag_d and cal_flag_h and cal_flag_w

                        if cal_flag:
                            input_a_view = pypto.view(
                                input_a_tensor,
                                [tile_batch, cin, tile_din, tile_hin, tile_win],
                                [batch_offset, 0, din_offset, hin_offset, win_offset],
                                valid_shape=[tile_batch, cin, din_current, hin_current, win_current]
                            )
                            input_b_view = input_b_tensor[cout_offset:cout_offset + tile_cout, 0:cin, 0:kd, 0:kh, 0:kw]
                            
                            padding = (update_pad_d[0], update_pad_d[1], update_pad_h[0], update_pad_h[1], update_pad_w[0], update_pad_w[1])
                            strides = [stride_d, stride_h, stride_w]
                            dilations = [dilation_d, dilation_h, dilation_w]
                            
                            extend_params = {"bias_tensor": input_c_tensor[cout_offset:cout_offset + tile_cout]} if use_bias else {}

                            output_view = pypto.conv(
                                input_a_view, input_b_view,
                                pypto.DT_FP16,
                                strides,
                                padding,
                                dilations,
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

    if conv_type == "conv3d":
        torch_padding = (padding[0], padding[2], padding[4])
    elif conv_type == "conv2d":
        torch_padding = (padding[0], padding[2])
    else:
        torch_padding = padding[0]

    a = torch.randn(fmap_shape, dtype=dtype, device=device)
    b = torch.randn(weight_shape, dtype=dtype, device=device)
    c = torch.randn(bias_shape, dtype=dtype, device=device) if use_bias else torch.zeros(bias_shape, dtype=dtype, device=device)

    bias_tensor = c if use_bias else None
    if conv_type == "conv3d":
        expected = torch.conv3d(a, b, bias=bias_tensor, padding=torch_padding, stride=strides, dilation=dilations, groups=1)
    elif conv_type == "conv2d":
        expected = torch.conv2d(a, b, bias=bias_tensor, padding=torch_padding, stride=strides, dilation=dilations, groups=1)
    else:
        expected = torch.conv1d(a, b, bias=bias_tensor, padding=torch_padding, stride=strides, dilation=dilations, groups=1)

    out = torch.empty(out_shape, dtype=dtype, device=device)
    start_time = time.time()
    
    params = {
        "shape": [fmap_shape, weight_shape, out_shape],
        "strides": strides,
        "padding": padding,
        "dilations": dilations,
        "use_bias": use_bias,
        "tile_config": case_config["tile_config"]
    }
    
    if conv_type == "conv3d":
        conv3d_dynamic_kernel(a, b, c, out, params)
    elif conv_type == "conv2d":
        conv2d_dynamic_kernel(a, b, c, out, params)
    else:
        conv1d_dynamic_kernel(a, b, c, out, params)
    
    end_time = time.time()
    print(f"Kernel execution time: {end_time - start_time:.4f} seconds")
    print(f"Input shape: {fmap_shape}")
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