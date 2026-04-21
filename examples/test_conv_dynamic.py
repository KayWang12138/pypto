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
Conv2D Dynamic Tiling Examples for PyPTO

This file demonstrates dynamic tiling on multiple axes for convolution operations:
- Basic mode: cout + hout (2-axis dynamic tiling)
- Full mode: batch + cout + hout + wout (4-axis dynamic tiling)
- Conv1D mode: cout (1-axis dynamic tiling)

Usage:
    python test_conv_dynamic.py --test basic       # Run basic 2-axis dynamic test
    python test_conv_dynamic.py --test full        # Run full 4-axis dynamic test
    python test_conv_dynamic.py --test conv1d      # Run conv1d test
    python test_conv_dynamic.py --test all         # Run all tests
    python test_conv_dynamic.py --device_id 0      # Specify NPU device
    python test_conv_dynamic.py --run_mode sim     # Use simulation mode
"""

import argparse
import os
import sys
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose

global_run_mode = pypto.RunMode.NPU


# ============================================================================
# Helper Functions for Calculating Input Offsets
# ============================================================================

def cal_hin_idxoffset(hout_idx, tile_hout, ho, hin, kh, strideh, dilationh, pad_top, pad_bottom):
    """
    Calculate hin offset and current size based on hout_idx.
    
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


def cal_win_idxoffset(wout_idx, tile_wout, wo, win, kw, stridew, dilationw, pad_left, pad_right):
    """
    Calculate win offset and current size based on wout_idx (新增).
    
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
# Conv2D Kernel: Basic Mode (cout + hout dynamic tiling)
# ============================================================================

@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode},
                    debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 0})
def conv2d_normal_kernel(
    input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_c_tensor: pypto.Tensor([pypto.DYNAMIC]),
    output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    params: dict):
    """
    Conv2D kernel with dynamic tiling on cout + hout axes (原实现).
    
    Dynamic axes:
    - cout: tile_cout = 64
    - hout: tile_hout = 3
    """
    _, _, hin, win = params["shape"][0]
    _, cin, kh, kw = params["shape"][1]
    batch, cout, ho, wo = params["shape"][2]
    
    padding = [0, 0, 0, 0]
    strides = [1, 1]
    dilations = [1, 1]
    
    pypto.set_conv_tile_shapes(
        pypto.pypto_impl.TileL1Info(
            tileHin=5,
            tileHout=3,
            tileWin=32,
            tileWout=32,
            tileCinFmap=32,
            tileCinWeight=32,
            tileN=64,
            tileBatch=1
        ),
        pypto.pypto_impl.TileL0Info(
            tileH=3,
            tileW=32,
            tileK=288,
            tileN=32
        )
    )
    
    pypto.set_vec_tile_shapes(1, 64, 3, 32)
    
    tile_cout = pypto.symbolic_scalar(64)
    tile_hout = pypto.symbolic_scalar(3)
    tile_hin = (3 - 1) * strides[0] + (kh - 1) * dilations[0] + 1
    
    cout_loop = (cout + tile_cout - 1) // tile_cout
    hout_loop = (ho + tile_hout - 1) // tile_hout

    for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
        for hout_idx in pypto.loop(0, hout_loop, 1, name="LOOP_L1_houtnIdx", idx_name="hout_idx"):
            cal_flag, hin_offset, hin_current, update_pad = \
                cal_hin_idxoffset(hout_idx, tile_hout, ho, hin, kh, strides[0], dilations[0], padding[0], padding[1])
            
            cout_offset = cout_idx * tile_cout
            hout_offset = hout_idx * tile_hout
            
            if cal_flag:
                input_a_view = pypto.view(
                    input_a_tensor, 
                    [batch, cin, tile_hin, win], 
                    [0, 0, hin_offset, 0], 
                    valid_shape=[batch, cin, hin_current, win]
                )
                input_b_view = input_b_tensor[cout_offset:cout_offset + tile_cout, 0:cin, 0:kh, 0:kw]
                input_c_view = input_c_tensor[cout_offset:cout_offset + tile_cout]
                
                output_view = pypto.conv(
                    input_a_view, input_b_view,
                    pypto.DT_FP16,
                    strides,
                    (0, 0, 0, 0),
                    dilations,
                    extend_params={"bias_tensor": input_c_view},
                    groups=1
                )
                
                pypto.assemble(output_view, [0, cout_offset, hout_offset, 0], output_c_tensor)


# ============================================================================
# Conv2D Kernel: Full Mode (batch + cout + hout + wout dynamic tiling)
# ============================================================================

@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode},
                    debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 0})
def conv2d_full_dynamic_kernel(
    input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_c_tensor: pypto.Tensor([pypto.DYNAMIC]),
    output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    params: dict):
    """
    Conv2D kernel with dynamic tiling on 4 axes (新增完整动态版本).
    
    Dynamic axes:
    - batch: tile_batch = 2
    - cout: tile_cout = 32
    - hout: tile_hout = 3
    - wout: tile_wout = 16
    """
    _, _, hin, win = params["shape"][0]
    _, cin, kh, kw = params["shape"][1]
    batch, cout, ho, wo = params["shape"][2]
    
    padding = [0, 0, 0, 0]
    strides = [1, 1]
    dilations = [1, 1]
    
    pypto.set_conv_tile_shapes(
        pypto.pypto_impl.TileL1Info(
            tileHin=5,
            tileHout=3,
            tileWin=16,
            tileWout=16,
            tileCinFmap=32,
            tileCinWeight=32,
            tileN=32,
            tileBatch=2
        ),
        pypto.pypto_impl.TileL0Info(
            tileH=3,
            tileW=16,
            tileK=288,
            tileN=16
        )
    )
    
    pypto.set_vec_tile_shapes(1, 64, 3, 32)
    
    tile_batch = pypto.symbolic_scalar(2)
    tile_cout = pypto.symbolic_scalar(32)
    tile_hout = pypto.symbolic_scalar(3)
    tile_wout = pypto.symbolic_scalar(16)
    
    batch_loop = (batch + tile_batch - 1) // tile_batch
    cout_loop = (cout + tile_cout - 1) // tile_cout
    hout_loop = (ho + tile_hout - 1) // tile_hout
    wout_loop = (wo + tile_wout - 1) // tile_wout
    
    tile_hin = (tile_hout - 1) * strides[0] + (kh - 1) * dilations[0] + 1
    tile_win = (tile_wout - 1) * strides[1] + (kw - 1) * dilations[1] + 1
    
    for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_L1_batchIdx", idx_name="batch_idx"):
        for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
            for hout_idx in pypto.loop(0, hout_loop, 1, name="LOOP_L1_houtIdx", idx_name="hout_idx"):
                for wout_idx in pypto.loop(0, wout_loop, 1, name="LOOP_L1_woutIdx", idx_name="wout_idx"):
                    
                    batch_offset = batch_idx * tile_batch
                    cout_offset = cout_idx * tile_cout
                    hout_offset = hout_idx * tile_hout
                    wout_offset = wout_idx * tile_wout
                    
                    cal_flag_h, hin_offset, hin_current, update_pad_h = \
                        cal_hin_idxoffset(hout_idx, tile_hout, ho, hin, kh, 
                                         strides[0], dilations[0], padding[0], padding[1])
                    
                    cal_flag_w, win_offset, win_current, update_pad_w = \
                        cal_win_idxoffset(wout_idx, tile_wout, wo, win, kw,
                                         strides[1], dilations[1], padding[2], padding[3])
                    
                    if cal_flag_h and cal_flag_w:
                        input_a_view = pypto.view(
                            input_a_tensor,
                            [tile_batch, cin, tile_hin, tile_win],
                            [batch_offset, 0, hin_offset, win_offset],
                            valid_shape=[tile_batch, cin, hin_current, win_current]
                        )
                        
                        input_b_view = input_b_tensor[cout_offset:cout_offset + tile_cout, 0:cin, 0:kh, 0:kw]
                        input_c_view = input_c_tensor[cout_offset:cout_offset + tile_cout]
                        
                        output_view = pypto.conv(
                            input_a_view, input_b_view,
                            pypto.DT_FP16,
                            strides,
                            (update_pad_h[0], update_pad_h[1], update_pad_w[0], update_pad_w[1]),
                            dilations,
                            extend_params={"bias_tensor": input_c_view},
                            groups=1
                        )
                        
                        pypto.assemble(
                            output_view,
                            [batch_offset, cout_offset, hout_offset, wout_offset],
                            output_c_tensor
                        )


# ============================================================================
# Conv1D Kernel (cout dynamic tiling)
# ============================================================================

@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode},
                    debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1})
def conv1d_normal_kernel(
    input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_c_tensor: pypto.Tensor([pypto.DYNAMIC]),
    output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    params: dict):
    """
    Conv1D kernel with dynamic tiling on cout axis.
    
    Dynamic axes:
    - cout: tile_cout = 16
    """
    _, cin_fmap, win = params["shape"][0]
    _, cin_weight, kw = params["shape"][1]
    batch, cout, wo = params["shape"][2]
    
    padding = [1, 1]
    strides = [1]
    dilations = [1]
    
    pypto.set_conv_tile_shapes(
        pypto.pypto_impl.TileL1Info(
            tileHin=1,
            tileHout=1,
            tileWin=2048,
            tileWout=2048,
            tileCinFmap=16,
            tileCinWeight=16,
            tileN=16,
            tileBatch=1
        ),
        pypto.pypto_impl.TileL0Info(
            tileH=1,
            tileW=2048,
            tileK=16,
            tileN=16
        )
    )
    
    pypto.set_vec_tile_shapes(16)
    
    tile_cout = pypto.symbolic_scalar(16)
    cout_loop = (cout + tile_cout - 1) // tile_cout

    for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
        cout_offset = cout_idx * tile_cout
        input_a_view = input_a_tensor[0:batch, 0:cin_fmap, 0:win]
        input_b_view = input_b_tensor[cout_offset:cout_offset + tile_cout, 0:cin_weight, 0:kw]
        input_c_view = input_c_tensor[cout_offset:cout_offset + tile_cout]
        
        output_view = pypto.conv(
            input_a_view, input_b_view,
            pypto.DT_FP16,
            strides,
            padding,
            dilations,
            extend_params={"bias_tensor": input_c_view},
            groups=1
        )
        
        output_c_tensor[0:batch, cout_offset:cout_offset + tile_cout, 0:wo] = output_view


# ============================================================================
# Test Functions
# ============================================================================

def test_conv2d_basic(device_id: int = None):
    """Test conv2d with basic dynamic tiling (cout + hout)"""
    print("=" * 60)
    print("Test: Conv2D Basic Mode (cout + hout dynamic)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'
    
    dtype = torch.float16
    fmap_shape = (1, 32, 8, 34)
    weight_shape = (64, 32, 3, 3)
    bias_shape = (64,)
    out_shape = (1, 64, 6, 32)
    
    a = torch.randn(fmap_shape, dtype=dtype, device=device)
    b = torch.randn(weight_shape, dtype=dtype, device=device)
    c = torch.randn(bias_shape, dtype=dtype, device=device)
    expected = torch.conv2d(a, b, bias=c, padding=(0, 0), stride=(1, 1), dilation=1, groups=1)
    
    out = torch.empty(out_shape, dtype=dtype, device=device)
    conv2d_normal_kernel(a, b, c, out, {"shape": [fmap_shape, weight_shape, out_shape]})
    
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
        print(f"✓ Results match! Max diff: {torch.max(torch.abs(out - expected)).item():.6f}")
    
    print(f"Input shape: {fmap_shape}")
    print(f"Weight shape: {weight_shape}")
    print(f"Output shape: {out_shape}")
    print(f"Tile config: cout_tile=64, hout_tile=3")
    print(f"Loop counts: cout_loop=1, hout_loop=2")
    print("✓ Basic conv2d test completed successfully")


def test_conv2d_full_dynamic(device_id: int = None):
    """Test conv2d with full dynamic tiling (batch + cout + hout + wout)"""
    print("=" * 60)
    print("Test: Conv2D Full Dynamic Mode (batch + cout + hout + wout)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'
    
    dtype = torch.float16
    
    fmap_shape = (4, 32, 8, 64)
    weight_shape = (64, 32, 3, 3)
    bias_shape = (64,)
    out_shape = (4, 64, 6, 62)
    
    a = torch.randn(fmap_shape, dtype=dtype, device=device)
    b = torch.randn(weight_shape, dtype=dtype, device=device)
    c = torch.randn(bias_shape, dtype=dtype, device=device)
    
    expected = torch.conv2d(a, b, bias=c, padding=(0, 0), stride=(1, 1), dilation=1, groups=1)
    
    out = torch.empty(out_shape, dtype=dtype, device=device)
    conv2d_full_dynamic_kernel(a, b, c, out, {"shape": [fmap_shape, weight_shape, out_shape]})
    
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
        print(f"✓ Results match! Max diff: {torch.max(torch.abs(out - expected)).item():.6f}")
    
    print(f"Input shape: {fmap_shape}")
    print(f"Weight shape: {weight_shape}")
    print(f"Output shape: {out_shape}")
    print(f"Tile config: batch_tile=2, cout_tile=32, hout_tile=3, wout_tile=16")
    print(f"Loop counts: batch_loop=2, cout_loop=2, hout_loop=2, wout_loop=4")
    print("✓ Full dynamic conv2d test completed successfully")


def test_conv1d_basic(device_id: int = None):
    """Test conv1d with cout dynamic tiling"""
    print("=" * 60)
    print("Test: Conv1D Basic Mode (cout dynamic)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'
    
    dtype = torch.float16
    fmap_shape = (1, 16, 2048)
    weight_shape = (16, 16, 3)
    bias_shape = (16,)
    out_shape = (1, 16, 2048)
    
    a = torch.randn(fmap_shape, dtype=dtype, device=device)
    b = torch.randn(weight_shape, dtype=dtype, device=device)
    c = torch.zeros(bias_shape, dtype=dtype, device=device)
    
    expected = torch.conv1d(a, b, bias=c, padding=1, stride=1, dilation=1, groups=1)
    
    out = torch.empty(out_shape, dtype=dtype, device=device)
    conv1d_normal_kernel(a, b, c, out, {"shape": [fmap_shape, weight_shape, out_shape]})
    
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
        print(f"✓ Results match! Max diff: {torch.max(torch.abs(out - expected)).item():.6f}")
    
    print(f"Input shape: {fmap_shape}")
    print(f"Weight shape: {weight_shape}")
    print(f"Output shape: {out_shape}")
    print(f"Tile config: cout_tile=16")
    print(f"Loop counts: cout_loop=1")
    print("✓ Conv1D test completed successfully")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Conv Dynamic Tiling Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test basic          Run basic 2-axis dynamic test
  %(prog)s --test full           Run full 4-axis dynamic test
  %(prog)s --test conv1d         Run conv1d test
  %(prog)s --test all            Run all tests
  %(prog)s --device_id 0         Use NPU device 0
  %(prog)s --run_mode sim        Use simulation mode
        """
    )
    
    parser.add_argument('--device_id', type=int, default=0, help='NPU device ID')
    parser.add_argument('--run_mode', type=str, default='npu', choices=['npu', 'sim'],
                       help='Run mode: npu or sim')
    parser.add_argument('--test', type=str, default='basic', 
                       choices=['basic', 'full', 'conv1d', 'all'],
                       help='Test case to run')
    
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
        if args.test == 'basic':
            test_conv2d_basic(args.device_id)
        elif args.test == 'full':
            test_conv2d_full_dynamic(args.device_id)
        elif args.test == 'conv1d':
            test_conv1d_basic(args.device_id)
        elif args.test == 'all':
            test_conv2d_basic(args.device_id)
            print("\n" + "=" * 60 + "\n")
            test_conv2d_full_dynamic(args.device_id)
            print("\n" + "=" * 60 + "\n")
            test_conv1d_basic(args.device_id)
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()