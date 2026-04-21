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
    python test_conv_dynamic.py --test conv2d        # Run conv2d test
    python test_conv_dynamic.py --test conv1d        # Run conv1d test
    python test_conv_dynamic.py --test all           # Run all tests
    python test_conv_dynamic.py --device_id 0        # Specify NPU device
    python test_conv_dynamic.py --run_mode sim       # Use simulation mode
"""

import argparse
import os
import sys
import pypto
import torch
import numpy as np
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
    Calculate hin offset and current size based on hout_idx (for Conv2D).
    
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
# Unified Conv Dynamic Kernel (supports Conv1D and Conv2D)
# ============================================================================

@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode},
                    debug_options={"compile_debug_mode": 0, "runtime_debug_mode": 1})
def conv_dynamic_kernel(
    input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_c_tensor: pypto.Tensor([pypto.DYNAMIC]),
    output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    params: dict):
    """
    Unified convolution kernel with dynamic tiling.
    Automatically adapts to Conv1D (3D) or Conv2D (4D) based on tensor dimensions.
    
    Dynamic tiling axes:
    Conv2D (4D):
      - batch: tile_batch = 2 (前端循环，TileShape.tileBatch=1 固定)
      - cout: tile_cout = 32 (TileShape动态)
      - hout: tile_hout = 3 (TileShape动态)
      - wout: tile_wout = 16 (前端循环)
    
    Conv1D (3D):
      - batch: tile_batch = 2 (前端循环，TileShape.tileBatch=1 固定)
      - cout: tile_cout = 16 (TileShape动态)
      - wout: tile_wout = 1024 (TileShape动态)
    
    注意：tileBatch 必须为 1，batch轴通过前端循环切分
    """
    is_conv2d = len(params["shape"][0]) == 4  # 判断是否为 Conv2D

    # 解析 shape 参数
    if is_conv2d:
        _, _, hin, win = params["shape"][0]
        _, cin, kh, kw = params["shape"][1]
        batch, cout, ho, wo = params["shape"][2]
        hin_dim = hin
        kh_dim = kh
        stride_h = params.get("strides", [1, 1])[0]
        dilation_h = params.get("dilations", [1, 1])[0]
        pad_h = params.get("padding", [0, 0, 0, 0])[:2]  # top, bottom
    else:
        _, cin, win = params["shape"][0]
        _, cin_weight, kw = params["shape"][1]
        batch, cout, wo = params["shape"][2]
        hin = 1  # Conv1D 没有 h 维度
        hin_dim = None
        kh = 1
        kh_dim = None
        ho = 1
        stride_h = 1
        dilation_h = 1
        pad_h = [0, 0]

    # 通用参数
    stride_w = params.get("strides", [1] if not is_conv2d else [1, 1])[-1]
    dilation_w = params.get("dilations", [1] if not is_conv2d else [1, 1])[-1]
    pad_w = params.get("padding", [1, 1] if not is_conv2d else [0, 0, 0, 0])[-2:]  # left, right

    # TileShape 配置（根据 Conv 类型）
    if is_conv2d:
        tile_l1_config = pypto.pypto_impl.TileL1Info(
            tileHin=4,
            tileHout=4,
            tileWin=16,
            tileWout=16,
            tileCinFmap=32,
            tileCinWeight=32,
            tileN=32,
            tileBatch=1  # 必须为 1
        )
        tile_l0_config = pypto.pypto_impl.TileL0Info(
            tileH=4,
            tileW=16,
            tileK=288,
            tileN=32
        )
        vec_tile_config = (1, 32, 16, 16)

        tile_batch = pypto.symbolic_scalar(1)
        tile_cout = pypto.symbolic_scalar(64)
        tile_hout = pypto.symbolic_scalar(16)
        tile_wout = pypto.symbolic_scalar(16)
    else:
        # Conv1D: 与 Conv2D 保持一致的动态切分模式
        # batch: 前端循环，cout: TileShape，wout: 前端循环

        tile_l1_config = pypto.pypto_impl.TileL1Info(
            tileHin=1,
            tileHout=1,
            tileWin=16,
            tileWout=16,
            tileCinFmap=16,
            tileCinWeight=16,
            tileN=16,
            tileBatch=1  # 必须为 1
        )
        tile_l0_config = pypto.pypto_impl.TileL0Info(
            tileH=1,
            tileW=16,
            tileK=16,
            tileN=16
        )
        vec_tile_config = (1, 16, 16)

        # 前端循环切分大小
        tile_batch = pypto.symbolic_scalar(1)     # batch 前端循环
        tile_cout = pypto.symbolic_scalar(16)     # cout TileShape 切分
        tile_hout = pypto.symbolic_scalar(1)      # hout 固定为 1（Conv1D无h维度）
        tile_wout = pypto.symbolic_scalar(16)  # wout 前端循环
        
        # 计算前端循环切分需要的 win 尺寸
        tile_win = (tile_wout - 1) * stride_w + (kw - 1) * dilation_w + 1

    pypto.set_conv_tile_shapes(tile_l1_config, tile_l0_config)
    pypto.set_vec_tile_shapes(*vec_tile_config)

    # 计算循环次数
    batch_loop = (batch + tile_batch - 1) // tile_batch
    cout_loop = (cout + tile_cout - 1) // tile_cout
    hout_loop = (ho + tile_hout - 1) // tile_hout if is_conv2d else 1
    wout_loop = (wo + tile_wout - 1) // tile_wout

    # 计算输入 tile 大小
    if is_conv2d:
        tile_hin = (tile_hout - 1) * stride_h + (kh - 1) * dilation_h + 1
        tile_win = (tile_wout - 1) * stride_w + (kw - 1) * dilation_w + 1
    else:
        tile_win = (tile_wout - 1) * stride_w + (kw - 1) * dilation_w + 1

    # 嵌套循环（根据 Conv 类型调整）
    for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_L1_batchIdx", idx_name="batch_idx"):
        for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
            for hout_idx in (pypto.loop(0, hout_loop, 1, name="LOOP_L1_houtIdx", idx_name="hout_idx") 
                             if is_conv2d else pypto.loop(0, 1, 1, name="LOOP_L1_houtIdx", idx_name="hout_idx")):
                for wout_idx in pypto.loop(0, wout_loop, 1, name="LOOP_L1_woutIdx", idx_name="wout_idx"):

                    # 计算偏移量
                    batch_offset = batch_idx * tile_batch
                    cout_offset = cout_idx * tile_cout
                    hout_offset = hout_idx * tile_hout if is_conv2d else 0
                    wout_offset = wout_idx * tile_wout

                    # 计算 win 区域
                    cal_flag_w, win_offset, win_current, update_pad_w = \
                        cal_win_idxoffset(wout_idx, tile_wout, wo, win, kw,
                                         stride_w, dilation_w, pad_w[0], pad_w[1])

                    # 计算 hin 区域（仅 Conv2D）
                    if is_conv2d:
                        cal_flag_h, hin_offset, hin_current, update_pad_h = \
                            cal_hin_idxoffset(hout_idx, tile_hout, ho, hin, kh,
                                             stride_h, dilation_h, pad_h[0], pad_h[1])
                        cal_flag = cal_flag_h and cal_flag_w
                    else:
                        cal_flag = cal_flag_w

                    if cal_flag:
                        # 创建输入视图
                        if is_conv2d:
                            input_a_view = pypto.view(
                                input_a_tensor,
                                [tile_batch, cin, tile_hin, tile_win],
                                [batch_offset, 0, hin_offset, win_offset],
                                valid_shape=[tile_batch, cin, hin_current, win_current]
                            )
                            input_b_view = input_b_tensor[cout_offset:cout_offset + tile_cout, 0:cin, 0:kh, 0:kw]
                            # padding = (update_pad_h[0], update_pad_h[1], update_pad_w[0], update_pad_w[1])
                            padding = (0, 0, 0, 0)
                            strides = [stride_h, stride_w]
                            dilations = [dilation_h, dilation_w]
                        else:
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
                        
                        input_c_view = input_c_tensor[cout_offset:cout_offset + tile_cout]

                        # 执行 conv
                        output_view = pypto.conv(
                            input_a_view, input_b_view,
                            pypto.DT_FP16,
                            strides,
                            padding,
                            dilations,
                            extend_params={"bias_tensor": input_c_view},
                            groups=1
                        )

                        # 组装结果
                        if is_conv2d:
                            pypto.assemble(
                                output_view,
                                [batch_offset, cout_offset, hout_offset, wout_offset],
                                output_c_tensor
                            )
                        else:
                            pypto.assemble(
                                output_view,
                                [batch_offset, cout_offset, wout_offset],
                                output_c_tensor
                            )


# ============================================================================
# Test Functions
# ============================================================================

def test_conv2d(device_id: int = None):
    """Test Conv2D with unified dynamic kernel"""
    print("=" * 60)
    print("Test: Conv2D (Unified Dynamic Kernel)")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float16

    fmap_shape = (4, 32, 34, 34)
    weight_shape = (128, 32, 3, 3)
    bias_shape = (128,)
    out_shape = (4, 128, 32, 32)

    a = torch.randn(fmap_shape, dtype=dtype, device=device)
    b = torch.randn(weight_shape, dtype=dtype, device=device)
    c = torch.randn(bias_shape, dtype=dtype, device=device)

    expected = torch.conv2d(a, b, bias=c, padding=(0, 0), stride=(1, 1), dilation=1, groups=1)

    out = torch.empty(out_shape, dtype=dtype, device=device)
    conv_dynamic_kernel(a, b, c, out, {
        "shape": [fmap_shape, weight_shape, out_shape],
        "strides": [1, 1],
        "padding": [0, 0, 0, 0],
        "dilations": [1, 1]
    })

    print(f"Input shape: {fmap_shape}")
    print(f"Weight shape: {weight_shape}")
    print(f"Output shape: {out_shape}")
    
    if global_run_mode == pypto.RunMode.NPU:
        all_pass = compare_precision(out, expected, rtol=1e-3, atol=1e-3, max_errors=100)
        if all_pass:
            print("✓ Conv2D test completed successfully - All points passed!")
        else:
            print("✗ Conv2D test failed - Some points did not match tolerance")


def test_conv1d(device_id: int = None):
    """Test Conv1D with unified dynamic kernel"""
    print("=" * 60)
    print("Test: Conv1D (Unified Dynamic Kernel)")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float16
    fmap_shape = (2, 16, 2048)
    weight_shape = (16, 16, 3)
    bias_shape = (16,)
    out_shape = (2, 16, 2048)

    a = torch.randn(fmap_shape, dtype=dtype, device=device)
    b = torch.randn(weight_shape, dtype=dtype, device=device)
    c = torch.zeros(bias_shape, dtype=dtype, device=device)

    expected = torch.conv1d(a, b, bias=c, padding=1, stride=1, dilation=1, groups=1)

    out = torch.empty(out_shape, dtype=dtype, device=device)
    conv_dynamic_kernel(a, b, c, out, {
        "shape": [fmap_shape, weight_shape, out_shape],
        "strides": [1],
        "padding": [1, 1],
        "dilations": [1]
    })

    print(f"Input shape: {fmap_shape}")
    print(f"Weight shape: {weight_shape}")
    print(f"Output shape: {out_shape}")
    
    if global_run_mode == pypto.RunMode.NPU:
        all_pass = compare_precision(out, expected, rtol=1e-3, atol=1e-3, max_errors=100)
        if all_pass:
            print("✓ Conv1D test completed successfully - All points passed!")
        else:
            print("✗ Conv1D test failed - Some points did not match tolerance")



# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Conv Dynamic Tiling Examples (Unified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test conv2d         Run conv2d test
  %(prog)s --test conv1d         Run conv1d test
  %(prog)s --test all            Run all tests
  %(prog)s --device_id 0         Use NPU device 0
  %(prog)s --run_mode sim        Use simulation mode

重要约束：
  - TileL1Info.tileBatch 必须为 1（硬件约束）
  - batch 轴动态切分通过前端循环实现
        """
    )
    
    parser.add_argument('--device_id', type=int, default=0, help='NPU device ID')
    parser.add_argument('--run_mode', type=str, default='npu', choices=['npu', 'sim'],
                       help='Run mode: npu or sim')
    parser.add_argument('--test', type=str, default='conv2d', 
                       choices=['conv2d', 'conv1d', 'all'],
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
        if args.test == 'conv2d':
            test_conv2d(args.device_id)
        elif args.test == 'conv1d':
            test_conv1d(args.device_id)
        elif args.test == 'all':
            test_conv2d(args.device_id)
            print("\n" + "=" * 60 + "\n")
            test_conv1d(args.device_id)
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()