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
Matrix Multiplication (matmul) Operation Examples for PyPTO

This file contains all matrix multiplication examples merged into a single file.
You can run all examples or select specific ones using command-line arguments.

Usage:
    python matmul_ops.py              # Run all examples
    python matmul_ops.py --list       # List all available examples
    python matmul_ops.py matmul::test_matmul_basic    # Run a specific case
"""

import argparse
import os
import sys
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose
# import pdb

global_run_mode = pypto.RunMode.NPU

def cal_hin_idxoffset(hout_idx, tile_hout, ho, hin, kh, strideh, dilationh, pad_top, pad_bottom):
    pad_top_current = pypto.symbolic_scalar(0)
    pad_botom_curent = pypto.symbolic_scalar(0)
    hin_offset = hout_idx * tile_hout * strideh - pad_top
    need_hin = (tile_hout - 1) * strideh + (kh - 1) * dilationh + 1
    if pypto.cond(hin_offset + need_hin <= 0):
        #当前块全部落在上pad，可以直接跳过conv逻辑
        return False, 0, 0, (pad_top_current, pad_botom_curent)
    if pypto.cond(hin - hin_offset <= 0):
        #全部落在下pad中
        return False, 0, 0, (pad_top_current, pad_botom_curent)
    pad_top_current = (-hin_offset).max(0)
    hin_current = hin_offset + need_hin
    pad_botom_curent = (hin_current - hin).max(0)
    hin_offset = hin_offset.max(0)
    hin_current = (hin - hin_offset).min(need_hin)
    return True, hin_offset, hin_current, (pad_top_current, pad_botom_curent)


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode},
                    debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 0})
def conv2d_normal_kernel(
    input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_c_tensor: pypto.Tensor([pypto.DYNAMIC]),
    output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    params: dict):
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
    # pypto.set_pass_options(cube_l1_reuse_setting={-1 : 1}, cube_nbuffer_setting={-1 : 1})
    pypto.set_vec_tile_shapes(1, 64, 3, 32)
    tile_cout = pypto.symbolic_scalar(64)
    tile_hout = pypto.symbolic_scalar(3)
    tile_hin = (3 - 1) * strides[0] + (kh - 1) * dilations[0] + 1
    cout_loop = (cout + tile_cout - 1) // tile_cout
    hout_loop = (ho + tile_hout - 1) // tile_hout

    for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
        for hout_idx in pypto.loop(0, hout_loop, 1, name="LOOP_L1_houtnIdx", idx_name="hout_idx"):
            # pdb.set_trace()
            cal_flag, hin_offset, hin_current, update_pad = \
                cal_hin_idxoffset(hout_idx, tile_hout, ho, hin, kh, strides[0], dilations[0], padding[0], padding[1])
            cout_offset = cout_idx * tile_cout
            hout_offset = hout_idx * tile_hout
            if cal_flag:
                # input_a_view = input_a_tensor[0:batch, 0:cin, hin_offset:hin_offset + hin_current, 0:win]
                input_a_view = pypto.view(input_a_tensor, [batch, cin, tile_hin, win], [0, 0, hin_offset, 0], valid_shape=[batch, cin, hin_current, win])
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
                print("lxw========", output_view.shape)
                print("lxw========", output_view.base())
                # output_c_tensor[0:batch, cout_offset: cout_offset + tile_cout, hout_offset:hout_offset + tile_hout, 0:wo] = output_view
                pypto.assemble(output_view, [0, cout_offset, hout_offset, 0], output_c_tensor)

def test_conv2d_basic(device_id: int = None):
    """Test basic conv"""
    print("=" * 60)
    print("Test: Basic Conv")
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
    expected = torch.conv2d(a, b, bias=c, padding=(0,0), stride=(1,1), dilation=1, groups=1)

    out = torch.empty(out_shape, dtype=dtype, device=device)
    conv2d_normal_kernel(a, b, c, out, {"shape": [[1, 32, 8, 34], [64, 32, 3, 3], [1, 64, 6, 32]]})
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic matrix completed successfully")


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode},
                    debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1})
def conv1d_normal_kernel(
    input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_c_tensor: pypto.Tensor([pypto.DYNAMIC]),
    output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    params: dict):
    _, cin_fmap, win = params["shape"][0]
    _, cin_weight, kw = params["shape"][1]
    batch, cout, wo = params["shape"][2]
    pading = [1, 1]
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
    # pypto.set_pass_options(cube_l1_reuse_setting={-1 : 1}, cube_nbuffer_setting={-1 : 1})
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
                pading,
                dilations,
                extend_params={"bias_tensor": input_c_view},
                groups=1
            )
        output_c_tensor[0:batch, cout_offset: cout_offset + cout_idx, 0:wo] = output_view

def test_conv1d_basic(device_id: int = None):
    """Test basic conv"""
    print("=" * 60)
    print("Test: Basic Conv")
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
    conv1d_normal_kernel(a, b, c, out, {"shape": [[1, 16, 2048], [16, 16, 3], [1, 16, 2048]]})
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic matrix completed successfully")

test_conv2d_basic(0)
# test_conv1d_basic(0)