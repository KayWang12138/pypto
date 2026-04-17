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

global_run_mode = pypto.RunMode.NPU

@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode},
                    debug_options={"compile_debug_mode": 1}
                    )
def matmul_normal_kernel(
    input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    params):
    m, n = output_c_tensor.shape
    k = params["input_shape"][0][1]
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [32, 32])
    pypto.set_vec_tile_shapes(32, 32)
    tile_m = pypto.symbolic_scalar(32)
    tile_n = pypto.symbolic_scalar(64)
    m_loop = (m + tile_m - 1) // tile_m
    n_loop = (n + tile_n - 1) // tile_n

    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_LO_mIdx", idx_name="m_idx"):

        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
            m_offset = m_idx * tile_m
            n_offset = n_idx * tile_n
            if params["a_trans"]:
                input_a_view = input_a_tensor[0:k, m_offset:m_offset + tile_m]
            else:
                input_a_view = input_a_tensor[m_offset:m_offset + tile_m, 0:k]
            if params["b_trans"]:
                input_b_view = input_b_tensor[n_offset:n_offset + tile_n, 0:k]
            else:
                input_b_view = input_b_tensor[0:k, n_offset:n_offset + tile_n]

            output_view = pypto.matmul(input_a_view, input_b_view, out_dtype=pypto.DT_FP32,
                                    a_trans=params["a_trans"], b_trans=params["b_trans"], c_matrix_nz=params["c_matrix_nz"])
            output_c_tensor[m_offset: m_offset + tile_m, n_offset: n_offset + tile_n] = output_view

def test_matmul_basic(device_id: int = None):
    """Test basic matrix multiplication"""
    print("=" * 60)
    print("Test: Basic Matrix Multiplication")
    print("=" * 60)
    
    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'
    
    dtype = torch.float32
    a_shape = [48, 64]
    b_shape = [64, 128]
    c_shape = [48, 128]
    a = torch.ones(a_shape, dtype=dtype, device=device)
    b = torch.ones(b_shape, dtype=dtype, device=device)
    expected = torch.matmul(a, b)

    out = torch.empty(c_shape, dtype=dtype, device=device)
    matmul_normal_kernel(a, b, out, params={"input_shape": [a_shape, b_shape], "a_trans": False, "b_trans": False, "c_matrix_nz": False, "output_dtype": dtype})
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic matrix multiplication completed successfully")


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode},
                    debug_options={"compile_debug_mode": 0, "runtime_debug_mode": 1}
                    )
def conv2d_normal_kernel(
    input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_c_tensor: pypto.Tensor([pypto.DYNAMIC]),
    output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    params: dict):
    _, _, hin, win = params["shape"][0]
    _, cin, kh, kw = params["shape"][1]
    batch, cout, ho, wo = params["shape"][2]
    pading = [1, 1, 1, 1]
    strides = [1, 1]
    dilations = [1, 1]
    pypto.set_conv_tile_shapes(
            pypto.pypto_impl.TileL1Info(
                tileHin=3,
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
    pypto.set_vec_tile_shapes(1, 32, 3, 32)
    tile_cout = pypto.symbolic_scalar(64)
    cout_loop = (cout + tile_cout - 1) // tile_cout

    for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
        cout_offset = cout_idx * tile_cout
        input_a_view = input_a_tensor[0:batch, 0:cin, 0:hin, 0:win]
        input_b_view = input_b_tensor[cout_offset:cout_offset + tile_cout, 0:cin, 0:kh, 0:kw]
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
        output_c_tensor[0:batch, cout_offset: cout_offset + cout_idx, 0:ho, 0:wo] = output_view

def test_conv2d_basic(device_id: int = None):
    """Test basic conv"""
    print("=" * 60)
    print("Test: Basic Conv")
    print("=" * 60)
    
    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'
    
    dtype = torch.float16
    fmap_shape = (1, 32, 3, 32)
    weight_shape = (256, 32, 3, 3)
    bias_shape = (256,)
    out_shape = (1, 256, 3, 32)
    a = torch.randn(fmap_shape, dtype=dtype, device=device)
    b = torch.randn(weight_shape, dtype=dtype, device=device)
    c = torch.randn(bias_shape, dtype=dtype, device=device)
    expected = torch.conv2d(a, b, bias=c, padding=(1,1), stride=(1,1), dilation=1, groups=1)

    out = torch.empty(out_shape, dtype=dtype, device=device)
    conv2d_normal_kernel(a, b, c, out, {"shape": [[1, 32, 3, 32], [256, 32, 3, 3], [1, 256, 3, 32]]})
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic matrix completed successfully")


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode},
                    debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1}
                    )
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

# test_matmul_basic(0)
test_conv2d_basic(0)
# test_conv1d_basic(0)
