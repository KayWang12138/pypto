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
                    debug_options={"compile_debug_mode": 0},
                    use_cache=False)
def conv_normal_kernel(
    input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    input_c_tensor: pypto.Tensor([pypto.DYNAMIC]),
    output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC]),
    params: dict):
    _, _, hin, win = params["shape"][0]
    _, cin, kh, kw = params["shape"][1]
    batch, cout, ho, wo = params["shape"][2]
    pading = [1, 1, 1, 1]
    strides = [2, 2]
    dilations = [1, 1]
    pypto.set_conv_tile_shapes(
            pypto.pypto_impl.TileL1Info(
                tileHin=128,
                tileHout=2,
                tileWin=64,
                tileWout=64,
                tileCinFmap=16,
                tileCinWeight=16,
                tileN=64,
                tileBatch=1
            ),
            pypto.pypto_impl.TileL0Info(
                tileH=2,
                tileW=64,
                tileK=16,
                tileN=64
            )
        )
    pypto.set_vec_tile_shapes(1, 64, 2, 64)
    tile_cout = pypto.symbolic_scalar(512)
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

def test_conv_basic(device_id: int = None):
    """Test basic conv"""
    print("=" * 60)
    print("Test: Basic Conv")
    print("=" * 60)
    
    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'
    
    dtype = torch.float16
    fmap_shape = (1, 32, 128, 128)
    weight_shape = (512, 32, 3, 3)
    bias_shape = (512,)
    out_shape = (1, 512, 64, 64)
    a = torch.randn(fmap_shape, dtype=dtype, device=device)
    b = torch.randn(weight_shape, dtype=dtype, device=device)
    c = torch.zeros(bias_shape, dtype=dtype, device=device)
    expected = torch.conv2d(a, b, padding=(1,1), stride=(2,2), dilation=1, groups=1)

    out = torch.empty(out_shape, dtype=dtype, device=device)
    conv_normal_kernel(a, b, c, out, {"shape": [[1, 32, 128, 128], [512, 32, 3, 3], [1, 512, 64, 64]]})
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic matrix completed successfully")

test_conv_basic(0)