#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
from dataclasses import dataclass, field
from typing import Optional
import os
import numpy as np
import torch
import torch_npu
import pypto
import pytest
from numpy.testing import assert_allclose
import torch.nn.functional as F
from pypto import pypto_impl
import multiprocessing as mp

def create_conv_kernel(fmap_shape, weight_shape, bias_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations, groups = 1):
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
    )

    def conv_kernel(
        fmap: pypto.Tensor(fmap_shape, dtype),
        weight: pypto.Tensor(weight_shape, dtype),
        bias: pypto.Tensor(bias_shape, dtype)
    ) -> pypto.Tensor(out_shape, dtype):
        pypto.set_conv_tile_shapes(tileL1Info, tileL0Info)
        extend_params = {'bias_tensor': bias}
        output = pypto.conv(fmap, weight, dtype, strides, pads, dilations, extend_params = extend_params, groups = groups)
        return output

    return conv_kernel

def create_conv_kernel_no_bias(fmap_shape, weight_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations, groups = 1):
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
    )

    def conv_kernel(
        fmap: pypto.Tensor(fmap_shape, dtype),
        weight: pypto.Tensor(weight_shape, dtype),
    ) -> pypto.Tensor(out_shape, dtype):
        pypto.set_conv_tile_shapes(tileL1Info, tileL0Info)
        extend_params = {}
        output = pypto.conv(fmap, weight, dtype, strides, pads, dilations, extend_params = extend_params, groups = groups)
        return output

    return conv_kernel

def test_conv2d_fp16_basic_with_bias():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 1, 16)
    weight_shape = (16, 16, 1, 1)
    bias_shape = (16,)
    out_shape = (1, 16, 1, 16)
    dtype = pypto.DT_FP16
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 16,
        tileWout = 16,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 16,
        tileK = 16,
        tileN = 16
    )
    strides = [1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1]
    dtype_torch = torch.float16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c = torch.rand(bias_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel(fmap_shape, weight_shape, bias_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations)(a, b, c)
    golden = torch.nn.functional.conv2d(a, b, c, stride=1, padding=0)

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv2d_bf16_group_2():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 32, 1, 16)
    weight_shape = (32, 16, 1, 1)
    out_shape = (1, 32, 1, 16)
    dtype = pypto.DT_BF16
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 16,
        tileWout = 16,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 16,
        tileK = 16,
        tileN = 16
    )
    strides = [1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1]
    dtype_torch = torch.bfloat16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel_no_bias(fmap_shape, weight_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations, 2)(a, b)
    golden = torch.nn.functional.conv2d(a, b, stride=1, padding=0, groups=2)

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv2d_fp32_strides_2():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 5, 16)
    weight_shape = (16, 16, 3, 3)
    out_shape = (1, 16, 2, 7)
    dtype = pypto.DT_FP32
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 16,
        tileWout = 16,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 16,
        tileK = 16,
        tileN = 16
    )
    strides = [2, 2]
    pads = [0, 0, 0, 0]
    dilations = [1, 1]
    dtype_torch = torch.float32
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel_no_bias(fmap_shape, weight_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations)(a, b)
    golden = torch.nn.functional.conv2d(a, b, stride=(2,2), padding=0)

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv2d_fp16_dilation2():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (3, 16, 5, 16)
    weight_shape = (32, 16, 3, 3)
    bias_shape = (32,)
    out_shape = (3, 32, 1, 12)
    dtype = pypto.DT_FP16
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 16,
        tileWout = 16,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 16,
        tileK = 16,
        tileN = 16
    )
    strides = [1, 1]
    pads = [0, 0, 0, 0]
    dilations = [2, 2]
    dtype_torch = torch.float16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c = torch.rand(bias_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel(fmap_shape, weight_shape, bias_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations)(a, b, c)
    golden = torch.nn.functional.conv2d(a, b, c, stride=1, padding=0,dilation=(2,2))

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv2d_fp32_pad_1_w32():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 5, 32)
    weight_shape = (16, 16, 3, 3)
    out_shape = (1, 16, 5, 32)
    dtype = pypto.DT_FP32
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 32,
        tileWout = 32,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 32,
        tileK = 16,
        tileN = 16
    )
    strides = [1, 1]
    pads = [1, 1, 1, 1]
    dilations = [1, 1]
    dtype_torch = torch.float32
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel_no_bias(fmap_shape, weight_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations)(a, b)
    golden = torch.nn.functional.conv2d(a, b, stride=1, padding=(1,1))

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv1d_fp16_basic_with_bias():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 16)
    weight_shape = (16, 16, 1)
    bias_shape = (16,)
    out_shape = (1, 16, 16)
    dtype = pypto.DT_FP16
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 16,
        tileWout = 16,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 16,
        tileK = 16,
        tileN = 16
    )
    strides = [1]
    pads = [0, 0]
    dilations = [1]
    dtype_torch = torch.float16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c = torch.rand(bias_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel(fmap_shape, weight_shape, bias_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations)(a, b, c)
    golden = torch.nn.functional.conv1d(a, b, c, stride=1, padding=0)

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv1d_bf16_group_2():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 32, 16)
    weight_shape = (32, 16, 1)
    out_shape = (1, 32, 16)
    dtype = pypto.DT_BF16
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 16,
        tileWout = 16,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 16,
        tileK = 16,
        tileN = 16
    )
    strides = [1]
    pads = [0, 0]
    dilations = [1]
    dtype_torch = torch.bfloat16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel_no_bias(fmap_shape, weight_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations, 2)(a, b)
    golden = torch.nn.functional.conv1d(a, b, stride=1, padding=0, groups=2)

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv1d_fp32_strides_2():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 16)
    weight_shape = (16, 16, 3)
    out_shape = (1, 16, 7)
    dtype = pypto.DT_FP32
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 16,
        tileWout = 16,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 16,
        tileK = 16,
        tileN = 16
    )
    strides = [2]
    pads = [0, 0]
    dilations = [1]
    dtype_torch = torch.float32
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel_no_bias(fmap_shape, weight_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations)(a, b)
    golden = torch.nn.functional.conv1d(a, b, stride=(2), padding=0)

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv1d_fp16_dilation2():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (3, 16, 16)
    weight_shape = (32, 16, 3)
    bias_shape = (32,)
    out_shape = (3, 32, 12)
    dtype = pypto.DT_FP16
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 16,
        tileWout = 16,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 16,
        tileK = 16,
        tileN = 16
    )
    strides = [1]
    pads = [0, 0]
    dilations = [2]
    dtype_torch = torch.float16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c = torch.rand(bias_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel(fmap_shape, weight_shape, bias_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations)(a, b, c)
    golden = torch.nn.functional.conv1d(a, b, c, stride=1, padding=0,dilation=(2))

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv1d_fp32_pad_1_w32():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 32)
    weight_shape = (16, 16, 3)
    out_shape = (1, 16, 32)
    dtype = pypto.DT_FP32
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 32,
        tileWout = 32,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 32,
        tileK = 16,
        tileN = 16
    )
    strides = [1]
    pads = [1, 1]
    dilations = [1]
    dtype_torch = torch.float32
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel_no_bias(fmap_shape, weight_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations)(a, b)
    golden = torch.nn.functional.conv1d(a, b, stride=1, padding=(1))

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv3d_fp16_basic_with_bias():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 1, 1, 16)
    weight_shape = (16, 16, 1, 1, 1)
    bias_shape = (16,)
    out_shape = (1, 16, 1, 1, 16)
    dtype = pypto.DT_FP16
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 16,
        tileWout = 16,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 16,
        tileK = 16,
        tileN = 16
    )
    strides = [1, 1, 1]
    pads = [0, 0, 0, 0, 0, 0]
    dilations = [1, 1, 1]
    dtype_torch = torch.float16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c = torch.rand(bias_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel(fmap_shape, weight_shape, bias_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations)(a, b, c)
    golden = torch.nn.functional.conv3d(a, b, c, stride=1, padding=0)

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv3d_bf16_group_2():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (5, 32, 2, 2, 16)
    weight_shape = (16, 16, 1, 1, 1)
    out_shape = (5, 16, 2, 2, 16)
    dtype = pypto.DT_BF16
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 16,
        tileWout = 16,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 16,
        tileK = 16,
        tileN = 16
    )
    strides = [1, 1, 1]
    pads = [0, 0, 0, 0, 0, 0]
    dilations = [1, 1, 1]
    dtype_torch = torch.bfloat16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel_no_bias(fmap_shape, weight_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations, 2)(a, b)
    golden = torch.nn.functional.conv3d(a, b, stride=1, padding=0, groups=2)

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv3d_fp32_strides_2():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (2, 16, 5, 16, 16)
    weight_shape = (16, 16, 3, 3, 3)
    out_shape = (2, 16, 2, 7, 7)
    dtype = pypto.DT_FP32
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 16,
        tileWout = 16,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 16,
        tileK = 16,
        tileN = 16
    )
    strides = [2, 2, 2]
    pads = [0, 0, 0, 0, 0, 0]
    dilations = [1, 1, 1]
    dtype_torch = torch.float32
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel_no_bias(fmap_shape, weight_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations)(a, b)
    golden = torch.nn.functional.conv3d(a, b, stride=(2,2,2), padding=0)

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def test_conv3d_fp32_pad_1_w32():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 5, 3, 32)
    weight_shape = (16, 16, 3, 3, 3)
    out_shape = (1, 16, 5, 3, 32)
    dtype = pypto.DT_FP32
    tileL1Info = pypto_impl.TileL1Info(
        tileHin = 1,
        tileHout = 1,
        tileWin = 32,
        tileWout = 32,
        tileCinFmap = 16,
        tileCinWeight = 16,
        tileN = 16,
        tileBatch = 1
    )
    tileL0Info = pypto_impl.TileL0Info(
        tileH = 1,
        tileW = 32,
        tileK = 16,
        tileN = 16
    )
    strides = [1, 1, 1]
    pads = [1, 1, 1, 1, 1, 1]
    dilations = [1, 1, 1]
    dtype_torch = torch.float32
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')

    output_npu = create_conv_kernel_no_bias(fmap_shape, weight_shape, out_shape, dtype, tileL1Info, tileL0Info, strides, pads, dilations)(a, b)
    golden = torch.nn.functional.conv3d(a, b, stride=1, padding=(1,1,1))

    assert torch.allclose(output_npu.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)

def run_single_test_subprocess(testcase):
    p = mp.Process(target=testcase)
    p.start()
    p.join(timeout=60)
    assert p.exitcode == 0

@pytest.mark.soc("950")
def run_case():
    test_cases = [
        test_conv2d_fp16_basic_with_bias,
        test_conv2d_bf16_group_2,
        test_conv2d_fp32_strides_2,
        test_conv2d_fp16_dilation2,
        test_conv2d_fp32_pad_1_w32,
        test_conv1d_fp16_basic_with_bias,
        test_conv1d_bf16_group_2,
        test_conv1d_fp32_strides_2,
        test_conv1d_fp16_dilation2,
        test_conv1d_fp32_pad_1_w32,
        test_conv3d_fp16_basic_with_bias,
        test_conv3d_bf16_group_2,
        test_conv3d_fp32_strides_2,
        test_conv3d_fp32_pad_1_w32
    ]
    for test in test_cases:
        run_single_test_subprocess(test)