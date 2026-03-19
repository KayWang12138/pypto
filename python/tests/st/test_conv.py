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
import os
import multiprocessing as mp
from typing import Optional
import numpy as np
import pytest
from numpy.testing import assert_allclose
import torch
import torch_npu
import pypto
from pypto import pypto_impl
import torch.nn.functional as F


def create_conv_kernel(fmap_shape, weight_shape, bias_shape, out_shape, dtype, tile_l1_info, tile_l0_info, strides, 
    pads, dilations, groups=1):
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
    )

    def conv_kernel(
        fmap: pypto.Tensor(fmap_shape, dtype),
        weight: pypto.Tensor(weight_shape, dtype),
        bias: pypto.Tensor(bias_shape, dtype),
        out: pypto.Tensor(out_shape, dtype)
    ):
        pypto.set_conv_tile_shapes(tile_l1_info, tile_l0_info)
        extend_params = {'bias_tensor': bias}
        output = pypto.conv(fmap, weight, dtype, strides, pads, dilations, extend_params=extend_params, groups=groups)
        out.move(output)

    return conv_kernel


@pytest.mark.soc("950")
def test_conv2d_fp16_basic_with_bias():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 5, 32)
    weight_shape = (16, 8, 3, 3)
    bias_shape = (16,)
    out_shape = (1, 16, 2, 15)
    dtype = pypto.DT_FP16
    tile_l1_info = pypto_impl.TileL1Info(
        tileHin=1,
        tileHout=1,
        tileWin=32,
        tileWout=16,
        tileCinFmap=16,
        tileCinWeight=16,
        tileN=16,
        tileBatch=1
    )
    tile_l0_info = pypto_impl.TileL0Info(
        tileH=1,
        tileW=16,
        tileK=16,
        tileN=16
    )
    strides = [2, 2]
    pads = [1, 1, 1, 1]
    dilations = [2, 2]
    dtype_torch = torch.float16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c = torch.rand(bias_shape, dtype=dtype_torch, device='npu')
    d = torch.zeros(out_shape, dtype=dtype_torch, device='npu')

    create_conv_kernel(fmap_shape, weight_shape, bias_shape, out_shape, dtype, tile_l1_info, tile_l0_info,
        strides, pads, dilations, 2)(a, b, c, d)
    golden = torch.nn.functional.conv2d(a, b, c, stride=(2, 2), padding=(1, 1), dilation=(2, 2), groups=2)

    assert torch.allclose(d.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950")
def test_conv1d_fp16_basic_with_bias():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 32)
    weight_shape = (16, 8, 3)
    bias_shape = (16,)
    out_shape = (1, 16, 15)
    dtype = pypto.DT_FP16
    tile_l1_info = pypto_impl.TileL1Info(
        tileHin=1,
        tileHout=1,
        tileWin=32,
        tileWout=16,
        tileCinFmap=16,
        tileCinWeight=16,
        tileN=16,
        tileBatch=1
    )
    tile_l0_info = pypto_impl.TileL0Info(
        tileH=1,
        tileW=16,
        tileK=16,
        tileN=16
    )
    strides = [2]
    pads = [1, 1]
    dilations = [2]
    dtype_torch = torch.float16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c = torch.rand(bias_shape, dtype=dtype_torch, device='npu')
    d = torch.zeros(out_shape, dtype=dtype_torch, device='npu')

    create_conv_kernel(fmap_shape, weight_shape, bias_shape, out_shape, dtype, tile_l1_info, tile_l0_info,
        strides, pads, dilations, 2)(a, b, c, d)
    golden = torch.nn.functional.conv1d(a, b, c, stride=(2), padding=(1), dilation=(2), groups=2)

    assert torch.allclose(d.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950")
def test_conv3d_fp16_basic_with_bias():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    fmap_shape = (1, 16, 5, 5, 32)
    weight_shape = (16, 8, 3, 3, 3)
    bias_shape = (16,)
    out_shape = (1, 16, 2, 2, 15)
    dtype = pypto.DT_FP16
    tile_l1_info = pypto_impl.TileL1Info(
        tileHin=1,
        tileHout=1,
        tileWin=32,
        tileWout=16,
        tileCinFmap=16,
        tileCinWeight=16,
        tileN=16,
        tileBatch=1
    )
    tile_l0_info = pypto_impl.TileL0Info(
        tileH=1,
        tileW=16,
        tileK=16,
        tileN=16
    )
    strides = [2, 2, 2]
    pads = [1, 1, 1, 1, 1, 1]
    dilations = [2, 2, 2]
    dtype_torch = torch.float16
    a = torch.rand(fmap_shape, dtype=dtype_torch, device='npu')
    b = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    c = torch.rand(bias_shape, dtype=dtype_torch, device='npu')
    d = torch.zeros(out_shape, dtype=dtype_torch, device='npu')

    create_conv_kernel(fmap_shape, weight_shape, bias_shape, out_shape, dtype, tile_l1_info, tile_l0_info,
        strides, pads, dilations, 2)(a, b, c, d)
    golden = torch.nn.functional.conv3d(a, b, c, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(2, 2, 2), groups=2)

    assert torch.allclose(d.cpu().to(dtype_torch), golden.cpu().to(dtype_torch), atol=1e-3, rtol=1e-3)