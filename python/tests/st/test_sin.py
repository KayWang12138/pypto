#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import os
import pto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose
import torch_npu



def test_sin_shape_dim():
    """Test whether the ouput shape is correct"""

    x_shape = [4, 4]
    dtype = pto.DT_FP32
    x = pto.tensor(x_shape, dtype)

    with pto.function("SIN_SHAPE", x, static=True):
        pto.set_vec_tile_shapes(4, 4)
        res = pto.sin(x)
        torch_case_tensor = torch.randn((4, 4), dtype = torch.float32)
        torch_case_res = torch.sin(torch_case_tensor)
        assert res.shape == list(torch_case_res.shape)

def test_sin_FP32():
    """Test whether the ouput of FP32 is correct"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 4]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    res = pto.tensor(x_shape, dtype)

    with pto.function("SIN_CONTENT_FP32", [x], [res]):
        for _ in pto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pto.set_vec_tile_shapes(4, 4)
            res.move(pto.sin(x))
            del res

    x_tensor = torch.rand(4, 4, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto.runtime._device_run_once_data_from_host([x_tensor], [res_tensor])
    expected = torch.sin(x_tensor)
    assert_allclose(res_tensor.flatten(), expected.flatten(), atol=1e-3, verbose=True)
    pto.runtime._device_fini()

def test_sin_FP16():
    """Test whether the ouput of FP16 shape is correct"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 4]
    dtype = pto.DT_FP16
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    res = pto.tensor(x_shape, dtype)

    with pto.function("SIN_CONTENT_FP16", [x], [res]):
        for _ in pto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pto.set_vec_tile_shapes(4, 4)
            res.move(pto.sin(x))
            del res

    x_tensor = torch.rand(4, 4, dtype=torch.float16) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float16)
    pto.runtime._device_run_once_data_from_host([x_tensor], [res_tensor])
    expected = torch.sin(x_tensor)
    assert_allclose(res_tensor.flatten(), expected.flatten(), atol=1e-3, verbose=True)
    pto.runtime._device_fini()

def test_tensor_sin_FP32():
    """Test whether the ouput of FP32 is correct"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 4]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    res = pto.tensor(x_shape, dtype)

    with pto.function("TENSOR_SIN_CONTENT_FP32", [x], [res]):
        for _ in pto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pto.set_vec_tile_shapes(4, 4)
            res.move(x.sin())
            del res

    x_tensor = torch.rand(4, 4, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto.runtime._device_run_once_data_from_host([x_tensor], [res_tensor])
    expected = torch.sin(x_tensor)
    assert_allclose(res_tensor.flatten(), expected.flatten(), atol=1e-3, verbose=True)
    pto.runtime._device_fini()
