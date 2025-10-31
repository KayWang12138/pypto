#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import pto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose

GRAPH_T = pto.GraphType.TENSOR_GRAPH
FUNC_T = pto.FunctionType.STATIC

def test_sigmoid_shape_dim():
    """Test whether the ouput shape is correct"""
    
    x_shape = [4, 4]
    dtype = pto.DT_FP32
    x = pto.tensor(x_shape, dtype)

    with pto.pto_function("SIGMOID_SHAPE", GRAPH_T, FUNC_T, x):
        pto.set_vec_tile_shapes(4, 4)
        res = pto.sigmoid(x)
        torch_case_tensor = torch.randn((4, 4), dtype = torch.float32)
        torch_case_res = torch.sigmoid(torch_case_tensor)
        assert res.shape == list(torch_case_res.shape)

def test_sigmoid_FP32():
    """Test whether the ouput of FP32 is correct"""

    x_shape = [4, 4]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    res = pto.tensor(x_shape, dtype)

    with pto.function("SIGMOID_CONTENT_FP32", [x], [res]):
        with pto.loop_function("LOOP_L0", "a_idx", pto.loop_range(1)) as aloop:
            for a_idx in aloop:
                pto.set_vec_tile_shapes(4, 4)
                res.move(pto.sigmoid(x))
                del res
    
    torch_tensor = np.random.uniform(-100, 100, [4, 4]).astype(np.float32)
    x_data = torch_tensor.flatten().tolist()
    res_data = np.random.uniform(-100, 100, [4, 4]).astype(np.float32)
    res_data = res_data.flatten().tolist()

    pto.runtime._device_run_once_data_from_host([x_data], [res_data])

    assert_allclose(res_data, torch.sigmoid(torch.tensor(torch_tensor)).flatten().tolist(), atol = 1e-3, verbose = True)

    pto.runtime._device_fini()

def test_sigmoid_FP16():
    """Test whether the ouput of FP16 shape is correct"""

    x_shape = [4, 4]
    dtype = pto.DT_FP16
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    res = pto.tensor(x_shape, dtype)

    with pto.function("SIGMOID_CONTENT_FP16", [x], [res]):
        with pto.loop_function("LOOP_L0", "a_idx", pto.loop_range(1)) as aloop:
            for a_idx in aloop:
                pto.set_vec_tile_shapes(4, 4)
                res.move(pto.sigmoid(x))
                del res
    
    torch_tensor = np.random.uniform(-100, 100, [4, 4]).astype(np.float16)
    x_data = torch_tensor.flatten().tolist()
    res_data = np.random.uniform(-100, 100, [4, 4]).astype(np.float16)
    res_data = res_data.flatten().tolist()

    pto.runtime._device_run_once_data_from_host([x_data], [res_data])

    assert_allclose(res_data, torch.sigmoid(torch.tensor(torch_tensor)).flatten().tolist(), atol = 1e-3, verbose = True)

    pto.runtime._device_fini()

def test_tensor_sigmoid_FP32():
    """Test whether the ouput of FP32 is correct"""

    x_shape = [4, 4]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    res = pto.tensor(x_shape, dtype)

    with pto.function("TENSOR_SIGMOID_CONTENT_FP32", [x], [res]):
        with pto.loop_function("LOOP_L0", "a_idx", pto.loop_range(1)) as aloop:
            for a_idx in aloop:
                pto.set_vec_tile_shapes(4, 4)
                res.move(x.sigmoid())
                del res
    
    torch_tensor = np.random.uniform(-100, 100, [4, 4]).astype(np.float32)
    x_data = torch_tensor.flatten().tolist()
    res_data = np.random.uniform(-100, 100, [4, 4]).astype(np.float32)
    res_data = res_data.flatten().tolist()

    pto.runtime._device_run_once_data_from_host([x_data], [res_data])

    assert_allclose(res_data, torch.sigmoid(torch.tensor(torch_tensor)).flatten().tolist(), atol = 1e-3, verbose = True)

    pto.runtime._device_fini()