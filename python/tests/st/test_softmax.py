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

def test_softmax_shape_dim():
    """Test whether the ouput shape is correct"""
    
    x_shape = [4, 4]
    dtype = pto.DT_FP32
    x = pto.tensor(x_shape, dtype)
    dim = -1

    with pto.pto_function("SOFTMAX_SHAPE", GRAPH_T, FUNC_T, x):
        pto.set_vec_tile_shapes(32, 32)
        res = pto.softmax(x, dim)
        torch_case_tensor = torch.randn((4, 4), dtype = torch.float32)
        torch_case_res = torch.softmax(torch_case_tensor, dim)
        assert res.shape == list(torch_case_res.shape)

def test_softmax_FP32():
    """Test whether the ouput of FP32 is correct"""

    x_shape = [4, 4]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    res = pto.tensor(x_shape, dtype)
    dim = -1

    with pto.function("SOFTMAX_CONTENT_FP32", [x], [res]):
        for _ in pto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pto.set_vec_tile_shapes(32, 32)
            res.move(pto.softmax(x, dim))
            del res
    
    x_tensor = torch.rand(4, 4, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto.runtime._device_run_once_data_from_host([x_tensor], [res_tensor])
    expected = torch.softmax(x_tensor, dim)
    assert_allclose(res_tensor.flatten(), expected.flatten(), atol=1e-3, verbose=True)
    pto.runtime._device_fini()

def test_tensor_softmax_FP32():
    """Test whether the ouput of FP32 is correct"""

    x_shape = [4, 4]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    res = pto.tensor(x_shape, dtype)
    dim = -1

    with pto.function("TENSOR_SOFTMAX_CONTENT_FP32", [x], [res]):
        for _ in pto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pto.set_vec_tile_shapes(32, 32)
            res.move(x.softmax(dim))
            del res
    
    x_tensor = torch.rand(4, 4, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto.runtime._device_run_once_data_from_host([x_tensor], [res_tensor])
    expected = torch.softmax(x_tensor, dim)
    assert_allclose(res_tensor.flatten(), expected.flatten(), atol=1e-3, verbose=True)
    pto.runtime._device_fini()
