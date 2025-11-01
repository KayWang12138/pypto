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

GRAPH_T = pto.GraphType.TENSOR_GRAPH
FUNC_T = pto.FunctionType.STATIC

def test_view_basic_shape():
    """Test whether the ouput shape is correct"""

    x_shape = [32, 64]
    dtype = pto.DT_FP32
    x = pto.tensor(x_shape, dtype)
    view_shape = [32, 32]
    offset = [0, 32]
    with pto.pto_function("VIEW_SHAPE", GRAPH_T, FUNC_T, x):
        pto.set_vec_tile_shapes(32, 32)
        res = pto.view(x, view_shape, offset)

    assert res.shape == view_shape

def test_view_content_equal():
    """Test whether the output content has changed"""

    x_shape = [4, 8]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    view_shape = [4, 4]
    offset = [0, 4]
    res = pto.tensor(view_shape, dtype)

    with pto.function("VIEW_CONTENT", [x], [res]):
        with pto.loop_function("LOOP_L0", "a_idx", pto.loop_range(1)) as aloop:
            for a_idx in aloop:
                pto.set_vec_tile_shapes(4, 4)
                res.move(pto.view(x, view_shape, offset))
                del res
    
    torch_tensor = torch.rand(4, 8, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto.runtime._device_run_once_data_from_host([torch_tensor], [res_tensor])
    
    expected = torch_tensor[0:4, 4:8]
    assert torch.equal(res_tensor.flatten(), expected.flatten())
    pto.runtime._device_fini()

def test_view_content_equal_validshape():
    """Test whether the output content has changed with validshape"""
    x_shape = [4, 4]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    view_shape = [4, 4]
    offset = [2, 0]
    validshape = [2, 4]
    res = pto.tensor(view_shape, dtype)

    with pto.function("VIEW_CONTENT_VALIDSHAPE", [x], [res]):
        with pto.loop_function("LOOP_L0", "a_idx", pto.loop_range(1)) as aloop:
            for a_idx in aloop:
                pto.set_vec_tile_shapes(4, 4)
                res.move(pto.view(x, view_shape, offset, valid_shape=validshape))
                del res
    
    torch_tensor = torch.rand(4, 4, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto.runtime._device_run_once_data_from_host([torch_tensor], [res_tensor])
     
    expected = torch_tensor[2:4, 0:4]
    assert torch.equal(res_tensor.flatten()[:2 * 4], expected.flatten())
     
    pto.runtime._device_fini()

def test_tensor_view_content_equal():
    """Test whether the output content has changed"""

    x_shape = [4, 8]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    view_shape = [4, 4]
    offset = [0, 4]
    res = pto.tensor(view_shape, dtype)

    with pto.function("Tensor_VIEW_CONTENT", [x], [res]):
        with pto.loop_function("LOOP_L0", "a_idx", pto.loop_range(1)) as aloop:
            for a_idx in aloop:
                pto.set_vec_tile_shapes(4, 4)
                res.move(x.view(view_shape, offset))
                del res
    
    torch_tensor = torch.rand(4, 8, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto.runtime._device_run_once_data_from_host([torch_tensor], [res_tensor])
    
    expected = torch_tensor[0:4, 4:8]
    assert torch.equal(res_tensor.flatten(), expected.flatten())
     
    pto.runtime._device_fini()

def test_syntactic_sugar_view_content_equal():
    """Test whether the output content has changed"""

    x_shape = [4, 8]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    view_shape = [4, 4]
    offset = [0, 4]
    res = pto.tensor(view_shape, dtype)

    with pto.function("SURGER_VIEW_CONTENT", [x], [res]):
        with pto.loop_function("LOOP_L0", "a_idx", pto.loop_range(1)) as aloop:
            for a_idx in aloop:
                pto.set_vec_tile_shapes(4, 4)
                res.move(x[ : offset[0] + view_shape[0], offset[1] : offset[1] + view_shape[1]])
                del res
    
    torch_tensor = torch.rand(4, 8, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto.runtime._device_run_once_data_from_host([torch_tensor], [res_tensor])
     
    expected = torch_tensor[:, 4:8]
    assert torch.equal(res_tensor.flatten(), expected.flatten())
     
    pto.runtime._device_fini()
