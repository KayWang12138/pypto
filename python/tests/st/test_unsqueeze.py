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
import os
import pto
import pytest
import torch
import numpy as np
import torch_npu


def test_unsqueeze_shape_dim():
    """Test whether the ouput shape is correct"""

    shape = [8, 16, 16]
    dtype = pto.DT_FP32
    x = pto.tensor(shape, dtype)
    dim = 0
    with pto.function("UNSQUEEZE_SHAPE", x, static=True):
        pto.set_vec_tile_shapes(8, 8, 8, 8)

        #Test each valid dim:[-4, -3, -2, -1, 0, 1, 2, 3]
        for dim in range(-4, 4, 1):
            res = pto.unsqueeze(x, dim)
            torch_case_tensor = torch.randn((8, 16, 16), dtype = torch.float32)
            torch_case_res = torch.unsqueeze(torch_case_tensor, dim)
            assert res.shape == list(torch_case_res.shape)

def test_unsqueeze_content_equal():
    """Test whether the output content has changed"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = [2, 2]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(shape, dtype)
    res = pto.tensor([1, 2, 2], dtype)
    dim = 0
    with pto.function("UNSQUEEZE_CONTENT", [x], [res]):
        for _ in pto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pto.set_vec_tile_shapes(2, 2, 2)
            res.move(pto.unsqueeze(x, dim))
            del res

    torch_case_tensor = torch.rand(2, 2, dtype=torch.float32)
    res_tensor = torch.zeros((1,) + torch_case_tensor.shape, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([torch_case_tensor], [res_tensor])

    torch_case_res = torch.unsqueeze(torch_case_tensor, dim)

    assert torch.equal(res_tensor.flatten(), torch_case_res.flatten())
    pto.runtime._device_fini()

def test_tensor_unsqueeze_shape_dim():
    """Test whether the ouput shape is correct"""

    shape = [8, 16, 16]
    dtype = pto.DT_FP32
    x = pto.tensor(shape, dtype)
    dim = 1
    with pto.function("TENSOR_UNSQUEEZE_SHAPE", x, static=True):
        pto.set_vec_tile_shapes(8, 8, 8, 8)

        res = x.unsqueeze(dim)
        torch_case_tensor = torch.randn((8, 16, 16), dtype = torch.float32)
        torch_case_res = torch.unsqueeze(torch_case_tensor, dim)
        assert res.shape == list(torch_case_res.shape)
