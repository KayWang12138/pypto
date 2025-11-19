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
import torch_npu


def test_slice_neg_index():
    """Test negative index"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 8]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    res = pto.tensor(x_shape, dtype)

    with pto.function("SLICE_NEG_INDEX", [x], [res]):
        for _ in pto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pto.set_vec_tile_shapes(4, 4)
            res[:] = (x[-3:-1, -2:-1])
            del res

    torch_tensor = torch.rand(4, 8, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(2, 1, dtype=torch.float32)
    pto.runtime._device_run_once_data_from_host([torch_tensor], [res_tensor])
    expected = torch_tensor[-3:-1, -2:-1]

    assert torch.equal(res_tensor.flatten(), expected.flatten())
    pto.runtime._device_fini()


def test_slice_int_index():
    """Test mix use of slice and int"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 8, 8, 8, 8]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    res = pto.tensor(x_shape, dtype)

    with pto.function("SLICE_INT_INDEX", [x], [res]):
        for _ in pto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pto.set_vec_tile_shapes(4, 4, 4, 4, 4)
            res[:] = x[-2, -3:8, :, 1:4, 2]
            del res

    torch_tensor = torch.rand(4, 8, 8, 8, 8, dtype=torch.float32) * 200 - 100
    res_tensor = torch.zeros(3, 8, 3, dtype=torch.float32)
    pto.runtime._device_run_once_data_from_host([torch_tensor], [res_tensor])
    expected = torch_tensor[-2, -3:8, :, 1:4, 2]

    assert torch.equal(res_tensor.flatten(), expected.flatten())
    pto.runtime._device_fini()


def test_slice_ellipsis_index():
    """Test mix use of ellipsis, slice and int"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x_shape = [4, 8, 8, 8]
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    x = pto.tensor(x_shape, dtype)
    res1 = pto.tensor(x_shape, dtype)
    res2 = pto.tensor(x_shape, dtype)
    res3 = pto.tensor(x_shape, dtype)
    res4 = pto.tensor(x_shape, dtype)

    with pto.function("SLICE_INT_ELLIPSIS_INDEX", [x], [res1, res2, res3, res4]):
        for _ in pto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pto.set_vec_tile_shapes(4, 4, 4, 4)
            res1[:] = x[..., 2]
            res2[:] = x[1:2, :, ..., 3:5]
            res3[:] = x[2, 3, ...]
            res4[:] = x[...] + 0.0
            del res1
            del res2
            del res3
            del res4

    torch_tensor = torch.rand(4, 8, 8, 8, dtype=torch.float32) * 200 - 100
    res1_tensor = torch.zeros(4, 8, 8, dtype=torch.float32)
    res2_tensor = torch.zeros(1, 8, 8, 2, dtype=torch.float32)
    res3_tensor = torch.zeros(8, 8, dtype=torch.float32)
    res4_tensor = torch.zeros(4, 8, 8, 8, dtype=torch.float32)
    pto.runtime._device_run_once_data_from_host([torch_tensor], [res1_tensor, res2_tensor, res3_tensor, res4_tensor])
    expected1 = torch_tensor[..., 2]
    expected2 = torch_tensor[1:2, :, ..., 3:5]
    expected3 = torch_tensor[2, 3, ...]
    expected4 = torch_tensor[...]

    assert torch.equal(res1_tensor.flatten(), expected1.flatten())
    assert torch.equal(res2_tensor.flatten(), expected2.flatten())
    assert torch.equal(res3_tensor.flatten(), expected3.flatten())
    assert torch.equal(res4_tensor.flatten(), expected4.flatten())
    pto.runtime._device_fini()
