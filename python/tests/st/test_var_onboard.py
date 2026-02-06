#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import os
import math
import copy
import pytest
import numpy as np
import torch
import pypto
import torch_npu


def var_2dim_tensor_proc(b, s, dim, cprrection, keepdim):
    pypto.runtime._device_init()
    input_shape = (b, s)
    if dim is None or len(dim) == 0:
        dst_shape = (1)
    elif isinstance(dim, int):
        dst_shape = (len(input_shape) - dim - 1)
    else if len(dim) == 1:
        dst_shape = (len(input_shape) - dim[0] - 1)
    else:
        dst_shape = (1)

    view_shape = (b, s)
    tile_shape = (b, s)

    input_tensor = pypto.tensor(input_shape, pypto.DT_FP32, "PTO_TENSOR_SELF")
    dst_tensor = pypto.tensor(dst_shape, pypto.DT_FP32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(indices_shape[0] / view_shape[0])
    s_loop_num = math.ceil(indices_shape[1] / view_shape[1])
    with pypto.function("MAIN", input_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                tmp_dst_tensor = pypto.tensor(view_shape, pypto.DT_FP32, "PTO_TENSOR_TMP")
                view_tensor_input = pypto.view(self_tensor, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pypto.symbolic_scalar(self_shape[0]) -
                        b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(self_shape[1]) -
                        s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1]))])
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_dst_tensor.move(
                    pypto.scatter_(view_tensor_input, dim, correction=correction, keepdim=keepdim))
                pypto.assemble(tmp_dst_tensor, [b_idx * view_shape[0], s_idx * view_shape[1]], dst_tensor)

    assert isinstance(dst_tensor, pypto.tensor)

    input0_tensor = torch.rand(*self_shape, dtype=torch.float32)
    c_tensor = torch.zeros_like(input0_tensor)

    pto_input0_tensor = pypto.from_torch(input0_tensor, "input0_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_input0_tensor, pto_c_tensor)

    result = input0_tensor.var(input0_tensor, dim, correction=correction, keepdim=keepdim)

    assert torch.equal(c_tensor, result)
    pypto.runtime._device_fini()


def test_var0_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 2
    s = 8
    dim = None
    cprrection = 1
    keepdim = False
    var_2dim_tensor_proc(b, s, dim, cprrection, keepdim)


def test_var1_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 2
    s = 8
    dim = None
    cprrection = 1
    keepdim = False
    var_2dim_tensor_proc(scatter_para, False)

