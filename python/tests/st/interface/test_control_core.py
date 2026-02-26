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
"""
import contextlib
import os

import pypto
import pytest
import torch
import torch_npu
import pytest


# def dynamic function
@pypto.jit(
    runtime_options={"max_cube_blockdim": 6}
)
def matmul_add_max_cube_blockdim(in_tensor0, in_tensor1, in_tensor2, out_tensor, m, k, n, tiling=None):
    a = in_tensor0
    b = in_tensor1
    c = in_tensor2
    d = out_tensor
    pypto.set_vec_tile_shapes(tiling, tiling)
    pypto.set_cube_tile_shapes(
        [tiling, tiling], [tiling, tiling], [tiling, tiling])
    for _ in pypto.loop(1, name="s0", idx_name="i"):
        a0 = pypto.view(a, [n, k], [0, 0])
        b0 = pypto.view(b, [k, m], [0, 0])
        d.move(pypto.add(pypto.matmul(a0, b0, pypto.DT_INT32), c))

@pypto.jit
def matmul_add_rts(in_tensor0, in_tensor1, in_tensor2, out_tensor, m, k, n, tiling=None):
    a = in_tensor0
    b = in_tensor1
    c = in_tensor2
    d = out_tensor
    pypto.set_vec_tile_shapes(tiling, tiling)
    pypto.set_cube_tile_shapes(
        [tiling, tiling], [tiling, tiling], [tiling, tiling])
    for _ in pypto.loop(1, name="s0", idx_name="i"):
        a0 = pypto.view(a, [n, k], [0, 0])
        b0 = pypto.view(b, [k, m], [0, 0])
        d.move(pypto.add(pypto.matmul(a0, b0, pypto.DT_INT32), c))


def test_rts_control_cores():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    stream1 = torch.npu.current_stream()
    torch.npu.set_stream_limit(stream1, 15, 27)

    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8

    # prepare data
    c_data_list = []
    d_data_list = []

    count = 16

    a_rawdata = torch.tensor([[1] * k] * n)
    b_rawdata = torch.tensor([[1] * m] * k)
    a_data = a_rawdata.to(dtype=torch.int8, device=f'npu:{device_id}')
    b_data = b_rawdata.to(dtype=torch.int8, device=f'npu:{device_id}')

    for idx in range(count):
        c_rawdata = torch.tensor([[idx] * m] * n)
        c_data = c_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
        c_data_list.append(c_data)

        d_data = torch.zeros((n, m), dtype=torch.int32,
                             device=f'npu:{device_id}')
        d_data_list.append(d_data)

        # def inputs and outputs
        inputs = [a_data, b_data, c_data]
        outputs = [d_data]
        pto_inputs = [pypto.from_torch(
            tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
        pto_outputs = [pypto.from_torch(
            tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
        if idx == 0:
            matmul_add_rts(pto_inputs[0], pto_inputs[1], pto_inputs[2],
                    pto_outputs[0], m, k, n, tiling=tiling)
        else if idx == 1:
            matmul_add_max_cube_blockdim(pto_inputs[0], pto_inputs[1], pto_inputs[2],
                    pto_outputs[0], m, k, n, tiling=tiling)

    torch_npu.npu.synchronize()

    for idx in range(count):
        # get data and compare result
        d_data_inlist = [c for r in d_data_list[idx].cpu().tolist() for c in r]
        assert d_data_inlist == [k + idx] * len(d_data_inlist)
