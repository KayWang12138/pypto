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

import numpy as np
import torch
import torch_npu


def test_device_run_data_from_host_numpy():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 16
    n, m, k = tiling * 1, tiling * 1, tiling * 1

    pto.runtime._device_init()

    a = pto.tensor((n, m, k), pto.DT_FP32, "PTO_TENSOR_a")
    b = pto.tensor((n, m, k), pto.DT_FP32, "PTO_TENSOR_b")

    pto.set_vec_tile_shapes(tiling, tiling, tiling)
    with pto.function("MAIN", [a], [b]):
        for idx in pto.loop(10, name="s0", idx_name="idx"):
            if pto.cond(idx == 0):
                b.move(pto.add(a, a))
            else:
                b.move(pto.add(a, b))
    assert isinstance(b, pto.tensor)

    a_tensor = torch.rand(n, m, k, dtype=torch.float32) * 2 - 1
    b_tensor = torch.zeros(n, m, k, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    golden = 11 * a_tensor

    assert torch.allclose(golden, b_tensor, atol=1e-5)
    pto.runtime._device_fini()


def test_device_run_data_from_host_torch():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 8
    n, m = tiling * 1, tiling * 1

    pto.runtime._device_init()

    a = pto.tensor((n, m), pto.DT_FP32, "PTO_TENSOR_a")
    b = pto.tensor((n, m), pto.DT_FP32, "PTO_TENSOR_b")

    pto.set_vec_tile_shapes(tiling, tiling)
    with pto.function("MAIN", [a], [b]):
        for k in pto.loop(10, name="s0", idx_name="k"):
            if pto.cond(k == 0):
                b.move(pto.add(a, a))
            else:
                b.move(pto.add(a, b))
    assert isinstance(b, pto.tensor)

    a_tensor = torch.rand(n, m, dtype=torch.float32)
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    golden = 11 * a_tensor

    assert torch.allclose(golden, b_tensor, atol=1e-5)
    pto.runtime._device_fini()


def test_device_run_data_from_host():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 32
    n, m = tiling * 1, tiling * 1

    pto.runtime._device_init()

    a = pto.tensor((n, m), pto.DT_INT32, "PTO_TENSOR_a")
    b = pto.tensor((n, m), pto.DT_INT32, "PTO_TENSOR_b")

    pto.set_vec_tile_shapes(tiling, tiling)
    with pto.function("MAIN", [a], [b]):
        for k in pto.loop(10, name="s0", idx_name="k"):
            if pto.cond(k == 0):
                b.move(pto.add(a, a))
            else:
                b.move(pto.add(a, b))
    assert isinstance(b, pto.tensor)

    a_tensor = torch.arange(n * m, dtype=torch.int32).reshape(n, m)
    b_tensor = torch.zeros(n, m, dtype=torch.int32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])
    golden = 11 * a_tensor

    assert torch.equal(golden, b_tensor)
    pto.runtime._device_fini()


# def dynamic function
@pto.jit
def cust_dyn_func(in_tensors, out_tensors, tiling = None):
    a = in_tensors[0]
    b = out_tensors[0]
    pto.set_vec_tile_shapes(tiling, tiling)
    with pto.function("MAIN", [a], [b]):
        for k in pto.loop(10, name="s0", idx_name="k"):
            if pto.cond(k == 0):
                b.move(pto.add(a, a))
            else:
                b.move(pto.add(a, b))
    assert isinstance(b, pto.tensor)


def test_device_run_data_from_device():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 32
    n, m = tiling * 1, tiling * 1

    # prepare data
    a_rawdata = torch.tensor([[k * 100 + v for v in range(m)] for k in range(n)])
    a_data = a_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
    b_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')
    # def inputs and outputs
    inputs = [a_data]
    outputs = [b_data]
    cust_dyn_func(inputs, outputs, tiling)

    pto.runtime._device_synchronize()
    # get data and compare result
    a_data_cpu = a_data.cpu()
    b_data_cpu = b_data.cpu()
    # verify
    a_data_list = [c for r in a_data_cpu.tolist() for c in r]
    b_data_list = [c for r in b_data_cpu.tolist() for c in r]
    assert b_data_list == [v * 11 for v in a_data_list]

    c_rawdata = torch.tensor([[k * 1000 + v for v in range(m)] for k in range(n)])
    c_data = a_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
    d_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')
    cust_dyn_func([c_data], [d_data])
    c_data_list = [c for r in c_data.cpu().tolist() for c in r]
    d_data_list = [c for r in d_data.cpu().tolist() for c in r]
    assert d_data_list == [v * 11 for v in c_data_list]


# def dynamic function
@pto.jit
def matmul_add(in_tensors, out_tensors, m, k, n, tiling = None):
    a = in_tensors[0]
    b = in_tensors[1]
    c = in_tensors[2]
    d = out_tensors[0]
    pto.set_vec_tile_shapes(tiling, tiling)
    pto.set_cube_tile_shapes([tiling, tiling], [tiling, tiling], [tiling, tiling])
    with pto.function("MAIN", [a, b, c], [d]):
        for i in pto.loop(1, name="s0", idx_name="i"):
            a0 = pto.view(a, [n, k], [0, 0])
            b0 = pto.view(b, [k, m], [0, 0])
            d.move(pto.add(pto.matmul(a0, b0, pto.DT_INT32), c))
            del a0
            del b0


def test_device_run_data_from_device_mix_nodep():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

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

        d_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')
        d_data_list.append(d_data)

        # def inputs and outputs
        inputs = [a_data, b_data, c_data]
        outputs = [d_data]
        matmul_add(inputs, outputs, m, k, n, tiling=tiling)

    pto.runtime._device_synchronize()

    for idx in range(count):
        # get data and compare result
        d_data_inlist = [c for r in d_data_list[idx].cpu().tolist() for c in r]
        assert d_data_inlist == [k + idx] * len(d_data_inlist)
