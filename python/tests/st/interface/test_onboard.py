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
import os
import pypto

import numpy as np
import torch
import torch_npu


def test_device_run_data_from_host_numpy():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 16
    n, m, k = tiling * 1, tiling * 1, tiling * 1

    pypto.runtime._device_init()

    a = pypto.tensor((n, m, k), pypto.DT_FP32, "PTO_TENSOR_a")
    b = pypto.tensor((n, m, k), pypto.DT_FP32, "PTO_TENSOR_b")

    pypto.set_vec_tile_shapes(tiling, tiling, tiling)
    with pypto.function("MAIN", a, b):
        for idx in pypto.loop(10, name="s0", idx_name="idx"):
            if pypto.cond(idx == 0):
                b.move(pypto.add(a, a))
            else:
                b.move(pypto.add(a, b))
    assert isinstance(b, pypto.tensor)

    a_tensor = torch.rand(n, m, k, dtype=torch.float32) * 2 - 1
    b_tensor = torch.zeros(n, m, k, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    golden = 11 * a_tensor

    assert torch.allclose(golden, b_tensor, atol=1e-5)
    pypto.runtime._device_fini()


def test_device_run_data_from_host_torch():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 8
    n, m = tiling * 1, tiling * 1

    pypto.runtime._device_init()

    a = pypto.tensor((n, m), pypto.DT_FP32, "PTO_TENSOR_a")
    b = pypto.tensor((n, m), pypto.DT_FP32, "PTO_TENSOR_b")

    pypto.set_vec_tile_shapes(tiling, tiling)
    with pypto.function("MAIN", a, b):
        for k in pypto.loop(10, name="s0", idx_name="k"):
            if pypto.cond(k == 0):
                b.move(pypto.add(a, a))
            else:
                b.move(pypto.add(a, b))
    assert isinstance(b, pypto.tensor)

    a_tensor = torch.rand(n, m, dtype=torch.float32)
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    golden = 11 * a_tensor

    assert torch.allclose(golden, b_tensor, atol=1e-5)
    pypto.runtime._device_fini()


def test_device_run_data_from_host():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 32
    n, m = tiling * 1, tiling * 1

    pypto.runtime._device_init()

    a = pypto.tensor((n, m), pypto.DT_INT32, "PTO_TENSOR_a")
    b = pypto.tensor((n, m), pypto.DT_INT32, "PTO_TENSOR_b")

    pypto.set_vec_tile_shapes(tiling, tiling)
    with pypto.function("MAIN", a, b):
        for k in pypto.loop(10, name="s0", idx_name="k"):
            if pypto.cond(k == 0):
                b.move(pypto.add(a, a))
            else:
                b.move(pypto.add(a, b))
    assert isinstance(b, pypto.tensor)

    a_tensor = torch.arange(n * m, dtype=torch.int32).reshape(n, m)
    b_tensor = torch.zeros(n, m, dtype=torch.int32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)
    golden = 11 * a_tensor

    assert torch.equal(golden, b_tensor)
    pypto.runtime._device_fini()


# def dynamic function
def create_cust_dyn_func(shape, tiling=None):
    @pypto.frontend.jit()
    def cust_dyn_func(
        a: pypto.Tensor(shape, pypto.DT_INT32),
        b: pypto.Tensor(shape, pypto.DT_INT32)
    ):
        pypto.set_vec_tile_shapes(tiling, tiling)

        for k in pypto.loop(10, name="s0", idx_name="k"):
            if k == 0:
                b.move(pypto.add(a, a))
            else:
                b.move(pypto.add(a, b))
    return cust_dyn_func


def test_device_run_data_from_device():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)

    # prepare data
    a_rawdata = torch.tensor(
        [[k * 100 + v for v in range(m)] for k in range(n)],
        dtype=torch.int32
    )
    a_data = a_rawdata.to(device=f'npu:{device_id}')
    b_data = torch.zeros(shape, dtype=torch.int32, device=f'npu:{device_id}')

    kernel = create_cust_dyn_func(shape, tiling=tiling)
    kernel(a_data, b_data)

    torch_npu.npu.synchronize()
    # get data and compare result
    a_data_cpu = a_data.cpu()
    b_data_cpu = b_data.cpu()
    # verify
    a_data_list = [c for r in a_data_cpu.tolist() for c in r]
    b_data_list = [c for r in b_data_cpu.tolist() for c in r]
    assert b_data_list == [v * 11 for v in a_data_list]

    c_rawdata = torch.tensor(
        [[k * 1000 + v for v in range(m)] for k in range(n)],
        dtype=torch.int32
    )
    c_data = c_rawdata.to(device=f'npu:{device_id}')
    d_data = torch.zeros(shape, dtype=torch.int32, device=f'npu:{device_id}')

    kernel(c_data, d_data)

    torch_npu.npu.synchronize()

    c_data_list = [c for r in c_data.cpu().tolist() for c in r]
    d_data_list = [c for r in d_data.cpu().tolist() for c in r]
    assert d_data_list == [v * 11 for v in c_data_list]


# def dynamic function
def create_matmul_add(m, k, n, tiling=None):
    shape_a = (n, k)
    shape_b = (k, m)
    shape_c = (n, m)
    shape_d = (n, m)
    @pypto.frontend.jit()
    def matmul_add(
        a: pypto.Tensor(shape_a, pypto.DT_INT8),
        b: pypto.Tensor(shape_b, pypto.DT_INT8),
        c: pypto.Tensor(shape_c, pypto.DT_INT32)
    ) -> pypto.Tensor(shape_d, pypto.DT_INT32):
        pypto.set_vec_tile_shapes(tiling, tiling)
        pypto.set_cube_tile_shapes(
            [tiling, tiling], [tiling, tiling], [tiling, tiling])
        d = pypto.tensor(shape_d, pypto.DT_INT32)
        for _ in pypto.loop(1, name="s0", idx_name="i"):
            a0 = pypto.view(a, [n, k], [0, 0])
            b0 = pypto.view(b, [k, m], [0, 0])
            d.move(pypto.add(pypto.matmul(a0, b0, pypto.DT_INT32), c))
        return d

    return matmul_add


def test_device_run_data_from_device_mix_nodep():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8


    # prepare data
    d_data_list = []

    count = 16

    a_data = torch.tensor([[1] * k] * n, dtype=torch.int8, device=f'npu:{device_id}')
    b_data = torch.tensor([[1] * m] * k, dtype=torch.int8, device=f'npu:{device_id}')

    # Create kernel once
    kernel = create_matmul_add(m, k, n, tiling=tiling)

    for idx in range(count):
        c_data = torch.tensor([[idx] * m] * n, dtype=torch.int32, device=f'npu:{device_id}')

        d_data = kernel(a_data, b_data, c_data)
        d_data_list.append(d_data)

    torch_npu.npu.synchronize()

    for idx in range(count):
        # get data and compare result
        d_data_inlist = [c for r in d_data_list[idx].cpu().tolist() for c in r]
        assert d_data_inlist == [k + idx] * len(d_data_inlist)


class InferControlflowShape:

    def __init__(self):
        s = 32
        self.b_vec = [1024, 128, 64, 32]
        self.cache_shapes = {b: [(b, s), (b, s), (b, s)] for b in self.b_vec}

    def __call__(self, *args):
        if not args:
            return list(self.cache_shapes.values())
        for b in self.b_vec:
            if args[0][0] >= b:
                return self.cache_shapes[b]
        raise ValueError(f"invalid shape {args[0]}")


@pypto.jit(
    infer_controlflow_shape=InferControlflowShape()
)
def infer_shape_kenrel(a, b, c):
    pypto.set_vec_tile_shapes(16, 16)
    for i in pypto.loop(0, a.shape[0], 32):
        ta = a[i: i + 32, :]
        tb = b[i: i + 32, :]
        c[i:, 0:] = ta + tb


def test_infer_shape():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    device = f'npu:{device_id}'
    for b in [2048, 1024, 512, 256, 128, 64, 32]:
        a = torch.randn((b, 32), device=device)
        b = torch.randn((b, 32), device=device)
        c = torch.zeros_like(a, device=device)
        g = a + b

        infer_shape_kenrel(
            pypto.from_torch(a, dynamic_axis=[0]),
            pypto.from_torch(b, dynamic_axis=[0]),
            pypto.from_torch(c, dynamic_axis=[0]),
        )
        torch.npu.synchronize()
        torch.testing.assert_close(c, g)
