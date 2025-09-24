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


def test_device_run_data_from_host_numpy():
    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0)
    tiling = 16
    n, m, k = tiling * 1, tiling * 1, tiling * 1

    pto.device_init()

    a = pto.tensor(pto.DataType.DT_FP32, (n, m, k), "PTO_TENSOR_a")
    b = pto.tensor(pto.DataType.DT_FP32, (n, m, k), "PTO_TENSOR_b")

    pto.set_vec_tile_shapes(tiling, tiling, tiling)
    with pto.dyn_function("MAIN", [a], [b]):
        with pto.loop_function("s0", "idx", pto.loop_range_(10)) as rlf:
            for idx in rlf:
                if pto.cond(idx == 0):
                    b.move(pto.add(a, a))
                else:
                    b.move(pto.add(a, b))
    assert isinstance(b, pto.tensor)

    a_data = np.random.uniform(-1, 1, [n, m, k]).astype(np.float32)
    b_data = np.zeros((n, m, k))

    pto.device_run_once_data_from_host([a_data], [b_data])

    golden = 11 * a_data

    assert np.allclose(golden, b_data, atol=1e-5)
    pto.device_fini()


def test_device_run_data_from_host_torch():
    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0)
    tiling = 8
    n, m = tiling * 1, tiling * 1

    pto.device_init()

    a = pto.tensor(pto.DataType.DT_FP32, (n, m), "PTO_TENSOR_a")
    b = pto.tensor(pto.DataType.DT_FP32, (n, m), "PTO_TENSOR_b")

    pto.set_vec_tile_shapes(tiling, tiling)
    with pto.dyn_function("MAIN", [a], [b]):
        with pto.loop_function("s0", "k", pto.loop_range_(10)) as rlf:
            for k in rlf:
                if pto.cond(k == 0):
                    b.move(pto.add(a, a))
                else:
                    b.move(pto.add(a, b))
    assert isinstance(b, pto.tensor)

    a_data = torch.rand(n, m)
    b_data = torch.zeros(n, m)

    pto.device_run_once_data_from_host([a_data], [b_data])

    golden = 11 * a_data

    assert torch.allclose(golden, b_data, atol=1e-5)
    pto.device_fini()


def test_device_run_data_from_host():
    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0)
    tiling = 32
    n, m = tiling * 1, tiling * 1

    pto.device_init()

    a = pto.tensor(pto.DataType.DT_INT32, (n, m), "PTO_TENSOR_a")
    b = pto.tensor(pto.DataType.DT_INT32, (n, m), "PTO_TENSOR_b")

    pto.set_vec_tile_shapes(tiling, tiling)
    with pto.dyn_function("MAIN", [a], [b]):
        with pto.loop_function("s0", "k", pto.loop_range_(10)) as rlf:
            for k in rlf:
                if pto.cond(k == 0):
                    b.move(pto.add(a, a))
                else:
                    b.move(pto.add(a, b))
    assert isinstance(b, pto.tensor)

    a_data = list(range(n * m))
    b_data = list([0] * n * m)

    pto.device_run_once_data_from_host([a_data], [b_data])

    assert b_data == [v * 11 for v in range(n * m)]
    pto.device_fini()


def test_device_run_data_from_device():
    try:
        import torch
        import torch_npu
    except e as ImportError:
        torch = None
        torch_npu = None

    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0)
    tiling = 32
    n, m = tiling * 1, tiling * 1

    # prepare data
    a_rawdata = torch.tensor([[k * 100 + v for v in range(m)] for k in range(n)])
    a_data = a_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
    b_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')

    # def inputs and outputs
    inputs = [a_data]
    outputs = [b_data]

    # def dynamic function
    @pto.jit
    def cust_dyn_func():
        a = pto.tensor(pto.DataType.DT_INT32, (n, m), "PTO_TENSOR_a")
        b = pto.tensor(pto.DataType.DT_INT32, (n, m), "PTO_TENSOR_b")
        pto.set_vec_tile_shapes(tiling, tiling)
        with pto.dyn_function("MAIN", [a], [b]):
            with pto.loop_function("s0", "k", pto.loop_range_(10)) as rlf:
                for k in rlf:
                    if pto.cond(k == 0):
                        b.move(pto.add(a, a))
                    else:
                        b.move(pto.add(a, b))
        assert isinstance(b, pto.tensor)

    cust_dyn_func(inputs, outputs)

    # get data and compare result
    a_data_cpu = a_data.cpu()
    b_data_cpu = b_data.cpu()

    a_data_list = [c for r in a_data_cpu.tolist() for c in r]
    b_data_list = [c for r in b_data_cpu.tolist() for c in r]
    assert b_data_list == [v * 11 for v in a_data_list]
