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
from utils import dyn_function, loop_function, record_if_branch


# FIXME(anastasios): Fix after pypackage.Monkey patching for now
pto.dyn_function = dyn_function
pto.loop_function = loop_function
pto.cond = record_if_branch


def test_device_run_data_from_host():
    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0)
    tiling = 32
    n, m = tiling * 1, tiling * 1

    pto._DeviceInit()

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

    pto._DeviceRunOnceDataFromHost([a_data], [b_data])

    assert b_data == [v * 11 for v in range(n * m)]
    pto._DeviceFini()

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

    pto._DeviceInit()

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

    a_rawdata = torch.tensor([[k * 100 + v for v in range(m)] for k in range(n)])
    a_data = a_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
    b_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')
    stream = torch.npu.current_stream()
    stream.synchronize()

    pto._DeviceRunOnceDataFromDevice([a_data.data_ptr()], [b_data.data_ptr()], stream.npu_stream)

    stream.synchronize()

    a_data_cpu = a_data.cpu()
    b_data_cpu = b_data.cpu()

    a_data_list = [c for r in a_data_cpu.tolist() for c in r]
    b_data_list = [c for r in b_data_cpu.tolist() for c in r]
    assert b_data_list == [v * 11 for v in a_data_list]
    pto._DeviceFini()