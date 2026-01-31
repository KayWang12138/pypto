#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import pypto

import torch
import torch_npu


@pypto.frontend.jit
def cust_dyn_func_add_int(
    a: pypto.Tensor([32, 32], pypto.DT_INT32),
    b: pypto.Tensor([32, 32], pypto.DT_INT32),
    ) -> pypto.Tensor([32, 32], pypto.DT_FP32):
    pypto.set_vec_tile_shapes(32, 32)
    for _ in pypto.loop(1, name="s0", idx_name="k"):
        c = pypto.add(a, b)
    return c


@pypto.frontend.jit
def cust_dyn_func_add_fp32(
    a: pypto.Tensor([32, 32], pypto.DT_FP32),
    b: pypto.Tensor([32, 32], pypto.DT_FP32),
    ) -> pypto.Tensor([32, 32], pypto.DT_FP32):
    pypto.set_vec_tile_shapes(32, 32)
    for _ in pypto.loop(1, name="s0", idx_name="k"):
        c = pypto.add(a, b)
    return c


def device_run():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    device = 'npu:0'

    # prepare data
    a_data = torch.rand([32, 32], dtype=torch.float, device=device)
    b_data = torch.rand([32, 32], dtype=torch.float, device=device)
    c_data = cust_dyn_func_add_fp32(a_data, b_data)
    torch_npu.npu.synchronize()

    golden = torch.add(a_data, b_data)
    assert torch.allclose(golden.cpu(), c_data.cpu(), atol=1e-5)

    c_data_int = cust_dyn_func_add_int(a_data, b_data)
    torch_npu.npu.synchronize()

    assert (not torch.allclose(golden.cpu(), c_data_int.cpu(), atol=1e-5))


def test_run_from_torch():
    device_run()