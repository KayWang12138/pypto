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
"""
profiling of aicpu pref  test for PyPTO
"""
import json
from typing import List, Dict
import contextlib
import os

import pypto
import pytest
import torch
import torch_npu
import pytest

# #include "tilefwk/aicore_print.h"
# AiCorePrintGmTensor(param->ctx, (__gm__ bfloat16_t*)gmTensor_1.GetAddr(), 1024, 0);

@pypto.frontend.jit(debug_options=dict(runtime_debug_mode=1))
# @pypto.frontend.jit
def matmul_add(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    tiling = 32
    # n, k, m = tiling * 8, tiling * 8, tiling * 8
    n, k, m = tiling, tiling, tiling
    pypto.set_vec_tile_shapes(tiling, tiling)
    pypto.set_cube_tile_shapes(
        [tiling, tiling], [tiling, tiling], [tiling, tiling])
    for _ in pypto.loop(1, name="s0", idx_name="i"):
        a0 = pypto.view(a, [n, k], [0, 0])
        b0 = pypto.view(b, [k, m], [0, 0])
        out.move(pypto.add(pypto.matmul(a0, b0, pypto.DT_FP32), c))


def test_device_run_data_from_device_mix_nodep():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    # n, k, m = tiling * 8, tiling * 8, tiling * 8
    n, k, m = tiling, tiling, tiling

    # prepare data
    c_data_list = []
    d_data_list = []

    count = 1

    # a_rawdata = torch.tensor([[1] * k] * n)
    increasing_tensor = torch.arange(1, n * k + 1)
    a_rawdata = increasing_tensor.reshape(n, k)
    b_rawdata = torch.tensor([[1] * m] * k)
    a_data = a_rawdata.to(dtype=torch.float32, device=f'npu:{device_id}')
    b_data = b_rawdata.to(dtype=torch.float32, device=f'npu:{device_id}')

    for idx in range(count):
        c_rawdata = torch.tensor([[idx] * m] * n)
        c_data = c_rawdata.to(dtype=torch.float32, device=f'npu:{device_id}')
        c_data_list.append(c_data)

        d_data = torch.zeros((n, m), dtype=torch.float32,
                             device=f'npu:{device_id}')
        d_data_list.append(d_data)

        # def inputs and outputs
        matmul_add(a_data, b_data, c_data, d_data)
    torch_npu.npu.synchronize()


if __name__ == "__main__":
    test_device_run_data_from_device_mix_nodep()