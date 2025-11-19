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
import numpy as np
import torch
from numpy.testing import assert_allclose
import torch_npu


F_1 = 1.0
SHAPE = [8, 32]
DTYPE = pto.DT_FP32


def test_assmble_2d():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pto.runtime._device_init()
    x = pto.tensor(SHAPE, DTYPE)
    out = pto.tensor(SHAPE, DTYPE)
    with pto.function("main", [x], [out]):
        pto.set_vec_tile_shapes(8, 8)
        for a_idx in pto.loop(4, name="LOOP_assemble_L0", idx_name="a_idx"):
            tmp = pto.view(x, [8, 8], [0, a_idx * 8])
            add_tensor = pto.add(tmp, F_1)
            offset = a_idx * 8
            if pto.cond(a_idx == 0):
                # syntactic_sugar call: out[0:, :]
                out[0:, :] = add_tensor
            elif pto.cond(a_idx == 1):
                # syntactic_sugar call
                out[0:, offset:] = add_tensor
            elif pto.cond(a_idx == 2):
                # tensor call
                out.assemble(add_tensor, [0, offset])
            else:
                # function call
                pto.assemble(add_tensor, [0, offset], out)
            del add_tensor
            del tmp

    torch_tensor = torch.ones(SHAPE, dtype=torch.float32)
    res_data = torch.ones(SHAPE, dtype=torch.float32) * 3
    golden = torch.zeros(SHAPE, dtype=torch.float32)
    golden[:, :32] = 2
    pto.runtime._device_run_once_data_from_host([torch_tensor], [res_data])
    assert_allclose(res_data, golden, atol=1e-5, verbose=True)
    pto.runtime._device_fini()


def test_assmble_1d():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pto.runtime._device_init()
    x = pto.tensor([24], DTYPE)
    out = pto.tensor([24], DTYPE)
    with pto.function("main", [x], [out]):
        pto.set_vec_tile_shapes(8)
        for a_idx in pto.loop(2, name="LOOP_assemble_L0", idx_name="a_idx"):
            tmp = pto.view(x, [8], [a_idx * 8])
            add_tensor = pto.add(tmp, F_1)
            # syntactic_sugar call
            out[a_idx * 8:] = add_tensor
    torch_tensor = torch.ones([24], dtype=torch.float32)

    res_data = torch.ones([24], dtype=torch.float32) * 3

    golden = torch.zeros([24], dtype=torch.float32)
    golden[:16] = 2
    pto.runtime._device_run_once_data_from_host([torch_tensor], [res_data])
    assert_allclose(res_data, golden, atol=1e-5, verbose=True)
    pto.runtime._device_fini()
