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
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose
import torch_npu


def test_vector_operation_range():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    size = 32
    view_shape = (16,)
    tile_shape = (8,)
    start_data = 1.0
    end_data = 32.1
    step_data = 1.0

    pto.runtime._device_init()
    pto.set_codegen_option("support_dynamic_unaligned", True)

    a = pto.tensor((1, 1, 1), pto.DT_FP32, "Range_TENSOR_a")
    b = pto.tensor((size,), pto.DT_FP32, "Range_TENSOR_b")
    start = 1.0
    end = 32.1
    step = 1.0

    with pto.function("RANGE", [a], [b]):
        for b_idx in pto.loop(1, name="LOOP_L0_b_idex", idx_name="b_idx"):
            pto.set_vec_tile_shapes(tile_shape[0])
            res = pto.tensor()
            res.move(pto.arange(start, end, step))
            pto.assemble(res, [b_idx * view_shape[0]], b)
            del res
    a_tensor = torch.rand([1, 1, 1], dtype=torch.float32) * 99.999 + 0.001
    res_tensor = torch.zeros(size, dtype=torch.float32)
    pto.runtime._device_run_once_data_from_host([a_tensor], [res_tensor])

    expected = torch.arange(start_data, end_data, step_data)
    assert_allclose(res_tensor.flatten(), expected.flatten(), rtol=1e-6, atol=1e-7)

    pto.runtime._device_fini()
