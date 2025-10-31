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


@pytest.mark.skip(reason="Dep operation interface")
def test_vector_operation_range():
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
    start = pto.element(pto.DT_FP32, start_data)
    end = pto.element(pto.DT_FP32, end_data)
    step = pto.element(pto.DT_FP32, step_data)

    with pto.function("RANGE", [a], [b]):
        loop_range_b = pto.loop_range(1)
        with pto.loop_function("LOOP_L0_b_idex", "b_idx", loop_range_b) as bloop:
            for b_idx in bloop:
                pto.set_vec_tile_shapes(tile_shape[0])
                res = pto.tensor()
                res.move(pto.range(start, end, step))
                pto.assemble(res, [b_idx * view_shape[0]], b)
                del res
    a_tensor = np.random.uniform(0.001, 100, [1, 1, 1]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = list([0] * size)

    pto.runtime._device_run_once_data_from_host([a_data], [b_data])

    golden_data = np.arange(start_data, end_data, step_data)
    assert(np.allclose(b_data, golden_data, rtol=1e-6, atol=1e-7))
    pto.runtime._device_fini()
