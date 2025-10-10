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


def test_vector_operation_rsqrt():
    pto.set_codegen_config("SUPPORT_DYNAMIC_UNALIGNED", True)
    dtype = pto.DataType.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.DeviceInit()
    a = pto.tensor(shape, pto.DataType.DT_FP32, "RSQRT_TENSOR_a")
    b = pto.tensor(shape, pto.DataType.DT_FP32, "RSQRT_TENSOR_b")

    with pto.dyn_function("RSQRT", [a], [b]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_RSQRT_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_RSQRT_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, view_shape, [min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                          pto.symbolic_scalar(n)), min(pto.symbolic_scalar(m) - b_idx * view_shape[1],
                                          pto.symbolic_scalar(m))], [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.rsqrt(tile_a))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                        del tile_a
    a_tensor = np.random.uniform(0, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = list([0] * n * m)
    pto.DeviceRunOnceDataFromHost([a_data], [b_data])
    assert_allclose(np.array(b_data),
                    np.array(torch.rsqrt(torch.tensor(a_tensor)).flatten().tolist()), rtol=3e-3, atol=3e-3)
    pto.DeviceFini()

