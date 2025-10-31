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

import pto
import pytest
import numpy as np
from numpy.testing import assert_allclose

F_1 = 1.0
SHAPE = [8, 24]
DTYPE = pto.DT_FP32


def prepare_test_data(shape):
    torch_tensor = np.ones(shape, dtype=np.float32)
    x_data = torch_tensor.flatten().tolist()

    res_data = np.ones(shape, dtype=np.float32) * 3
    res_data = res_data.flatten().tolist()

    golden = np.zeros(shape, dtype=np.float32)
    golden[:, :16] = 2
    golden_data = golden.flatten().tolist()

    return x_data, res_data, golden_data


def test_assmble_function_call():
    pto.runtime._device_init()
    x = pto.tensor(SHAPE, DTYPE)
    out = pto.tensor(SHAPE, DTYPE)
    with pto.function("main", [x], [out]):
        pto.set_vec_tile_shapes(8, 8)
        with pto.loop_function("LOOP_assemble_L0", "a_idx", pto.loop_range(2)) as aloop:
            for a_idx in aloop:
                tmp = pto.view(x, [8, 8], [0, a_idx * 8])
                add_tensor = pto.add(tmp, F_1)
                # function call
                pto.assemble(add_tensor, [0, a_idx * 8], out)
    x_data, res_data, golden_data = prepare_test_data(SHAPE)
    pto.runtime._device_run_once_data_from_host([x_data], [res_data])
    assert_allclose(res_data, golden_data, atol=1e-5, verbose=True)
    pto.runtime._device_fini()


def test_assmble_tensor_call():
    pto.runtime._device_init()
    x = pto.tensor(SHAPE, DTYPE)
    out = pto.tensor(SHAPE, DTYPE)
    with pto.function("main", [x], [out]):
        pto.set_vec_tile_shapes(8, 8)
        with pto.loop_function("LOOP_assemble_L0", "a_idx", pto.loop_range(2)) as aloop:
            for a_idx in aloop:
                tmp = pto.view(x, [8, 8], [0, a_idx * 8])
                add_tensor = pto.add(tmp, F_1)
                # tensor call
                out.assemble(add_tensor, [0, a_idx * 8])
    x_data, res_data, golden_data = prepare_test_data(SHAPE)
    pto.runtime._device_run_once_data_from_host([x_data], [res_data])
    assert_allclose(res_data, golden_data, atol=1e-5, verbose=True)
    pto.runtime._device_fini()


def test_assmble_syntactic_sugar():
    pto.runtime._device_init()
    x = pto.tensor(SHAPE, DTYPE)
    out = pto.tensor(SHAPE, DTYPE)
    with pto.function("main", [x], [out]):
        pto.set_vec_tile_shapes(8, 8)
        with pto.loop_function("LOOP_assemble_L0", "a_idx", pto.loop_range(2)) as aloop:
            for a_idx in aloop:
                tmp = pto.view(x, [8, 8], [0, a_idx * 8])
                add_tensor = pto.add(tmp, F_1)
                # syntactic_sugar call
                out[0:, a_idx * 8:] = add_tensor
    x_data, res_data, golden_data = prepare_test_data(SHAPE)
    pto.runtime._device_run_once_data_from_host([x_data], [res_data])
    assert_allclose(res_data, golden_data, atol=1e-5, verbose=True)
    pto.runtime._device_fini()
