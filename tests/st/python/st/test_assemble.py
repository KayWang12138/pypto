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


def test_assemble_basic():
    pto.DeviceInit()
    dtype = pto.DT_FP32

    input_shape = [8, 8]
    x = pto.tensor(input_shape, dtype)
    output_shape = [8, 16]
    out = pto.tensor(output_shape, dtype)
    with pto.function("main", [x], [out]):
        with pto.loop_function("LOOP_assemble_L0", "a_idx", pto.loop_range(1)) as aloop:
            for _ in aloop:
                pto.set_vec_tile_shapes(8, 8)
                pto.assemble(x, [0, 0], out)

    torch_tensor = np.ones((8, 8))
    x_data = torch_tensor.flatten().tolist()
    
    res_data = np.ones((8, 16), dtype=np.float32) * 3
    res_data = res_data.flatten().tolist()

    golden_data = np.zeros((8, 16), dtype=np.float32)
    golden_data[:, :8] = 1
    golden_data = golden_data.flatten().tolist()

    pto.DeviceRunOnceDataFromHost([x_data], [res_data])
    assert_allclose(res_data, golden_data, atol=1e-5, verbose=True)
    pto.DeviceFini()


@pytest.mark.skip(reason="Dep operation interface")
def test_view_assemble():
    f_1 = 1.0
    pto.DeviceInit()
    dtype = pto.DT_FP32

    input_shape = [8, 24]
    x = pto.tensor(input_shape, dtype)
    output_shape = [8, 24]
    out = pto.tensor(output_shape, dtype)
    with pto.function("main", [x], [out]):
        pto.set_vec_tile_shapes(8, 8)
        with pto.loop_function("LOOP_assemble_L0", "a_idx", pto.loop_range(2)) as aloop:
            for a_idx in aloop:
                tmp = pto.view(x, [8, 8], [0, a_idx * 8])
                add_tensor = pto.add_s(tmp, pto.element(pto.DT_FP32, f_1))
                pto.assemble(add_tensor, [0, a_idx * 8], out)

    torch_tensor = np.ones((8, 24))
    x_data = torch_tensor.flatten().tolist()
    
    res_data = np.ones((8, 24), dtype=np.float32) * 3
    res_data = res_data.flatten().tolist()

    golden_data = np.zeros((8, 24), dtype=np.float32)
    golden_data[:, :16] = 2
    golden_data = golden_data.flatten().tolist()

    pto.DeviceRunOnceDataFromHost([x_data], [res_data])
    assert_allclose(res_data, golden_data, atol=1e-5, verbose=True)
    pto.DeviceFini()

