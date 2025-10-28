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
import math
import torch
import pytest
import numpy as np

import pto
from pto import (
    tensor, element, view, symbolic_scalar,
    loop_range, loop_function, function,
    set_vec_tile_shapes, set_codegen_option,
    device_init, device_run_once_data_from_host, device_fini,
)


@pytest.mark.skip(reason="Dep operation interface")
def test_maxs():
    scalar_data = 5
    first_dim, second_dim = 128, 128
    view_shape, tile_shape = (64, 64), (32, 32)
    device_init()
    set_codegen_option("support_dynamic_unaligned", True)

    x = tensor((first_dim, second_dim), pto.DT_INT32, "Operand1")
    y = tensor((first_dim, second_dim), pto.DT_INT32, "Output")
    scalar = pto.element(pto.DT_INT32, scalar_data)

    bloop_range = math.ceil(first_dim / view_shape[0])
    sloop_range = math.ceil(second_dim / view_shape[1])
    first_view_shape, second_view_shape = view_shape

    bloop = loop_range(bloop_range)
    sloop = loop_range(sloop_range)
    with function("MaxS", [x], [y]), \
            loop_function("LOOP_L0_bIdx", "bIdx", bloop) as bloop_ctx, \
            loop_function("LOOP_L1_sIdx", "sIdx", sloop) as sloop_ctx:
        for b_idx in bloop_ctx:
            for s_idx in sloop_ctx:
                tile_tensor_0 = view(
                    x, view_shape,
                    [
                        pto.min(
                            symbolic_scalar(first_dim) - b_idx * first_view_shape,
                            symbolic_scalar(first_view_shape)
                        ),
                        pto.min(
                            symbolic_scalar(second_dim) - s_idx * second_view_shape,
                            symbolic_scalar(second_view_shape)
                        ),
                    ],
                    [b_idx * first_view_shape, s_idx * second_view_shape]
                )
                set_vec_tile_shapes(*tile_shape)
                res = tensor()
                res.move(pto.maxs(tile_tensor_0, scalar))
                pto.assemble(
                    res,
                    [b_idx * first_view_shape, s_idx * second_view_shape],
                    y,
                )
                del tile_tensor_0, res

    nx_data = np.random.uniform(-10, 10, [first_dim, second_dim]).astype(np.int32)
    ny_data = np.zeros([first_dim, second_dim]).astype(np.int32)
    x_data = nx_data.flatten().tolist()
    y_data = ny_data.flatten().tolist()
    device_run_once_data_from_host([x_data], [y_data])
    golden_data = np.maximum(x_data, scalar_data)
    assert(np.allclose(y_data, golden_data, rtol=1e-9, atol=1e-10))
    device_fini()
