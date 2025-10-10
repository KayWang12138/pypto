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

GRAPH_T = pto.graph_type.TENSOR_GRAPH
FUNC_T = pto.function_type.STATIC


def test_matrix_matmul():
    dtype = pto.data_type.DT_FP32
    a = pto.tensor((32, 64), dtype, "A")
    b = pto.tensor((64, 32), dtype, "B")
    c = None

    with pto.pto_function("MATMUL", GRAPH_T, FUNC_T, a, b):
        pto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pto.matmul(dtype, a, b)
        d = pto.matmul(dtype, a, b, a_trans=True, b_trans=True)

    assert isinstance(c, pto.tensor)
    expected_c_shape = [32, 32]
    actual_c_shape = c.get_shape()
    assert (expected_c_shape == actual_c_shape)

    assert isinstance(d, pto.tensor)
    expected_d_shape = [64, 64]
    actual_d_shape = d.get_shape()
    assert (actual_d_shape == expected_d_shape)


def test_matrix_batch_matmul():
    dtype = pto.data_type.DT_FP32
    a = pto.tensor((2, 64, 32), dtype, "A")
    b = pto.tensor((2, 32, 64), dtype, "B")
    c = None

    with pto.pto_function("BATCH_MATMUL", GRAPH_T, FUNC_T, a, b):
        pto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pto.batch_matmul(dtype, a, b)
        d = pto.batch_matmul(dtype, a, b, a_trans=True, b_trans=True)

    assert isinstance(c, pto.tensor)
    expected_c_shape = [2, 64, 64]
    actual_c_shape = c.get_shape()
    assert (expected_c_shape == actual_c_shape)

    assert isinstance(d, pto.tensor)
    expected_d_shape = [2, 32, 32]
    actual_d_shape = d.get_shape()
    assert (actual_d_shape == expected_d_shape)
