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

GRAPH_T = pto.GraphType.TENSOR_GRAPH
FUNC_T = pto.FunctionType.STATIC


def test_matrix_matmul():
    dtype = pto.DT_FP32
    a = pto.tensor((32, 64), dtype, "A")
    b = pto.tensor((64, 32), dtype, "B")
    c = None

    with pto.pto_function("MATMUL", GRAPH_T, FUNC_T, a, b):
        pto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pto.matmul(dtype, a, b)
        d = pto.matmul(dtype, a, b, a_trans=True, b_trans=True)

    assert isinstance(c, pto.tensor)
    assert c.shape == [32, 32]

    assert isinstance(d, pto.tensor)
    assert d.shape == [64, 64]


def test_matrix_batch_matmul():
    dtype = pto.DT_FP32
    a = pto.tensor((2, 64, 32), dtype, "A")
    b = pto.tensor((2, 32, 64), dtype, "B")
    c = None

    with pto.pto_function("BATCH_MATMUL", GRAPH_T, FUNC_T, a, b):
        pto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pto.batch_matmul(dtype, a, b)
        d = pto.batch_matmul(dtype, a, b, a_trans=True, b_trans=True)

    assert isinstance(c, pto.tensor)
    assert c.shape == [2, 64, 64]

    assert isinstance(d, pto.tensor)
    assert d.shape == [2, 32, 32]
