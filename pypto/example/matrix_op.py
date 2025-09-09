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
Run:
    GLOBAL_LOG_LEVEL=1 python example/matrix_op.py | tee run_vadd_py.log

Confirm same output as `vector_add` in `cpp_reference`
"""

import pto
from utils import pto_function

GRAPH_T = pto.graph_type.TENSOR_GRAPH
FUNC_T = pto.function_type.STATIC


def matrix_matmul():
    dtype = pto.DataType.DT_FP32
    a = pto.tensor(dtype, (32, 64), "A")
    b = pto.tensor(dtype, (64, 32), "B")
    c = None

    with pto_function("MATMUL", GRAPH_T, FUNC_T, a, b):
        pto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
        c = pto.matmul(dtype, a, b)
        d = pto.matmul(dtype, a, b, a_trans=True, b_trans=True)

    assert isinstance(c, pto.tensor)
    print(c.get_shape())
    assert isinstance(d, pto.tensor)
    print(d.get_shape())


if __name__ == "__main__":
    matrix_matmul()
