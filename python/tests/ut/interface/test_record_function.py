# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import pto
import pytest


def test_record_function():
    dtype = pto.DT_FP16
    shape = (8, 8)
    a = pto.tensor(shape, dtype, "tensor_a")
    b = pto.tensor(shape, dtype, "tensor_b")
    c = None

    with pto.function("ADD", a, b, static=True):
        pto.set_vec_tile_shapes(8, 8)
        c = pto.add(a, b)

    print(pto.dump())
    # Replace True with False to see graph
    assert isinstance(c, pto.tensor)


def test_begin_inplaceadd_end_function():
    dtype = pto.DT_FP16
    shape = (8, 8)
    a = pto.tensor(shape, dtype, "tensor_a")
    b = pto.tensor(shape, dtype, "tensor_b")
    c = None

    with pto.function("ADD_INPLACE", a, b, static=True):
        pto.set_vec_tile_shapes(8, 8)
        c = a + b

    print(pto.dump())
    assert isinstance(c, pto.tensor)


def test_empty_begin_end_function():
    dtype = pto.DT_FP16
    a = pto.tensor((8, 8), dtype, "tensor_a")

    with pto.function("MAIN", a, static=True):
        pto.set_vec_tile_shapes(8, 8)

    assert True
