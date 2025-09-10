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
    GLOBAL_LOG_LEVEL=1 python example/vector_op.py | tee run_vadd_py.log

Confirm same output as `vector_add` in `cpp_reference`
"""

import pto
from utils import pto_function

GRAPH_T = pto.graph_type.TENSOR_GRAPH
FUNC_T = pto.function_type.STATIC


def vector_add():
    dtype = pto.DataType.DT_FP32
    shape = (128, 2, 64, 128)
    a = pto.tensor(dtype, shape, "A")
    b = pto.tensor(dtype, shape, "B")
    c = None

    with pto_function("ADD", GRAPH_T, FUNC_T, a, b):
        pto.set_vec_tile_shapes(32, 1, 16, 32)
        c = pto.add(a, b)

    assert isinstance(c, pto.tensor)


def vector_sub():
    dtype = pto.DataType.DT_FP32
    shape = (128, 2, 64, 128)
    a = pto.tensor(dtype, shape, "A")
    b = pto.tensor(dtype, shape, "B")
    c = None

    with pto_function("SUB", GRAPH_T, FUNC_T, a, b):
        pto.set_vec_tile_shapes(32, 1, 16, 32)
        c = pto.sub(a, b)

    assert isinstance(c, pto.tensor)


def vector_mul():
    dtype = pto.DataType.DT_FP32
    shape = (128, 2, 64, 128)
    a = pto.tensor(dtype, shape, "A")
    b = pto.tensor(dtype, shape, "B")
    c = None

    with pto_function("MUL", GRAPH_T, FUNC_T, a, b):
        pto.set_vec_tile_shapes(32, 1, 16, 32)
        c = pto.mul(a, b)

    assert isinstance(c, pto.tensor)


def vector_div():
    dtype = pto.DataType.DT_FP32
    shape = (128, 2, 64, 128)
    a = pto.tensor(dtype, shape, "A")
    b = pto.tensor(dtype, shape, "B")
    c = None

    with pto_function("DIV", GRAPH_T, FUNC_T, a, b):
        pto.set_vec_tile_shapes(32, 1, 16, 32)
        c = pto.div(a, b)

    assert isinstance(c, pto.tensor)


def vector_view():
    dtype = pto.DataType.DT_FP32
    shape = (128, 2, 64, 128)
    a = pto.tensor(dtype, shape, "A")
    b = pto.tensor(dtype, shape, "B")
    c = None

    with pto_function("VIEW", GRAPH_T, FUNC_T, a, b):
        pto.set_vec_tile_shapes(32, 1, 16, 32)
        pto.set_cube_tile_shapes((16, 16), (16, 16), (16, 16))
        print("vec tile shapes:", pto.get_vec_tile_shapes())
        c = pto.add(
            pto.view(a, [64, 1, 32, 64], [0, 0, 0, 0]),
            pto.view(b, [64, 1, 32, 64], [0, 0, 0, 0]),
        )

    assert isinstance(c, pto.tensor)


def vector_cast_exp():
    dtype = pto.DataType.DT_FP32
    shape = (128, 2, 64, 128)
    a = pto.tensor(dtype, shape, "A")
    b = None

    with pto_function("exp", GRAPH_T, FUNC_T, a):
        pto.set_vec_tile_shapes(32, 1, 16, 32)
        b = pto.cast(pto.exp(a), pto.DataType.DT_FP16, pto.CastMode.CAST_FLOOR)

    assert isinstance(b, pto.tensor)
    print(b.get_shape())
    print(b.get_dtype())


def vector_element():
    a = pto.element(pto.DataType.DT_FP32, 1.0)
    b = pto.element(pto.DataType.DT_INT64, 2)
    c = pto.element(pto.DataType.DT_UINT64, 3)
    print(
        a.get_data_type(),
        a.get_signed_data(),
        a.get_unsigned_data(),
        a.get_float_data(),
    )
    print(
        b.get_data_type(),
        b.get_signed_data(),
        b.get_unsigned_data(),
        b.get_float_data(),
    )
    print(
        c.get_data_type(),
        c.get_signed_data(),
        c.get_unsigned_data(),
        c.get_float_data(),
    )
    # DataType.DT_INT64 2 2 1e-323
    # DataType.DT_UINT64 3 3 1.5e-323
    # b c have both signed and unsigned data, which is not expected

    dtype = pto.DataType.DT_FP32
    shape = (128, 2, 64, 128)
    d = pto.tensor(dtype, shape, "D")
    e = None
    f = None
    with pto_function("ELEMENT", GRAPH_T, FUNC_T, d):
        pto.set_vec_tile_shapes(32, 1, 16, 32)
        e = pto.add_s(d, a)  # add element to tensor
        f = pto.mul_s(d, a)
    print(e.get_shape(), e.get_dtype())
    print(f.get_shape(), f.get_dtype())


def vector_maximum():
    dtype = pto.DataType.DT_FP32
    shape = (128, 2, 64, 128)
    a = pto.tensor(dtype, shape, "A")
    b = pto.tensor(dtype, shape, "B")
    c = None

    with pto_function("MAXIMUM", GRAPH_T, FUNC_T, a, b):
        pto.set_vec_tile_shapes(32, 1, 16, 32)
        c = pto.maximum(a, b)

    assert isinstance(c, pto.tensor)
    print(c.get_shape())
    print(c.get_dtype())


def vector_row_sum_single():
    dtype = pto.DataType.DT_FP32
    shape = (128, 2, 64, 128)
    a = pto.tensor(dtype, shape, "A")
    b = None

    with pto_function("ROW_SUM_SINGLE", GRAPH_T, FUNC_T, a):
        pto.set_vec_tile_shapes(32, 1, 16, 32)
        b = pto.row_sum_single(a)

    assert isinstance(b, pto.tensor)
    print(b.get_shape())
    print(b.get_dtype())


def vector_row_max_single():
    dtype = pto.DataType.DT_FP32
    shape = (128, 2, 64, 128)
    a = pto.tensor(dtype, shape, "A")
    b = None

    with pto_function("ROW_MAX_SINGLE", GRAPH_T, FUNC_T, a):
        pto.set_vec_tile_shapes(32, 1, 16, 32)
        b = pto.row_max_single(a)

    assert isinstance(b, pto.tensor)
    print(b.get_shape())
    print(b.get_dtype())


def vector_rms_norm():
    dtype = pto.DataType.DT_FP32
    shape = (128, 2, 64, 128)
    a = pto.tensor(dtype, shape, "A")
    b = None

    with pto_function("RMS_NORM", GRAPH_T, FUNC_T, a):
        pto.set_vec_tile_shapes(16, 1, 16, 16)
        b = pto.rms_norm(a)

    assert isinstance(b, pto.tensor)
    print(b.get_shape())
    print(b.get_dtype())


def vector_reciprocal():
    dtype = pto.DataType.DT_FP32
    shape = (128, 2, 64, 128)
    a = pto.tensor(dtype, shape, "A")
    b = None

    with pto_function("RECIPROCAL", GRAPH_T, FUNC_T, a):
        pto.set_vec_tile_shapes(32, 1, 16, 32)
        b = pto.reciprocal(pto.transpose(a, [2, 3]))

    assert isinstance(b, pto.tensor)
    print(b.get_shape())
    print(b.get_dtype())


def vector_assemble():
    dtype = pto.DataType.DT_FP32
    shape = (128, 128)
    offsets = (0, 0)
    tensor = pto.tensor(dtype, shape, "tensor")
    c = None

    assemble_input = [(tensor, offsets)]

    with pto_function("ASSEMBLE", GRAPH_T, FUNC_T, tensor):
        pto.set_vec_tile_shapes(128, 128)
        c = pto.assemble(assemble_input)

    assert isinstance(c, pto.tensor)


if __name__ == "__main__":
    vector_add()
    vector_sub()
    vector_mul()
    vector_div()
    vector_view()
    vector_cast_exp()
    vector_element()
    vector_maximum()
    vector_row_sum_single()
    vector_row_max_single()
    vector_rms_norm()
    vector_reciprocal()
    vector_assemble()
