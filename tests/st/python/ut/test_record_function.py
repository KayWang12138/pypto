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
import pto
import pytest


def test_record_function():
    dtype = pto.DataType.DT_FP16
    shape = (8, 8)
    a = pto.tensor(dtype, shape, "tensor_a")
    b = pto.tensor(dtype, shape, "tensor_b")
    c = None

    graph_t = pto.graph_type.TENSOR_GRAPH
    func_t = pto.function_type.STATIC

    pto.begin_function("ADD", graph_t, func_t, a, b)
    pto.set_vec_tile_shapes(8, 8)
    c = pto.add(a, b)
    pto.end_function("ADD", False)
    # del recorder

    print(pto.dump())
    # Replace True with False to see graph
    assert isinstance(c, pto.tensor)


def test_begin_inplaceadd_end_function():
    dtype = pto.DataType.DT_FP16
    func_type = pto.function_type.STATIC
    shape = (8, 8)
    a = pto.tensor(dtype, shape, "tensor_a")
    b = pto.tensor(dtype, shape, "tensor_b")
    c = None

    graph_t = pto.graph_type.TENSOR_GRAPH
    func_t = pto.function_type.STATIC

    pto.begin_function("ADD_INPLACE", graph_t, func_t, a, b)
    pto.set_vec_tile_shapes(8, 8)
    c = a + b
    pto.end_function("ADD_INPLACE", False)

    print(pto.dump())
    assert isinstance(c, pto.tensor)


def test_empty_begin_end_function():
    dtype = pto.DataType.DT_FP16
    a = pto.tensor(dtype, (8, 8), "tensor_a")

    graph_t = pto.graph_type.TENSOR_GRAPH
    func_t = pto.function_type.STATIC

    pto.begin_function("MAIN", graph_t, func_t, a)
    pto.set_vec_tile_shapes(8, 8)
    pto.end_function("MAIN", False)

    assert True


def test_record_function_static():
    dtype = pto.DataType.DT_FP16
    a = pto.tensor(dtype, (8, 8), "tensor_a")
    b = pto.tensor(dtype, (8, 8), "tensor_b")
    c = pto.tensor(dtype, (8, 8), "tensor_c")

    func_cfg = pto.func_config(pto.function_type.STATIC)
    recorder = pto.record_func("ADD_FNC", func_cfg, [a, b, c])
    pto.set_vec_tile_shapes(8, 8)
    c.move(pto.add(a, b))
    del recorder

    assert True
