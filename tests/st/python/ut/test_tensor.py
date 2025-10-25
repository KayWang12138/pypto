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

GRAPH_T = pto.GraphType.TENSOR_GRAPH
FUNC_T = pto.FunctionType.STATIC


def test_init_tensor():
    dtype = pto.DT_FP16
    shape = [32, 1]

    a = pto.tensor(shape, dtype, "a")
    assert a.name == "a"
    assert a.dtype == dtype
    assert a.shape == shape
    assert a.dim() == len(shape)

    b = pto.tensor([-1, 2], dtype, "b")
    assert b.dtype == dtype
    assert b.name == "b"
    with pytest.raises(ValueError):
        # dynamic shape could not be compared
        assert b.shape == [-1, 2]


def test_init_tensor_no_name():
    expected_dtype = pto.DT_FP16
    shape = [32, 1]
    tensor = pto.tensor(shape, expected_dtype)

    assert tensor.dtype == expected_dtype
    assert tensor.shape == shape

def test_tensor_add_plus_op():
    dtype = pto.DT_FP16
    shape = [8, 8]
    a = pto.tensor(shape, dtype, "tensor_a")
    b = pto.tensor(shape, dtype, "tensor_b")

    with pto.pto_function("ADD", GRAPH_T, FUNC_T, a, b):
        pto.set_vec_tile_shapes(8, 8)
        c = a + b

    assert c.shape == shape
    assert c.dtype == dtype


@pytest.mark.skip(reason="Dep operation interface")
def test_tensor_add_tensor_element():
    dtype = pto.DT_FP16
    shape = [8, 8]
    a = pto.tensor(shape, dtype, "tensor_a")

    with pto.pto_function("ADD", GRAPH_T, FUNC_T, a):
        pto.set_vec_tile_shapes(8, 8)
        elem = pto.element(dtype, 3.14)
        c = a + elem

    assert c.shape == shape
    assert c.dtype == dtype


@pytest.mark.skip(reason="Dep operation interface")
def test_tensor_add_element_tensor():
    dtype = pto.DT_FP16
    shape = [8, 8]
    a = pto.tensor(shape, dtype, "tensor_a")

    with pto.pto_function("ADD", GRAPH_T, FUNC_T, a):
        pto.set_vec_tile_shapes(8, 8)
        elem = pto.element(dtype, 3.14)
        c = elem + a

    assert c.shape == shape
    assert c.dtype == dtype


def test_tensor_sub_op():
    dtype = pto.DT_FP16
    shape = [8, 8]
    a = pto.tensor(shape, dtype, "tensor_a")
    b = pto.tensor(shape, dtype, "tensor_b")

    with pto.pto_function("SUB", GRAPH_T, FUNC_T, a, b):
        pto.set_vec_tile_shapes(8, 8)
        c = a - b

    assert c.shape == shape
    assert c.dtype == dtype


@pytest.mark.skip(reason="Dep operation interface")
def test_tensor_subs_tensor_element():
    dtype = pto.DT_FP16
    shape = [8, 8]
    a = pto.tensor(shape, dtype, "tensor_a")

    with pto.pto_function("SUBS", GRAPH_T, FUNC_T, a):
        pto.set_vec_tile_shapes(8, 8)
        elem = pto.element(dtype, 3.14)
        c = a - elem

    assert c.shape == shape
    assert c.dtype == dtype
