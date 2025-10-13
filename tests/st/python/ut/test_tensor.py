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

GRAPH_T = pto.graph_type.TENSOR_GRAPH
FUNC_T = pto.function_type.STATIC


def test_init_tensor():
    expected_dtype = pto.data_type.DT_FP16
    shape = [32, 1]
    tensor = pto.tensor(shape, expected_dtype, "tensor_a")

    assert tensor.get_dtype() == expected_dtype, "[Tensor] get_dtype is wrong!"
    assert tensor.shape == shape, "[Tensor] shape is wrong!"
    assert tensor.get_shape() == shape, "[Tensor] get_shape is wrong!"


def test_init_tensor_no_name():
    expected_dtype = pto.data_type.DT_FP16
    shape = [32, 1]
    tensor = pto.tensor(shape, expected_dtype)

    assert tensor.get_dtype() == expected_dtype, "[Tensor] get_dtype is wrong!"
    assert tensor.shape == shape, "[Tensor] shape is wrong!"


def test_tensor_get_shape():
    dtype = pto.data_type.DT_FP16
    expected_shape = [32, 1]
    tensor = pto.tensor(expected_shape, dtype, "tensor_a")
    actual = tensor.get_shape()

    assert (
        actual == expected_shape
    ), f"[Tensor] get_shape is wrong. Got {actual}. Expected {expected_shape}"


def test_tensor_shape_property():
    dtype = pto.data_type.DT_FP16
    expected_shape = [32, 1]
    tensor = pto.tensor(expected_shape, dtype, "tensor_a")
    actual = tensor.shape

    assert (
        actual == expected_shape
    ), f"[Tensor] get_shape is wrong. Got {actual}. Expected {expected_shape}"


def test_tensor_add_plus_op():
    dtype = pto.DataType.DT_FP16
    shape = [8, 8]
    a = pto.tensor(shape, dtype, "tensor_a")
    b = pto.tensor(shape, dtype, "tensor_b")

    with pto.pto_function("ADD", GRAPH_T, FUNC_T, a, b):
        pto.set_vec_tile_shapes(8, 8)
        c = a + b

    assert c.shape == shape
    assert c.dtype == dtype


def test_tensor_add_tensor_element():
    dtype = pto.DataType.DT_FP16
    shape = [8, 8]
    a = pto.tensor(shape, dtype, "tensor_a")

    with pto.pto_function("ADD", GRAPH_T, FUNC_T, a):
        pto.set_vec_tile_shapes(8, 8)
        elem = pto.element(dtype, 3.14)
        c = a + elem

    assert c.shape == shape
    assert c.dtype == dtype


def test_tensor_add_element_tensor():
    dtype = pto.DataType.DT_FP16
    shape = [8, 8]
    a = pto.tensor(shape, dtype, "tensor_a")

    with pto.pto_function("ADD", GRAPH_T, FUNC_T, a):
        pto.set_vec_tile_shapes(8, 8)
        elem = pto.element(dtype, 3.14)
        c = elem + a

    assert c.shape == shape
    assert c.dtype == dtype


def test_tensor_sub_op():
    dtype = pto.DataType.DT_FP16
    shape = [8, 8]
    a = pto.tensor(shape, dtype, "tensor_a")
    b = pto.tensor(shape, dtype, "tensor_b")

    with pto.pto_function("SUB", GRAPH_T, FUNC_T, a, b):
        pto.set_vec_tile_shapes(8, 8)
        c = a - b

    assert c.shape == shape
    assert c.dtype == dtype


def test_tensor_subs_tensor_element():
    dtype = pto.DataType.DT_FP16
    shape = [8, 8]
    a = pto.tensor(shape, dtype, "tensor_a")

    with pto.pto_function("SUBS", GRAPH_T, FUNC_T, a):
        pto.set_vec_tile_shapes(8, 8)
        elem = pto.element(dtype, 3.14)
        c = a - elem

    assert c.shape == shape
    assert c.dtype == dtype
