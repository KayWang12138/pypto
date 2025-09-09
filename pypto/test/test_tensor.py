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


def test_init_tensor():
    expected_dtype = pto.DataType.DT_FP16
    shape = [32, 1]
    tensor = pto.tensor(expected_dtype, shape, "tensor_a")

    assert tensor.get_dtype() == expected_dtype, "[Tensor] get_dtype is wrong!"
    assert tensor.shape == shape, "[Tensor] shape is wrong!"
    assert tensor.get_shape() == shape, "[Tensor] get_shape is wrong!"


def test_init_tensor_no_name():
    expected_dtype = pto.DataType.DT_FP16
    shape = [32, 1]
    tensor = pto.tensor(expected_dtype, shape)

    assert tensor.get_dtype() == expected_dtype, "[Tensor] get_dtype is wrong!"
    assert tensor.shape == shape, "[Tensor] shape is wrong!"


def test_tensor_get_shape():
    dtype = pto.DataType.DT_FP16
    expected_shape = [32, 1]
    tensor = pto.tensor(dtype, expected_shape, "tensor_a")
    actual = tensor.get_shape()

    assert (
        actual == expected_shape
    ), f"[Tensor] get_shape is wrong. Got {actual}. Expected {expected_shape}"


def test_tensor_shape_property():
    dtype = pto.DataType.DT_FP16
    expected_shape = [32, 1]
    tensor = pto.tensor(dtype, expected_shape, "tensor_a")
    actual = tensor.shape

    assert (
        actual == expected_shape
    ), f"[Tensor] get_shape is wrong. Got {actual}. Expected {expected_shape}"
