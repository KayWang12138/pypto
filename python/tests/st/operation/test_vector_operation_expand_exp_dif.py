#!/usr/bin/env python3
# coding: utf-8
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
from test_case_class_vector_operations import ExpandExpDifTestCase


def test_tensor_expand_exp_dif_0():
    original_shape = (16, 128)
    input_tensors = [
        {
            "name": "A",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [-10, 10],
        },
        {
            "name": "B",
            "shape": (16, 1),
            "dtype": "fp32",
            "data_range": [-10, 10],
        }
    ]
    output_tensors = [
        {
            "name": "C",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (16, 128)
    tile_shape = (16, 32)
    test_case = ExpandExpDifTestCase(
        0,
        "Expand_exp_dif_test_0",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {},
    )
    test_case.exec(False)

def test_tensor_expand_exp_dif_1():
    original_shape = (16, 128)
    input_tensors = [
        {
            "name": "A",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [-10, 10],
        },
        {
            "name": "B",
            "shape": (1, 128),
            "dtype": "fp32",
            "data_range": [-10, 10],
        }
    ]
    output_tensors = [
        {
            "name": "C",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (16, 128)
    tile_shape = (16, 32)
    test_case = ExpandExpDifTestCase(
        0,
        "Expand_exp_dif_test_1",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {},
    )
    test_case.exec(False)
