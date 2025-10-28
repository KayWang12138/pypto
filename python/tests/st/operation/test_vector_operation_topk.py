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
""" """
import pytest

from test_case_class_vector_operations import TopKTestCase
from test_case_desc import TensorDesc


@pytest.mark.skip(reason="There is a probability of failure")
def test_tensor_topk():
    original_shape = (64, 64)
    k = 10
    axis = 1
    output_shape = tuple(
        [k if index == axis else value for index, value in enumerate(original_shape)]
    )
    input_tensors = [TensorDesc("A", original_shape, "fp32", [-100, 100])]
    output_tensors = [
        TensorDesc("Value", output_shape, "fp32", [-0, 0]),
        TensorDesc("Index", output_shape, "int32", [0, 0]),
    ]
    view_shape = (32, 64)
    tile_shape = (32, 64)
    test_case = TopKTestCase(
        0,
        "TopK_test_0",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"count": k, "dims": "[-1]", "islargest": "[1]"},
    )
    test_case.exec(True)
