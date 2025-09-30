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
from test_case_class_vector_operations import ScalarSubSTestCase
from test_case_desc import TensorDesc


def test_tensor_scalar_subs():
    original_shape = (64, 64)
    input_tensors = [TensorDesc("A", original_shape, "fp32", [-100, 100])]
    output_tensors = [TensorDesc("B", original_shape, "fp32", [-100, 100])]
    view_shape = (32, 32)
    tile_shape = (32, 32)
    test_case = ScalarSubSTestCase(
        0,
        "ScalarSubS_test_0",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"scalar": 10, "reverse": 0},
    )
    test_case.exec(False)
