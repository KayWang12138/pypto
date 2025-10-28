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
import json
import os
from test_case_class_vector_operations import (
    AddTestCase,
    CastTestCase,
    ExpTestCase,
    ScalarAddSTestCase,
    ScalarDivSTestCase,
    ScalarMaxSTestCase,
    ScalarMulSTestCase,
    ScalarSubSTestCase,
    TopKTestCase,
    TransposeTestCase,
)
from test_case_desc import TensorDesc

_op_to_cls = {
    "Add": AddTestCase,
    "Cast": CastTestCase,
    "Exp": ExpTestCase,
    "ScalarAddS": ScalarAddSTestCase,
    "ScalarSubS": ScalarSubSTestCase,
    "ScalarMulS": ScalarMulSTestCase,
    "ScalarDivS": ScalarDivSTestCase,
    "ScalarMaxS": ScalarMaxSTestCase,
    "Transpose": TransposeTestCase,
    "TopK": TopKTestCase,
}

need_binary_compare = ("Cast", "ScalarMaxS", "Transpose", "TopK")


def test_vec_op_test_case():
    test_case_info = json.loads(os.getenv("CUR_TEST_CASE_INFO"))
    case_op = test_case_info["operation"]
    cls = _op_to_cls.get(case_op, None)
    if cls is None:
        raise ValueError(f"class {case_op}TestCase has not defined.")

    test_case = cls(
        test_case_info.get("case_index"),
        test_case_info.get("case_name"),
        [
            TensorDesc.from_dict(tensor)
            for tensor in test_case_info.get("input_tensors")
        ],
        [
            TensorDesc.from_dict(tensor)
            for tensor in test_case_info.get("output_tensors")
        ],
        test_case_info.get("view_shape"),
        test_case_info.get("tile_shape"),
        test_case_info.get("params"),
    )
    test_case.exec(case_op in need_binary_compare)
