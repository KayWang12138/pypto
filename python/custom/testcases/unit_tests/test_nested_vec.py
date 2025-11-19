#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
from pypto.module import AscppModule
from pypto.utils import CustStruct, TensorMap, AggregationVec, Vector, Tensor, Var, Tuple, ConfigMap, Shape
from pypto.utils import DATATYPE as DT


class TestNestedVec(AscppModule):
    def init(self):
        ...

    def forward(self, vec1: Vector = None, vec2: Vector = None):
        vec1_1 = vec1[0]
        vec1_1_1 = vec1_1[1]
        val = vec1_1_1[3]

        vec2_1 = vec2[0]
        val2 = vec2_1[1]

        new_vec = Vector(Vector('int'))
        new_vec_1 = new_vec[0]
        new_val = new_vec_1[0]


if __name__ == '__main__':
    v1 = Vector(Vector(Vector('int')))
    v2 = Vector(Vector('int'))

    mod = TestNestedVec()
    mod(v1, v2)

    mod.gen_code('./generatedcpp/test_nested_vec.cpp')
