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
from pypto.module import AscppModule
from pypto.utils import CustStruct, Vector, Tensor, Var, Tuple, ConfigMap, Shape
from pypto.utils import DATATYPE as DT
from pypto.flowcontrol import If


class IntsToCheck(CustStruct):
    v1: int
    op1: int
    op2: int
    len: int
    v2: int
    ind: int
    val: int


class TestVector(AscppModule):
    def init(self):
        ...

    def forward(self, t1: Tensor = None, t2: Tensor = None, itc: CustStruct = None):
        # Test 1
        vec1 = Vector('Tensor', init_vec=[t1, t2])
        vec2 = Vector('Tensor')
        vec3 = Vector('int', init_vec=[234, 222, itc.v1])
        with If(itc.op1 < itc.op2):
            vec4 = Vector('int', init_len=itc.len)
        vec5 = Vector('int', init_len=100, init_val=itc.v2)

        tmp1 = t1 * vec1[1]
        tmp2 = vec2.size()
        vec2.emplace_back(t2)
        vec5[itc.ind] = itc.val

        shape1 = t1.shape
        shape1.emplace_back(1)
        return t1 * t2


if __name__ == "__main__":
    t1 = Tensor()
    t2 = Tensor()
    x = IntsToCheck()

    mod = TestVector()

    mod(t1, t2, x)

    mod.gen_code("./generatedcpp/test_vector.cpp")
