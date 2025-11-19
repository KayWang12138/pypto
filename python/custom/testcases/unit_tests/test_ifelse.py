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
from pypto.utils import CustStruct, Vector, Tensor, Var, Tuple, ConfigMap, Shape
from pypto.utils import DATATYPE as DT
from pypto.flowcontrol import If, Else, Elif


class IntsToCheck(CustStruct):
    v1: int
    v2: int
    v3: int


class TestIfElse(AscppModule):
    def init(self):
        ...

    def forward(self, t1: Tensor = None, t2: Tensor = None, itc: CustStruct = None):
        # Test 1
        with If(itc.v1):
            # test += and *=
            t1 *= t2
            t1 *= 5
            t1 += t2
            t1 *= t2

        # Test 2: if-elif-else
        with If(itc.v1):
            t1 += t2
        with Elif(itc.v2):
            t1 += 2
        with Elif(itc.v3):
            t1 += 3
        with Else():
            t1 *= t2

        # Test 3: nested
        with If(itc.v1):
            with If(itc.v2):
                t1 += 3
            with Else():
                t2 += 4
        with Else():
            with If(itc.v2):
                t1 += 5
            with Else():
                t2 += 6

        # Test 4: no else
        with If(itc.v1):
            t2 *= 0.1
        with Elif(itc.v2):
            t2 *= 0.2

        return t1 * t2


if __name__ == "__main__":
    t1 = Tensor()
    t2 = Tensor()
    x = IntsToCheck()

    mod = TestIfElse()

    mod(t1, t2, x)

    mod.gen_code("./generatedcpp/test_ifelse.cpp")
