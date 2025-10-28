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

from pypto.module import AscppModule
from pypto.utils import CustStruct, Vector, Tensor, Var, Tuple, ConfigMap, Shape
from pypto.utils import DATATYPE as DT
from pypto.flowcontrol import If, Else


class Type1(CustStruct):
    int_setting: int
    bool_setting: bool


class Type2(CustStruct):
    int_value: int
    vec: Vector[int]
    nested_vec: Vector[Vector[int]]


class Type3(CustStruct):
    str_setting: str


class TestGenCustStruct(AscppModule):
    def init(self, config: Type1 = None, config2: Type3 = None):
        self.int_setting = config.int_setting
        self.bool_setting = config.bool_setting
        with If(config2.str_setting == "yes"):
            self.int_setting += 5

    def forward(self, t1: Tensor = None, settings: Type2 = None):
        return_tensor = Tensor(force_declare=True)
        with If((self.int_setting > 10) & self.bool_setting):
            return_tensor.assign(t1 + settings.int_value + settings.vec[0] + settings.nested_vec[0][0])
        with Else():
            return_tensor.assign(t1 + settings.int_value + settings.vec[1] + settings.nested_vec[1][1])
        return return_tensor


if __name__ == "__main__":
    t1 = Tensor()
    type1_config = Type1()
    type2_config = Type2()
    type3_config = Type3()

    mod = TestGenCustStruct(type1_config, type3_config)

    mod(t1, type2_config)

    mod.gen_code("./generatedcpp/test_gen_custstruct.cpp")
