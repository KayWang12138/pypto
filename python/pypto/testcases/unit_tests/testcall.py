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
from pypto.utils import CustStruct, TensorMap, Vector, Tensor, Var, Tuple, ConfigMap, Shape
from pypto.utils import DATATYPE as DT
from pypto.stub_fun import rms_norm, cast


class FuncC(AscppModule):
    def init(self):
        ...

    def forward(self, x: Tensor = None):
        x = rms_norm(x)
        return x


class FuncB(AscppModule):
    def init(self):
        self.func_c = FuncC()

    def forward(self, x: Tensor = None):
        x = cast(x, DT.fp32)
        return self.func_c(x)


class FuncA(AscppModule):
    def init(self):
        self.func_b = FuncB()
        self.func_c = FuncC()

    def forward(self, x: Tensor = None):
        x = self.func_c(x)
        x = self.func_b(x)
        return x


if __name__ == '__main__':
    x = Tensor()
    res = FuncA()
    res(x)
    res.gen_code("./generatedcpp/test_call.cpp")
