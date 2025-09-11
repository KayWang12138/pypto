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
from pypto.utils import Tensor
from pypto.stub_fun import sigmoid


class Linear(AscppModule):
    def init(self, input_shape=None, output_shape=None):
        self.y = input_shape

    def forward(self, x=None):
        x = x * self.y
        return x


class Layer(AscppModule):
    def init(self, input_shape=None, output_shape=None):
        self.layer1 = Linear(input_shape, output_shape)

    def forward(self, inp=None):
        x = self.layer1(inp)
        return x


if __name__ == "__main__":
    z = Layer(21, 5)
    input_tensor = Tensor()
    z(input_tensor)
    z.gen_code('./generatedcpp/test_module.cpp')
