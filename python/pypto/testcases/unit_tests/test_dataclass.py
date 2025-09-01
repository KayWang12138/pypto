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

from pyascpp.module import AscppModule
from pyascpp.utils import ConfigMap, Tensor, DATATYPE


class NNConfig(ConfigMap):
    input_size: int = 7
    hidden_size: int = 64
    output_size: int


class SimpleNN(AscppModule):
    def init(self, config: NNConfig = None):
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size

        self.w1 = Tensor([self.input_size, self.hidden_size], DATATYPE.fp32)
        self.w2 = Tensor([self.hidden_size, self.output_size], DATATYPE.fp32)

    def forward(self, input_tsr: Tensor = None):
        h1 = input_tsr * self.w1
        h2 = h1 * self.w2
        return h2


if __name__ == "__main__":
    config = NNConfig()
    network = SimpleNN(config)
    network(Tensor())
    network.gen_code("generatedcpp/test_dataclass.cpp")
