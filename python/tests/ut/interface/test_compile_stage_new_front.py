#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import pypto
import torch


def test_all_compile_stages():
    def compile_add(a, b):
        shape = (4, 4)
        dtype = pypto.DT_FP32

        @pypto.frontend.jit(
            runtime_options={"run_mode": 1}
        )
        def kernel(a: pypto.Tensor(shape, dtype),
                   b: pypto.Tensor(shape, dtype)) -> pypto.Tensor(shape, dtype):
            pypto.set_vec_tile_shapes(4, 4)
            c = a + b
            return c
        return kernel(a, b)

    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3

    output_c = compile_add(a, b)


if __name__ == "__main__":
    test_all_compile_stages()