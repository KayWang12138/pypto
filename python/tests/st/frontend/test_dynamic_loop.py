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
import pypto

@pypto.frontend.jit()
def dynamic_loop(
    a: pypto.Tensor((128, 128), pypto.DT_FP32),
) -> pypto.Tensor((128, 128), pypto.DT_FP32):
    pypto.set_vec_tile_shapes(64, 64)
    b = pypto.tensor((128, 128), pypto.DT_FP32)
    for k in range(10):
        b[:] = pypto.add(a, a)
        if k < 2:
            b[:] = pypto.add(b, a)
        else:
            b[:] = pypto.sub(b, a)

        if k < 5:
            b[:] = pypto.mul(b, a)
        else:
            b[:] = pypto.div(b, a)
        b[:] = pypto.sub(b, a)
    return b


print(pypto.dump())
