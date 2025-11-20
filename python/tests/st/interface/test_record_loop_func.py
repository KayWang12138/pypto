# -----------------------------------------------------------------------------------------------------------
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
import pto
import sys
import os

def init_tensors():
    dtype = pto.DT_FP32
    shape = (128, 128)
    a = pto.tensor(shape, dtype, "a")
    b = pto.tensor(shape, dtype, "b")
    c = pto.tensor(shape, dtype, "c")
    return a, b, c


def test_dynamic_loop_nomacro():
    a, b, c = init_tensors()
    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)
        for k in pto.loop(10, name="LOOP", idx_name="k"):
            b.move(pto.add(a, a))

            if pto.cond(k < 2):
                b.move(pto.add(b, a))
            else:
                b.move(pto.sub(b, a))

            if pto.cond(k < 5):
                b.move(pto.mul(b, a))
            else:
                b.move(pto.div(b, a))
            c.move(pto.sub(b, a))

    assert isinstance(b, pto.tensor)
