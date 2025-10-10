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
"""
"""
import pto
import sys
import os

def init_tensors():
    dtype = pto.data_type.DT_FP32
    shape = (128, 128)
    a = pto.tensor(shape, dtype, "a")
    b = pto.tensor(shape, dtype, "b")
    c = pto.tensor(shape, dtype, "c")
    return a, b, c


def test_dynamic_loop_nomacro():
    a, b, c = init_tensors()
    with pto.dyn_function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        with pto.loop_function(
            "LOOP",
            "k",
            pto.loop_range(10),
        ) as rlf:

            for k in rlf:
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
