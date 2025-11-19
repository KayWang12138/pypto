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
import pto

def init_tensors():
    dtype = pto.DT_FP32
    shape = (128, 128)
    a = pto.tensor(shape, dtype, "a")
    b = pto.tensor(shape, dtype, "b")
    c = pto.tensor(shape, dtype, "c")
    return a, b, c


def test_tensor_setitem_inside_loop():
    a, b, c = init_tensors()
    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)
        for k in pto.loop(10, name="LOOP", idx_name="k"):
            b[:] = pto.add(a, a)

            if pto.cond(k < 2):
                b[:] = pto.add(b, a)
            else:
                b[:] = pto.sub(b, a)

            if pto.cond(k < 5):
                b[:] = pto.mul(b, a)
            else:
                b[:] = pto.div(b, a)
            c[:] = pto.sub(b, a)

    assert isinstance(b, pto.tensor)


def test_tensor_assmble_slice():
    a, b, c = init_tensors()
    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        for k in pto.loop(10, name="LOOP", idx_name="k"):
            b[k*16:, 0:] = pto.add(a, a)

            if pto.cond(k < 2):
                b[k*16:, 0:] = pto.add(a, a)
            else:
                b[k*16:, 0:] = pto.sub(a, a)

            if pto.cond(k < 5):
                b[0:, :k*16] = pto.mul(a, a)
            else:
                b[0:, :k*16] = pto.div(a, a)
            c[:] = pto.sub(b, a)

    assert isinstance(c, pto.tensor)


def test_set_tensor_data():
    a = pto.tensor([32, 32], pto.DT_INT32, "a")
    b = pto.tensor([32, 32], pto.DT_INT32, "b")
    c = pto.tensor([32, 32], pto.DT_INT32, "c")
    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)
        sym_a = a[0, 0]
        sym_b = b[0, 0]
        assert isinstance(sym_a, pto.SymbolicScalar)
        assert isinstance(sym_b, pto.SymbolicScalar)
        c[0, 0] = sym_a + sym_b
        c[1, 1] = 1
