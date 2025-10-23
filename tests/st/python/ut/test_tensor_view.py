
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

def init_tensors():
    dtype = pto.DT_FP32
    shape = (128, 128)
    a = pto.tensor(shape, dtype, "a")
    b = pto.tensor(shape, dtype, "b")
    c = pto.tensor(shape, dtype, "c")
    return a, b, c

def test_tensor_view():
    a, b, c = init_tensors()
    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        for k in pto.loop(10):
            a_view = pto.view(a, (16, 16), (k*16, k*16))
            b_view = pto.view(b, (16, 16), (k*16, k*16))
            a_view = a[k*16:(k+1)*16, k*16:(k+1)*16]
            b_view = b[:16, :16]

            assert isinstance(a_view, pto.tensor)
            assert isinstance(b_view, pto.tensor)
            assert a_view.shape == [16, 16]
            assert b_view.shape == [16, 16]

def test_tensor_getitem():
    a, b, c = init_tensors()
    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        for k in pto.loop(10):
            a_view = a[k*16:(k+1)*16, k*16:(k+1)*16]
            b_view = b[:16, :16]

            assert isinstance(a_view, pto.tensor)
            assert isinstance(b_view, pto.tensor)
            assert a_view.shape == [16, 16]
            assert b_view.shape == [16, 16]
