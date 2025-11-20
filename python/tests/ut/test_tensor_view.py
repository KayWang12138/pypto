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
            a_view = a[k*16:(k+1)*16, k*16:(k+1)*16]
            b_view = b[:16, :16]

            assert isinstance(a_view, pto.tensor)
            assert isinstance(b_view, pto.tensor)
            assert a_view.shape == [16, 16]
            assert b_view.shape == [16, 16]


def test_tensor_get_tensor_data():
    a = pto.tensor((128, 128), pto.DT_INT32, "a")
    with pto.function("MAIN", [a], []):
        pto.set_vec_tile_shapes(16, 16)
        t = a[0, 0]


def test_slice_neg_index():
    """Test negative index"""
    x_shape = [4, 8]
    dtype = pto.DT_FP32
    x = pto.tensor(x_shape, dtype)

    with pto.function("SLICE_NEG_INDEX", static=True):
        pto.set_vec_tile_shapes(4, 4)
        res = x[-3:-1, -2:-1]
        assert res.shape == [2, 1]


def test_slice_int_index():
    """Test mix use of slice and int"""
    x_shape = [4, 8, 8, 8, 8]
    dtype = pto.DT_FP32
    x = pto.tensor(x_shape, dtype)

    with pto.function("SLICE_INT_INDEX", static=True):
        pto.set_vec_tile_shapes(4, 4, 4, 4, 4)
        res = x[-2, -3:8, :, 1:4, 2]
        assert res.shape == [3, 8, 3]


def test_slice_ellipsis_index():
    """Test mix use of ellipsis, slice and int"""
    x_shape = [4, 8, 8, 8]
    dtype = pto.DT_FP32
    x = pto.tensor(x_shape, dtype)

    with pto.function("SLICE_INT_ELLIPSIS_INDEX", static=True):
        pto.set_vec_tile_shapes(4, 4, 4, 4)
        res1 = x[..., 2]
        res2 = x[1:2, :, ..., 3:5]
        res3 = x[2, 3, ...]
        res4 = x[...] + 0.0
        assert res1.shape == [4, 8, 8]
        assert res2.shape == [1, 8, 8, 2]
        assert res3.shape == [8, 8]
        assert res4.shape == [4, 8, 8, 8]