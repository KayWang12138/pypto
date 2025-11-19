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


def test_pto_loop_end_only():
    a, b, c = init_tensors()
    pto.reset()

    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        for k in pto.loop(10):
            b.move(pto.add(a, a))

            if pto.cond(k < 2):
                b.move(pto.add(b, a))
            else:
                b.move(pto.sub(b, a))

            if pto.cond(k < 5):
                b.move(pto.mul(b, a))
            else:
                b.move(pto.div(b, a))
            b.move(pto.sub(b, a))

    assert isinstance(b, pto.tensor)


def test_pto_loop_end_only_with_custom_name():
    a, b, c = init_tensors()
    pto.reset()

    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        for k in pto.loop(10, name="LOOP"):
            b.move(pto.add(a, a))

            if pto.cond(k < 5):
                b.move(pto.add(b, a))
            else:
                b.move(pto.sub(b, a))

            if pto.cond(k < 3):
                b.move(pto.mul(b, a))
            else:
                b.move(pto.div(b, a))
            b.move(pto.sub(b, a))

    assert isinstance(b, pto.tensor)


def test_pto_loop_start_end():
    a, b, c = init_tensors()
    pto.reset()

    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        for k in pto.loop(1, 10):
            b.move(pto.add(a, a))

            if pto.cond(k < 7):
                b.move(pto.add(b, a))
            else:
                b.move(pto.sub(b, a))

            if pto.cond(k < 8):
                b.move(pto.mul(b, a))
            else:
                b.move(pto.div(b, a))
            b.move(pto.sub(b, a))

    assert isinstance(b, pto.tensor)


def test_pto_loop_start_end_step():
    a, b, c = init_tensors()
    pto.reset()

    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        for k in pto.loop(1, 10, 2):
            b.move(pto.add(a, a))

            if pto.cond(k < 6):
                b.move(pto.add(b, a))
            else:
                b.move(pto.sub(b, a))

            if pto.cond(k < 2):
                b.move(pto.add(b, a))
            else:
                b.move(pto.div(b, a))
            b.move(pto.sub(b, a))

    assert isinstance(b, pto.tensor)


def test_pto_loop_start_end_step_and_name():
    a, b, c = init_tensors()
    pto.reset()

    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        for k in pto.loop(1, 10, 2, name="LOOP"):
            b.move(pto.add(a, a))

            if pto.cond(k < 3):
                b.move(pto.mul(b, a))
            else:
                b.move(pto.sub(b, a))

            if pto.cond(k < 8):
                b.move(pto.mul(b, a))
            else:
                b.move(pto.div(b, a))
            b.move(pto.sub(b, a))

    assert isinstance(b, pto.tensor)


def test_pto_loop_start_end_step_and_name():
    a, b, c = init_tensors()
    pto.reset()

    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        for k in pto.loop(1, 10, 2, name="LOOP"):
            b.move(pto.add(a, a))

            if pto.cond(k < 5):
                b.move(pto.mul(b, a))

    assert isinstance(b, pto.tensor)


def test_pto_loop_unroll_n_submit_before_loop():
    a, b, c = init_tensors()
    pto.reset()

    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        for k in pto.loop(
            1, 10, 2, name="LOOP", submit_before_loop=True
        ):

            if pto.cond(k < 5):
                b.move(pto.sub(b, a))
            if pto.cond(1):
                b.move(pto.add(b, a))
            if pto.cond(pto.is_loop_end(k)):
                b.move(pto.add(b, a))

            b.move(pto.sub(a, a))

    assert isinstance(b, pto.tensor)


def test_loop_issue52():
    pto.runtime._device_init()

    a = pto.tensor((128, 128), pto.DT_FP32, "a")
    b = pto.tensor((128, 128), pto.DT_FP32, "b")
    c = pto.tensor((128, 128), pto.DT_FP32, "c")

    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        for i in pto.loop(a.shape[0] // 16):
            for j in pto.loop(a.shape[1] // 16):
                view_a = a[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16]
                view_b = b[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16]
                assert isinstance(view_a, pto.tensor)
                assert isinstance(view_b, pto.tensor)
                c[i * 16:, j * 16:] = view_b + view_a

    pto.runtime._device_fini()


def test_if_true():
    A = pto.tensor((64, 64), pto.DT_FP32, "A")
    B = pto.tensor((64, 64), pto.DT_FP32, "B")

    pto.set_semantic_label("IF_TRUE")
    with pto.function("MAIN", [A], [A]):
        for _ in pto.loop(1):
            pto.set_vec_tile_shapes(16, 16)
            if pto.cond(True):
                B[:] = A + 2
            else:
                B[:] = A - 2


def test_loop_manual_unroll():
    pto.runtime._device_init()
    A = pto.tensor((-1, 64), pto.DT_FP32, "A")
    B = pto.tensor((-1, 64), pto.DT_FP32, "B")

    with pto.function("MAIN", [A], [B]):
        pto.set_vec_tile_shapes(64, 64)
        for b, k in pto.loop_unroll(A.shape[0] // 64, unroll_list=[1, 2, 4]):
            def inner(nb, nk):
                tile_a = A[nb * 64:(nb + nk) * 64, :]
                tile_a = tile_a + 2
                B[nb * 64:, :] = tile_a
            inner(b, k)

    pto.runtime._device_fini()


def test_loop_manual_unroll_const():
    A = pto.tensor((64, 64), pto.DT_FP32, "A")
    B = pto.tensor((64, 64), pto.DT_FP32, "B")

    k_list = []
    pto.runtime._device_init()
    with pto.function("MAIN", [A], [B]):
        pto.set_vec_tile_shapes(64, 64)
        for _, k in pto.loop_unroll(1, 8, unroll_list=[1, 2, 4]):
            k_list.append(k)

            def inner():
                B[:] = A + 1
            inner()
    assert k_list == [4, 2, 1]
    pto.runtime._device_fini()
