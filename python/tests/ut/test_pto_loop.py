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


def test_pto_loop_start_end_step_and_name_unroll():
    a, b, c = init_tensors()
    pto.reset()

    with pto.function("MAIN", [a, b], [c]):
        pto.set_vec_tile_shapes(16, 16)

        for k in pto.loop(1, 10, 2, name="LOOP", unroll_list={1}):
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
            1, 10, 2, name="LOOP", unroll_list=set(), submit_before_loop=True
        ):

            if pto.cond(k < 5):
                b.move(pto.sub(b, a))
            if pto.cond(1):
                b.move(pto.add(b, a))
            if pto.cond(pto.is_loop_end(k, 0)):
                b.move(pto.add(b, a))

            b.move(pto.sub(a, a))

    assert isinstance(b, pto.tensor)
