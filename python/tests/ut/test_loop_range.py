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
import logging
import pto

def test_init_loop_range_end():
    expected_val = 123
    print("aaa")
    scalar = pto.symbolic_scalar(expected_val)
    print("bbbb")
    loop = pto.controller._loop_range(scalar)

    print("11111")
    loop_end = loop.end()
    print("22222")
    print(type(loop_end.base()))

    assert loop_end.is_concrete() == True
    print("2222")
    assert loop_end.concrete() == expected_val
    print("4444")


def test_init_loop_range_begin_end():
    expected_begin, expected_end = 11, 22
    sym_begin = pto.symbolic_scalar(expected_begin)
    sym_end = pto.symbolic_scalar(expected_end)
    loop = pto.controller._loop_range(sym_begin, sym_end)

    loop_begin = loop.begin()
    loop_end = loop.end()

    assert loop_begin.is_concrete() == True
    assert loop_begin.concrete() == expected_begin

    assert loop_end.is_concrete() == True
    assert loop_end.concrete() == expected_end


def test_init_loop_range_begin_end_step():
    expected_begin, expected_end = 11, 22
    expected_step = 2
    sym_begin = pto.symbolic_scalar(expected_begin)
    sym_end = pto.symbolic_scalar(expected_end)
    sym_step = pto.symbolic_scalar(expected_step)
    loop = pto.controller._loop_range(sym_begin, sym_end, sym_step)

    loop_begin = loop.begin()
    loop_end = loop.end()
    loop_step = loop.step()

    assert loop_begin.is_concrete() == True
    assert loop_begin.concrete() == expected_begin

    assert loop_end.is_concrete() == True
    assert loop_end.concrete() == expected_end

    assert loop_step.is_concrete() == True
    assert loop_step.concrete() == expected_step


def test_loop_dump():
    expected_begin, expected_end = 11, 22
    expected_step = 2
    sym_begin = pto.symbolic_scalar(expected_begin)
    sym_end = pto.symbolic_scalar(expected_end)
    sym_step = pto.symbolic_scalar(expected_step)
    loop = pto.controller._loop_range(sym_begin, sym_end, sym_step)

    expected = f"LoopRange({expected_begin}, {expected_end}, {expected_step})"
    actual = str(loop)

    assert actual == expected


def test_init_loop_range_end_implicit_conversion_from_int():
    expected_val = 123
    loop = pto.controller._loop_range(expected_val)

    loop_end = loop.end()

    assert loop_end.is_concrete() == True
    assert loop_end.concrete() == expected_val
