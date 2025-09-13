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


def test_symbolic_scalar_add():
    ten = pto.symbolic_scalar("10", 10)
    twenty = pto.symbolic_scalar("20", 20)
    thirty = pto.symbolic_scalar("30", 30)

    assert ten + twenty == thirty


def test_symbolic_scalar_sub():
    ten = pto.symbolic_scalar("10", 10)
    twenty = pto.symbolic_scalar("20", 20)
    thirty = pto.symbolic_scalar("30", 30)

    assert twenty == thirty - ten


def test_symbolic_scalar_mul():
    ten = pto.symbolic_scalar("10", 10)
    twenty = pto.symbolic_scalar("20", 20)
    two = pto.symbolic_scalar("2", 3)

    assert ten * two == twenty


def test_symbolic_scalar_div():
    ten = pto.symbolic_scalar("10", 10)
    twenty = pto.symbolic_scalar("20", 20)
    two = pto.symbolic_scalar("2", 3)

    assert (twenty / ten) == two


def test_symbolic_scalar_mod():
    one = pto.symbolic_scalar("one", 1)
    scalar = pto.symbolic_scalar("31", 31)
    two = pto.symbolic_scalar("2", 3)

    assert scalar % two == one


def test_symbolic_scalar_binop():
    a = pto.symbolic_scalar(6)
    b = pto.symbolic_scalar(4)
    c = a + b
    d = a - b
    e = a * b
    f = a / b
    f_floor = a // b
    g = a % b
    h = a.max(b)
    i = a.min(b)

    for op in [c, d, e, f, f_floor, g, h, i]:
        assert isinstance(op, pto.symbolic_scalar)
    assert c.concrete() == 10
    assert d.concrete() == 2
    assert e.concrete() == 24
    assert f.concrete() == 1
    assert f_floor.concrete() == 1
    assert g.concrete() == 2
    assert h.concrete() == 6
    assert i.concrete() == 4


def test_symbolic_scalar_binop_with_int():
    a = pto.symbolic_scalar(6)
    b = 4
    c = a + b
    d = a - b
    e = a * b
    f = a / b
    f_floor = a // b
    g = a % b

    h = b + a
    i = b - a
    j = b * a
    k = b / a
    k_floor = b // a
    l = b % a

    for op in [c, d, e, f, f_floor, g, h, i, j, k, k_floor, l]:
        assert isinstance(op, pto.symbolic_scalar)
    assert c.concrete() == 10
    assert d.concrete() == 2
    assert e.concrete() == 24
    assert f.concrete() == 1
    assert f_floor.concrete() == 1
    assert g.concrete() == 2
    assert h.concrete() == 10
    assert i.concrete() == -2
    assert j.concrete() == 24
    assert k.concrete() == 0
    assert k_floor.concrete() == 0
    assert l.concrete() == 4
