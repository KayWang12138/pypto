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

from .util import get_var_str
from ..utils import CodeHelper, Instruction, Var


def var_min(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'auto v{inst.dst.idx} = std::min({s1}, {s2});')


def var_max(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'auto v{inst.dst.idx} = std::max({s1}, {s2});')


def var_mul(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'auto v{inst.dst.idx} = {s1} * {s2};')


def var_inplace_mul(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'v{inst.dst.idx} = {s1} * {s2};')


def var_add(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'auto v{inst.dst.idx} = {s1} + {s2};')


def var_inplace_add(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'v{inst.dst.idx} = {s1} + {s2};')


def var_sub(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'auto v{inst.dst.idx} = {s1} - {s2};')


def var_inplace_sub(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'v{inst.dst.idx} = {s1} - {s2};')


def var_true_div(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'auto v{inst.dst.idx} = (float) ({s1}) / {s2};')


def var_inplace_true_div(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'v{inst.dst.idx} = (float) ({s1}) / {s2};')


def var_floor_div(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'auto v{inst.dst.idx} = (int) ({s1} / {s2});')


def var_inplace_floor_div(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'v{inst.dst.idx} = (int) ({s1} / {s2});')


def var_mod(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'auto v{inst.dst.idx} = {s1} % {s2};')


def var_inplace_mod(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'v{inst.dst.idx} = {s1} % {s2};')


def var_pow(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'auto v{inst.dst.idx} = (float) Pow({s1}, {s2});')


def var_inplace_pow(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'v{inst.dst.idx} = (float) Pow({s1}, {s2});')
