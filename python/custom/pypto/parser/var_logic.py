#!/usr/bin/env python3
# coding: utf-8
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
from .util import get_var_str
from ..utils import CodeHelper, Instruction, Var


def check_inst_src_dtype1(inst: Instruction):
    if not isinstance(inst.src[0], Var):
        raise Exception()
    if not isinstance(inst.src[1], Var):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()


def check_inst_src_dtype2(inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float, str)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()


def check_inst_src_dtype3(inst: Instruction):
    if not isinstance(inst.src[0], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()


def var_or(h: CodeHelper, inst: Instruction):
    check_inst_src_dtype1(inst)

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'auto v{inst.dst.idx} = {s1} || {s2};')


def var_and(h: CodeHelper, inst: Instruction):
    check_inst_src_dtype1(inst)

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'auto v{inst.dst.idx} = {s1} && {s2};')


def var_inv(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Var):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    s1 = get_var_str(inst.src[0])

    h(f'auto v{inst.dst.idx} = !{s1};')


def var_compare_eq(h: CodeHelper, inst: Instruction):
    check_inst_src_dtype2(inst)

    s1 = get_var_str(inst.src[0])
    if isinstance(inst.src[1], str):
        s2 = "\"" + inst.src[1] + "\""
    else:
        s2 = get_var_str(inst.src[1])

    h(f'bool v{inst.dst.idx} = ({s1} == {s2});')


def var_compare_neq(h: CodeHelper, inst: Instruction):
    check_inst_src_dtype2(inst)

    s1 = get_var_str(inst.src[0])
    if isinstance(inst.src[1], str):
        s2 = "\"" + inst.src[1] + "\""
    else:
        s2 = get_var_str(inst.src[1])

    h(f'bool v{inst.dst.idx} = ({s1} != {s2});')


def var_compare_gt(h: CodeHelper, inst: Instruction):
    check_inst_src_dtype3(inst)

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'bool v{inst.dst.idx} = ({s1} > {s2});')


def var_compare_ge(h: CodeHelper, inst: Instruction):
    check_inst_src_dtype3(inst)

    s1 = get_var_str(inst.src[0])
    s2 = get_var_str(inst.src[1])

    h(f'bool v{inst.dst.idx} = ({s1} >= {s2});')
