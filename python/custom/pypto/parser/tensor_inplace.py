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
from .util import get_var_str, get_scalar_dtype
from ..utils import CodeHelper, Instruction, Tensor, Var


def inplace_add(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'tsr{inst.dst.idx} = Add(tsr{inst.src[0].idx}, tsr{inst.src[1].idx});')


def inplace_sub(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'tsr{inst.dst.idx} = Sub(tsr{inst.src[0].idx}, tsr{inst.src[1].idx});')


def inplace_mul(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'tsr{inst.dst.idx} = Mul(tsr{inst.src[0].idx}, tsr{inst.src[1].idx});')


def inplace_div(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'tsr{inst.dst.idx} = Div(tsr{inst.src[0].idx}, tsr{inst.src[1].idx});')


def retrieve_const_str(inst: Instruction) -> str:
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    if isinstance(inst.src[1], float):
        const_str = "(float) " + get_var_str(inst.src[1])
    else:
        const_str = get_var_str(inst.src[1])
    return const_str


def inplace_muls(h: CodeHelper, inst: Instruction):
    const_str = retrieve_const_str(inst)
    const_str = get_scalar_dtype(inst.src[1]) + ', ' + const_str
    h(f'tsr{inst.dst.idx} = MulS(tsr{inst.src[0].idx}, Element({const_str}));')


def inplace_subs(h: CodeHelper, inst: Instruction):
    const_str = retrieve_const_str(inst)
    const_str = get_scalar_dtype(inst.src[1]) + ', ' + const_str
    h(f'tsr{inst.dst.idx} = Sub(tsr{inst.src[0].idx}, Element({const_str}));')


def inplace_divs(h: CodeHelper, inst: Instruction):
    const_str = retrieve_const_str(inst)
    const_str = get_scalar_dtype(inst.src[1]) + ', ' + const_str
    h(f'tsr{inst.dst.idx} = DivS(tsr{inst.src[0].idx}, Element({const_str}));')


def inplace_adds(h: CodeHelper, inst: Instruction):
    const_str = retrieve_const_str(inst)
    const_str = get_scalar_dtype(inst.src[1]) + ', ' + const_str
    h(f'tsr{inst.dst.idx} = AddS(tsr{inst.src[0].idx}, Element({const_str}));')
