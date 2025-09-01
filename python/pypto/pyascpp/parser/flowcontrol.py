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


def if_control(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, bool)):
        raise Exception()

    h(f'if ({get_var_str(inst.src[0])}) ' + '{')
    h.ir()


def elif_control(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, bool)):
        raise Exception()

    h(f'else if ({get_var_str(inst.src[0])}) ' + '{')
    h.ir()


def else_control(h: CodeHelper, inst: Instruction):
    h('else {')
    h.ir()


def close_bracket(h: CodeHelper, inst: Instruction):
    h.il()
    h('}')


def start_loop(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Var):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int)):
        raise Exception()
    if not isinstance(inst.src[2], (Var, int)):
        raise Exception()
    if not isinstance(inst.src[3], (Var, int)):
        raise Exception()

    ctr_var = inst.src[0]
    ctr = get_var_str(ctr_var)
    start = get_var_str(inst.src[1])
    end = get_var_str(inst.src[2])
    stride = get_var_str(inst.src[3])
    h(f"for ({ctr_var.dtype} {ctr} = {start}; {ctr} < {end}; {ctr} += {stride}) " + "{")
    h.ir()


def end_loop(h: CodeHelper, inst: Instruction):
    h.il()
    h('}')


def continue_loop(h: CodeHelper, inst: Instruction):
    h('continue;')


def break_loop(h: CodeHelper, inst: Instruction):
    h('break;')
