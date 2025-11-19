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
from collections.abc import Callable
from ..utils import Instruction, Var, CodeHelper


def run(i: Instruction, h: 'CodeHelper'):
    h(f'{i.name}.Compute();')


def assign_var(i: Instruction, h: CodeHelper):
    h(f'{i.v.name} = {i.src};')

def create_var(i: Instruction, h: CodeHelper):
    v: Var = i.v
    h(f'{v.dtype} {v.name} = {i.value};')


# flow control
def start_loop(i: Instruction, h: CodeHelper):
    h(f'for (int {i.name}={i.start}; {i.name}<{i.end}; {i.name}+={i.step}){{')
    h.ir()

def end_loop(i: Instruction, h: CodeHelper):
    h.il()
    h('}')

def start_if(i: Instruction, h: CodeHelper):
    h(f'if ({i.cond}){{')
    h.ir()

def start_elif(i: Instruction, h: CodeHelper):
    h(f'else if({i.cond}){{')
    h.ir()

def start_else(i: Instruction, h: CodeHelper):
    h('else {')
    h.ir()

def end_if(i: Instruction, h: CodeHelper):
    h.il()
    h('}')


INST_MAPPING: dict[str, Callable[[Instruction, 'CodeHelper'], None]] = {
    'RUN'                 : run,
    'ASSIGNVAR'           : assign_var,
    'CREATEVAR'           : create_var,
    # flowcontrol
    'STARTLOOP'           : start_loop,
    'ENDLOOP'             : end_loop,
    'STARTIF'             : start_if,
    'STARTELIF'           : start_elif,
    'STARTELSE'           : start_else,
    'ENDIF'               : end_if,
}

def parse_all(inst_list: list[Instruction], h: 'CodeHelper'):
    for i in inst_list:
        INST_MAPPING[i._inst](i, h)

