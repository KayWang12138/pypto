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

from .util import get_var_str, get_scalar_dtype
from ..utils import CodeHelper, Instruction, Tensor, Var


def check_inst_src_type(inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int, float)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()


def muls(h: CodeHelper, inst: Instruction):
    check_inst_src_type(inst)

    if isinstance(inst.src[1], float):
        const_str = "(float) " + get_var_str(inst.src[1])
    else:
        const_str = get_var_str(inst.src[1])
    const_str = get_scalar_dtype(inst.src[1]) + ', ' + const_str
    h(f'auto tsr{inst.dst.idx} = MulS(tsr{inst.src[0].idx}, Element({const_str}));')


def adds(h: CodeHelper, inst: Instruction):
    check_inst_src_type(inst)

    if isinstance(inst.src[1], float):
        const_str = "(float) " + get_var_str(inst.src[1])
    else:
        const_str = get_var_str(inst.src[1])
    const_str = get_scalar_dtype(inst.src[1]) + ', ' + const_str
    h(f'auto tsr{inst.dst.idx} = AddS(tsr{inst.src[0].idx}, Element({const_str}));')


def subs(h: CodeHelper, inst: Instruction):
    check_inst_src_type(inst)

    if isinstance(inst.src[1], float):
        const_str = "(float) " + get_var_str(inst.src[1])
    else:
        const_str = get_var_str(inst.src[1])
    const_str = get_scalar_dtype(inst.src[1]) + ', ' + const_str
    h(f'auto tsr{inst.dst.idx} = SubS(tsr{inst.src[0].idx}, Element({const_str}));')


def divs(h: CodeHelper, inst: Instruction):
    check_inst_src_type(inst)

    if isinstance(inst.src[1], float):
        const_str = "(float) " + get_var_str(inst.src[1])
    else:
        const_str = get_var_str(inst.src[1])
    const_str = get_scalar_dtype(inst.src[1]) + ', ' + const_str
    h(f'auto tsr{inst.dst.idx} = DivS(tsr{inst.src[0].idx}, Element({const_str}));')
