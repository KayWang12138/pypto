# -----------------------------------------------------------------------------------------------------------
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
from ..utils import CodeHelper, Instruction, Tensor, Var


def abs(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = Abs(tsr{inst.src[0].idx});')


def exp(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = Exp(tsr{inst.src[0].idx});')


def log(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    raise NotImplementedError("AscendC++ does not have a log() function as of time of implementation")


def sqrt(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = Sqrt(tsr{inst.src[0].idx});')


def reciprocal(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = Reciprocal(tsr{inst.src[0].idx});')


def round(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (str, Var)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    datatype_str = get_var_str(inst.src[1])
    h(f'auto tsr{inst.dst.idx} = Round(tsr{inst.src[0].idx}, {datatype_str});')


def logical_not(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = LogicalNot(tsr{inst.src[0].idx});')


def sin(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = Sin(tsr{inst.src[0].idx});')


def cos(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = Cos(tsr{inst.src[0].idx});')


def rotate_half(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = RotateHalf(tsr{inst.src[0].idx});')
