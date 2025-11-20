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
from ..utils import CodeHelper, Instruction, Tensor


def add(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = Add(tsr{inst.src[0].idx}, tsr{inst.src[1].idx});')


def sub(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = Sub(tsr{inst.src[0].idx}, tsr{inst.src[1].idx});')


def mul(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = Mul(tsr{inst.src[0].idx}, tsr{inst.src[1].idx});')


def div(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = Div(tsr{inst.src[0].idx}, tsr{inst.src[1].idx});')


def maximum(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = Maximum(tsr{inst.src[0].idx}, tsr{inst.src[1].idx});')
