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
from typing import Sequence
import numpy as np
from ..utils import Tensor, Instruction


def matmul(i: Instruction):
    src0 = i.src[1].data
    src1 = i.src[2].data
    if i.src[3]:
        src0 = src0.T
    if i.src[4]:
        src1 = src1.T
    i.dst.data = src0.astype(np.float32) @ src1.astype(np.float32)


def muls(i: Instruction):
    i.dst.data = i.src[0].data * i.src[1].data


def adds(i: Instruction):
    i.dst.data = i.src[0].data + i.src[1].data


def exp(i: Instruction):
    i.dst.data = np.exp(i.src[0].data)


def tsrdiv(i: Instruction):
    i.dst.data = i.src[0].data / i.src[1].data


def tsrmul(i: Instruction):
    i.dst.data = i.src[0].data * i.src[1].data


def call_func(i: Instruction):
    inputs = i.src[1:]
    from .. import context
    ret = context.sim.run(i.src[0], inputs)
    if isinstance(ret, Tensor):
        i.dst.data = ret.data
    elif isinstance(ret, Sequence):
        for r, o in zip(ret, i.dst):
            o.data = r.data


INST_EXEC_MAPPING = {
    'matmul': matmul,
    'muls': muls,
    'adds': adds,
    'exp': exp,
    'tsrdiv': tsrdiv,
    'tsrmul': tsrmul,
    'call_func': call_func,
}


def run_all(inst_list: list[Instruction]):
    for i in inst_list:
        if i.inst in INST_EXEC_MAPPING:
            INST_EXEC_MAPPING[i.inst](i)