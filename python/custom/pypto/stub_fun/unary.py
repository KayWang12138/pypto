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

from typing import Union

from .. import context
from ..utils import Tensor, Var, Instruction


def exp(a: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('exp', [a], result))
    return result


def sqrt(a: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('sqrt', [a], result))
    return result


def log(a: Tensor, num_sqrt: int = 1) -> Tensor:
    if context.active_module is None:
        raise Exception()

    '''
    Hack to increase log precision.
    True value: log(50000) = 10.819778284410283
    Estimation, looping 20 times: 4.9577470605015135
    Estimation, num_sqrt = 1: 9.26238429713684
    Estimation, num_sqrt = 2: 10.816767148350074
    '''
    from .misc import zeros
    for _ in range(num_sqrt):  # hack to increase log precision
        a = sqrt(a)
    base = (a - 1) / (a + 1)
    current = Tensor(force_declare=True)
    current.eq(base)
    base = (base * base)
    result = zeros(a.shape)

    from ..flowcontrol import Loop
    with Loop(0, 20, 1) as i:
        result += current / (2 * i + 1)
        current *= base
    multiplier = 2
    multiplier *= 2 ** num_sqrt
    return multiplier * result


def abs(a: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('abs', [a], result))
    return result


def reciprocal(a: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('reciprocal', [a], result))
    return result


def floor(a: Tensor, datatype: Union[str, Var]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('floor', [a, datatype], result))
    return result


def round(a: Tensor, datatype: Union[str, Var]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('round', [a, datatype], result))
    return result


def logical_not(a: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('logical_not', [a], result))
    return result


def sin(a: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('sin', [a], result))
    return result


def cos(a: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('cos', [a], result))
    return result


def rotate_half(a: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('rotate_half', [a], result))
    return result
