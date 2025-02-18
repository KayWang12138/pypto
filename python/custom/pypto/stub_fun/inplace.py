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
from typing import Union

from .. import context
from ..utils import Tensor, Var, Instruction


def inplace_muls(a: Tensor, b: Union[Var, int, float]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('inplace_muls', [a, b], a))
    return a


def inplace_mul(a: Tensor, b: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('inplace_mul', [a, b], a))
    return a


def inplace_add(a: Tensor, b: Tensor) -> Tensor:  # a += b
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('inplace_add', [a, b], a))
    return a


def inplace_adds(a: Tensor, b: Union[Var, int, float]) -> Tensor:  # a += b
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('inplace_adds', [a, b], a))
    return a


def inplace_sub(a: Tensor, b: Tensor) -> Tensor:  # a -= b
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('inplace_sub', [a, b], a))
    return a


def inplace_subs(a: Tensor, b: Union[Var, int, float]) -> Tensor:  # a -= b
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('inplace_subs', [a, b], a))
    return a


def inplace_div(a: Tensor, b: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('inplace_div', [a, b], a))
    return a


def inplace_divs(a: Tensor, b: Union[Var, int, float]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('inplace_divs', [a, b], a))
    return a
