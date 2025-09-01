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


def imuls(a: Tensor, b: Union[Var, int, float]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('imuls', [a, b], a))
    return a


def imul(a: Tensor, b: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('itsrmul', [a, b], a))
    return a


def iadd(a: Tensor, b: Tensor) -> Tensor:  # a += b
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('itsradd', [a, b], a))
    return a


def iadds(a: Tensor, b: Union[Var, int, float]) -> Tensor:  # a += b
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('iadds', [a, b], a))
    return a


def isub(a: Tensor, b: Tensor) -> Tensor:  # a -= b
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('itsrsub', [a, b], a))
    return a


def isubs(a: Tensor, b: Union[Var, int, float]) -> Tensor:  # a -= b
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('isubs', [a, b], a))
    return a


def idiv(a: Tensor, b: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('itsrdiv', [a, b], a))
    return a


def idivs(a: Tensor, b: Union[Var, int, float]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('idivs', [a, b], a))
    return a
