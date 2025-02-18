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

from .unary import abs
from .. import context
from ..utils import Tensor, Var


def lt(a: Tensor, b: Union[Tensor, Var, int, float]):
    if context.active_module is None:
        raise Exception()
    diff = b - a
    eps = 1e-8
    diff -= eps
    diff /= tabs(diff)
    mask = (diff + 1) / 2  # now mask is either 0 or 1
    return mask


def le(a: Tensor, b: Union[Tensor, Var, int, float]):
    if context.active_module is None:
        raise Exception()
    diff = b - a
    eps = 1e-8
    diff += eps
    diff /= abs(diff)
    mask = (diff + 1) / 2  # now mask is either 0 or 1
    return mask


def gt(a: Tensor, b: Union[Tensor, Var, int, float]):
    if context.active_module is None:
        raise Exception()
    diff = a - b
    eps = 1e-8
    diff -= eps
    diff /= abs(diff)
    mask = (diff + 1) / 2  # now mask is either 0 or 1
    return mask


def ge(a: Tensor, b: Union[Tensor, Var, int, float]):
    if context.active_module is None:
        raise Exception()
    diff = a - b
    eps = 1e-8
    diff += eps
    diff /= abs(diff)
    mask = (diff + 1) / 2  # now mask is either 0 or 1
    return mask
