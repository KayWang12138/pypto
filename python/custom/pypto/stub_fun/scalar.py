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
from ..utils import Var, Instruction, get_obj_dtype


def var_min(a: Union[Var, int, float], b: Union[Var, int, float]):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_var(dtype=get_obj_dtype(a))
    context.active_module.add_inst(Instruction('var_min', [a, b], result))
    return result


def var_max(a: Union[Var, int, float], b: Union[Var, int, float]):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_var(dtype=get_obj_dtype(a))
    context.active_module.add_inst(Instruction('var_max', [a, b], result))
    return result


def var_ceil_div(a: Union[Var, int], b: Union[Var, int]) -> Union[Var, int]:
    return (a + (b - 1)) // b
