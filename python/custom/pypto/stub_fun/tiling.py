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
from typing import Union, Sequence

from .. import context
from ..utils import Var, Instruction, Vector, Shape, CustStruct


def update_record_tile_op(a: Union[Var, int]):
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('update_record_tile_op', [a], None))


def set_vec_tile_shapes(a: Union[Sequence[Union[Var, int]], Vector, Shape]):
    if context.active_module is None:
        raise Exception()
    if not isinstance(a, (Sequence, Vector, Shape)):
        raise Exception()
    context.active_module.add_inst(Instruction('set_vec_tile_shapes', [a], None))


def set_tile_shape(a: Union[Var, int], b: Union[Var, int]):
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('set_tile_shape', [a, b], None))


def set_cube_tile_shapes(a: Sequence[Union[Var, int]], b: Sequence[Union[Var, int]], c: Sequence[Union[Var, int]]):
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('set_cube_tile_shapes', [a, b, c], None))


def set_c1_cube_config(a: CustStruct):
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('set_c1_cube_config', [a], None))


def set_c2_cube_config(a: CustStruct):
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('set_c2_cube_config', [a], None))


def set_matrix_size(a: Sequence[Union[Var, int]]):
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('set_matrix_size', [a], None))
