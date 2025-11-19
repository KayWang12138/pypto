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
from typing import Sequence

from .util import get_var_str, get_vartype_str, get_curlybrace_list
from ..utils import CodeHelper, Instruction, CustStruct, Tensor, Vector, Var, Shape
from .misc import call_module_function


def update_record_tile_op(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int)):
        raise Exception()

    s0 = get_var_str(inst.src[0])

    h(f'Program::GetInstance().GetConfig().UpdateRecordTileOperation({s0});')


def set_vec_tile_shapes(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Vector, Shape, list)):
        raise Exception()

    if isinstance(inst.src[0], list):
        s0 = get_curlybrace_list(inst.src[0])
    else:
        s0 = get_var_str(inst.src[0])

    h(f'TileShape::Current().SetVecTile({s0});')


def set_tile_shape(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int)):
        raise Exception()

    s0 = get_var_str(inst.src[0])
    s1 = get_var_str(inst.src[1])

    h(f'Program::GetInstance().GetConfig().SetTileShape({s0}, {s1});')


def set_cube_tile_shapes(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Shape, list)):
        raise Exception()
    if not isinstance(inst.src[1], (Shape, list)):
        raise Exception()
    if not isinstance(inst.src[2], (Shape, list)):
        raise Exception()

    if isinstance(inst.src[0], list):
        s0 = get_curlybrace_list(inst.src[0])
    else:
        s0 = get_var_str(inst.src[0])

    if isinstance(inst.src[1], list):
        s1 = get_curlybrace_list(inst.src[1])
    else:
        s1 = get_var_str(inst.src[1])

    if isinstance(inst.src[2], list):
        s2 = get_curlybrace_list(inst.src[2])
    else:
        s2 = get_var_str(inst.src[2])

    h(f'TileShape::Current().SetCubeTile({s0}, {s1}, {s2});')


def set_c1_cube_config(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], CustStruct):
        raise Exception()

    h(f'SetC1CubeConfig(stru{inst.src[0].idx});')


def set_c2_cube_config(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], CustStruct):
        raise Exception()

    h(f'SetC2CubeConfig(stru{inst.src[0].idx});')


def set_matrix_size(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Shape, list)):
        raise Exception()

    if isinstance(inst.src[0], list):
        s0 = get_curlybrace_list(inst.src[0])
    else:
        s0 = get_var_str(inst.src[0])

    h(f'TileShape::Current().SetMatrixSize({s0});')
