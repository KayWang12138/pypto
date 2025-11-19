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
from typing import Union

from .. import context
from ..utils import Tensor, Var, Shape, Instruction, Vector


def view(a: Tensor, new_shape: Union[Shape, list[Union[int, Var]]], \
         new_offset: Union[Shape, list[Union[int, Var]]]):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('view', [a, new_shape, new_offset], result))
    return result


def dview(a: Tensor, new_shape: Union[Shape, list[Union[int, Var]]], \
          new_offset: Union[Shape, list[Union[int, Var]]]):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('dview', [a, new_shape, new_offset], result))
    return result


def dview_pad(a: Tensor, new_shape: Union[Shape, list[Union[int, Var]]], \
             new_offset: Union[Shape, list[Union[int, Var]]], block_size: Union[Shape, list[Union[int, Var]]]):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('dview_pad', [a, new_shape, new_offset, block_size], result))
    return result


def shape_size(a: Shape) -> Var:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_var(dtype='int')
    context.active_module.add_inst(Instruction('get_shape_size', [a], result))
    return result


def shape_emplace_back(a: Shape, val: Union[Var, int]):
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('shape_emplace_back', [a, val], None))


def reshape(a: Tensor, new_shape: Union[Shape, list[Union[int, Var]]]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('reshape', [a, new_shape], result))
    return result


def unsqueeze(old: Tensor, unsqueeze_dim_num: Union[int, Var]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('unsqueeze', [old, unsqueeze_dim_num], result))
    return result


def transpose(a: Tensor, new_shape: Union[Shape, list[Union[int, Var]]]):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('transpose', [a, new_shape], result))
    return result


def concat(a: Union[Vector, list[Tensor]], axis: Union[int, Var]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('concat', [a, axis], result))
    return result


def expand(a: Tensor, b: Union[list, Shape]):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('expand', [a, b], result))
    return result


def pad(a: Tensor, pad_shape: Union[Shape, list[Union[int, Var]]]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('pad', [a, pad_shape], result))
    return result
