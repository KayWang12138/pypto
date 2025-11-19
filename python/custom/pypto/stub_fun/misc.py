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
from typing import Union, Optional, Sequence, Any

from .. import context
from ..utils import Tensor, Instruction, Var, TensorMap, AggregationVec, Vector, Shape
from ..utils import DATATYPE


def add_comment(*comments: Sequence[Any]):
    if context.active_module is None:
        raise Exception()
    comment = ' '.join([str(comm) for comm in comments])
    context.active_module.add_inst(Instruction('add_comment', [comment], None))


def data_type(a: Tensor) -> Var:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_var('DataType')
    context.active_module.add_inst(Instruction('get_data_type', [a], result))
    return result


def tensor_to_aggregation_vec(a: TensorMap) -> AggregationVec:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_aggregationvec()
    context.active_module.add_inst(Instruction('tensor_to_aggregation_vec', [a], result))
    return result


def is_not_null_ptr(a: Tensor) -> Var:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_var('bool')
    context.active_module.add_inst(Instruction('is_not_null_ptr', [a], result))
    return result


def continue_loop():
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('continue_loop', [], None))


def break_loop():
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('break_loop', [], None))


def call_module_function(func_name: str, return_type: Optional[str], args: Sequence) -> Union[Var, Tensor, None]:
    if context.active_module is None:
        raise Exception()
    if return_type is None:
        result = None
    elif return_type == "Tensor":
        result = context.active_module.create_tensor()
    elif return_type == "Var":
        result = context.active_module.create_var()
    else:
        raise NotImplementedError
    context.active_module.add_inst(Instruction('call_module_function', [func_name, return_type, args], result))
    return result


def zeros(shape: Union[list, Vector, Shape], dtype: Optional[str] = None) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    if dtype is None:
        dtype = DATATYPE.fp32
    context.active_module.add_inst(Instruction('vector_duplicate', ['0.0f', shape, dtype], result))
    return result


def ones(shape: Union[list, Vector, Shape], dtype: Optional[str] = None) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    if dtype is None:
        dtype = DATATYPE.fp32
    context.active_module.add_inst(Instruction('vector_duplicate', ['1.0f', shape, dtype], result))
    return result
