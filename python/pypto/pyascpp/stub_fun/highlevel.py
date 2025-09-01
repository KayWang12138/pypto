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

from typing import Union, Optional

from .. import context
from ..utils import Tensor, Var, Instruction, AggregationVec, Tuple, Vector, Shape, get_obj_dtype


def sigmoid(a: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('sigmoid', [a], result))
    return result


def cast(a: Tensor, dtype: Union[str, Var], cast_mode: Optional[str] = None) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('cast', [a, dtype, cast_mode], result))
    return result


def rms_norm(a: Tensor, b: Optional[Tensor] = None, c: Optional[Union[Var, int, float]] = None) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    if b is None and c is None:
        context.active_module.add_inst(Instruction('rms_norm', [a], result))
    else:
        context.active_module.add_inst(Instruction('rms_norm', [a, b, c], result))
    return result


def matmul(a: Tensor, b: Tensor, dtype: Union[Var, str], c: Optional[Tensor] = None):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    if c is None:
        context.active_module.add_inst(Instruction('matmul', [dtype, a, b, a.is_transpose, b.is_transpose], result))
    else:
        context.active_module.add_inst(
            Instruction('matmul', [dtype, a, b, a.is_transpose, b.is_transpose, c], result))
    return result


def batchmatmul(dtype: Union[Var, str], a: Tensor, b: Tensor):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('batchmatmul', [dtype, a, b, a.is_transpose, b.is_transpose], result))
    return result


def topk(operand: Tensor, k: Union[Var, int], axis: Union[Var, int], is_largest: bool = True) -> Tuple:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tuple(["Tensor", "Tensor"])
    context.active_module.add_inst(Instruction('topk', [operand, k, axis, is_largest], result))
    return result


def rowmaxsingle(a: Tensor, axis: Optional[Union[int, Var]] = None) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    if axis is not None:
        context.active_module.add_inst(Instruction('rowmaxsingle', [a, axis], result))
    else:
        context.active_module.add_inst(Instruction('rowmaxsingle', [a], result))
    return result


def rowsumsingle(a: Tensor, axis: Optional[Union[int, Var]] = None) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    if axis is not None:
        context.active_module.add_inst(Instruction('rowsumsingle', [a, axis], result))
    else:
        context.active_module.add_inst(Instruction('rowsumsingle', [a], result))
    return result


def tsum(a: Tensor, axis: Union[int, Var]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('sum', [a, axis], result))
    return result


def tmax(a: Tensor, axis: Union[int, Var]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('max', [a, axis], result))
    return result


def scatterupdate(a: Tensor, b: Tensor, c: Tensor, axis: Union[int, Var]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('scatterupdate', [a, b, c, axis], result))
    return result


def tensorindex(a: Tensor, b: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('tensorindex', [a, b], result))
    return result


def softmax(a: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('softmax', [a], result))
    return result


def softmaxnew(a: Tensor) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('softmaxnew', [a], result))
    return result


def assemble(a: AggregationVec) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('assemble', [a], result))
    return result


def argsort(a: Tensor, axis: Optional[Union[int, Var]] = None, is_largest: Optional[bool] = None):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    if axis is None:
        context.active_module.add_inst(Instruction('argsort', [a], result))
    else:
        if is_largest is None:
            context.active_module.add_inst(Instruction('argsort', [a, axis], result))
        else:
            context.active_module.add_inst(Instruction('argsort', [a, axis, is_largest], result))
    return result


def scatterelement(a: Tensor, b: Tensor, s: Union[int, float, Var], axis: Union[int, Var]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('scatterelement', [a, b, s, axis], result))
    return result


def gather(a: Tensor, b: Tensor, axis: Union[int, Var]) -> Tensor:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('gather', [a, b, axis], result))
    return result


def gatherelement(a: Tensor, b: Tensor, axis: Union[int, Var]):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('gatherelement', [a, b, axis], result))
    return result


def indexput(a: Tensor, b: list, c: Tensor):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('indexput', [a, b, c], result))
    return result


def reduce(a: Vector, b: str):
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tensor()
    context.active_module.add_inst(Instruction('reduce', [a, b], result))
    return result


def quant(a: Tensor, b: Optional[Union['Var', bool]] = None, c: Optional[Union['Var', bool]] = None,
          d: Optional[Tensor] = None) -> Tuple:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_tuple(['Tensor', 'Tensor'])
    if b is not None and c is not None and d is not None:
        context.active_module.add_inst(Instruction('quant', [a, b, c, d], result))
    else:
        context.active_module.add_inst(Instruction('quant', [a], result))
    return result


def dassemble(a: Tensor, b: Union[Shape, list[Union[int, Var]], Vector], c: Tensor):
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('dassemble', [a, b, c], None))


def get_mask_row(a: list[tuple[Union[int, float], Union[int, Var]]]):
    from .misc import zeros, ones
    from .shape import concat, unsqueeze
    if context.active_module is None:
        raise Exception()
    lst = []
    for item in a:
        if item[0] == 0:
            lst.append(zeros([item[1]]))
        elif item[0] == 1:
            lst.append(ones([item[1]]))
    row = concat(lst, axis=0)
    return unsqueeze(row, 0)


def less_than0_inv(a: Union[Var, int], b: Union[Var, int]):
    if context.active_module is None:
        raise Exception()
    if isinstance(a, int):
        if a >= 0:
            return a
        if isinstance(b, int):
            return a + b
    res = context.active_module.create_var(dtype=get_obj_dtype(a))
    context.active_module.add_inst(Instruction('less_than0_inv', [a, b], res))
    return res
