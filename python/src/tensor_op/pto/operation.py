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
# pyright: reportReturnType=false
"""
"""
import typing
from typing import Optional, Union

from pto import pto_impl

from .element import Element
from .pto_utils import to_syms
from .symbolic_scalar import SymbolicScalar
from .tensor import Tensor


def _to_base(arg):
    if isinstance(arg, (Tensor, Element, SymbolicScalar)):
        return arg.base()
    elif isinstance(arg, (list, tuple)):
        return [_to_base(a) for a in arg]
    else:
        return arg


def op_wrapper(func):
    def wrapper(*args, **kwargs):
        args = _to_base(args)
        assert isinstance(args, (list, tuple))
        out = func(*args, **kwargs)
        if out is None:
            return None
        elif isinstance(out, pto_impl.Tensor):
            return Tensor.from_base(out)
        else:
            return out
    return wrapper


@op_wrapper
def add(left, right) -> Tensor:
    return pto_impl.add(left, right)


@op_wrapper
def sub(left, right) -> Tensor:
    return pto_impl.sub(left, right)


@op_wrapper
def mul(left, right) -> Tensor:
    return pto_impl.mul(left, right)


@op_wrapper
def div(left, right) -> Tensor:
    return pto_impl.div(left, right)


def view(a, shapes, *args) -> Tensor:
    shapes = [int(s) for s in shapes]
    if len(args) == 1:
        out = pto_impl.view(a.base(), shapes, to_syms(args[0]))
    elif len(args) == 2:
        out = pto_impl.view(a.base(), shapes, to_syms(
            args[0]), to_syms(args[1]))
    else:
        raise RuntimeError("Invalid arguments")
    return Tensor.from_base(out)


def assemble(*args) -> Optional[Tensor]:
    if len(args) == 2:
        tenors = typing.cast(list[pto_impl.Tensor], _to_base(args[0]))
        return Tensor.from_base(pto_impl.assemble(tenors, args[1]))
    elif len(args) == 3:
        pto_impl.assemble(args[0].base(), to_syms(args[1]), args[2].base())
        return None
    else:
        raise RuntimeError("Invalid arguments")


def min(a: 'SymbolicScalar | int', b: 'SymbolicScalar | int') -> 'SymbolicScalar':
    if isinstance(a, int):
        a = SymbolicScalar(a)
    return a.min(b)


def max(a: 'SymbolicScalar | int', b: 'SymbolicScalar | int') -> 'SymbolicScalar':
    if isinstance(a, int):
        a = SymbolicScalar(a)
    return a.max(b)


@op_wrapper
def exp(a) -> Tensor:
    return pto_impl.exp(a)


@op_wrapper
def transpose(a, axis) -> Tensor:
    return pto_impl.transpose(a, axis)


@op_wrapper
def abs(a) -> Tensor:
    return pto_impl.abs(a)


@op_wrapper
def reciprocal(a) -> Tensor:
    return pto_impl.reciprocal(a)


@op_wrapper
def rsqrt(a) -> Tensor:
    return pto_impl.rsqrt(a)


@op_wrapper
def sqrt(a) -> Tensor:
    return pto_impl.sqrt(a)


@op_wrapper
def neg(a) -> Tensor:
    return pto_impl.neg(a)


@op_wrapper
def log(a, base) -> Tensor:
    return pto_impl.log(a, base)


@op_wrapper
def cast(a, dtype, mode) -> Tensor:
    return pto_impl.cast(a, dtype, mode)


@op_wrapper
def add_s(a, b) -> Tensor:
    return pto_impl.add_s(a, b)


@op_wrapper
def sub_s(a, b) -> Tensor:
    return pto_impl.sub_s(a, b)


@op_wrapper
def mul_s(a, b) -> Tensor:
    return pto_impl.mul_s(a, b)


@op_wrapper
def div_s(a, b) -> Tensor:
    return pto_impl.div_s(a, b)


@op_wrapper
def row_max_single(a, axis=-1) -> Tensor:
    return pto_impl.row_max_single(a, axis)


@op_wrapper
def row_sum_single(a, axis=-1) -> Tensor:
    return pto_impl.row_sum_single(a, axis)


@op_wrapper
def matmul(dtype, a, b, **kwargs) -> Tensor:
    a_trans = kwargs.get("a_trans", False)
    b_trans = kwargs.get("b_trans", False)
    c_matrix_nz = kwargs.get("c_matrix_nz", False)
    return pto_impl.matmul(dtype, a, b, a_trans, b_trans, c_matrix_nz)


@op_wrapper
def batch_matmul(dtype, a, b, **kwargs) -> Tensor:
    a_trans = kwargs.get("a_trans", False)
    b_trans = kwargs.get("b_trans", False)
    c_matrix_nz = kwargs.get("c_matrix_nz", False)
    return pto_impl.batch_matmul(dtype, a, b, a_trans, b_trans, c_matrix_nz)
