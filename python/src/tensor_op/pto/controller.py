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
# pyright: reportOptionalMemberAccess=false
"""
"""

import inspect
import logging
from contextlib import contextmanager
from typing import List, Optional, Set, Tuple, Union

import pto
from pto import pto_impl

from .pto_utils import to_sym
from .symbolic_scalar import SymbolicScalar
from .tensor import Tensor

logging.basicConfig(level=logging.DEBUG)


def set_vec_tile_shapes(*shapes: int):
    pto_impl.SetVecTile(*shapes)


def get_vec_tile_shapes() -> List[int]:
    return pto_impl.GetVecTile()


def set_cube_tile_shapes(m: List[int], k: List[int], n: List[int]):
    pto_impl.SetCubeTile(m, k, n)


def set_build_static(static: bool):
    pto_impl.SetBuildStatic(static)


def set_semantic_label(label: str):
    pto_impl.SetSemanticLabel(label)


def bytes_of(dtype: pto.DataType) -> int:
    return pto_impl.BytesOf(dtype)

def set_host_config(key, val):
    pto_impl.SetOption(f"host.{key}", val)


def set_codegen_config(key, val):
    pto_impl.SetCodeGenOption(key, val)


def set_pass_config(key, val):
    pto_impl.SetOption(f"pass.{key}", val)


def begin_function(
    name: str,
    graph_type: pto_impl.GraphType,
    func_type: pto_impl.FunctionType,
    *args
) -> pto_impl.RecordFunc:
    args = [arg.base() for arg in args]
    return pto_impl.BeginFunction(name, graph_type, func_type, *args)


def end_function(name: str, generate_call: bool = True):
    pto_impl.EndFunction(name, generate_call)


class LoopRange:
    def __init__(self, start, stop=None, step: Union[int, SymbolicScalar] = 1):
        if stop is None:
            start, stop = 0, start
        self._base = pto_impl.LoopRange(
            to_sym(start), to_sym(stop), to_sym(step))

    def begin(self) -> SymbolicScalar:
        return SymbolicScalar.from_base(self._base.Begin())

    def end(self) -> SymbolicScalar:
        return SymbolicScalar.from_base(self._base.End())

    def step(self) -> SymbolicScalar:
        return SymbolicScalar.from_base(self._base.Step())

    def __str__(self) -> str:
        return self._base.Dump()

    def __repr__(self) -> str:
        return f"LoopRange({self._base.Dump()})"

    def base(self) -> pto_impl.LoopRange:
        return self._base


loop_range = LoopRange


def is_loop_begin(scalar: SymbolicScalar, begin: Union[int, SymbolicScalar]):
    nbegin = to_sym(begin)
    return pto_impl.IsLoopBegin(scalar.base(), nbegin)


def is_loop_end(scalar: SymbolicScalar, end: Union[int, SymbolicScalar]):
    nend = to_sym(end)
    return pto_impl.IsLoopEnd(scalar.base(), nend)


@contextmanager
def function(
    name: str,
    in_tensors: List[Tensor],
    out_tensors: List[Tensor],
    inplace_tensors: Optional[List[Tuple[Tensor, Tensor]]] = None,
    **kwargs
):
    if inplace_tensors is None:
        inplace_tensors = []

    if "static" in kwargs:
        set_build_static(kwargs["static"])

    inputs = [t.base() for t in in_tensors]
    outputs = [t.base() for t in out_tensors]
    inplaces = [(t1.base(), t2.base()) for t1, t2 in inplace_tensors]

    func = None
    try:
        func = pto_impl.RecordFunc(name, inputs, outputs, inplaces)
        yield func
    except Exception as e:
        logging.error("Record function %s failed: %s", name, e)
        raise
    finally:
        del func


def cond(scalar: pto.SymbolicScalar):
    frame = inspect.currentframe().f_back
    return pto_impl.RecordIfBranch(scalar.base(), frame.f_code.co_filename, frame.f_lineno)


class _LoopFunction:

    class Iterator:
        def __init__(self, iter):
            self.iter = iter

        def __next__(self):
            return SymbolicScalar.from_base(self.iter.__next__())

    def __init__(self, *args):
        self._base = pto_impl.RecordLoopFunc(*args)

    def __iter__(self):
        return self.Iterator(self._base.__iter__())


@contextmanager
def loop_function(
    name: str,
    loop_name: str,
    loop_range_: pto.LoopRange,
    unroll_list: Set[int] = set(),
    submit_before_loop: bool = False,
):
    if unroll_list is None:
        unroll_list = set()

    rlf = None
    try:
        rlf = _LoopFunction(
            name,
            pto_impl.FunctionType.DYNAMIC_LOOP,
            loop_name,
            loop_range_.base(),
            unroll_list,
            submit_before_loop,
        )
        yield rlf
    except Exception as e:
        logging.error("Record loop function %s failed: %s", name, e)
        raise
    finally:
        del rlf


@contextmanager
def pto_function(name: str, graph_type: pto_impl.GraphType, func_type: pto_impl.FunctionType, *args):
    try:
        yield begin_function(name, graph_type, func_type, *args)
    except Exception as e:
        logging.error("Record function %s failed: %s", name, e)
        raise
    finally:
        pto_impl.EndFunction(name, False)


def loop(start, stop=None, step=1, **kwargs):
    name = kwargs.get("name", "LOOP")
    idx_name = kwargs.get("idx_name", "K")
    unroll_list = kwargs.get("unroll_list", set())
    submit_before_loop = kwargs.get("submit_before_loop", False)
    with loop_function(
        name, idx_name, loop_range(
            start, stop, step), unroll_list, submit_before_loop
    ) as rlf:
        for k in rlf:
            yield k
