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

from pto import pto_impl

from .enum import * # noqa
from .pto_utils import to_sym
from .symbolic_scalar import SymbolicScalar, SymInt
from .tensor import Tensor

logging.basicConfig(level=logging.DEBUG)


def set_vec_tile_shapes(*shapes: int):
    """ set the tile shapes in vector computation

    This operation sets the value of the tile shapes
    in each dimension in vector computation.

    Parameters
    ----------
    shapes: *int
        the values of the tile shape in each dimension

    Returns
    -------
    None

    Examples
    --------
    >>> import pto
    >>> pto.set_vec_tile_shapes(1, 1, 8, 8)
    >>> print(pto.get_vec_tile_shapes())
    [1, 1, 8, 8]

    """
    # implementation
    pto_impl.SetVecTile(*shapes)


def get_vec_tile_shapes() -> List[int]:
    """ get the tile shapes in vector computation

    This operation returns the value of the tile shapes
    in each dimension in vector computation.

    Parameters
    ----------
    None

    Returns
    -------
    List of integers. The values in the list represent the
    tile shape in each dimension respectively

    Examples
    --------
    >>> import pto
    >>> pto.set_vec_tile_shapes([1, 1, 8, 8])
    >>> print(pto.get_vec_tile_shapes())
    [1, 1, 8, 8]

    """
    # implementation
    return pto_impl.GetVecTile()


def set_cube_tile_shapes(m: List[int], k: List[int], n: List[int], set_l1_tile: bool = False):
    """ set the tile shapes in cube computation

    This operation sets the value of the tile shapes
    in each dimension in cube computation of left and right matrix,
    together with the cache level (L1/L0).

    Parameters
    ----------
    m: List[int]
        the value of the tile shape in m dimension.
        The length of the the list must be 2.

    k: List[int]
        the value of the tile shape in k dimension
        The length of the the list must be 2.

    n: List[int]
        the value of the tile shape in n dimension
        The length of the the list must be 2.

    set_l1_tile: bool
        whether the tile shape is set for L1 or L0.
        default is false (i.e. set for L0)

    Returns
    -------
    None

    Examples
    --------
    >>> import pto
    >>> pto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128], True)
    >>> print(pto.get_cube_tile_shapes())
    [[16, 16], [256, 512], [128, 128], True]

    """
    # implementation
    pto_impl.SetCubeTile(m, k, n, set_l1_tile)


def get_cube_tile_shapes() -> List[Union[List, bool]]:
    """ get the tile shapes in cube computation

    This operation gets the value of the tile shapes
    in each dimension in cube computation of left and right matrix,
    together with the cache level (L1/L0).

    Parameters
    ----------
    None

    Returns
    -------
    return List[Union[List, bool]]
    The list includes the tile shape information of both left and
    right matrix, together with the cache level (L1/L0).

    Examples
    --------
    >>> import pto
    >>> pto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128], True)
    >>> print(pto.get_cube_tile_shapes())
    [[16, 16], [256, 512], [128, 128], True]

    """
    # implementation
    return pto_impl.GetCubeTile()

def set_matrix_size(size: List[int]):
    pto_impl.SetMatrixSize(size)

def set_build_static(static: bool):
    pto_impl.SetBuildStatic(static)


def bytes_of(dtype: DataType) -> int:
    ''' return the number of bytes of the current datatype

    Parameters
    ----------
    dtype: pto.DataType
        datatype to be determined the number of bytes

    Returns
    -------
    int: the size of bytes the datatype contains

    Examples
    --------
    >>> import pto
    >>> print(pto.bytes_of(pto.DataType.DT_FP32))
        4
    '''
    # implementation
    return pto_impl.BytesOf(dtype)


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
    ''' Determines if the current iteration is the start of loop
    This function returns a boolean value which specifys whether
    the current iteration is the beginning of the loop

    Parameters
    ----------
    scalar: SymbolicScalar
        current loop index
    begin: Union[int, SymbolicScalar]
        begin loop index

    Returns
    -------
    SymbolicScalar : expression to determine if currently at loop start

    Examples
    --------
    >>> import pto
    >>> for s2_idx in pto.loop(0, bn_per_batch, 1, name="LOOP_L4_s2_SA", idx_name="s2_idx",
            unroll_list=pto.powers_of_2(1)):
            if pto.cond(pto.is_loop_begin(s2_idx, 0)):
                ...
    '''
    # implementation
    nbegin = to_sym(begin)
    return pto_impl.IsLoopBegin(scalar.base(), nbegin)


def is_loop_end(scalar: SymbolicScalar, end: Union[int, SymbolicScalar]):
    ''' Determines if the current iteration is the end of loop
    This function returns a boolean value which specifys whether
    the current iteration is the end of the loop

    Parameters
    ----------
    scalar: SymbolicScalar
        current loop index
    end: Union[int, SymbolicScalar]
        end loop index

    Returns
    -------
    SymbolicScalar : expression to determine if currently at loop end

    Examples
    --------
    >>> import pto
    >>> for s2_idx in pto.loop(0, bn_per_batch, 1, name="LOOP_L4_s2_SA", idx_name="s2_idx",
            unroll_list=pto.powers_of_2(1)):
            if pto.cond(pto.is_loop_end(s2_idx, 0)):
                ...
    '''
    # implementation
    nend = to_sym(end)
    return pto_impl.IsLoopEnd(scalar.base(), nend)


@contextmanager
def function(
    name: str,
    in_tensors: List[Tensor],
    out_tensors: List[Tensor],
    **kwargs
):
    """ defining the function

    This API record the function and dataflow user has defined. A computing
    graph will be built based on the recorded function.

    Parameters
    ----------
    name: str
        The name of the function
    in_tensors: List[Tensor]
        The list of input tensors
    out_tensors: List[Tensor]
        The list of output tensors

    Returns
    -------
    return the function in pypto framework. Operations will be added
    under this API. It will produce the computing graph of the function
    in the end.

    Examples
    --------
    >>> import pto
    >>> with pto.function("main", [a, b], c):
            pto.set_vec_tile_shapes(16, 16)
            for _ in pto.loop(0, b_loop, 1, name, = "LOOP_L0_bIdx_mla_prolog",
                idx_name = "b_idx"):
                c[:] = a+b

    """
    # implementation
    if "static" in kwargs:
        set_build_static(kwargs["static"])

    inputs = [t.base() for t in in_tensors]
    outputs = [t.base() for t in out_tensors]

    func = None
    try:
        func = pto_impl.RecordFunc(name, inputs, outputs, [])
        yield func
    except Exception as e:
        logging.error("Record function %s failed: %s", name, e)
        raise
    finally:
        del func


def cond(scalar: SymbolicScalar):
    """ set up a conditional computation. Use as a "if" condition in python.

    Parameters
    ----------
    scalar: pto.SymbolicScalar
        expression to determine if condition is true or not

    Returns
    -------
    return a generator, which will be used for setting up the
    "if" in building computing graph

    Examples
    --------
    >>> import pto
    >>> if pto.cond(pto.is_loop_begin(bn, 0)):
            pass
        elif pto.cond(pto.is_loop_end(bn, 0)):
            pass
        else:
            pass
    """
    # implementation
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
    loop_range_: LoopRange,
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


def loop(start: SymInt, end: Optional[SymInt] = None, step: Optional[SymInt] = None,
         unroll_times: Optional[List[int]] = None, **kwargs):
    """ set up a loop computation. Use as a for loop in python.

    Parameters
    ----------
    start: int
        the initial value of the variable in a for loop
    end: Optional[int]
        the ending value for the for loop
    step: Optional[int]
        The increment amount of the looping value
    unroll_times: Optional[list[int]]
        The number of loop layer which is unrolled

    Returns
    -------
    return a generator, which will be used for setting up the
    for loop in building computing graph

    Examples
    --------
    >>> import pto
    >>> with pto.loop(0, 10, 1, power_of_2(max_unroll_times), loop_name, iter_name):
            if pto.cond(k==0):
                b[:] = a + a
            else:
                b[:] = a + b

    """
    # implementation
    step = 1 if step is None else step
    name = kwargs.get("name", "LOOP")
    idx_name = kwargs.get("idx_name", "K")
    unroll_list = kwargs.get("unroll_list", set())
    submit_before_loop = kwargs.get("submit_before_loop", False)
    with loop_function(
        name, idx_name, loop_range(
            start, end, step), unroll_list, submit_before_loop
    ) as rlf:
        for k in rlf:
            yield k
