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
import itertools
from contextlib import contextmanager
from typing import List, Optional, Set, Tuple, Union, Iterator, overload

from . import pto_impl

from .enum import *  # noqa
from .pto_utils import to_sym, set_source_location, clear_source_location
from .symbolic_scalar import SymbolicScalar, SymInt
from .tensor import Tensor

logging.basicConfig(level=logging.DEBUG)


__all__ = [
    "set_vec_tile_shapes",
    "get_vec_tile_shapes",
    "set_cube_tile_shapes",
    "get_cube_tile_shapes",
    "set_matrix_size",

    "function",
    "loop",
    "is_loop_begin",
    "is_loop_end",
    "cond",
]


class Controller:
    _loop_idx_generator = itertools.count(0)

    @classmethod
    def next_loop_idx(cls) -> int:
        return next(cls._loop_idx_generator)

    @classmethod
    def reset(cls):
        cls._loop_idx_generator = itertools.count(0)


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


def begin_function(
    name: str,
    graph_type: pto_impl.GraphType,
    func_type: pto_impl.FunctionType,
    *args
) -> pto_impl.RecordFunc:
    args = [arg.base() for arg in args]
    Controller.reset()
    return pto_impl.BeginFunction(name, graph_type, func_type, *args)


def end_function(name: str, generate_call: bool = True):
    pto_impl.EndFunction(name, generate_call)


class LoopRange:
    def __init__(self, start, stop=None, step: Union[int, SymbolicScalar] = 1):
        if stop is None:
            start, stop = 0, start
        self._base = pto_impl.LoopRange(
            to_sym(start), to_sym(stop), to_sym(step))
        self._start = self._base.Begin()
        self._stop = self._base.End()

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


_loop_range = LoopRange


def is_loop_begin(scalar: SymbolicScalar):
    ''' Determines if the current iteration is the start of loop
    This function returns a boolean value which specifys whether
    the current iteration is the beginning of the loop

    Parameters
    ----------
    scalar: SymbolicScalar
        current loop index

    Returns
    -------
    SymbolicScalar : expression to determine if currently at loop start

    Examples
    --------
    >>> for s2_idx in pto.loop(0, bn_per_batch, 1, name="LOOP_L4_s2_SA", idx_name="s2_idx",
            unroll_list=pto.powers_of_2(1)):
            if pto.cond(pto.is_loop_begin(s2_idx)):
                ...
    '''
    if not hasattr(scalar, "_loop_begin"):
        raise ValueError("not loop index")
    # implementation
    return SymbolicScalar.from_base(
        pto_impl.IsLoopBegin(to_sym(scalar), getattr(scalar, "_loop_begin")))


def is_loop_end(scalar: SymbolicScalar):
    ''' Determines if the current iteration is the end of loop
    This function returns a boolean value which specifys whether
    the current iteration is the end of the loop

    Parameters
    ----------
    scalar: SymbolicScalar
        current loop index

    Returns
    -------
    SymbolicScalar : expression to determine if currently at loop end

    Examples
    --------
    >>> for s2_idx in pto.loop(0, bn_per_batch, 1, name="LOOP_L4_s2_SA", idx_name="s2_idx",
            unroll_list=pto.powers_of_2(1)):
            if pto.cond(pto.is_loop_end(s2_idx)):
                ...
    '''
    if not hasattr(scalar, "_loop_end"):
        raise ValueError("not loop index")
    # implementation
    return SymbolicScalar.from_base(
        pto_impl.IsLoopEnd(to_sym(scalar), getattr(scalar, "_loop_end")))


@overload
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
    >>> with pto.function("main", [a, b], c):
            pto.set_vec_tile_shapes(16, 16)
            for _ in pto.loop(0, b_loop, 1, name, = "LOOP_L0_bIdx_mla_prolog",
                idx_name = "b_idx"):
                c[:] = a+b

    """
    ...


@overload
def function(name: str, *args, **kwargs):
    """
    The function with only name and tensors. The function type is static.

    Parameters
    ----------
    name: str
        The name of the function
    *args: List[Tensor]
        The list of input and output tensors
    static: bool, optional
        Whether the function is static or not. Default is True.

    Returns
    -------
    return the function in pypto framework. Operations will be added
    under this API. It will produce the computing graph of the function
    in the end.

    Examples
    --------
    >>> with pto.function("main", a, b, c, static=True):
            c[:] = a+b

    """
    ...


@contextmanager
def function(name: str, *args, **kwargs):
    if "static" in kwargs:
        set_build_static(kwargs["static"])
        try:
            Controller.reset()
            set_source_location(level=2)
            yield begin_function(name, pto_impl.GraphType.TENSOR_GRAPH,
                                 pto_impl.FunctionType.STATIC, *args)
            clear_source_location()
        except Exception as e:
            logging.error("Record function %s failed: %s", name, e)
            raise
        finally:
            end_function(name)
    else:
        in_tensors, out_tensors = args[0], args[1]
        inputs = [t.base() for t in in_tensors]
        outputs = [t.base() for t in out_tensors]
        func = None
        try:
            Controller.reset()
            set_source_location(level=2)
            func = pto_impl.RecordFunc(name, inputs, outputs, [])
            clear_source_location()
            yield func
        except Exception as e:
            logging.error("Record function %s failed: %s", name, e)
            raise
        finally:
            del func


def cond(scalar: SymInt):
    """ set up a conditional computation. Use as a "if" condition in python.

    Parameters
    ----------
    scalar: Union[int, SymbolicScalar]
        expression to determine if condition is true or not

    Returns
    -------
    return a generator, which will be used for setting up the
    "if" in building computing graph

    Examples
    --------
    >>> if pto.cond(pto.is_loop_begin(bn)):
            pass
        elif pto.cond(pto.is_loop_end(bn)):
            pass
        elif pto.cond(1):
            pass
        else:
            pass
    """
    # implementation
    stack = inspect.stack()[1]
    return pto_impl.RecordIfBranch(to_sym(scalar), stack.filename, stack.lineno)


class _LoopFunction:

    class Iterator:
        def __init__(self, iter, begin, end):
            self._iter = iter
            self._begin = begin
            self._end = end

        def __next__(self):
            scalar = SymbolicScalar.from_base(self._iter.__next__())
            setattr(scalar, "_loop_begin", self._begin)
            setattr(scalar, "_loop_end", self._end)
            return scalar

    def __init__(self, name, loop_name, loop_range, unroll_list, submit_before_loop):
        loop_range = loop_range.base()
        self._base = pto_impl.RecordLoopFunc(name, pto_impl.FunctionType.DYNAMIC_LOOP,
                                             loop_name, loop_range,
                                             unroll_list, submit_before_loop)
        self._begin = loop_range.Begin()
        self._end = loop_range.End()

    def __iter__(self):
        return self.Iterator(self._base.__iter__(), self._begin, self._end)


@contextmanager
def _loop_function(
    name: str,
    loop_name: str,
    loop_range: LoopRange,
    unroll_list: Optional[Set[int]] = None,
    submit_before_loop: bool = False,
):
    if unroll_list is None:
        unroll_list = set()

    rlf = None
    try:
        set_source_location(level=3)
        rlf = _LoopFunction(name, loop_name, loop_range,
                            unroll_list, submit_before_loop)
        clear_source_location()
        yield rlf
    except Exception as e:
        logging.error("Record loop function %s failed: %s", name, e)
        raise
    finally:
        del rlf


@overload
def loop(stop: SymInt, /, **kwargs) -> Iterator[SymInt]:
    """ Create a symbolic loop ranging from 0 to `stop` (exclusive).

    Parameters
    ----------
    stop : SymInt
        The end value (exclusive) of the loop range.
    kwargs :
        See base `loop()` documentation for shared keyword arguments.

    Returns
    -------
    Iterator[SymInt]
        A generator over symbolic integers representing each iteration variable.


    Examples
    --------
    with pto.loop(10, name="LOOP_L0_bIdx", idx_name="bIdx"):
        if pto.cond(k==0):
            b[:] = a + a
        else:
            b[:] = a + b
    """
    ...


@overload
def loop(start: SymInt, stop: SymInt, step: Optional[SymInt] = 1, /, **kwargs) -> Iterator[SymInt]:
    """ Create a symbolic loop ranging from `start` to `stop` (exclusive), incrementing by `step`.

    Parameters
    ----------
    start : SymInt
        Start value.
    stop : SymInt
        End value (exclusive).
    step : Optional[SymInt], default=1
        Increment for each iteration.
    kwargs :
        See base `loop()` documentation for shared keyword arguments.

    Returns
    -------
    Iterator[SymInt]
        A generator over symbolic integers.

    Examples
    --------
    with pto.loop(0, 10, 1, name="LOOP_L0_bIdx", idx_name="bIdx"):
        if pto.cond(k==0):
            b[:] = a + a
        else:
            b[:] = a + b
    """
    ...


def loop(
    *args,
    **kwargs,
) -> Iterator[SymInt]:
    """ set up a loop computation. Use as a for loop in python.

    Parameters
    ----------
    kwargs:
        name: str
            The name of the loop
        idx_name: str
            The name of the loop index
        unroll_list: Set[int]
            The number of loop layer which is unrolled

    Returns
    --------
    return a generator, which will be used for setting up the
    for loop in building computing graph
    """
    nargs = len(args)
    if nargs == 1:
        start, stop, step = 0, args[0], 1
    elif nargs == 2:
        start, stop, step = args[0], args[1], 1
    elif nargs == 3:
        start, stop, step = args
    else:
        raise TypeError(
            f"loop() takes 1 to 3 positional arguments but {nargs} were given")

    # implementation
    loop_idx = Controller.next_loop_idx()
    name = kwargs.get("name", f"loop_{loop_idx}")
    idx_name = kwargs.get("idx_name", f"loop_idx_{loop_idx}")
    unroll_list = kwargs.get("unroll_list", set())
    submit_before_loop = kwargs.get("submit_before_loop", False)
    with _loop_function(
        name, idx_name, _loop_range(
            start, stop, step), unroll_list, submit_before_loop
    ) as rlf:
        for k in rlf:
            yield k
