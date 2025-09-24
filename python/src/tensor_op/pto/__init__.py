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
"""
"""

import sys
import importlib.resources

sys.path.append(str(importlib.resources.files(__package__)))
del importlib
del sys

from contextlib import contextmanager
from typing import List, Tuple
import inspect

from pto import pto_impl
from pto.pto_impl import *
from .runtime import device_init, device_fini, device_run_once_data_from_host, device_run_once_data_from_device, jit
cond = pto_impl.record_if_branch

@contextmanager
def dyn_function(
    name: str,
    in_tensors: List[pto_impl.tensor],
    out_tensors: List[pto_impl.tensor],
    inplace_tensors: List[Tuple[pto_impl.tensor, pto_impl.tensor]] = None,
) -> pto_impl.record_func:
    if inplace_tensors is None:
        inplace_tensors = []
    func_cfg = pto_impl.func_config(pto_impl.function_type.DYNAMIC)
    record_func = pto_impl.record_func(
        name, func_cfg, in_tensors, out_tensors, inplace_tensors
    )
    print(f"Entering DYNAMIC function: {name}")
    try:
        yield record_func
    except Exception as e:
        print(f"Caught exception: {e}")
        raise
    finally:
        del record_func
        print(f"Exiting DYNAMIC function: {name}")


@contextmanager
def loop_function(
    name: str,
    loop_name: str,
    loop_range_: pto_impl.loop_range_,
    unroll_list: set[int] = None,
    submit_before_loop: bool = False,
) -> pto_impl.record_loop_func:
    if unroll_list is None:
        unroll_list = set()
    rlf = None
    print(f"Entering LOOP function: {name}")
    try:
        rlf = pto_impl.record_loop_func(
            name,
            pto_impl.function_type.DYNAMIC_LOOP,
            loop_name,
            loop_range_,
            unroll_list,
            submit_before_loop,
        )
        yield rlf
    except Exception as e:
        print(f"Caught exception: {e}")
        raise
    finally:
        del rlf
        print(f"Exiting LOOP function: {name}")


@contextmanager
def pto_function(
    name: str, graph_type: pto_impl.graph_type, func_type: pto_impl.function_type, *args
):
    print(f"Entering context: {name}")
    try:
        yield pto_impl.begin_function(name, graph_type, func_type, *args)
    except Exception as e:
        print(f"Caught exception: {e}")
        raise
    finally:
        pto_impl.end_function(name, False)
        print(f"Exiting context: {name}")


def record_if_branch(scalar: pto_impl.symbolic_scalar):
    frame = inspect.currentframe().f_back
    return pto_impl.record_if_branch(scalar, frame.f_code.co_filename, frame.f_lineno)


def convert_to_symbolic(value):
    if value is None:
        return value
    if isinstance(value, int):
        return pto_impl.symbolic_scalar(value)
    if isinstance(value, list):
        return [convert_to_symbolic(v) for v in value]
    if not isinstance(value, pto_impl.symbolic_scalar):
        raise TypeError(
            f"Expected value to be int, list, or symbolic_scalar, but got {type(value)}"
        )
    return value


def loop_range(start, end=None, step=None):
    start = convert_to_symbolic(start)
    end = convert_to_symbolic(end)
    step = convert_to_symbolic(step)
    if end is None:
        end = start
        start = pto_impl.symbolic_scalar(0)
    if step is None:
        step = pto_impl.symbolic_scalar(1)
    return pto_impl.loop_range_(start, end, step)


def view(tensor, shape, offset, valid_shape=None):
    if valid_shape is not None:
        # dview_pad
        offset = convert_to_symbolic(offset)
        valid_shape = convert_to_symbolic(valid_shape)
        return pto_impl.view_(tensor, shape, offset, valid_shape)
    else:
        if not all(isinstance(o, int) for o in offset):
            offset = convert_to_symbolic(offset)
        return pto_impl.view_(tensor, shape, offset)


def reshape(tensor, shape, valid_shape=None):
    if valid_shape is None:
        return pto_impl.reshape_(tensor, shape)
    valid_shape = convert_to_symbolic(valid_shape)
    return pto_impl.reshape_(tensor, shape, valid_shape)


def assemble(tensor, offset=None, dest=None):
    if offset is None and dest is None:
        return pto_impl.assemble_(tensor)
    offset = convert_to_symbolic(offset)
    return pto_impl.assemble_(tensor, offset, dest)


def is_loop_begin(scalar, begin):
    begin = convert_to_symbolic(begin)
    return pto_impl.is_loop_begin_(scalar, begin)


def is_loop_end(scalar, end):
    end = convert_to_symbolic(end)
    return pto_impl.is_loop_end_(scalar, end)
