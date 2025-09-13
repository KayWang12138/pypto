#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
from typing import List, Tuple
from contextlib import contextmanager
import inspect
import pto


@contextmanager
def dyn_function(
    name: str, in_tensors: List[pto.tensor], out_tensors: List[pto.tensor], 
    inplace_tensors: List[Tuple[pto.tensor, pto.tensor]] = None,
) -> pto.record_func:
    if inplace_tensors is None:
        inplace_tensors = []
    func_cfg = pto.func_config(pto.function_type.DYNAMIC)
    record_func = pto.record_func(
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
    name: str, loop_name: str, loop_range_: pto.loop_range_,
    unroll_list: set[int] = None, submit_before_loop: bool = False
) -> pto.record_loop_func:
    if unroll_list is None:
        unroll_list = set()
    rlf = None
    print(f"Entering LOOP function: {name}")
    try:
        rlf = pto.record_loop_func(
            name,
            pto.function_type.DYNAMIC_LOOP,
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


# TODO: move common utils into `pto` Python package
@contextmanager
def pto_function(
    name: str, graph_type: pto.graph_type, func_type: pto.function_type, *args
):
    print(f"Entering context: {name}")
    try:
        yield pto.begin_function(name, graph_type, func_type, *args)
    except Exception as e:
        print(f"Caught exception: {e}")
        raise
    finally:
        # TODO(anastasios): make false input param.
        pto.end_function(name, False)
        print(f"Exiting context: {name}")


def record_if_branch(scalar: pto.symbolic_scalar):
    frame = inspect.currentframe().f_back
    return pto.record_if_branch(scalar, frame.f_code.co_filename, frame.f_lineno)


def convert_to_symbolic(value):
    if value is None:
        return value
    if isinstance(value, int):
        return pto.symbolic_scalar(value)
    if isinstance(value, list):
        return [convert_to_symbolic(v) for v in value]
    if not isinstance(value, pto.symbolic_scalar):
        raise TypeError(f"Expected value to be int, list, or symbolic_scalar, but got {type(value)}")
    return value


def loop_range(start, end=None, step=None):
    start = convert_to_symbolic(start)
    end = convert_to_symbolic(end)
    step = convert_to_symbolic(step)
    if end is None:
        end = start
        start = pto.symbolic_scalar(0)
    if step is None:
        step = pto.symbolic_scalar(1)
    return pto.loop_range_(start, end, step)


def view(tensor, shape, offset, valid_shape=None):
    if valid_shape is not None:
        # dview_pad
        offset = convert_to_symbolic(offset)
        valid_shape = convert_to_symbolic(valid_shape)
        return pto.view_(tensor, shape, offset, valid_shape)
    else:
        if not all(isinstance(o, int) for o in offset):
            offset = convert_to_symbolic(offset)
        return pto.view_(tensor, shape, offset)


def reshape(tensor, shape, valid_shape=None):
    if valid_shape is None:
        return pto.reshape_(tensor, shape)
    valid_shape = convert_to_symbolic(valid_shape)
    return pto.reshape_(tensor, shape, valid_shape)


def assemble(tensor, offset=None, dest=None):
    if offset is None and dest is None:
        return pto.assemble_(tensor)
    offset = convert_to_symbolic(offset)
    return pto.assemble_(tensor, offset, dest)


def is_loop_begin(scalar, begin):
    begin = convert_to_symbolic(begin)
    return pto.is_loop_begin_(scalar, begin)


def is_loop_end(scalar, end):
    end = convert_to_symbolic(end)
    return pto.is_loop_end_(scalar, end)