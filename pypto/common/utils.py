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
import pto
from contextlib import contextmanager
from typing import List
import inspect


@contextmanager
def dyn_function(
    name: str, in_tensors: List[pto.tensor], out_tensors: List[pto.tensor]
) -> pto.record_func:
    func_cfg = pto.func_config(pto.function_type.DYNAMIC)
    record_func = pto.record_func(
        name, func_cfg, in_tensors, out_tensors, []
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
    name: str, loop_name: str, loop_range: pto.loop_range
) -> pto.record_loop_func:
    # TODO(anastasios): make it an input parameter
    rlf = None
    empty_set_ints = set()
    print(f"Entering LOOP function: {name}")
    try:
        rlf = pto.record_loop_func(
            name,
            pto.function_type.DYNAMIC_LOOP,
            loop_name,
            loop_range,
            empty_set_ints,
            False,
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
