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

from contextlib import contextmanager
from typing import List, Tuple
import inspect
import logging
from pto import pto_impl

logging.basicConfig(level=logging.DEBUG)

set_vec_tile_shapes = pto_impl.SetVecTile
get_vec_tile_shapes = pto_impl.GetVecTile
set_cube_tile_shapes = pto_impl.SetCubeTile
set_config = pto_impl.SetConfig
set_matrix_size = pto_impl.SetMatrixSize
set_semantic_label = pto_impl.SetSemanticLabel
set_operation_config = pto_impl.SetOperationConfig
set_pass_config = pto_impl.SetPassConfig
set_host_config = pto_impl.SetHostConfig
set_codegen_config = pto_impl.SetCodeGenConfig
bytes_of = pto_impl.BytesOf
dump = pto_impl.Dump
powers_of_2 = pto_impl.PowersOf2
begin_function = pto_impl.BeginFunction
end_function = pto_impl.EndFunction

record_func = RecordFunc = pto_impl.RecordFunc
record_loop_func = RecordLoopFunc = pto_impl.RecordLoopFunc
record_if_branch = RecordIfBranch = pto_impl.RecordIfBranch
vec_tile = VecTile = pto_impl.VecTile
function_config = FunctionConfig = pto_impl.FunctionConfig

TileShape = pto_impl.TileShape
TileShape.reset = TileShape.Reset
TileShape.to_string = TileShape.toString
TileShape.get_vec_tile_shapes = TileShape.GetVecTile
TileShape.set_vec_tile_shapes = TileShape.SetVecTile
TileShape.set_dist_rank_id = TileShape.SetDistRankId
tile_shape = TileShape

LoopRange = pto_impl.LoopRange
LoopRange.begin = LoopRange.Begin
LoopRange.end = LoopRange.End
LoopRange.step = LoopRange.Step
LoopRange.dump = LoopRange.Dump
loop_range = LoopRange

is_loop_begin = pto_impl.IsLoopBegin
is_loop_end = pto_impl.IsLoopEnd


@contextmanager
def dyn_function(
    name: str,
    in_tensors: List[pto_impl.tensor],
    out_tensors: List[pto_impl.tensor],
    inplace_tensors: List[Tuple[pto_impl.tensor, pto_impl.tensor]] = None,
) -> pto_impl.RecordFunc:
    if inplace_tensors is None:
        inplace_tensors = []
    func_cfg = pto_impl.FunctionConfig(pto_impl.FunctionType.DYNAMIC)
    record_func_ = pto_impl.RecordFunc(
        name, func_cfg, in_tensors, out_tensors, inplace_tensors
    )
    logging.debug("Entering DYNAMIC function: %s", name)
    try:
        yield record_func_
    except Exception as e:
        logging.debug("Caught exception: %s", e)
        raise
    finally:
        del record_func_
        logging.debug("Exiting DYNAMIC function: %s", name)


def cond(scalar: pto_impl.symbolic_scalar):
    frame = inspect.currentframe().f_back
    return pto_impl.RecordIfBranch(scalar, frame.f_code.co_filename, frame.f_lineno)


@contextmanager
def loop_function(
    name: str,
    loop_name: str,
    loop_range_: pto_impl.LoopRange,
    unroll_list: set[int] = None,
    submit_before_loop: bool = False,
) -> pto_impl.RecordLoopFunc:
    if unroll_list is None:
        unroll_list = set()
    rlf = None
    logging.debug("Entering LOOP function: %s", name)
    try:
        rlf = pto_impl.RecordLoopFunc(
            name,
            pto_impl.FunctionType.DYNAMIC_LOOP,
            loop_name,
            loop_range_,
            unroll_list,
            submit_before_loop,
        )
        yield rlf
    except Exception as e:
        logging.debug("Caught exception: %s", e)
        raise
    finally:
        del rlf
        logging.debug("Exiting LOOP function: %s", name)


@contextmanager
def pto_function(
    name: str, graph_type: pto_impl.GraphType, func_type: pto_impl.FunctionType, *args
):
    logging.debug("Entering context: %s", name)
    try:
        yield pto_impl.BeginFunction(name, graph_type, func_type, *args)
    except Exception as e:
        logging.debug("Caught exception: %s", e)
        raise
    finally:
        # TODO(anastasios): make false input param.
        pto_impl.EndFunction(name, False)
        logging.debug("Exiting context: %s", name)
