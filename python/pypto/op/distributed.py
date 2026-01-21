#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO"""
from typing import List, Union, Optional, Tuple
from enum import Enum
from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..tensor import Tensor
from ..enum import DataType, AtomicType
from ..symbolic_scalar import SymbolicScalar
from .._controller import loop

class ShmemMemType(Enum):
    WIN_IN = 0
    WIN_EXP = 1


Signal = Tensor


class CommConfig:
    def __init__(self, group_name: str, world_size: int, my_pe: SymbolicScalar):
        self.group_name = group_name
        self.world_size = world_size
        self.my_pe = my_pe


comm_config: Optional[CommConfig] = None
        

@op_wrapper
def create_shmem_tensor(
    group_name: str, 
    world_size: int,
    shape: List[int], 
    dtype: DataType,
) -> Tuple[Tensor, Signal, Tensor]:
    shmem_data = Tensor([world_size] + shape, dtype)
    shmem_signal = Tensor([world_size, world_size] + shape, DataType.DT_INT32)
    shmem_barrier_signal = Tensor([world_size] + [1, 1, 1, 8], DataType.DT_INT32)

    for _ in loop(1, name="CREATE_SHMEM_TENSOR", idx_name="_"):
        pypto_impl.CreateShmemData(group_name, world_size, dtype, shape, shmem_data.base(), ShmemMemType.WIN_IN.value)
        pypto_impl.CreateShmemSignal(group_name, shmem_data.base(), shmem_signal.base())
        pypto_impl.CreateShmemData(group_name, world_size, DataType.DT_INT32, [1, 1, 1, 8],
            shmem_barrier_signal.base(), ShmemMemType.WIN_EXP.value)
        
    global comm_config
    if comm_config is None:
        comm_config = CommConfig(group_name, world_size, pypto_impl.GetSymbolicScalarRankId(group_name))
    return shmem_data, shmem_signal, shmem_barrier_signal


@op_wrapper
def shmem_put(
    src: Tensor,
    dst_offset: List[Union[int, SymbolicScalar]],
    dst: Tensor,
    dst_rank: Union[int, SymbolicScalar],
    atomic_type: AtomicType = AtomicType.SET,
    pred_tokens: List[Tensor] = None,
) -> Tensor:
    dummy = pypto_impl.Nop(pred_tokens)
    dst_tile = pypto_impl.View(dst, [1, 1] + src.shape, [dst_rank] + dst_offset)
    return pypto_impl.ShmemPut(dummy, src, dst_tile, atomic_type)


@op_wrapper
def shmem_get(
    src: Tensor,
    src_rank: Union[int, SymbolicScalar],
    shape: List[int] = None,
    dst_offset: List[Union[int, SymbolicScalar]] = None,
    pred_tokens: List[Tensor] = None,
) -> Tensor:
    dummy = pypto_impl.Nop(pred_tokens)
    src_tile = pypto_impl.View(src, [1] + shape, [src_rank] + dst_offset)
    return pypto_impl.ShmemGet(dummy, src_tile)


@op_wrapper
def shmem_signal(
    dst: Signal,
    dst_rank: Union[int, SymbolicScalar],
    shapes: List[List[int]] = None,
    offsets: List[List[Union[int, SymbolicScalar]]] = None,
    atomic_type: AtomicType = AtomicType.SET,
    pred_tokens: List[Tensor] = None,
) -> Tensor:
    if len(shapes) > 1:
        raise ValueError(f"Currently not supporting shapes list size > 1 (current size: {len(shapes)})")
    dummy = pypto_impl.Nop(pred_tokens)
    out_dummys= []

    for idx in range(len(shapes)):
        dst_tile = pypto_impl.View(dst, [1, 1] + shapes[idx], [dst_rank, dst_rank] + offsets[idx])
        out_dummy = pypto_impl.ShmemSignal(dummy, dst_tile, atomic_type)
        out_dummys.append(out_dummy)
    return pypto_impl.Nop(out_dummys)


@op_wrapper
def shmem_wait(
    src: Signal,
    shapes: List[List[int]] = None,
    offsets: List[List[Union[int, SymbolicScalar]]] = None,
    expect_values: int = 0,
    pred_tokens: List[Tensor] = None,
    clear_flag: bool = False,
) -> Tensor:
    if len(shapes) > 1:
        raise ValueError(f"Currently not supporting shapes list size > 1 (current size: {len(shapes)})")
    dummy = pypto_impl.Nop(pred_tokens)
    out_dummys= []

    for idx in range(len(shapes)):
        src_tile = pypto_impl.View(src, [1, 1] + shapes[idx], [comm_config.my_pe, comm_config.my_pe] + offsets[idx])
        out_dummy = pypto_impl.WaitUntil(dummy, src_tile, expect_values, clear_flag)
        out_dummys.append(out_dummy)
    return pypto_impl.Nop(out_dummys)


@op_wrapper
def shmem_barrier_all(
    src: Tensor,
    group: str,
    pred_tokens: List[Tensor] = None,
) -> Tensor:
    dummy = pypto_impl.Nop(pred_tokens)
    return pypto_impl.ShmemBarrier(dummy, src, group, comm_config.world_size)


@op_wrapper
def shmem_clear(
    src: Tensor,
    shape: List[int] = None,
    offset: List[Union[int, SymbolicScalar]] = None,
    is_signal: bool = False,
    pred_tokens: List[Tensor] = None,
) -> Tensor:
    dummy = pypto_impl.Nop(pred_tokens)
    if is_signal:
        src_tile = pypto_impl.View(src, [1, 1] + shape, [comm_config.my_pe, 0] + offset)
        out_dummy =pypto_impl.ShmemSignalSet(dummy, src_tile)
    else:
        src_tile = pypto_impl.View(src, [1] + shape, [comm_config.my_pe] + offset)
        out_dummy =pypto_impl.ShmemDataSet(dummy, src_tile)
    return out_dummy



@op_wrapper
def my_symbolic_pe(group: str) -> SymbolicScalar:
    return pypto_impl.GetSymbolicScalarRankId(group)