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
    """Creates a symmetric tensor in shared memory.

    Parameters
    ----------
    group_name : str
        The name of the communication group.
    world_size : int
        The total number of processes in the communication group.
    shape : list of int
        The shape of the tensor to create.
    dtype : DataType
        The data type of the tensor.

    Returns
    -------
    shmem_data : Tensor
        A shared memory tensor used by communication operators to transport data.
    shmem_signal : Tensor
        A signal tensor used for synchronization between processes.
    shmem_barrier_signal : Tensor
        A barrier signal tensor to coordinate process execution.


    Examples
    --------
    shmem_data, shmem_signal, shmem_barrier_signal = pypto.distributed.create_shmem_tensor(
        "tp",
        8,
        shape=[64, 128],
        dtype=pypto.DT_FP16
    )
    """
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
    """Asynchronously sends local GM data to a remote GM.

    Parameters
    ----------
    src: Tensor
        The source tensor located in local GM.
    dst_offset : list of int
        The offset in the destination shared memory GM.
    dst: Tensor
        The destination tensor in shared memory GM (symmetric memory).
    dst_rank : int
        The rank of the destination device.
    atomic_type : AtomicType
        The type of atomic operation to apply during the data transfer.
    pred_tokens : Tensor
        Predicate tokens used to control the execution dependency of the operation.

    Returns
    -------
    Tensor
        Output predicate tokens representing the completion dependency of the operation.

    Examples
    --------
    Send local GM data to rank 1
    dst_tile = pypto.view(shmem_tensor, shape, offset)
    tile = pypto.distributed.shmem_put(
        local_tensor,
        offset,
        shmem_tensor, 
        dst_rank=1,
        atomic_type=pypto.AtomicType.SET,
        pred_tokens
    )
    """
    dummy = pypto_impl.Nop(pred_tokens)
    dst_tile = pypto_impl.View(dst, [1, 1] + src.shape, [dst_rank] + dst_offset)
    return pypto_impl.ShmemPut(dummy, src, dst_tile, atomic_type)


@op_wrapper
def shmem_get(
    src: Tensor,
    src_rank: Union[int, SymbolicScalar],
    shape: List[int] = None,
    offset: List[Union[int, SymbolicScalar]] = None,
    atomic_type: AtomicType = AtomicType.SET, 
    pred_tokens: List[Tensor] = None,
) -> Tensor:
    """Asynchronously fetches data from a remote GM to local GM.

    Parameters
    ----------
    src : Tensor
        The source tensor in remote GM.
    src_rank : int
        The rank of the source device.
    shape : list of int
        The shape of the destination tensor.
    offset : list of int
        The offset of the destination tensor in local GM.
    atomic_type : AtomicType
        The type of atomic operation.
    pred_tokens : Tensor
        Predicate tokens used as control dependencies.

    Returns
    -------
    Tensor
        The destination tensor in local GM.


    Examples
    --------
    Load data from a remote device to local GM
    shmem_tile = pypto.view(shmem_tensor, shape, offset)
    local_tensor = pypto.distributed.shmem_get(
        shmem_tile,
        src_rank=1,
        shape,
        offset,
        atomic_type=pypto.AtomicType.SET,
        pred_tokens
    )
    """
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
    """Writes a signal.

    Parameters
    ----------
    dst : Tensor
        The shared memory signal.
    dst_rank : int
        The target device id.
    shapes : list of int
        The shapes of the shared memory signal; currently only len(shapes) = 1 is supported.
    offsets : list of int
        The offsets of the signal in shared memory.
    atomic_type : AtomicType
        The type of atomic operation.
    pred_tokens : Tensor
        Predicate tokens used as control dependencies.

    Returns
    -------
    Tensor
        The destination tensor in local GM.

    Examples
    --------
    Load data from a remote device to local GM
    dummy = pypto.distributed.shmem_signal(
        shmem_signal,
        dst_rank=1,
        shapes,
        offsets,
        atomic_type=pypto.AtomicType.SET,
        pred_tokens
    )
    """
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
    cmp_value: int = 0,
    clear_flag: bool = False,
    pred_tokens: List[Tensor] = None,
) -> Tensor:
    """Waits for a signal.

    Parameters
    ----------
    src : Tensor
        The shared memory signal; can only be a local (this device) shmem signal.
    shape : list of int
        The shapes of the shared memory signal; currently only len(shape) = 1 is supported.
    offsets : list of int
        he offsets of the shmem signal.
    cmp_value : int
        The value to wait for.
    clear_flag : bool
        Whether to reset the signal after waiting.
    pred_tokens : Tensor
        Predicate tokens used as control dependencies.

    Returns
    -------
    Tensor
        Output predicate tokens.

    Examples
    --------
    dummy = pypto.distributed.shmem_wait(
        shmem_signal,
        shapes,
        offsets,
        cmp_value,
        clear_flag,
        pred_tokens
    )
    """
    if len(shapes) > 1:
        raise ValueError(f"Currently not supporting shapes list size > 1 (current size: {len(shapes)})")
    dummy = pypto_impl.Nop(pred_tokens)
    out_dummys= []

    for idx in range(len(shapes)):
        src_tile = pypto_impl.View(src, [1, 1] + shapes[idx], [comm_config.my_pe, comm_config.my_pe] + offsets[idx])
        out_dummy = pypto_impl.WaitUntil(dummy, src_tile, cmp_value, clear_flag)
        out_dummys.append(out_dummy)
    return pypto_impl.Nop(out_dummys)


@op_wrapper
def shmem_barrier_all(
    src: Tensor,
    group: str,
    pred_tokens: List[Tensor] = None,
) -> Tensor:
    """Synchronizes multiple devices within a communication group.

    Parameters
    ----------
    src : Tensor
        The shared memory barrier signal.
    group : str
        The name of the communication group.
    pred_tokens : Tensor
        Predicate tokens used as control dependencies.

    Returns
    -------
    Tensor
        Output predicate tokens.


    Examples
    --------
    Synchronize devices within the communication group
    result = pypto.matmul(A_tile, B_tile, pypto.DT_FP16)
    dummy = pypto.distributed.shmem_barrier_all(
        shmem_barrier_signal,
        "TP",
        pred_tokens
    )
    """
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
    """Synchronizes multiple devices within a communication group by clearing shared memory tensors.

    Parameters
    ----------
    src : Tensor
        The shmem tensor to be cleared; can be a data tensor or a signal tensor.
    shape : list of int
        The shape of the shmem tensor to be cleared; can be for data or signal.
    offset : list of int
        The offset of the shmem tensor to be cleared; can be for data or signal.
    is_signal : bool
        Whether the tensor is a signal.
    pred_tokens : Tensor
        Predicate tokens used as control dependencies.

    Returns
    -------
    Tensor
        Output predicate tokens.

    Examples
    --------
    Clear a shmem data tensor in shared memory
    data_clear_dummy = pypto.distributed.shmem_clear(
        shmem_data,
        shmem_shape,
        shmem_offset,
        False,
        pred_tokens
    )
    Clear a shmem signal tensor in shared memory
    data_clear_dummy = pypto.distributed.shmem_clear(
        shmem_signal,
        shmem_shape,
        shmem_offset,
        True,
        pred_tokens
    )
    """
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
    """Gets the symbolic PE.

    Parameters
    ----------
    group : str
        The name of the communication group.

    Returns
    -------
    symbolic scalar
        Represents my_pe.
    """
    return pypto_impl.GetSymbolicScalarRankId(group)