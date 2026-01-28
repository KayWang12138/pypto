#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO"""
from enum import Enum
from typing import Optional, Union

from pypto import pypto_impl
from pypto._controller import loop
from pypto.enum import DataType, AtomicType
from pypto._op_wrapper import op_wrapper
from pypto.symbolic_scalar import SymbolicScalar
from pypto.tensor import Tensor

class ShmemMemType(Enum):
    WIN_IN = 0
    WIN_EXP = 1


Signal = Tensor


class CommConfig:
    def __init__(self, group_name: str, world_size: int, my_pe: SymbolicScalar):
        self.group_name = group_name
        self.world_size = world_size
        self.my_pe = my_pe


comm_configs: dict[str, CommConfig] = {}
shmem_id_to_group: dict[int, str] = {}
        

@op_wrapper
def create_shmem_tensor(
    group_name: str, 
    world_size: int,
    shape: list[int], 
    dtype: DataType,
) -> tuple[Tensor, Signal]:
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


    Examples
    --------
    shmem_data, shmem_signal = pypto.distributed.create_shmem_tensor(
        "tp",
        8,
        [64, 128],
        pypto.DT_FP16,
    )
    """
    shmem_data = Tensor([world_size] + shape, dtype)
    shmem_signal = Tensor([world_size, world_size] + shape, DataType.DT_INT32)

    for _ in loop(1, name="CREATE_SHMEM_TENSOR", idx_name="_"):
        pypto_impl.CreateShmemData(group_name, world_size, dtype, shape, shmem_data.base(), ShmemMemType.WIN_IN.value)
        pypto_impl.CreateShmemSignal(group_name, shmem_data.base(), shmem_signal.base())
    
    if group_name not in comm_configs:
        comm_configs[group_name] = CommConfig(group_name, world_size, pypto_impl.GetSymbolicScalarRankId(group_name))
    
    shmem_id_to_group[shmem_data.base().Id()] = group_name
    shmem_id_to_group[shmem_signal.base().Id()] = group_name
    
    return shmem_data, shmem_signal


@op_wrapper
def create_shmem_barrier_signal(group_name: str, world_size: int) -> Signal:
    """Creates a barrier signal tensor in shared memory.

    Parameters
    ----------
    group_name : str
        The name of the communication group.
    world_size : int
        The total number of processes in the communication group.

    Returns
    -------
    shmem_barrier_signal : Tensor
        A barrier signal tensor to coordinate process execution.
    

    Examples
    --------
    shmem_barrier_signal = pypto.distributed.create_shmem_barrier_signal("tp", 8)
    """
    shmem_barrier_signal_shape = [1, 1, 1, 8]
    shmem_barrier_signal = Tensor([world_size] + shmem_barrier_signal_shape, DataType.DT_INT32)

    for _ in loop(1, name="CREATE_SHMEM_BARRIER_SIGNAL", idx_name="_"):
        pypto_impl.CreateShmemData(group_name, world_size, DataType.DT_INT32, shmem_barrier_signal_shape,
            shmem_barrier_signal.base(), ShmemMemType.WIN_EXP.value)
    
    if group_name not in comm_configs:
        comm_configs[group_name] = CommConfig(group_name, world_size, pypto_impl.GetSymbolicScalarRankId(group_name))
    
    shmem_id_to_group[shmem_barrier_signal.base().Id()] = group_name
        
    return shmem_barrier_signal


@op_wrapper
def shmem_put(
    src: Tensor,
    offsets: list[Union[int, SymbolicScalar]],
    dst: Tensor,
    dst_rank: Union[int, SymbolicScalar],
    *,
    pred_tokens: list[Tensor] = None,
    shmem_op: AtomicType = AtomicType.SET,
) -> Tensor:
    """Asynchronously sends local GM data to a remote GM.

    Parameters
    ----------
    src: Tensor
        The source tensor located in local GM.
    offsets : list of int
        The offset in the destination shared memory GM.
    dst: Tensor
        The destination tensor in shared memory GM (symmetric memory).
    dst_rank : int
        The rank of the destination device.
    pred_tokens : Tensor
        Predicate tokens used to control the execution dependency of the operation.
    shmem_op : AtomicType
        The type of atomic operation to apply during the data transfer.

    Returns
    -------
    Tensor
        Output predicate tokens representing the completion dependency of the operation.

    Examples
    --------
    Send local GM data to rank 1
    tile = pypto.distributed.shmem_put(
        local_tensor,
        offset,
        shmem_tensor, 
        dst_rank,
        pred_tokens=None,
        shmem_op=pypto.AtomicType.SET,
    )
    """
    dummy = pred_tokens if len(pred_tokens) == 1 else pypto_impl.Nop(pred_tokens)
    dst_tile = pypto_impl.View(dst, [1, 1] + src.shape, [dst_rank] + offsets)
    return pypto_impl.ShmemPut(dummy, src, dst_tile, shmem_op)


@op_wrapper
def shmem_get(
    src: Tensor,
    src_rank: Union[int, SymbolicScalar],
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    pred_tokens: list[Tensor] = None,
    valid_shape: Optional[list[Union[int, SymbolicScalar]]] = None,
    shmem_op: AtomicType = AtomicType.SET, 
) -> Tensor:
    """Asynchronously fetches data from a remote GM to local GM.

    Parameters
    ----------
    src : Tensor
        The source tensor in remote GM.
    src_rank : Union[int, SymbolicScalar]
        The rank of the source device.
    shape : list of int
        The shape of the destination tensor.
    offsets : list of int
        The offset of the destination tensor in local GM.
    pred_tokens : Tensor
        Predicate tokens used as control dependencies.
    valid_shape: list[int] = None
        Optional parameter to retrieve the effective data size of the schematic block.
        It is required that the valid_shape is smaller than the shape of the input.
    shmem_op : AtomicType
        The type of atomic operation.

    Returns
    -------
    Tensor
        The destination tensor in local GM.


    Examples
    --------
    Load data from a remote device to local GM
    local_tensor = pypto.distributed.shmem_get(
        src,
        src_rank,
        shape,
        offset,
        pred_tokens=None,
        valid_shape=None,
        shmem_op=pypto.AtomicType.SET,
    )
    """
    dummy = pred_tokens if len(pred_tokens) == 1 else pypto_impl.Nop(pred_tokens)
    src_tile = pypto_impl.View(src, [1] + shape, [src_rank] + offset)
    return pypto_impl.ShmemGet(dummy, src_tile)


@op_wrapper
def shmem_signal(
    dst: Signal,
    dst_rank: Union[int, SymbolicScalar],
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    pred_tokens: list[Tensor] = None,
    shmem_op: AtomicType = AtomicType.SET,
) -> Tensor:
    """Writes a signal.

    Parameters
    ----------
    dst : Tensor
        The shared memory signal.
    dst_rank : int
        The target device id.
    shape : list of int
        The shapes of the shared memory signal;
    offset : list of int
        The offsets of the signal in shared memory.
    pred_tokens : Tensor
        Predicate tokens used as control dependencies.
    shmem_op : AtomicType
        The type of atomic operation.

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
        shape,
        offset,
        pred_tokens=None,
        shmem_op=pypto.AtomicType.SET,
    )
    """
    dummy = pred_tokens if len(pred_tokens) == 1 else pypto_impl.Nop(pred_tokens)
    dst_tile = pypto_impl.View(dst, [1, 1] + shape, [dst_rank, dst_rank] + offset)
    out_dummy = pypto_impl.ShmemSignal(dummy, dst_tile, shmem_op)
    return out_dummy


@op_wrapper
def shmem_wait(
    src: Signal,
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    cmp_value: int = 0,
    *,
    pred_tokens: list[Tensor] = None,
    clear_flag: bool = False,
) -> Tensor:
    """Waits for a signal.

    Parameters
    ----------
    src : Tensor
        The shared memory signal; can only be a local (this device) shmem signal.
    shape : list of int
        The shapes of the shared memory signal;
    offset : list of int
        he offsets of the shmem signal.
    cmp_value : int
        The value to wait for.
    pred_tokens : Tensor
        Predicate tokens used as control dependencies.
    clear_flag : bool
        Whether to reset the signal after waiting.

    Returns
    -------
    Tensor
        Output predicate tokens.

    Examples
    --------
    dummy = pypto.distributed.shmem_wait(
        shmem_signal,
        shape,
        offset,
        cmp_value,
        pred_tokens=None,
        clear_flag=False,
    )
    """
    dummy = pred_tokens if len(pred_tokens) == 1 else pypto_impl.Nop(pred_tokens)
    group_name = shmem_id_to_group[src.Id()]
    comm_config = comm_configs[group_name]

    src_tile = pypto_impl.View(src, [1, 1] + shape, [comm_config.my_pe, comm_config.my_pe] + offset)
    out_dummy = pypto_impl.WaitUntil(dummy, src_tile, cmp_value, clear_flag)
    return out_dummy


@op_wrapper
def shmem_barrier_all(
    src: Signal,
    pred_tokens: list[Tensor] = None,
) -> Tensor:
    """Synchronizes multiple devices within a communication group.

    Parameters
    ----------
    src : Tensor
        The shared memory barrier signal.
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
    dummy = pypto.distributed.shmem_barrier_all(shmem_barrier_signal,pred_tokens)
    """
    group_name = shmem_id_to_group[src.Id()]
    comm_config = comm_configs[group_name]
    dummy = pred_tokens if len(pred_tokens) == 1 else pypto_impl.Nop(pred_tokens)
    return pypto_impl.ShmemBarrier(dummy, src, group_name, comm_config.world_size)


@op_wrapper
def shmem_clear(
    src: Tensor,
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    pred_tokens: list[Tensor] = None,
    is_signal: bool = False,
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
    pred_tokens : Tensor
        Predicate tokens used as control dependencies.
    is_signal : bool
        Whether the tensor is a signal.

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
        pred_tokens=None,
        is_signal=False,
    )
    Clear a shmem signal tensor in shared memory
    data_clear_dummy = pypto.distributed.shmem_clear(
        shmem_signal,
        shmem_shape,
        shmem_offset,
        pred_tokens=None,
        is_signal=True,
    )
    """
    group_name = shmem_id_to_group[src.Id()]
    comm_config = comm_configs[group_name]
    dummy = pred_tokens if len(pred_tokens) == 1 else pypto_impl.Nop(pred_tokens)
    if is_signal:
        src_tile = pypto_impl.View(src, [1, 1] + shape, [comm_config.my_pe, 0] + offset)
        out = pypto_impl.ShmemSignalSet(dummy, src_tile)
    else:
        src_tile = pypto_impl.View(src, [1] + shape, [comm_config.my_pe] + offset)
        out = pypto_impl.ShmemDataSet(dummy, src_tile)
    return out



@op_wrapper
def my_symbolic_pe(group_name: str) -> SymbolicScalar:
    """Gets the symbolic PE.

    Parameters
    ----------
    group_name : str
        The name of the communication group.

    Returns
    -------
    symbolic scalar
        Represents my_pe.
    
    Examples
    --------
       my_pe = pypto.distributed.my_symbolic_pe(group_name) 
    """
    return pypto_impl.GetSymbolicScalarRankId(group_name)