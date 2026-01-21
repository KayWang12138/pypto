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
from typing import List
from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..tensor import Tensor
from ..enum import DataType, AtomicType
from ..symbolic_scalar import SymbolicScalar

@op_wrapper
def create_shmem_data(
        group: str, 
        world_size: int, 
        dtype: DataType, 
        shape: List[int], 
        shmem_tensor: Tensor, 
        mem_type: int
) -> None:
    return pypto_impl.CreateShmemData(group, world_size, dtype, shape, shmem_tensor, mem_type)


@op_wrapper
def create_shmem_signal(
        group: str, 
        shmem_data: Tensor, 
        shmem_signal: Tensor
) -> None:
    return pypto_impl.CreateShmemSignal(group, shmem_data, shmem_signal)


@op_wrapper
def shmem_barrier(
        pred_token: Tensor,
        shmem_signal: Tensor,
        group: str,
        world_size: int,
        out: Tensor
) -> Tensor:
    return pypto_impl.ShmemBarrier(pred_token, shmem_signal, group, world_size, out)


@op_wrapper
def shmem_data_set(
        pred_token: Tensor,
        shmem_data: Tensor
) -> None:
    return pypto_impl.ShmemDataSet(pred_token, shmem_data)


@op_wrapper
def shmem_signal_set(
        pred_token: Tensor,
        shmem_signal: Tensor
) -> None:
    return pypto_impl.ShmemSignalSet(pred_token, shmem_signal)


@op_wrapper
def shmem_put(
        input: Tensor, 
        shmem_data_tile: Tensor, 
        pred_token: Tensor, 
        atomic_type: AtomicType
) -> Tensor:
    return pypto_impl.ShmemPut(input, shmem_data_tile, pred_token, atomic_type)


@op_wrapper
def shmem_get(
        dummy: Tensor, 
        shmem_data_tile: Tensor, 
        non_shmem_dtype: DataType.DT_BOTTOM, 
        atomic_type: AtomicType = AtomicType.SET
) -> Tensor:
    return pypto_impl.ShmemGet(dummy, shmem_data_tile, non_shmem_dtype, atomic_type)


@op_wrapper
def shmem_get_gm2ub(
        dummy: Tensor, 
        shmem_data_tile: Tensor, 
        non_shmem_dtype: DataType.DT_BOTTOM, 
        atomic_type: AtomicType = AtomicType.SET
) -> Tensor: 
    return pypto_impl.ShmemGetGm2Ub(dummy, shmem_data_tile, non_shmem_dtype, atomic_type)


@op_wrapper
def shmem_signal(
        dummy: Tensor, 
        shmem_signal_tile: Tensor, 
        atomic_type: AtomicType
) -> Tensor:
    return pypto_impl.ShmemSignal(dummy, shmem_signal_tile, atomic_type)


@op_wrapper
def wait_until(
        dummyIn: Tensor, 
        shmem_signal_tile: Tensor, 
        expected_sum: int, 
        reset_signal: bool = False
) -> Tensor:
    return pypto_impl.WaitUntil(dummyIn, shmem_signal_tile, expected_sum, reset_signal)


@op_wrapper
def get_symbolic_scalar_rank_id(group: str) -> SymbolicScalar:
    return pypto_impl.GetSymbolicScalarRankId(group)