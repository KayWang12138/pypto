# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO distributed operations."""

from . import pypto_impl
from .tensor import Tensor
from .enum import DataType, DistReduceType, AtomicType
from ._utils import experimental
from . import config as _config


def _get_base(t):
    if hasattr(t, "base"):
        return t.base()
    return t


def _tensor_to_bytes(t):
    try:
        return t.cpu().numpy().tobytes()
    except Exception:
        return bytes(t.cpu().tolist())


@experimental
def init_hccl_comm_from_torch_pg(pg=None):
    """Initialize HCCL comm from torch.distributed."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    if pg is None:
        pg = dist.group.WORLD
    rank = dist.get_rank(pg)
    world = dist.get_world_size(pg)
    import os
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("RANK_ID", str(rank))
    os.environ.setdefault("LOCAL_RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    os.environ.setdefault("RANK_SIZE", str(world))
    os.environ.setdefault("LOCAL_WORLD_SIZE", str(world))
    os.environ.setdefault("HCCL_WORLD_SIZE", str(world))
    try:
        backend = pg._get_backend(torch.device("npu"))
        if hasattr(backend, "get_hccl_comm") and hasattr(backend, "get_hccl_comm_name"):
            hccl_handle = backend.get_hccl_comm(rank)
            group_name = backend.get_hccl_comm_name(rank)
            if hccl_handle and group_name:
                _config.set_distributed_options(hccl_handle=[hccl_handle], hccl_group_name=[group_name])
                return hccl_handle, group_name
    except Exception:
        pass
    root_size = pypto_impl.GetHcclRootInfoSize()
    root_tensor = torch.empty((root_size,), dtype=torch.uint8, device="npu")
    if rank == 0:
        root_info = pypto_impl.HcclGetRootInfo()
        cpu_bytes = torch.tensor(list(root_info), dtype=torch.uint8, device="cpu")
        root_tensor.copy_(cpu_bytes.to(device="npu"))
    dist.broadcast(root_tensor, src=0, group=pg)
    root_bytes = _tensor_to_bytes(root_tensor)
    hccl_handle = pypto_impl.HcclCommInitRootInfo(root_bytes, rank, world)
    dist.barrier(pg)
    group_name = pypto_impl.HcclGetCommName(hccl_handle)
    _config.set_distributed_options(hccl_handle=[hccl_handle], hccl_group_name=[group_name])
    return hccl_handle, group_name


@experimental
def create_shmem_tensor(rank_size: int, group: str, dtype: DataType, shape: list, mem_type: int = 0) -> Tensor:
    """Create a shared-memory tensor."""
    return pypto_impl.CreateShmemTensor(rank_size, group, dtype, shape, mem_type)


@experimental
def shmem_barrier(pred_token: Tensor, shmem_signal: Tensor, group: str, out: Tensor):
    """Barrier synchronize across ranks."""
    return pypto_impl.ShmemBarrier(_get_base(pred_token), _get_base(shmem_signal), group, _get_base(out))


@experimental
def shmem_set(pred_token: Tensor, shmem_tensor: Tensor) -> Tensor:
    """Initialize shared-memory data."""
    return pypto_impl.ShmemSet(_get_base(pred_token), _get_base(shmem_tensor))


@experimental
def shmem_all_gather(tensor: Tensor, barrier_dummy: Tensor, group: str, out: Tensor):
    """AllGather via shared memory."""
    return pypto_impl.ShmemAllGather(_get_base(tensor), _get_base(barrier_dummy), group, _get_base(out))


@experimental
def one_shot_shmem_all_reduce(tensor: Tensor, group: str, out: Tensor):
    """One-shot AllReduce via shared memory."""
    return pypto_impl.OneShotShmemAllReduce(_get_base(tensor), group, _get_base(out))


@experimental
def two_shot_shmem_all_reduce(tensor: Tensor, group: str, out: Tensor):
    """Two-shot AllReduce via shared memory."""
    return pypto_impl.TwoShotShmemAllReduce(_get_base(tensor), group, _get_base(out))


@experimental
def shmem_reduce_scatter(tensor: Tensor, group: str, reduce_type: DistReduceType, out: Tensor):
    """ReduceScatter via shared memory."""
    return pypto_impl.ShmemReduceScatter(_get_base(tensor), group, reduce_type, _get_base(out))


MoeConfig = pypto_impl.MoeConfig


@experimental
def shmem_moe_dispatch(token_tensor: Tensor, token_expert_table: Tensor, expand_x: Tensor,
                       valid_cnt: Tensor, combine_info: Tensor, group: str, moe_config: MoeConfig):
    """MoE dispatch via shared memory."""
    return pypto_impl.MoeDispatch(_get_base(token_tensor), _get_base(token_expert_table),
                                  _get_base(expand_x), _get_base(valid_cnt), _get_base(combine_info),
                                  group, moe_config)


@experimental
def shmem_moe_combine(input_tensor: Tensor, combine_info: Tensor, recv_counts: Tensor, scale: Tensor,
                      group: str, rank_size: int, total_expert_num: int, out: Tensor):
    """MoE combine via shared memory."""
    return pypto_impl.ShmemMoeCombine(_get_base(input_tensor), _get_base(combine_info),
                                      _get_base(recv_counts), _get_base(scale), group, rank_size, total_expert_num,
                                      _get_base(out))


@experimental
def shmem_moe_combine_ffn_fused(input_tensor: Tensor, combine_info: Tensor, recv_counts: Tensor, scale: Tensor,
                                ffn_weight: Tensor, group: str, rank_size: int, total_expert_num: int, out: Tensor):
    """MoE combine + FFN fused via shared memory."""
    return pypto_impl.ShmemMoeCombineFfnFused(_get_base(input_tensor), _get_base(combine_info),
                                              _get_base(recv_counts), _get_base(scale), _get_base(ffn_weight),
                                              group, rank_size, total_expert_num, _get_base(out))


@experimental
def shmem_put(in_tensor: Tensor, shmem_data_tile: Tensor, barrier_dummy: Tensor,
              tile_count: int, atomic_type: AtomicType = AtomicType.SET) -> Tensor:
    """Write to shared memory."""
    return pypto_impl.ShmemPut(_get_base(in_tensor), _get_base(shmem_data_tile),
                               _get_base(barrier_dummy), tile_count, atomic_type)


@experimental
def shmem_put_ub2gm(in_tensor: Tensor, shmem_data_tile: Tensor, barrier_dummy: Tensor,
                    tile_count: int, atomic_type: AtomicType = AtomicType.SET) -> Tensor:
    """Write UB data to shared memory."""
    return pypto_impl.ShmemPutUb2Gm(_get_base(in_tensor), _get_base(shmem_data_tile),
                                    _get_base(barrier_dummy), tile_count, atomic_type)


@experimental
def shmem_get(dummy: Tensor, shmem_data_tile: Tensor,
              non_shmem_data_type: DataType = DataType.DT_BOTTOM,
              atomic_type: AtomicType = AtomicType.SET) -> Tensor:
    """Read from shared memory."""
    return pypto_impl.ShmemGet(_get_base(dummy), _get_base(shmem_data_tile),
                               non_shmem_data_type, atomic_type)


@experimental
def shmem_get_gm2ub(dummy: Tensor, shmem_data_tile: Tensor,
                    non_shmem_data_type: DataType = DataType.DT_BOTTOM,
                    atomic_type: AtomicType = AtomicType.SET) -> Tensor:
    """Read shared memory into UB."""
    return pypto_impl.ShmemGetGm2Ub(_get_base(dummy), _get_base(shmem_data_tile),
                                    non_shmem_data_type, atomic_type)


@experimental
def shmem_reduce(in_tensor: Tensor, shm_data: Tensor, dummy: Tensor, out: Tensor):
    """Reduce shared-memory data."""
    return pypto_impl.ShmemReduce(_get_base(in_tensor), _get_base(shm_data),
                                  _get_base(dummy), _get_base(out))


@experimental
def shmem_signal(dummy: Tensor, shmem_signal_tile: Tensor,
                 atomic_type: AtomicType = AtomicType.SET) -> Tensor:
    """Send a shared-memory signal."""
    return pypto_impl.ShmemSignal(_get_base(dummy), _get_base(shmem_signal_tile), atomic_type)


@experimental
def wait_until(dummy_in: Tensor, shmem_signal_tile: Tensor, tile_count: int,
               expected_sum: int, reset_signal: bool = False) -> Tensor:
    """Wait for shared-memory signals."""
    return pypto_impl.WaitUntil(_get_base(dummy_in), _get_base(shmem_signal_tile),
                                tile_count, expected_sum, reset_signal)
