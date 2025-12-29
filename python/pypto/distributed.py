# pypto/distributed.py

from . import pypto_impl
from .tensor import Tensor
from .enum import DataType

def create_shmem_tensor(rank_size: int, group_index: int, dtype: DataType, shape: list) -> Tensor:
    return pypto_impl.CreateShmemTensor(rank_size, group_index, dtype, shape)

def barrier(tensor: Tensor, group: str) -> Tensor:
    return pypto_impl.Barrier(tensor, group)

def shmem_all_gather(tensor: Tensor, dummy: Tensor, group: str):
    return pypto_impl.ShmemAllGather(tensor, dummy, group)

def two_shot_shmem_all_reduce(tensor: Tensor, group: str, out: Tensor):
    return pypto_impl.TwoShotShmemAllReduce(tensor.base(), group, out.base())

def create_shmem(rank_size: int, expert_num_per_rank: int, shmem_col: int, group_index: int, dtype: DataType, mem_type: int) -> Tensor:
    return pypto_impl.CreateShmem(rank_size, expert_num_per_rank, shmem_col, group_index, dtype, mem_type)

MoeConfig = pypto_impl.MoeConfig

def moe_dispatch(token_tensor: Tensor, token_expert_table: Tensor, group: str, moe_config: MoeConfig) -> tuple[Tensor, Tensor, Tensor]:
    return pypto_impl.MoeDispatch(token_tensor, token_expert_table, group, moe_config)
