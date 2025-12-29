# pypto/distributed.py

from . import pypto_impl
from .tensor import Tensor
from .enum import DataType

# def create_shmem_tensor(rank_size: int, group_index: int, dtype: DataType, shape: list) -> Tensor:
#     return pypto_impl.CreateShmemTensor(rank_size, group_index, dtype, shape)

# def create_shmem(rank_size: int, expert_num_per_rank: int, shmem_col: int, group_index: int, dtype: DataType, mem_type: int) -> Tensor:
#     return pypto_impl.CreateShmem(rank_size, expert_num_per_rank, shmem_col, group_index, dtype, mem_type)

def shmem_all_gather(tensor: Tensor, dummy: Tensor, group: str, out: Tensor):
    return pypto_impl.ShmemAllGather(tensor.base(), dummy.base(), group, out.base())

def one_shot_shmem_all_reduce(tensor: Tensor, group: str, out: Tensor):
    return pypto_impl.OneShotShmemAllReduce(tensor.base(), group, out.base())

def two_shot_shmem_all_reduce(tensor: Tensor, group: str, out: Tensor):
    return pypto_impl.TwoShotShmemAllReduce(tensor.base(), group, out.base())

DistReduceType = pypto_impl.DistReduceType
def shmem_reduce_scatter(tensor: Tensor, group: str, reduce_type: DistReduceType, out: Tensor):
    return pypto_impl.ShmemReduceScatter(tensor.base(), group, reduce_type, out.base())


def barrier(tensor: Tensor, group: str) -> Tensor:
    return pypto_impl.Barrier(tensor, group)

MoeConfig = pypto_impl.MoeConfig

# here we use the "shmem" prefix to align the naming convention with the other shmem functions
# Python: shmem_moe_dispatch, C++: MoeDispatch
def shmem_moe_dispatch(token_tensor: Tensor, token_expert_table: Tensor, expand_x: Tensor, valid_cnt: Tensor, combine_info: Tensor, group: str, moe_config: MoeConfig):
    return pypto_impl.MoeDispatch(token_tensor.base(), token_expert_table.base(), expand_x.base(), valid_cnt.base(), combine_info.base(), group, moe_config)

# no implementation
# def moe_combine(input_tensor: Tensor, scale: Tensor, combine_info: Tensor, group: str) -> Tensor:
#     return pypto_impl.MoeCombine(input_tensor, scale, combine_info, group)

def shmem_moe_combine(input_tensor: Tensor, combine_info: Tensor, scale: Tensor, group: str, rank_size: int, total_expert_num: int, out: Tensor):
    return pypto_impl.ShmemMoeCombine(input_tensor.base(), combine_info.base(), scale.base(), group, rank_size, total_expert_num, out.base())
