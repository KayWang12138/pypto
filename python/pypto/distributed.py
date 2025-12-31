# pypto/distributed.py

from . import pypto_impl
from .tensor import Tensor
from .enum import DataType, DistReduceType, AtomicType
from ._utils import experimental

def _get_base(t):
    if hasattr(t, "base"):
        return t.base()
    return t

@experimental
def create_shmem_tensor(rank_size: int, group: str, dtype: DataType, shape: list, mem_type: int = 0) -> Tensor:
    return pypto_impl.CreateShmemTensor(rank_size, group, dtype, shape, mem_type)

@experimental
def shmem_barrier(pred_token: Tensor, shmem_signal: Tensor, group: str, out: Tensor):
    return pypto_impl.ShmemBarrier(_get_base(pred_token), _get_base(shmem_signal), group, _get_base(out))

@experimental
def shmem_set(pred_token: Tensor, shmem_tensor: Tensor) -> Tensor:
    return pypto_impl.ShmemSet(_get_base(pred_token), _get_base(shmem_tensor))

@experimental
def shmem_all_gather(tensor: Tensor, barrier_dummy: Tensor, group: str, out: Tensor):
    return pypto_impl.ShmemAllGather(_get_base(tensor), _get_base(barrier_dummy), group, _get_base(out))

@experimental
def one_shot_shmem_all_reduce(tensor: Tensor, group: str, out: Tensor):
    return pypto_impl.OneShotShmemAllReduce(_get_base(tensor), group, _get_base(out))

@experimental
def two_shot_shmem_all_reduce(tensor: Tensor, group: str, out: Tensor):
    return pypto_impl.TwoShotShmemAllReduce(_get_base(tensor), group, _get_base(out))

@experimental
def shmem_reduce_scatter(tensor: Tensor, group: str, reduce_type: DistReduceType, out: Tensor):
    return pypto_impl.ShmemReduceScatter(_get_base(tensor), group, reduce_type, _get_base(out))

MoeConfig = pypto_impl.MoeConfig

# here we use the "shmem" prefix to align the naming convention with the other shmem functions
# Python: shmem_moe_dispatch, C++: MoeDispatch
@experimental
def shmem_moe_dispatch(token_tensor: Tensor, token_expert_table: Tensor, expand_x: Tensor, valid_cnt: Tensor, combine_info: Tensor, group: str, moe_config: MoeConfig):
    return pypto_impl.MoeDispatch(_get_base(token_tensor), _get_base(token_expert_table), _get_base(expand_x), _get_base(valid_cnt), _get_base(combine_info), group, moe_config)

# no implementation
# def moe_combine(input_tensor: Tensor, scale: Tensor, combine_info: Tensor, group: str) -> Tensor:
#     return pypto_impl.MoeCombine(input_tensor, scale, combine_info, group)

@experimental
def shmem_moe_combine(input_tensor: Tensor, combine_info: Tensor, scale: Tensor, group: str, rank_size: int, total_expert_num: int, out: Tensor):
    return pypto_impl.ShmemMoeCombine(_get_base(input_tensor), _get_base(combine_info), _get_base(scale), group, rank_size, total_expert_num, _get_base(out))


@experimental
def shmem_put(in_tensor: Tensor, shmem_data_tile: Tensor, barrier_dummy: Tensor, tile_count: int, atomic_type: AtomicType = AtomicType.SET) -> Tensor:
    return pypto_impl.ShmemPut(_get_base(in_tensor), _get_base(shmem_data_tile), _get_base(barrier_dummy), tile_count, atomic_type)

@experimental
def shmem_put_ub2gm(in_tensor: Tensor, shmem_data_tile: Tensor, barrier_dummy: Tensor, tile_count: int, atomic_type: AtomicType = AtomicType.SET) -> Tensor:
    return pypto_impl.ShmemPutUb2Gm(_get_base(in_tensor), _get_base(shmem_data_tile), _get_base(barrier_dummy), tile_count, atomic_type)

@experimental
def shmem_get(dummy: Tensor, shmem_data_tile: Tensor, non_shmem_data_type: DataType = DataType.DT_BOTTOM, atomic_type: AtomicType = AtomicType.SET) -> Tensor:
    return pypto_impl.ShmemGet(_get_base(dummy), _get_base(shmem_data_tile), non_shmem_data_type, atomic_type)

@experimental
def shmem_get_gm2ub(dummy: Tensor, shmem_data_tile: Tensor, non_shmem_data_type: DataType = DataType.DT_BOTTOM, atomic_type: AtomicType = AtomicType.SET) -> Tensor:
    return pypto_impl.ShmemGetGm2Ub(_get_base(dummy), _get_base(shmem_data_tile), non_shmem_data_type, atomic_type)

@experimental
def shmem_reduce(in_tensor: Tensor, shm_data: Tensor, dummy: Tensor, out: Tensor):
    return pypto_impl.ShmemReduce(_get_base(in_tensor), _get_base(shm_data), _get_base(dummy), _get_base(out))

@experimental
def shmem_signal(dummy: Tensor, shmem_signal_tile: Tensor, atomic_type: AtomicType = AtomicType.SET) -> Tensor:
    return pypto_impl.ShmemSignal(_get_base(dummy), _get_base(shmem_signal_tile), atomic_type)

@experimental
def wait_until(dummy_in: Tensor, shmem_signal_tile: Tensor, tile_count: int, expected_sum: int, reset_signal: bool = False) -> Tensor:
    return pypto_impl.WaitUntil(_get_base(dummy_in), _get_base(shmem_signal_tile), tile_count, expected_sum, reset_signal)
