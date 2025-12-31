import os
import warnings
warnings.filterwarnings("ignore", message=".*owner does not match the current owner.*")
import torch
import pypto
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Dict

hccl_comm = 0
hccl_comm_name = ""
hccl_comm_dict: Dict[str, int] = {}

distributed_options = {"hccl_handle": [], "hccl_group_name": []}

def setup_distributed(rank, world_size):
    global hccl_comm, hccl_comm_name, distributed_options, hccl_comm_dict
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    torch.npu.set_device(rank)
    
    pg = dist.group.WORLD
    backend = pg._get_backend(torch.device("npu"))

    hccl_comm = backend.get_hccl_comm(rank)
    hccl_comm_name = backend.get_hccl_comm_name(rank)

    distributed_options["hccl_handle"].append(hccl_comm)
    distributed_options["hccl_group_name"].append(hccl_comm_name)
    hccl_comm_dict[hccl_comm_name] = hccl_comm

    print(f"[Rank {rank}] hccl_handle: {hccl_comm}, hccl_group_name: {hccl_comm_name}")

    return hccl_comm, hccl_comm_name



@pypto.jit(distributed_options=distributed_options)
def all_reduce_kernel_v1(input_tensor: pypto.Tensor, output_tensor: pypto.Tensor, group_name: str, world_size: int) -> None:
    h, w = input_tensor.shape
    tileNum1 = 1
    tileNum2 = 1
    scatter_row = h // world_size
    pypto.set_dist_tile_shapes([scatter_row // tileNum1, tileNum1, scatter_row % tileNum1], [w // tileNum2, tileNum2, w % tileNum2], [1, world_size, 0])
    pypto.distributed.two_shot_shmem_all_reduce(input_tensor, group_name, output_tensor)

@pypto.jit
def all_reduce_kernel_v2(input_tensor: pypto.Tensor, output_tensor: pypto.Tensor, group_name: str, world_size: int) -> None:
    global distributed_options
    h, w = input_tensor.shape
    tileNum1 = 1
    tileNum2 = 1
    scatter_row = h // world_size
    pypto.set_distributed_options(hccl_handle=distributed_options["hccl_handle"], hccl_group_name=distributed_options["hccl_group_name"])
    pypto.set_dist_tile_shapes([scatter_row // tileNum1, tileNum1, scatter_row % tileNum1], [w // tileNum2, tileNum2, w % tileNum2], [1, world_size, 0])
    pypto.distributed.two_shot_shmem_all_reduce(input_tensor, group_name, output_tensor)

@pypto.jit
def all_reduce_kernel_v3(input_tensor: pypto.Tensor, output_tensor: pypto.Tensor, group_name: str, world_size: int) -> None:
    global distributed_options
    h, w = input_tensor.shape
    tileNum1 = 1
    tileNum2 = 1
    scatter_row = h
    pypto.set_distributed_options(hccl_handle=distributed_options["hccl_handle"], hccl_group_name=distributed_options["hccl_group_name"])
    pypto.set_dist_tile_shapes([scatter_row // tileNum1, tileNum1, scatter_row % tileNum1], [w // tileNum2, tileNum2, w % tileNum2], [1, world_size, 0])
    pypto.distributed.one_shot_shmem_all_reduce(input_tensor, group_name, output_tensor)

@pypto.jit
def all_reduce_kernel_two_shot(input_tensor: pypto.Tensor, output_tensor: pypto.Tensor, group_name: str, world_size: int) -> None:
    h, w = input_tensor.shape
    tileNum1 = 1
    tileNum2 = 1
    scatter_row = h // world_size
    pypto.set_distributed_options(hccl_handle=distributed_options["hccl_handle"], hccl_group_name=distributed_options["hccl_group_name"])
    pypto.set_dist_tile_shapes([scatter_row // tileNum1, tileNum1, scatter_row % tileNum1], [w // tileNum2, tileNum2, w % tileNum2], [1, world_size, 0])

    shmem_data_shape = [world_size, h // world_size, w]
    
    tile_count = 1
    shmem_signal_shape = [world_size, tile_count, 8]
    
    shmem_data_type = input_tensor.dtype
    if shmem_data_type == pypto.DataType.DT_BF16 or shmem_data_type == pypto.DataType.DT_FP16:
        shmem_data_type = pypto.DataType.DT_FP32
    
    shmem_data = pypto.create_shmem_tensor(world_size, group_name, shmem_data_type, shmem_data_shape)
    shmem_signal = pypto.create_shmem_tensor(world_size, group_name, pypto.DataType.DT_INT32, shmem_signal_shape)

    for dyn_rank_id in pypto.loop(0, world_size, 1, name="LOOP"):
        row_per_rank = h // world_size
        
        shmem_data_tile = pypto.view(shmem_data, [1, 1, row_per_rank, w], [dyn_rank_id, dyn_rank_id, 0, 0])
        shmem_signal_tile = pypto.view(shmem_signal, [1, 1, tile_count, 8], [dyn_rank_id, dyn_rank_id, 0, 0])
        in_tile = pypto.view(input_tensor, [row_per_rank, w], [row_per_rank * dyn_rank_id, 0])
        fake_barrier_dummy = pypto.tensor([1, 1], pypto.DataType.DT_INT32, "fakeBarrierDummy")

        dummy = pypto.shmem_put(in_tile, shmem_data_tile, fake_barrier_dummy, tile_count, pypto.AtomicType.ADD)
        dummy_signal = pypto.shmem_signal(dummy, shmem_signal_tile, pypto.AtomicType.ADD)
        
        dummy_local = pypto.wait_until(dummy_signal, shmem_signal_tile, tile_count, world_size)
        
        tmp = pypto.shmem_get(dummy_local, shmem_data_tile, input_tensor.dtype, pypto.AtomicType.SET)
        
        pypto.assemble(tmp, [row_per_rank * dyn_rank_id, 0], output_tensor)
    
    return
    
@pypto.jit
def all_reduce_kernel_one_shot(input_tensor: pypto.Tensor, output_tensor: pypto.Tensor, group_name: str, world_size: int) -> None:
    h, w = input_tensor.shape
    tileNum1 = 1
    tileNum2 = 1
    scatter_row = h
    pypto.set_distributed_options(hccl_handle=distributed_options["hccl_handle"], hccl_group_name=distributed_options["hccl_group_name"])
    pypto.set_dist_tile_shapes([scatter_row // tileNum1, tileNum1, scatter_row % tileNum1], [w // tileNum2, tileNum2, w % tileNum2], [1, world_size, 0])

    # Align with C++ OneShotShmemAllReduce: per-rank slice is size 1
    shmem_data_shape = [1, h, w]
    
    tile_count = 1
    shmem_signal_shape = [1, tile_count, 8]
    
    shmem_data_type = input_tensor.dtype
    if shmem_data_type == pypto.DataType.DT_BF16 or shmem_data_type == pypto.DataType.DT_FP16:
        shmem_data_type = pypto.DataType.DT_FP32
    
    shmem_data = pypto.create_shmem_tensor(world_size, group_name, shmem_data_type, shmem_data_shape)
    shmem_signal = pypto.create_shmem_tensor(world_size, group_name, pypto.DataType.DT_INT32, shmem_signal_shape)

    this_rank = pypto.get_dist_rank_id()

    for dyn_rank_id in pypto.loop(0, world_size, 1, name="LOOP_ONESHOT"):
        row_per_rank = h
        
        shmem_data_tile = pypto.view(shmem_data, [1, 1, row_per_rank, w], [dyn_rank_id, 0, 0, 0])
        shmem_signal_tile = pypto.view(shmem_signal, [1, 1, tile_count, 8], [dyn_rank_id, 0, 0, 0])
        fake_barrier_dummy = pypto.tensor([1, 1], pypto.DataType.DT_INT32, "fakeBarrierDummy")

        dummy = pypto.shmem_put(input_tensor, shmem_data_tile, fake_barrier_dummy, tile_count, pypto.AtomicType.ADD)
        dummy_signal = pypto.shmem_signal(dummy, shmem_signal_tile, pypto.AtomicType.ADD)
        
        if pypto.cond(this_rank == dyn_rank_id):
            dummy_local = pypto.wait_until(dummy_signal, shmem_signal_tile, tile_count, world_size)
            tmp = pypto.shmem_get(dummy_local, shmem_data_tile, input_tensor.dtype, pypto.AtomicType.SET)
            pypto.assemble(tmp, [0, 0], output_tensor)
    
    return



def run_all_reduce_test(rank, world_size):
    
    hccl_comm, hccl_comm_name = setup_distributed(rank, world_size)

    pypto.set_dist_rank_id(rank)

    shape = (1024, 1024)
    input_data = torch.ones(shape, dtype=torch.float32, device=f"npu") * (rank + 1)
    output_data = torch.zeros_like(input_data)

    # Convert to PyPto tensors
    input_pto = pypto.from_torch(input_data)
    output_pto = pypto.from_torch(output_data)

    all_reduce_kernel_v1(input_pto, output_pto, hccl_comm_name, world_size)
    # all_reduce_kernel_two_shot(input_pto, output_pto, hccl_comm_name, world_size)
    #all_reduce_kernel_one_shot(input_pto, output_pto, hccl_comm_name, world_size)
    
    # Verification
    expected_sum = sum(range(1, world_size + 1))
    expected_tensor = torch.ones(shape, dtype=torch.float32, device=f"npu:{rank}") * expected_sum
    
    if torch.allclose(output_data, expected_tensor, atol=1e-3):
        print(f"[Rank {rank}] SUCCESS! Output value: {output_data[0][0].item()} (Expected: {expected_sum})")
    else:
        print(f"[Rank {rank}] FAILED! Output value: {output_data[0][0].item()} (Expected: {expected_sum})")

if __name__ == "__main__":
    WORLD_SIZE = 8
    mp.spawn(run_all_reduce_test, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
