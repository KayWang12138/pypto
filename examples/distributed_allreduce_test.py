import os
import warnings
warnings.filterwarnings("ignore", message=".*owner does not match the current owner.*")
import torch
import pypto
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

import torch.distributed._symmetric_memory as symm_mem

hccl_comm = 0
hccl_comm_name = ""
hccl_comm_dict = {}

distributed_options = {"hccl_handle": 0, "hccl_group_name": ""}

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

    distributed_options["hccl_handle"] = hccl_comm
    distributed_options["hccl_group_name"] = hccl_comm_name
    hccl_comm_dict[hccl_comm_name] = hccl_comm

    print(f"[Rank {rank}] hccl_handle: {hccl_comm}, hccl_group_name: {hccl_comm_name}")

    return hccl_comm, hccl_comm_name



@pypto.jit(distributed_options=distributed_options)
def all_reduce_kernel_v1(input_tensor: pypto.Tensor, output_tensor: pypto.Tensor, group_name: str, world_size: int) -> None:
    h, w = input_tensor.shape
    pypto.set_dist_tile_shapes([h // world_size, 1, h % world_size], [w // 1, 1, w % 1], [1, world_size, 0])
    pypto.distributed.two_shot_shmem_all_reduce(input_tensor, group_name, output_tensor)

@pypto.jit
def all_reduce_kernel_v2(input_tensor: pypto.Tensor, output_tensor: pypto.Tensor, group_name: str, world_size: int) -> None:
    h, w = input_tensor.shape
    pypto.set_distributed_options(hccl_handle=hccl_comm_dict[group_name], hccl_group_name=group_name)
    pypto.set_dist_tile_shapes([h // world_size, 1, h % world_size], [w // 1, 1, w % 1], [1, world_size, 0])
    pypto.distributed.two_shot_shmem_all_reduce(input_tensor, group_name, output_tensor)


def run_all_reduce_test(rank, world_size):
    
    hccl_comm, hccl_comm_name = setup_distributed(rank, world_size)

    shape = (1024, 1024)
    input_data = torch.ones(shape, dtype=torch.float32, device=f"npu") * (rank + 1)
    output_data = torch.zeros_like(input_data)

    # Convert to PyPto tensors
    input_pto = pypto.from_torch(input_data)
    output_pto = pypto.from_torch(output_data)

    all_reduce_kernel_v2(input_pto, output_pto, hccl_comm_name, world_size)
    
    # Verification
    expected_sum = sum(range(1, world_size + 1))
    expected_tensor = torch.ones(shape, dtype=torch.float32, device=f"npu:{rank}") * expected_sum
    
    if torch.allclose(output_data, expected_tensor, atol=1e-3):
        print(f"[Rank {rank}] SUCCESS! Output value: {output_data[0][0].item()} (Expected: {expected_sum})")
    else:
        print(f"[Rank {rank}] FAILED! Output value: {output_data[0][0].item()} (Expected: {expected_sum})")

if __name__ == "__main__":
    WORLD_SIZE = 2 # Example with 2 ranks
    # Check if NPU is available
    if not torch.npu.is_available():
        print("Error: NPU not available. This example requires NPU devices.")
        exit(1)
        
    if torch.npu.device_count() < WORLD_SIZE:
        print(f"Error: Need at least {WORLD_SIZE} NPUs, found {torch.npu.device_count()}")
        exit(1)

    print(f"Running AllReduce test with {WORLD_SIZE} processes...")
    
    mp.spawn(run_all_reduce_test, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

