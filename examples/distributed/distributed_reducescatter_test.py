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

@pypto.jit
def reduce_scatter_kernel(input_tensor: pypto.Tensor, output_tensor: pypto.Tensor, group_name: str, world_size: int) -> None:
    global distributed_options
    h, w = input_tensor.shape
    
    tileNum1 = 2
    tileNum2 = 2
    scatter_row = h // world_size

    pypto.set_distributed_options(hccl_handle=distributed_options["hccl_handle"], hccl_group_name=distributed_options["hccl_group_name"])
    
    pypto.set_dist_tile_shapes([scatter_row // tileNum1, tileNum1, scatter_row % tileNum1], [w // tileNum2, tileNum2, w % tileNum2], [1, world_size, 0])
    
    reduce_type = pypto.distributed.DistReduceType.DIST_REDUCE_ADD
    pypto.distributed.shmem_reduce_scatter(input_tensor, group_name, reduce_type, output_tensor)

def run_reduce_scatter_test(rank, world_size):
    
    hccl_comm, hccl_comm_name = setup_distributed(rank, world_size)

    # Input Shape: [1024, 1024]
    # Output Shape: [1024 / WORLD_SIZE, 1024]
    M, N = 1024, 1024
    assert M % world_size == 0, "M must be divisible by world_size for this test"
    
    input_shape = (M, N)
    output_shape = (M // world_size, N)
    
    # Rank 0: all 1s, Rank 1: all 2s ...
    # input_data = torch.ones(input_shape, dtype=torch.float32, device="npu") * (rank + 1)
    input_data = torch.randn(input_shape, dtype=torch.float32, device="npu")
    output_data = torch.zeros(output_shape, dtype=torch.float32, device="npu")

    # Convert to PyPto tensors
    input_pto = pypto.from_torch(input_data)
    output_pto = pypto.from_torch(output_data)

    print(f"[Rank {rank}] Starting ReduceScatter kernel...")
    reduce_scatter_kernel(input_pto, output_pto, hccl_comm_name, world_size)
    
    # Verification
    # Use torch.distributed to calculate ground truth
    torch_input = input_data.clone()
    dist.all_reduce(torch_input, op=dist.ReduceOp.SUM)
    
    # Expected output is a slice of the global sum corresponding to this rank
    slice_h = M // world_size
    start_idx = rank * slice_h
    end_idx = start_idx + slice_h
    expected_tensor = torch_input[start_idx:end_idx, :]
    
    if torch.allclose(output_data, expected_tensor, atol=1e-3):
        print(f"[Rank {rank}] SUCCESS! Output matches ground truth.")
    else:
        print(f"[Rank {rank}] FAILED! Max diff: {(output_data - expected_tensor).abs().max().item()}")

if __name__ == "__main__":
    WORLD_SIZE = 4 # Example with 2 ranks
    # Check if NPU is available
    if not torch.npu.is_available():
        print("Error: NPU not available. This example requires NPU devices.")
        exit(1)
        
    if torch.npu.device_count() < WORLD_SIZE:
        print(f"Error: Need at least {WORLD_SIZE} NPUs, found {torch.npu.device_count()}")
        exit(1)

    print(f"Running ReduceScatter test with {WORLD_SIZE} processes...")
    
    mp.spawn(run_reduce_scatter_test, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
