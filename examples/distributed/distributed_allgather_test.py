import os
import warnings
warnings.filterwarnings("ignore", message=".*owner does not match the current owner.*")
import torch
import pypto
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    torch.npu.set_device(rank)

    pg = dist.group.WORLD
    hccl_comm, hccl_comm_name = pypto.distributed.init_hccl_comm_from_torch_pg(pg)
    print(f"[Rank {rank}] hccl_handle: {hccl_comm}, hccl_group_name: {hccl_comm_name}")

    distributed_options = {"hccl_handle": [hccl_comm], "hccl_group_name": [hccl_comm_name]}
    return hccl_comm_name, distributed_options


def run_all_gather_test(rank, world_size):

    hccl_comm_name, distributed_options = setup_distributed(rank, world_size)

    @pypto.jit(distributed_options=distributed_options)
    def all_gather_kernel(input_tensor: pypto.Tensor, dummy_tensor: pypto.Tensor,
                          result_holder: pypto.Tensor, group_name: str, world_size: int) -> None:
        h, w = input_tensor.shape
        tileNum1 = 8
        tileNum2 = 8

        pypto.set_vec_tile_shapes(h, w)
        pypto.set_dist_tile_shapes([h // tileNum1, tileNum1, h % tileNum1],
                                   [w // tileNum2, tileNum2, w % tileNum2],
                                   [1, world_size, 0])

        pypto.distributed.shmem_all_gather(input_tensor, dummy_tensor, group_name, result_holder)

    M, N = 512, 1024
    input_shape = (M, N)
    output_shape = (M * world_size, N)

    input_data = torch.randn(input_shape, dtype=torch.float32, device="npu")
    result_data = torch.zeros(output_shape, dtype=torch.float32, device="npu")
    dummy_data = torch.zeros((1, 1), dtype=torch.int32, device="npu")

    input_pto = pypto.from_torch(input_data)
    result_pto = pypto.from_torch(result_data)
    dummy_pto = pypto.from_torch(dummy_data)

    print(f"[Rank {rank}] Starting AllGather kernel...")
    all_gather_kernel(input_pto, dummy_pto, result_pto, hccl_comm_name, world_size)

    # Verification
    # Torch AllGather
    gather_list = [torch.zeros_like(input_data) for _ in range(world_size)]
    dist.all_gather(gather_list, input_data)
    expected_tensor = torch.cat(gather_list, dim=0)

    if torch.allclose(result_data, expected_tensor, atol=1e-3):
        print(f"[Rank {rank}] SUCCESS! Output matches ground truth.")
    else:
        print(f"[Rank {rank}] FAILED! Max diff: {(result_data - expected_tensor).abs().max().item()}")

if __name__ == "__main__":
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "2"))
    if not torch.npu.is_available():
        print("Error: NPU not available.")
        exit(1)

    if torch.npu.device_count() < WORLD_SIZE:
        print(f"Error: Need at least {WORLD_SIZE} NPUs, found {torch.npu.device_count()}")
        exit(1)

    print(f"Running AllGather test with {WORLD_SIZE} processes...")
    mp.spawn(run_all_gather_test, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
