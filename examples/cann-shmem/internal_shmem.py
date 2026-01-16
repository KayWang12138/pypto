import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*owner does not match the current owner.*")
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List, Dict
import pypto

def _get_verify_backend():
    backend = os.environ.get("PYPTO_SHMEM_VERIFY_BACKEND", "hccl").lower()
    if backend not in ("gloo", "hccl"):
        backend = "gloo"
    return backend


def _verify_result(input_data, result_data, world_size, rank, backend):
    if backend == "hccl":
        gather_list = [torch.zeros_like(input_data) for _ in range(world_size)]
        dist.all_gather(gather_list, input_data)
        expected_tensor = torch.cat(gather_list, dim=0)
        if torch.allclose(result_data, expected_tensor, atol=1e-3):
            print(f"[Rank {rank}] SUCCESS! Output matches ground truth.")
        else:
            print(f"[Rank {rank}] FAILED! Max diff: {(result_data - expected_tensor).abs().max().item()}")
        return

    input_cpu = input_data.cpu()
    gather_list = [torch.zeros_like(input_cpu) for _ in range(world_size)]
    dist.all_gather(gather_list, input_cpu)
    expected_cpu = torch.cat(gather_list, dim=0)
    result_cpu = result_data.cpu()
    if torch.allclose(result_cpu, expected_cpu, atol=1e-3):
        print(f"[Rank {rank}] SUCCESS! Output matches ground truth.")
    else:
        print(f"[Rank {rank}] FAILED! Max diff: {(result_cpu - expected_cpu).abs().max().item()}")


def compute(rank, world_size, verify_backend):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    import pypto

    @pypto.jit
    def all_gather_kernel(input_tensor: pypto.Tensor, dummy_tensor: pypto.Tensor, result_holder: pypto.Tensor, world_size: int) -> None:
        h, w = input_tensor.shape
        tileNum1 = 8
        tileNum2 = 8
        pypto.set_distributed_options(hccl_handle=[rank], hccl_group_name=["shmem_group"])

        pypto.set_vec_tile_shapes(h, w)
        pypto.set_dist_tile_shapes([h // tileNum1, tileNum1, h % tileNum1], [w // tileNum2, tileNum2, w % tileNum2], [1, world_size, 0])
        pypto.distributed.shmem_all_gather(input_tensor, dummy_tensor, "shmem_group", result_holder)

    M, N = 512, 1024
    input_shape = (M, N)
    output_shape = (M * world_size, N)

    input_data = torch.randn(input_shape, dtype=torch.float32, device="npu")
    result_data = torch.zeros(output_shape, dtype=torch.float32, device="npu")
    dummy_data = torch.zeros((1, 1), dtype=torch.int32, device="npu")

    print(hex(input_data.data_ptr()))
    print(hex(result_data.data_ptr()))
    print(hex(dummy_data.data_ptr()))


    input_pto = pypto.from_torch(input_data)
    result_pto = pypto.from_torch(result_data)
    dummy_pto = pypto.from_torch(dummy_data)

    # return
    print(f"[Rank {rank}] Starting AllGather kernel...")
    all_gather_kernel(input_pto, dummy_pto, result_pto, world_size)

    print(input_data[0:2, 0:2], result_data[0:2, 0:2])
    _verify_result(input_data, result_data, world_size, rank, verify_backend)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ["WORLD_SIZE"])
    verify_backend = _get_verify_backend()
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend=verify_backend, rank=rank, world_size=world_size)
    try:
        compute(local_rank, world_size, verify_backend)
    except Exception as e:
        print(f"Error: {e}")
        dist.destroy_process_group()
        print("init_test.py running error!")
        raise e
    else:
        dist.destroy_process_group()
        print("init_test.py running success!")
        sys.exit(0)
