import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*owner does not match the current owner.*")
if "SHMEM_HOME_PATH" not in os.environ:
    os.environ["SHMEM_HOME_PATH"] = "/usr/local/Ascend/shmem/latest"
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List, Dict
import pypto
import shmem as ash

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


def shmem_myinit(allocate_size: int):
    g_ash_size = 1 << 31 # 1GB
    g_malloc_size = allocate_size # 32MB
    G_IP_PORT = "tcp://127.0.0.1:19777"
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # TLS config
    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")
    # init
    attributes = ash.InitAttr()
    attributes.my_rank = rank
    attributes.n_ranks = world_size
    attributes.local_mem_size = g_ash_size
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    ret = ash.aclshmem_init(attributes)
    if ret != 0:
        raise ValueError('[ERROR] shmem_init failed')
    # alloc
    shmem_ptr = ash.aclshmem_malloc(g_malloc_size // 2)
    print(f'rank[{rank}]: shmem_ptr:{shmem_ptr} with type{type(shmem_ptr)}')
    # rank info
    my_pe, pe_count = ash.my_pe(), ash.pe_count()
    print(f'rank[{rank}]: my_pe:{my_pe} and pe_count:{pe_count}')
    if not (my_pe == rank and pe_count == world_size):
        raise ValueError('[ERROR] pe/world failed')
    # team info
    my_team_pe, team_pe_count = ash.team_my_pe(0), ash.team_n_pes(0)
    print(f'x: rank[{rank}]: t_my_pe:{my_team_pe} and t_pe_count:{team_pe_count}')

    return shmem_ptr

def npu_ptr_to_tensor(ptr, shape, dtype=torch.float32):
    # Build a tensor view from an NPU pointer.
    num_elements = 1
    for s in shape:
        num_elements *= s
    size_in_bytes = num_elements * dtype.itemsize

    storage = torch.UntypedStorage.from_ptr(
        ptr,
        size_in_bytes,
        device=torch.device(f'npu:{torch_npu.current_device()}')
    )

    return torch.as_tensor(storage, dtype=dtype).reshape(shape)


def compute(shmem_ptr):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _, hccl_group_name = pypto.distributed.init_hccl_comm_from_torch_pg(dist.group.WORLD)
    distributed_options = {"hccl_handle": [shmem_ptr], "hccl_group_name": [hccl_group_name]}

    @pypto.jit(distributed_options=distributed_options)
    def all_gather_kernel(
        input_tensor: pypto.Tensor,
        dummy_tensor: pypto.Tensor,
        result_holder: pypto.Tensor,
        group_name: str,
        world_size: int,
    ) -> None:
        h, w = input_tensor.shape
        tileNum1 = 8
        tileNum2 = 8

        pypto.set_vec_tile_shapes(h, w)
        pypto.set_dist_tile_shapes([h // tileNum1, tileNum1, h % tileNum1], [w // tileNum2, tileNum2, w % tileNum2], [1, world_size, 0])
        pypto.distributed.shmem_all_gather(input_tensor, dummy_tensor, group_name, result_holder)

    M, N = 512, 1024
    input_shape = (M, N)
    output_shape = (M * world_size, N)

    input_data = torch.randn(input_shape, dtype=torch.float32, device="npu")
    result_data = torch.zeros(output_shape, dtype=torch.float32, device="npu")
    dummy_data = torch.zeros((1, 1), dtype=torch.int32, device="npu")

    print(hex(input_data.data_ptr()))
    print(hex(result_data.data_ptr()))
    print(hex(dummy_data.data_ptr()))
    print(hex(shmem_ptr))


    input_pto = pypto.from_torch(input_data)
    result_pto = pypto.from_torch(result_data)
    dummy_pto = pypto.from_torch(dummy_data)

    # return
    print(f"[Rank {rank}] Starting AllGather kernel...")
    all_gather_kernel(input_pto, dummy_pto, result_pto, hccl_group_name, world_size)
    torch.npu.synchronize()

    _verify_result(input_data, result_data, world_size, rank, _get_verify_backend())


def shmem_myfinalize(shmem_ptr: int):
    # release and finalize
    if shmem_ptr is not None:
        ash.aclshmem_free(shmem_ptr)
    ash.aclshmem_finialize()



if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ["WORLD_SIZE"])
    verify_backend = _get_verify_backend()
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend=verify_backend, rank=rank, world_size=world_size)
    shmem_ptr = shmem_myinit(allocate_size=1 << 30)
    try:
        compute(shmem_ptr)
    except Exception as e:
        print(f"Error: {e}")
        # 如需排查 hang，可在此处加入 dist.barrier()
        shmem_myfinalize(shmem_ptr)
        dist.destroy_process_group()
        print("init_test.py running error!")
        raise e
    else:
        # 如需排查 teardown hang，可在此处加入 dist.barrier()
        shmem_myfinalize(shmem_ptr)
        dist.destroy_process_group()
        print("init_test.py running success!")
        sys.exit(0)
