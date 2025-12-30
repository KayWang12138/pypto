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
def moe_dispatch_kernel(token_tensor: pypto.Tensor, token_expert_table: pypto.Tensor,
                        group_name: str, moe_config: pypto.distributed.MoeConfig,
                        expand_x: pypto.Tensor, valid_cnt: pypto.Tensor, combine_info: pypto.Tensor) -> None:
    global distributed_options
    
    pypto.set_distributed_options(hccl_handle=distributed_options["hccl_handle"], hccl_group_name=distributed_options["hccl_group_name"])
    
    # MoeDispatch returns 3 tensors: expandX, validCnt, combineInfo
    pypto.distributed.shmem_moe_dispatch(token_tensor, token_expert_table, expand_x, valid_cnt, combine_info, group_name, moe_config)

def run_moe_dispatch_test(rank, world_size):
    hccl_comm, hccl_comm_name = setup_distributed(rank, world_size)
    
    # Parameters from test_codegen_dispatch.cpp
    # Adjusted for small test
    batch_size = 8
    hidden_size = 5120
    top_k = 8
    routing_expert_num = 160
    expert_num_per_rank = routing_expert_num // world_size
    
    # Setup MoeConfig
    moe_config = pypto.distributed.MoeConfig()
    moe_config.routedExpertNum = routing_expert_num
    moe_config.expertNumPerRank = expert_num_per_rank
    moe_config.rankNum = world_size
    
    # Input Data
    # token_tensor: [batch_size, hidden_size]
    token_tensor = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")
    
    # token_expert_table: [batch_size, top_k]
    # Randomly assign experts to tokens
    token_expert_table = torch.randint(0, routing_expert_num, (batch_size, top_k), dtype=torch.int32, device="npu")
    
    # Calculate output shapes
    if top_k * world_size < routing_expert_num:
        expand_x_rows = batch_size * top_k * world_size
    else:
        expand_x_rows = batch_size * routing_expert_num
    
    expand_x_data = torch.zeros(expand_x_rows, hidden_size, dtype=torch.bfloat16, device="npu")
    valid_cnt_data = torch.zeros(expert_num_per_rank, dtype=torch.int32, device="npu")
    combine_info_data = torch.zeros(expand_x_rows, 3, dtype=torch.int32, device="npu") # [expandXRowShape, 3]
    
    # Convert to PyPto tensors
    token_pto = pypto.from_torch(token_tensor)
    table_pto = pypto.from_torch(token_expert_table)
    expand_x_pto = pypto.from_torch(expand_x_data)
    valid_cnt_pto = pypto.from_torch(valid_cnt_data)
    combine_info_pto = pypto.from_torch(combine_info_data)
    
    print(f"[Rank {rank}] Starting MoeDispatch kernel...")
    moe_dispatch_kernel(token_pto, table_pto, hccl_comm_name, moe_config, expand_x_pto, valid_cnt_pto, combine_info_pto)
    
    # Verification
    # 1. Check valid_cnt sum. It should be roughly batch_size * top_k / world_size * world_size
    # Actually, global sum of valid_cnt across all ranks should equal batch_size * top_k * world_size (total tokens dispatched)
    
    # Local valid count sum
    local_valid_sum = valid_cnt_data.sum()
    
    # Global valid count sum
    dist.all_reduce(local_valid_sum, op=dist.ReduceOp.SUM)
    
    expected_total_tokens = batch_size * top_k * world_size
    
    print(f"[Rank {rank}] Local Valid Sum: {valid_cnt_data.sum().item()}")
    
    if rank == 0:
        if local_valid_sum.item() == expected_total_tokens:
            print(f"[Rank {rank}] SUCCESS! Total dispatched tokens matches: {local_valid_sum.item()}")
        else:
            print(f"[Rank {rank}] FAILED! Total dispatched tokens: {local_valid_sum.item()}, Expected: {expected_total_tokens}")

if __name__ == "__main__":
    WORLD_SIZE = 4
    if not torch.npu.is_available():
        print("Error: NPU not available.")
        exit(1)
    
    if torch.npu.device_count() < WORLD_SIZE:
        print(f"Error: Need at least {WORLD_SIZE} NPUs.")
        exit(1)

    print(f"Running MoeDispatch test with {WORLD_SIZE} processes...")
    mp.spawn(run_moe_dispatch_test, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

