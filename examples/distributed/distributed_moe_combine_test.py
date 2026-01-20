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
    hccl_comm, hccl_comm_name = pypto.distributed.init_hccl_comm_from_torch_pg(pg)

    distributed_options["hccl_handle"].append(hccl_comm)
    distributed_options["hccl_group_name"].append(hccl_comm_name)
    hccl_comm_dict[hccl_comm_name] = hccl_comm

    print(f"[Rank {rank}] hccl_handle: {hccl_comm}, hccl_group_name: {hccl_comm_name}")

    return hccl_comm, hccl_comm_name

@pypto.jit
def dispatch_kernel(token_tensor: pypto.Tensor, token_expert_table: pypto.Tensor,
                    group_name: str, moe_config: pypto.distributed.MoeConfig,
                    expand_x: pypto.Tensor, valid_cnt: pypto.Tensor, combine_info: pypto.Tensor) -> None:
    global distributed_options
    pypto.set_distributed_options(hccl_handle=distributed_options["hccl_handle"], hccl_group_name=distributed_options["hccl_group_name"])

    pypto.distributed.shmem_moe_dispatch(token_tensor, token_expert_table, expand_x, valid_cnt, combine_info, group_name, moe_config)

@pypto.jit
def combine_kernel(expert_out: pypto.Tensor, combine_info: pypto.Tensor, recv_counts: pypto.Tensor,
                   scale: pypto.Tensor, group_name: str, moe_config: pypto.distributed.MoeConfig,
                   combine_out: pypto.Tensor) -> None:
    global distributed_options
    pypto.set_distributed_options(hccl_handle=distributed_options["hccl_handle"], hccl_group_name=distributed_options["hccl_group_name"])

    pypto.distributed.shmem_moe_combine(expert_out, combine_info, recv_counts, scale, group_name,
                                        moe_config.rankNum, moe_config.routedExpertNum,
                                        combine_out)

def run_moe_combine_test(rank, world_size):
    hccl_comm, hccl_comm_name = setup_distributed(rank, world_size)

    # Parameters
    batch_size = 8
    hidden_size = 5120
    top_k = 8
    routing_expert_num = 160
    expert_num_per_rank = routing_expert_num // world_size

    moe_config = pypto.distributed.MoeConfig()
    moe_config.routedExpertNum = routing_expert_num
    moe_config.expertNumPerRank = expert_num_per_rank
    moe_config.rankNum = world_size

    # Input Data
    # token_tensor: [batch_size, hidden_size]
    # Use random data, but for verification of identity expert, let's use ones or specific pattern
    # If we use random data X, and Scale=1, TopK=2, Identity Expert:
    # Result should be X * 1 + X * 1 = 2X.
    token_tensor = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")

    # token_expert_table: [batch_size, top_k]
    token_expert_table = torch.randint(0, routing_expert_num, (batch_size, top_k), dtype=torch.int32, device="npu")

    # scale: [batch_size, top_k] - Weights for each selected expert
    # Set to all 1.0 for easy verification
    scale = torch.ones(batch_size, top_k, dtype=torch.float32, device="npu")

    # Holders for outputs
    if top_k * world_size < routing_expert_num:
        expand_x_rows = batch_size * top_k * world_size
    else:
        expand_x_rows = batch_size * routing_expert_num

    expand_x_data = torch.zeros(expand_x_rows, hidden_size, dtype=torch.bfloat16, device="npu")
    combine_info_data = torch.full((expand_x_rows, 3), -1, dtype=torch.int32, device="npu")
    valid_cnt_data = torch.zeros(expert_num_per_rank, dtype=torch.int32, device="npu")


    # Final output holder
    combine_out_data = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")

    # Convert to PyPto tensors
    token_pto = pypto.from_torch(token_tensor)
    table_pto = pypto.from_torch(token_expert_table)
    scale_pto = pypto.from_torch(scale)
    combine_out_pto = pypto.from_torch(combine_out_data)
    expand_x_pto = pypto.from_torch(expand_x_data)
    combine_info_pto = pypto.from_torch(combine_info_data)
    valid_cnt_pto = pypto.from_torch(valid_cnt_data)

    print(f"[Rank {rank}] Running Dispatch...")
    dispatch_kernel(token_pto, table_pto, hccl_comm_name, moe_config, expand_x_pto, valid_cnt_pto, combine_info_pto)

    torch.npu.synchronize()
    combine_info_cpu = combine_info_data.cpu().numpy()
    local_counts = torch.zeros((world_size, batch_size), dtype=torch.int32)
    for rank_id, token_id, k_offset in combine_info_cpu:
        if rank_id < 0:
            continue
        if 0 <= rank_id < world_size and 0 <= token_id < batch_size and 0 <= k_offset < top_k:
            local_counts[rank_id, token_id] += 1
    local_counts_npu = local_counts.to("npu")
    dist.all_reduce(local_counts_npu, op=dist.ReduceOp.SUM)
    torch.npu.synchronize()
    recv_counts = local_counts_npu[rank].to(torch.int32)
    recv_counts_pto = pypto.from_torch(recv_counts)

    # Mock Compute: expert_out = expand_x (Identity)
    expert_out_pto = expand_x_pto

    print(f"[Rank {rank}] Running Combine...")
    combine_kernel(expert_out_pto, combine_info_pto, recv_counts_pto, scale_pto,
                   hccl_comm_name, moe_config, combine_out_pto)

    gathered_expand = [torch.empty_like(expand_x_data) for _ in range(world_size)]
    gathered_info = [torch.empty_like(combine_info_data) for _ in range(world_size)]
    dist.all_gather(gathered_expand, expand_x_data)
    dist.all_gather(gathered_info, combine_info_data)
    torch.npu.synchronize()

    scale_cpu = scale.cpu()
    expected_out_fp32 = torch.zeros((batch_size, hidden_size), dtype=torch.float32)
    for src_rank in range(world_size):
        expand_x_cpu = gathered_expand[src_rank].cpu()
        combine_info_cpu = gathered_info[src_rank].cpu().numpy()
        for row_idx, (rank_id, token_id, k_offset) in enumerate(combine_info_cpu):
            if rank_id != rank:
                continue
            if token_id < 0 or token_id >= batch_size or k_offset < 0 or k_offset >= top_k:
                continue
            expected_out_fp32[token_id] += expand_x_cpu[row_idx].float() * scale_cpu[token_id, k_offset]
    expected_out = expected_out_fp32.to(token_tensor.dtype)

    # Allow error due to BF16 precision and accumulation.
    if torch.allclose(combine_out_data.cpu(), expected_out, rtol=1e-2, atol=1e-2):
        print(f"[Rank {rank}] SUCCESS! Output matches expected (Input * recv_counts).")
    else:
        diff = (combine_out_data.cpu() - expected_out).abs().max().item()
        print(f"[Rank {rank}] FAILED! Max diff: {diff}")
        print(f"Sample Output: {combine_out_data.cpu()[0, :5]}")
        print(f"Sample Expected: {expected_out[0, :5]}")

if __name__ == "__main__":
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "4"))
    if not torch.npu.is_available():
        print("Error: NPU not available.")
        exit(1)

    if torch.npu.device_count() < WORLD_SIZE:
        print(f"Error: Need at least {WORLD_SIZE} NPUs.")
        exit(1)

    print(f"Running Moe Dispatch+Combine test with {WORLD_SIZE} processes...")
    mp.spawn(run_moe_combine_test, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
