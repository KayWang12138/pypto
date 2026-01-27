import os
import warnings
warnings.filterwarnings("ignore", message=".*owner does not match the current owner.*")
import time
import torch
import pypto
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict

ROUTED_EXPERT_NUM = int(os.environ.get("ROUTED_EXPERT_NUM", "160"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
RANDOM_ROUTING = int(os.environ.get("RANDOM_ROUTING", "0")) != 0
INCLUDE_CONVERSION = int(os.environ.get("INCLUDE_CONVERSION", "1")) != 0

# Global variables for distributed options
hccl_comm = 0
hccl_comm_name = ""
hccl_comm_dict: Dict[str, int] = {}
distributed_options = {"hccl_handle": [], "hccl_group_name": []}

def setup_distributed(rank, world_size):
    """Setup distributed environment for PyPto"""
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


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    idx = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def run_moe_dispatch_test(rank, world_size, warmup_rounds=3, test_rounds=20):
    """Run PyPto implementation with multi-turn testing, timing, and memory tracking"""
    
    # Parameters
    batch_size = BATCH_SIZE
    hidden_size = 5120
    top_k = 8
    routing_expert_num = ROUTED_EXPERT_NUM
    expert_num_per_rank = routing_expert_num // world_size
    
    print(f"\n{'='*80}")
    print(f"[Rank {rank}] Starting multi-turn MoE Dispatch test (PyPto only)")
    print(f"[Rank {rank}]   Warmup rounds: {warmup_rounds}")
    print(f"[Rank {rank}]   Test rounds: {test_rounds}")
    print(f"{'='*80}\n")
    
    # ====== Step 1: Setup distributed environment ======
    print(f"[Rank {rank}] Step 1: Setting up distributed environment...")
    hccl_comm, hccl_comm_name = setup_distributed(rank, world_size)
    
    # ====== Step 2: Define PyPto kernel ======
    print(f"[Rank {rank}] Step 2: Defining PyPto kernel...")
    
    @pypto.jit(distributed_options=distributed_options)
    def moe_dispatch_kernel(token_tensor: pypto.Tensor, token_expert_table: pypto.Tensor,
                            group_name: str, moe_config: pypto.distributed.MoeConfig,
                            expand_x: pypto.Tensor, valid_cnt: pypto.Tensor, combine_info: pypto.Tensor) -> None:
        pypto.distributed.shmem_moe_dispatch(token_tensor, token_expert_table, expand_x, valid_cnt, combine_info,
                                             group_name, moe_config)
    
    # ====== Step 3: Generate shared input data ======
    print(f"[Rank {rank}] Step 3: Generating shared input data...")
    torch.manual_seed(42)  # Use fixed seed for reproducibility
    
    # token_tensor: [batch_size, hidden_size].
    token_tensor = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")
    
    # token_expert_table: [batch_size, top_k].
    token_expert_table = torch.randint(0, routing_expert_num, (batch_size, top_k), dtype=torch.int32, device="npu")
    
    # Calculate output shapes
    if top_k * world_size < routing_expert_num:
        expand_x_rows = batch_size * top_k * world_size
    else:
        expand_x_rows = batch_size * routing_expert_num
    
    print(f"[Rank {rank}]   token_tensor shape: {token_tensor.shape}, dtype: {token_tensor.dtype}")
    print(f"[Rank {rank}]   token_expert_table shape: {token_expert_table.shape}, dtype: {token_expert_table.dtype}")
    print(f"[Rank {rank}]   expand_x_rows: {expand_x_rows}")
    
    # Setup MoeConfig
    moe_config = pypto.distributed.MoeConfig()
    moe_config.routedExpertNum = routing_expert_num
    moe_config.expertNumPerRank = expert_num_per_rank
    moe_config.rankNum = world_size
    
    # ====== Step 4: Run PyPto implementation ======
    print(f"[Rank {rank}] Step 4: Running PyPto implementation...")
    
    pypto_times = []
    pypto_mem_used = []
    pypto_mem_peak = []
    pypto_result = None
    
    # Prepare output tensors (reused across rounds)
    expand_x_data = torch.zeros(expand_x_rows, hidden_size, dtype=torch.bfloat16, device="npu")
    valid_cnt_data = torch.zeros(expert_num_per_rank, dtype=torch.int32, device="npu")
    combine_info_data = torch.zeros(expand_x_rows, 3, dtype=torch.int32, device="npu")

    def wrap_tensors():
        return (
            pypto.from_torch(token_tensor),
            pypto.from_torch(token_expert_table),
            pypto.from_torch(expand_x_data),
            pypto.from_torch(valid_cnt_data),
            pypto.from_torch(combine_info_data),
        )

    # if not INCLUDE_CONVERSION:
    #     token_pto, table_pto, expand_x_pto, valid_cnt_pto, combine_info_pto = wrap_tensors()
    
    # Warmup rounds
    for warmup_round in range(warmup_rounds):
        print(f"[Rank {rank}]   Warmup round {warmup_round + 1}/{warmup_rounds}...")
        # if RANDOM_ROUTING:
        token_expert_table.random_(0, routing_expert_num)
    # if INCLUDE_CONVERSION:
        token_pto, table_pto, expand_x_pto, valid_cnt_pto, combine_info_pto = wrap_tensors()
        # Reset output tensors
        expand_x_data.zero_()
        valid_cnt_data.zero_()
        combine_info_data.zero_()
        torch.npu.synchronize()
        
        moe_dispatch_kernel(token_pto, table_pto, hccl_comm_name, moe_config, expand_x_pto, valid_cnt_pto, combine_info_pto)
        torch.npu.synchronize()
    
    # Test rounds
    for test_round in range(test_rounds):
        print(f"[Rank {rank}]   Test round {test_round + 1}/{test_rounds}...")
        # if RANDOM_ROUTING:
        #     token_expert_table.random_(0, routing_expert_num)
        token_expert_table = torch.randint(0, routing_expert_num, (batch_size, top_k), dtype=torch.int32, device="npu")

        # Reset output tensors
        expand_x_data.zero_()
        valid_cnt_data.zero_()
        combine_info_data.zero_()
        torch.npu.synchronize()
        
        # Record memory before PyPto execution
        torch.npu.reset_peak_memory_stats()
        mem_before_pypto = torch.npu.memory_allocated()
        
        # Record time before PyPto execution
        time_start_pypto = time.perf_counter()
        # if INCLUDE_CONVERSION:
        token_pto, table_pto, expand_x_pto, valid_cnt_pto, combine_info_pto = wrap_tensors()
        
        moe_dispatch_kernel(token_pto, table_pto, hccl_comm_name, moe_config, expand_x_pto, valid_cnt_pto, combine_info_pto)
        torch.npu.synchronize()
        
        # Record time after PyPto execution
        time_end_pypto = time.perf_counter()
        time_pypto = time_end_pypto - time_start_pypto
        
        # Record memory after PyPto execution
        mem_after_pypto = torch.npu.memory_allocated()
        mem_peak_pypto = torch.npu.max_memory_allocated()
        mem_used_pypto = mem_after_pypto - mem_before_pypto
        
        # Store results
        pypto_times.append(time_pypto)
        pypto_mem_used.append(mem_used_pypto)
        pypto_mem_peak.append(mem_peak_pypto)
        if pypto_result is None:
            pypto_result = (expand_x_data.clone(), valid_cnt_data.clone(), combine_info_data.clone())
    
    # Calculate average performance
    avg_time_pypto = sum(pypto_times) / len(pypto_times)
    avg_mem_used_pypto = sum(pypto_mem_used) / len(pypto_mem_used)
    avg_mem_peak_pypto = sum(pypto_mem_peak) / len(pypto_mem_peak)
    
    # Use results from first test round for verification
    expand_x_data, valid_cnt_data, combine_info_data = pypto_result
    
    print(f"[Rank {rank}]   PyPto results (averaged over {test_rounds} rounds):")
    print(f"[Rank {rank}]     expand_x: {expand_x_data.shape}")
    print(f"[Rank {rank}]     valid_cnt: {valid_cnt_data.shape}, sum: {valid_cnt_data.sum().item()}")
    print(f"[Rank {rank}]     combine_info: {combine_info_data.shape}")
    sorted_times = sorted(pypto_times)
    print(f"[Rank {rank}]   sorted_times: {sorted_times}")
    p50 = _percentile(sorted_times, 50)
    p90 = _percentile(sorted_times, 90)
    p99 = _percentile(sorted_times, 99)
    print(f"[Rank {rank}]   PyPto performance:")
    print(f"[Rank {rank}]     Execution time: {avg_time_pypto*1000:.3f} ms "
          f"(p50: {p50*1000:.3f}, p90: {p90*1000:.3f}, p99: {p99*1000:.3f}, "
          f"min: {sorted_times[0]*1000:.3f}, max: {sorted_times[-1]*1000:.3f})")
    print(f"[Rank {rank}]     Memory allocated: {avg_mem_used_pypto / 1024**2:.2f} MB")
    print(f"[Rank {rank}]     Peak memory: {avg_mem_peak_pypto / 1024**2:.2f} MB")
    
    # ====== Step 5: Final verification ======
    print(f"[Rank {rank}] Step 5: Final verification...")
    local_valid_sum_pypto = valid_cnt_data.sum()
    
    dist.all_reduce(local_valid_sum_pypto, op=dist.ReduceOp.SUM)
    expected_total = batch_size * top_k * world_size
    
    print(f"[Rank {rank}]   Local valid_cnt sum: {valid_cnt_data.sum().item()}")
    print(f"[Rank {rank}]   Global valid_cnt sum: {local_valid_sum_pypto.item()}")
    print(f"[Rank {rank}]   Expected total: {expected_total}")
    
    if rank == 0:
        if local_valid_sum_pypto.item() == expected_total:
            print(f"[Rank {rank}] ✅ Global valid_cnt sum matches expected: {local_valid_sum_pypto.item()}")
        else:
            print(f"[Rank {rank}] ❌ Global valid_cnt sum mismatch: {local_valid_sum_pypto.item()} vs {expected_total}")
    
    # ====== Step 6: Performance summary ======
    print(f"[Rank {rank}] Step 6: Performance summary...")
    
    print(f"[Rank {rank}]   Performance Summary (averaged over {test_rounds} test rounds):")
    print(f"[Rank {rank}]     {'Metric':<30} {'Value':<20} {'P50':<15} {'P90':<15}")
    print(f"[Rank {rank}]     {'-'*80}")
    print(f"[Rank {rank}]     {'Execution Time (ms)':<30} {avg_time_pypto*1000:<20.3f} {p50*1000:<15.3f} {p90*1000:<15.3f}")
    print(f"[Rank {rank}]     {'Memory Allocated (MB)':<30} {avg_mem_used_pypto/1024**2:<20.2f} {min(pypto_mem_used)/1024**2:<15.2f} {max(pypto_mem_used)/1024**2:<15.2f}")
    print(f"[Rank {rank}]     {'Peak Memory (MB)':<30} {avg_mem_peak_pypto/1024**2:<20.2f} {min(pypto_mem_peak)/1024**2:<15.2f} {max(pypto_mem_peak)/1024**2:<15.2f}")
    
    # ====== Final result ======
    print(f"\n{'='*80}")
    print(f"[Rank {rank}] ✅ PyPto MoE Dispatch test completed!")
    print(f"{'='*80}\n")
    
    dist.barrier()
    dist.destroy_process_group()
    
    return True

if __name__ == "__main__":
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "4"))
    WARMUP_ROUNDS = int(os.environ.get("WARMUP_ROUNDS", "3"))
    TEST_ROUNDS = int(os.environ.get("TEST_ROUNDS", "20"))
    
    if not torch.npu.is_available():
        print("Error: NPU not available.")
        exit(1)

    if BATCH_SIZE != 8:
        print(f"Error: PyPTO MoeDispatch only supports BATCH_SIZE=8, got {BATCH_SIZE}.")
        exit(1)

    if ROUTED_EXPERT_NUM % WORLD_SIZE != 0:
        print(f"Error: ROUTED_EXPERT_NUM ({ROUTED_EXPERT_NUM}) must be divisible by WORLD_SIZE ({WORLD_SIZE}).")
        exit(1)

    if torch.npu.device_count() < WORLD_SIZE:
        print(f"Error: Need at least {WORLD_SIZE} NPUs, but only {torch.npu.device_count()} available.")
        exit(1)

    print(f"\n{'='*80}")
    print(f"Running MoE Dispatch Multi-turn Test (PyPto Only)")
    print(f"  World Size: {WORLD_SIZE}")
    print(f"  Warmup Rounds: {WARMUP_ROUNDS}")
    print(f"  Test Rounds: {TEST_ROUNDS}")
    print(f"{'='*80}\n")
    
    mp.spawn(run_moe_dispatch_test, args=(WORLD_SIZE, WARMUP_ROUNDS, TEST_ROUNDS), nprocs=WORLD_SIZE, join=True)
    
    print(f"\n{'='*80}")
    print(f"Multi-turn test completed")
    print(f"{'='*80}\n")
