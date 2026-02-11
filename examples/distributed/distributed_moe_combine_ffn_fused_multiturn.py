import os
import time
import warnings
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import pypto

warnings.filterwarnings("ignore", message=".*owner does not match the current owner.*")

hccl_comm = 0
hccl_comm_name = ""
dispatch_group_name = ""
combine_group_name = ""
hccl_comm_dict: Dict[str, int] = {}

distributed_options = {"hccl_handle": [], "hccl_group_name": []}


def setup_distributed(rank: int, world_size: int) -> Tuple[int, str]:
    global hccl_comm, hccl_comm_name, dispatch_group_name, combine_group_name, distributed_options, hccl_comm_dict
    os.environ["RANK"] = str(rank)
    os.environ["RANK_ID"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    torch.npu.set_device(rank)

    pg = dist.group.WORLD
    use_shmem_group = os.environ.get("USE_SHMEM_GROUP", "0") == "1"
    if use_shmem_group:
        shmem_group_name = os.environ.get("SHMEM_GROUP_NAME", "shmem_group")
        hccl_comm = 0
        hccl_comm_name = shmem_group_name
        dispatch_group_name = f"{shmem_group_name}_dispatch"
        combine_group_name = f"{shmem_group_name}_combine"
        distributed_options["hccl_handle"] = [hccl_comm, hccl_comm]
        distributed_options["hccl_group_name"] = [dispatch_group_name, combine_group_name]
        hccl_comm_dict[dispatch_group_name] = hccl_comm
        hccl_comm_dict[combine_group_name] = hccl_comm
        print(f"[Rank {rank}] using shmem groups: dispatch={dispatch_group_name}, combine={combine_group_name}")
        return hccl_comm, hccl_comm_name

    hccl_comm, hccl_comm_name = pypto.distributed.init_hccl_comm_from_torch_pg(pg)
    dispatch_group_name = hccl_comm_name
    combine_group_name = hccl_comm_name
    distributed_options["hccl_handle"] = [hccl_comm, hccl_comm]
    distributed_options["hccl_group_name"] = [dispatch_group_name, combine_group_name]
    hccl_comm_dict[dispatch_group_name] = hccl_comm
    hccl_comm_dict[combine_group_name] = hccl_comm

    print(f"[Rank {rank}] hccl_handle: {hccl_comm}, dispatch_group: {dispatch_group_name}, combine_group: {combine_group_name}")
    return hccl_comm, hccl_comm_name


@pypto.jit(distributed_options=distributed_options)
def dispatch_kernel(
    token_tensor: pypto.Tensor,
    token_expert_table: pypto.Tensor,
    group_name: str,
    moe_config: pypto.distributed.MoeConfig,
    expand_x: pypto.Tensor,
    valid_cnt: pypto.Tensor,
    combine_info: pypto.Tensor,
) -> None:
    pypto.distributed.shmem_moe_dispatch(
        token_tensor, token_expert_table, expand_x, valid_cnt, combine_info, group_name, moe_config
    )


@pypto.jit(distributed_options=distributed_options)
def fused_kernel(
    expert_out: pypto.Tensor,
    combine_info: pypto.Tensor,
    recv_counts: pypto.Tensor,
    scale: pypto.Tensor,
    ffn_weight: pypto.Tensor,
    group_name: str,
    moe_config: pypto.distributed.MoeConfig,
    fused_out: pypto.Tensor,
) -> None:
    pypto.distributed.shmem_moe_combine_ffn_fused(
        expert_out,
        combine_info,
        recv_counts,
        scale,
        ffn_weight,
        group_name,
        moe_config.rankNum,
        moe_config.routedExpertNum,
        fused_out,
    )


def build_recv_counts(
    combine_info: torch.Tensor,
    batch_size: int,
    top_k: int,
    world_size: int,
    rank: int,
) -> torch.Tensor:
    combine_info_cpu = combine_info.cpu().numpy()
    local_counts = torch.zeros((world_size, batch_size), dtype=torch.int32)
    for rank_id, token_id, k_offset in combine_info_cpu:
        if rank_id < 0:
            continue
        if 0 <= rank_id < world_size and 0 <= token_id < batch_size and 0 <= k_offset < top_k:
            local_counts[rank_id, token_id] += 1
    local_counts_npu = local_counts.to("npu")
    dist.all_reduce(local_counts_npu, op=dist.ReduceOp.SUM)
    torch.npu.synchronize()
    return local_counts_npu[rank].to(torch.int32)


def validate_combine_info(
    combine_info: torch.Tensor,
    batch_size: int,
    top_k: int,
    world_size: int,
) -> Tuple[int, int, int]:
    combine_info_cpu = combine_info.cpu().numpy()
    bad_rank = 0
    bad_token = 0
    bad_k = 0
    for rank_id, token_id, k_offset in combine_info_cpu:
        if rank_id < 0:
            continue
        if rank_id >= world_size:
            bad_rank += 1
        if token_id < 0 or token_id >= batch_size:
            bad_token += 1
        if k_offset < 0 or k_offset >= top_k:
            bad_k += 1
    return bad_rank, bad_token, bad_k


def torch_combine_only(
    expand_all: torch.Tensor,
    info_all: torch.Tensor,
    scale: torch.Tensor,
    rank: int,
    batch_size: int,
    top_k: int,
) -> torch.Tensor:
    rank_ids = info_all[:, 0]
    token_ids = info_all[:, 1]
    k_offsets = info_all[:, 2]
    mask = (
        (rank_ids == rank)
        & (token_ids >= 0)
        & (token_ids < batch_size)
        & (k_offsets >= 0)
        & (k_offsets < top_k)
    )

    out_fp32 = torch.zeros((batch_size, expand_all.shape[1]), device=expand_all.device, dtype=torch.float32)
    if mask.sum().item() == 0:
        return out_fp32

    token_ids = token_ids[mask].to(torch.int64)
    k_offsets = k_offsets[mask].to(torch.int64)
    rows = expand_all[mask].float()
    scale_vals = scale[token_ids, k_offsets]
    scaled = rows * scale_vals.unsqueeze(1)
    out_fp32.index_add_(0, token_ids, scaled)
    return out_fp32


def torch_ffn_only(
    combine_fp32: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    gate = torch.matmul(combine_fp32, gate_w.float())
    up = torch.matmul(combine_fp32, up_w.float())
    intermediate = torch.nn.functional.silu(gate) * up
    return torch.matmul(intermediate, down_w.float())


def run_moe_combine_ffn_multiturn(rank: int, world_size: int) -> None:
    setup_distributed(rank, world_size)
    torch.manual_seed(0)
    torch.npu.manual_seed_all(0)

    debug_validate = os.environ.get("DEBUG_VALIDATE_COMBINE_INFO", "0") == "1"
    skip_combine = os.environ.get("SKIP_COMBINE", "0") == "1"

    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    hidden_size = int(os.environ.get("HIDDEN_SIZE", "5120"))
    top_k = int(os.environ.get("TOP_K", "8"))
    routing_expert_num = int(os.environ.get("ROUTED_EXPERT_NUM", "160"))
    intermediate_size = int(os.environ.get("INTERMEDIATE_SIZE", "1024"))
    warmup_rounds = int(os.environ.get("WARMUP_ROUNDS", "1"))
    test_rounds = int(os.environ.get("TEST_ROUNDS", "3"))

    if batch_size not in (8, 256):
        raise ValueError("BATCH_SIZE must be 8 or 256 for the current MoE combine validation.")
    if hidden_size != 5120:
        raise ValueError("HIDDEN_SIZE must be 5120 for the current MoE combine validation.")
    if top_k != 8:
        raise ValueError("TOP_K must be 8 for the current MoE combine validation.")
    if routing_expert_num != 160:
        raise ValueError("ROUTED_EXPERT_NUM must be 160 for the current MoE combine validation.")
    if intermediate_size <= 0 or intermediate_size > 65535:
        raise ValueError("INTERMEDIATE_SIZE must be in (0, 65535].")
    if warmup_rounds < 0 or test_rounds <= 0:
        raise ValueError("WARMUP_ROUNDS must be >= 0 and TEST_ROUNDS must be > 0.")

    moe_config = pypto.distributed.MoeConfig()
    moe_config.routedExpertNum = routing_expert_num
    moe_config.expertNumPerRank = routing_expert_num // world_size
    moe_config.rankNum = world_size

    token_scale = float(os.environ.get("TOKEN_SCALE", "0.02"))
    token_tensor = torch.empty(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")
    token_expert_table = torch.empty(batch_size, top_k, dtype=torch.int32, device="npu")
    scale = torch.empty(batch_size, top_k, dtype=torch.float32, device="npu")

    def randomize_round_inputs() -> None:
        if rank == 0:
            token_fp32 = torch.randn(batch_size, hidden_size, dtype=torch.float32, device="npu") * token_scale
            token_tensor.copy_(token_fp32.to(torch.bfloat16))

            expert_table_cpu = torch.empty((batch_size, top_k), dtype=torch.int32)
            expert_table_cpu.random_(0, routing_expert_num)
            token_expert_table.copy_(expert_table_cpu.to("npu"))

            routing_logits = torch.randn(batch_size, top_k, dtype=torch.float32, device="npu")
            scale.copy_(torch.softmax(routing_logits, dim=-1))
        else:
            token_tensor.zero_()
            token_expert_table.zero_()
            scale.zero_()

        dist.broadcast(token_tensor, src=0)
        dist.broadcast(token_expert_table, src=0)
        dist.broadcast(scale, src=0)

    if top_k * world_size < routing_expert_num:
        expand_x_rows = batch_size * top_k * world_size
    else:
        expand_x_rows = batch_size * routing_expert_num

    expand_x_data = torch.zeros(expand_x_rows, hidden_size, dtype=torch.bfloat16, device="npu")
    combine_info_data = torch.full((expand_x_rows, 3), -1, dtype=torch.int32, device="npu")
    valid_cnt_data = torch.zeros(moe_config.expertNumPerRank, dtype=torch.int32, device="npu")

    def reset_buffers() -> None:
        expand_x_data.zero_()
        combine_info_data.fill_(-1)
        valid_cnt_data.zero_()

    weight_scale = float(os.environ.get("FFN_WEIGHT_SCALE", "0.02"))
    gate_w = (torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device="npu") * weight_scale).contiguous()
    up_w = (torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device="npu") * weight_scale).contiguous()
    down_w = (torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device="npu") * weight_scale).contiguous()
    ffn_weight_flat = torch.cat([gate_w.view(-1), up_w.view(-1), down_w.view(-1)], dim=0).contiguous()
    ffn_weight_pto = pypto.from_torch(ffn_weight_flat)

    fused_out_data = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")

    def wrap_inputs():
        return (
            pypto.from_torch(token_tensor),
            pypto.from_torch(token_expert_table),
            pypto.from_torch(expand_x_data),
            pypto.from_torch(valid_cnt_data),
            pypto.from_torch(combine_info_data),
        )

    def run_dispatch(token_pto, table_pto, expand_x_pto, valid_cnt_pto, combine_info_pto) -> None:
        dispatch_kernel(token_pto, table_pto, dispatch_group_name, moe_config, expand_x_pto, valid_cnt_pto, combine_info_pto)

    def run_fused(recv_counts_pto, scale_pto, expand_x_pto, combine_info_pto) -> None:
        fused_kernel(
            expand_x_pto,
            combine_info_pto,
            recv_counts_pto,
            scale_pto,
            ffn_weight_pto,
            combine_group_name,
            moe_config,
            pypto.from_torch(fused_out_data),
        )

    def gather_inputs() -> Tuple[torch.Tensor, torch.Tensor]:
        gathered_expand: List[torch.Tensor] = [torch.empty_like(expand_x_data) for _ in range(world_size)]
        gathered_info: List[torch.Tensor] = [torch.empty_like(combine_info_data) for _ in range(world_size)]
        dist.all_gather(gathered_expand, expand_x_data)
        dist.all_gather(gathered_info, combine_info_data)
        return torch.cat(gathered_expand, dim=0), torch.cat(gathered_info, dim=0)

    def run_torch_baseline(expand_all: torch.Tensor, info_all: torch.Tensor) -> None:
        combine_fp32 = torch_combine_only(expand_all, info_all, scale, rank, batch_size, top_k)
        _ = torch_ffn_only(combine_fp32, gate_w, up_w, down_w)

    if rank == 0:
        print(f"[Rank 0] Warmup rounds: {warmup_rounds}, Test rounds: {test_rounds}")

    for _ in range(warmup_rounds):
        dist.barrier()
        reset_buffers()
        torch.npu.synchronize()
        dist.barrier()

        randomize_round_inputs()
        token_pto, table_pto, expand_x_pto, valid_cnt_pto, combine_info_pto = wrap_inputs()
        run_dispatch(token_pto, table_pto, expand_x_pto, valid_cnt_pto, combine_info_pto)
        torch.npu.synchronize()
        dist.barrier()

        if debug_validate and rank == 0:
            bad_rank, bad_token, bad_k = validate_combine_info(combine_info_data, batch_size, top_k, world_size)
            print(f"[Debug][Rank 0] Warmup invalid combine_info -> rank:{bad_rank} token:{bad_token} k:{bad_k}")

        recv_counts_round = build_recv_counts(combine_info_data, batch_size, top_k, world_size, rank)

        if skip_combine:
            dist.barrier()
            continue

        recv_counts_pto = pypto.from_torch(recv_counts_round)
        scale_pto = pypto.from_torch(scale)
        run_fused(recv_counts_pto, scale_pto, expand_x_pto, combine_info_pto)
        torch.npu.synchronize()
        dist.barrier()

    pypto_times: List[float] = []
    torch_times: List[float] = []
    for _ in range(test_rounds):
        dist.barrier()
        reset_buffers()
        torch.npu.synchronize()
        dist.barrier()

        randomize_round_inputs()
        token_pto, table_pto, expand_x_pto, valid_cnt_pto, combine_info_pto = wrap_inputs()
        run_dispatch(token_pto, table_pto, expand_x_pto, valid_cnt_pto, combine_info_pto)
        torch.npu.synchronize()
        dist.barrier()

        if debug_validate and rank == 0:
            bad_rank, bad_token, bad_k = validate_combine_info(combine_info_data, batch_size, top_k, world_size)
            print(f"[Debug][Rank 0] Test invalid combine_info -> rank:{bad_rank} token:{bad_token} k:{bad_k}")

        recv_counts_round = build_recv_counts(combine_info_data, batch_size, top_k, world_size, rank)
        recv_counts_pto = pypto.from_torch(recv_counts_round)
        scale_pto = pypto.from_torch(scale)

        if skip_combine:
            dist.barrier()
            continue

        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        run_fused(recv_counts_pto, scale_pto, expand_x_pto, combine_info_pto)
        torch.npu.synchronize()
        dist.barrier()
        pypto_times.append((time.perf_counter() - start) * 1000.0)

        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        expand_all, info_all = gather_inputs()
        run_torch_baseline(expand_all, info_all)
        torch.npu.synchronize()
        dist.barrier()
        torch_times.append((time.perf_counter() - start) * 1000.0)

    if skip_combine:
        if rank == 0:
            print("[Rank 0] SKIP_COMBINE=1, dispatch-only validation finished.")
        return

    expand_all, info_all = gather_inputs()
    torch_combine_fp32 = torch_combine_only(expand_all, info_all, scale, rank, batch_size, top_k)
    torch_out = torch_ffn_only(torch_combine_fp32, gate_w, up_w, down_w).to(torch.bfloat16)
    diff = (fused_out_data - torch_out).abs().max().item()
    if torch.allclose(fused_out_data, torch_out, rtol=2e-2, atol=2e-2):
        print(f"[Rank {rank}] SUCCESS! Fused output matches torch baseline. Max diff: {diff}")
    else:
        nan_count = int(torch.isnan(fused_out_data).sum().item())
        inf_count = int(torch.isinf(fused_out_data).sum().item())
        print(f"[Rank {rank}] FAILED! Max diff: {diff}, nan_count: {nan_count}, inf_count: {inf_count}")
        if nan_count > 0:
            bad_rows = torch.isnan(fused_out_data).any(dim=1).nonzero(as_tuple=False).view(-1).cpu().tolist()
            print(f"[Rank {rank}] NaN rows: {bad_rows}")

    if rank == 0:
        pypto_avg = sum(pypto_times) / len(pypto_times)
        torch_avg = sum(torch_times) / len(torch_times)
        speedup = torch_avg / pypto_avg if pypto_avg > 0 else 0.0
        print(f"[Rank 0] PyPTO avg: {pypto_avg:.3f} ms, Torch avg: {torch_avg:.3f} ms, Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    def _pick_free_port() -> int:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    os.environ.setdefault("MASTER_PORT", str(_pick_free_port()))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "4"))

    if not torch.npu.is_available():
        print("Error: NPU not available.")
        raise SystemExit(1)

    if torch.npu.device_count() < WORLD_SIZE:
        print(f"Error: Need at least {WORLD_SIZE} NPUs.")
        raise SystemExit(1)

    print(f"Running Moe Combine+FFN multi-turn test with {WORLD_SIZE} processes...")
    mp.spawn(run_moe_combine_ffn_multiturn, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
