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
hccl_comm_dict: Dict[str, int] = {}

distributed_options = {"hccl_handle": [], "hccl_group_name": []}


def setup_distributed(rank: int, world_size: int) -> Tuple[int, str]:
    global hccl_comm, hccl_comm_name, distributed_options, hccl_comm_dict
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
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
    pypto.set_distributed_options(
        hccl_handle=distributed_options["hccl_handle"],
        hccl_group_name=distributed_options["hccl_group_name"],
    )
    pypto.distributed.shmem_moe_dispatch(
        token_tensor, token_expert_table, expand_x, valid_cnt, combine_info, group_name, moe_config
    )


@pypto.jit
def combine_ffn_fused_kernel(expert_out: pypto.Tensor, combine_info: pypto.Tensor, recv_counts: pypto.Tensor,
                             scale: pypto.Tensor, ffn_weight: pypto.Tensor, group_name: str,
                             moe_config: pypto.distributed.MoeConfig, out: pypto.Tensor) -> None:
    global distributed_options
    pypto.set_distributed_options(
        hccl_handle=distributed_options["hccl_handle"],
        hccl_group_name=distributed_options["hccl_group_name"],
    )
    pypto.distributed.shmem_moe_combine_ffn_fused(
        expert_out, combine_info, recv_counts, scale, ffn_weight,
        group_name, moe_config.rankNum, moe_config.routedExpertNum, out
    )


@pypto.jit
def combine_kernel(expert_out: pypto.Tensor, combine_info: pypto.Tensor, recv_counts: pypto.Tensor,
                   scale: pypto.Tensor, group_name: str, moe_config: pypto.distributed.MoeConfig,
                   combine_out: pypto.Tensor) -> None:
    global distributed_options
    pypto.set_distributed_options(
        hccl_handle=distributed_options["hccl_handle"],
        hccl_group_name=distributed_options["hccl_group_name"],
    )
    pypto.distributed.shmem_moe_combine(
        expert_out, combine_info, recv_counts, scale, group_name,
        moe_config.rankNum, moe_config.routedExpertNum, combine_out
    )



def build_recv_counts(combine_info: torch.Tensor, batch_size: int, top_k: int, world_size: int, rank: int) -> torch.Tensor:
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


def torch_combine_and_ffn(expand_all: torch.Tensor, info_all: torch.Tensor, scale: torch.Tensor, rank: int,
                          batch_size: int, top_k: int, gate_w: torch.Tensor,
                          up_w: torch.Tensor, down_w: torch.Tensor,
                          use_scatter: bool) -> torch.Tensor:
    combine_fp32 = torch_combine_only(expand_all, info_all, scale, rank, batch_size, top_k, use_scatter)
    out_fp32 = torch_ffn_only(combine_fp32, gate_w, up_w, down_w)
    return out_fp32.to(torch.bfloat16)


def torch_combine_only(expand_all: torch.Tensor, info_all: torch.Tensor, scale: torch.Tensor, rank: int,
                       batch_size: int, top_k: int, use_scatter: bool) -> torch.Tensor:
    rank_ids = info_all[:, 0]
    token_ids = info_all[:, 1]
    k_offsets = info_all[:, 2]
    mask = (
        (rank_ids == rank) &
        (token_ids >= 0) & (token_ids < batch_size) &
        (k_offsets >= 0) & (k_offsets < top_k)
    )
    if mask.sum().item() == 0:
        return torch.zeros((batch_size, expand_all.shape[1]), device=expand_all.device, dtype=torch.float32)

    token_ids = token_ids[mask].to(torch.int64)
    k_offsets = k_offsets[mask].to(torch.int64)
    rows = expand_all[mask].float()
    scale_vals = scale[token_ids, k_offsets]
    scaled = rows * scale_vals.unsqueeze(1)

    out_fp32 = torch.zeros((batch_size, rows.shape[1]), device=expand_all.device, dtype=torch.float32)
    if use_scatter:
        index = token_ids.unsqueeze(1).expand(-1, rows.shape[1])
        out_fp32.scatter_add_(0, index, scaled)
    else:
        out_fp32.index_add_(0, token_ids, scaled)
    return out_fp32


def torch_ffn_only(combine_fp32: torch.Tensor, gate_w: torch.Tensor,
                   up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    gate = torch.matmul(combine_fp32, gate_w.float())
    up = torch.matmul(combine_fp32, up_w.float())
    intermediate = torch.nn.functional.silu(gate) * up
    out_fp32 = torch.matmul(intermediate, down_w.float())
    return out_fp32


def run_moe_combine_ffn_fused_test(rank: int, world_size: int) -> None:
    setup_distributed(rank, world_size)
    torch.manual_seed(0)
    torch.npu.manual_seed_all(0)

    # USE_TORCH_FFN=1 runs FFN with torch (stable path). USE_TORCH_FFN=0 uses shmem_moe_combine_ffn_fused.
    # The shmem combine path is not re-entrant, so avoid calling it multiple times per run.
    use_torch_ffn = os.environ.get("USE_TORCH_FFN", "1") == "1"

    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    hidden_size = int(os.environ.get("HIDDEN_SIZE", "5120"))
    top_k = int(os.environ.get("TOP_K", "8"))
    routing_expert_num = int(os.environ.get("ROUTED_EXPERT_NUM", "160"))
    intermediate_size = int(os.environ.get("INTERMEDIATE_SIZE", "1024"))
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

    moe_config = pypto.distributed.MoeConfig()
    moe_config.routedExpertNum = routing_expert_num
    moe_config.expertNumPerRank = routing_expert_num // world_size
    moe_config.rankNum = world_size

    token_tensor = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")
    token_expert_table = torch.randint(0, routing_expert_num, (batch_size, top_k), dtype=torch.int32, device="npu")
    scale = torch.ones(batch_size, top_k, dtype=torch.float32, device="npu")

    if top_k * world_size < routing_expert_num:
        expand_x_rows = batch_size * top_k * world_size
    else:
        expand_x_rows = batch_size * routing_expert_num

    expand_x_data = torch.zeros(expand_x_rows, hidden_size, dtype=torch.bfloat16, device="npu")
    combine_info_data = torch.full((expand_x_rows, 3), -1, dtype=torch.int32, device="npu")
    valid_cnt_data = torch.zeros(moe_config.expertNumPerRank, dtype=torch.int32, device="npu")

    token_pto = pypto.from_torch(token_tensor)
    table_pto = pypto.from_torch(token_expert_table)
    scale_pto = pypto.from_torch(scale)
    expand_x_pto = pypto.from_torch(expand_x_data)
    combine_info_pto = pypto.from_torch(combine_info_data)
    valid_cnt_pto = pypto.from_torch(valid_cnt_data)

    gate_w = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device="npu").contiguous()
    up_w = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device="npu").contiguous()
    down_w = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device="npu").contiguous()
    ffn_weight_flat = torch.cat(
        [gate_w.view(-1), up_w.view(-1), down_w.view(-1)], dim=0
    ).contiguous()
    ffn_weight_pto = pypto.from_torch(ffn_weight_flat)

    print(f"[Rank {rank}] Running dispatch...")
    dispatch_kernel(token_pto, table_pto, hccl_comm_name, moe_config, expand_x_pto, valid_cnt_pto, combine_info_pto)
    torch.npu.synchronize()

    recv_counts = build_recv_counts(combine_info_data, batch_size, top_k, world_size, rank)
    recv_counts_pto = pypto.from_torch(recv_counts)

    combine_out_data = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")
    combine_out_pto = pypto.from_torch(combine_out_data)
    fused_out_data = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")
    fused_out_pto = pypto.from_torch(fused_out_data)

    def fused_step() -> None:
        if use_torch_ffn:
            torch_out_fp32 = torch_ffn_only(combine_out_data.float(), gate_w, up_w, down_w)
            fused_out_data.copy_(torch_out_fp32.to(torch.bfloat16))
        else:
            combine_ffn_fused_kernel(
                expand_x_pto, combine_info_pto, recv_counts_pto, scale_pto, ffn_weight_pto,
                hccl_comm_name, moe_config, fused_out_pto
            )

    combine_ms = None
    if use_torch_ffn:
        def combine_step() -> None:
            combine_kernel(
                expand_x_pto, combine_info_pto, recv_counts_pto, scale_pto, hccl_comm_name, moe_config,
                combine_out_pto
            )

        print(f"[Rank {rank}] Running combine only...")
        dist.barrier()
        torch.npu.synchronize()
        combine_start = time.perf_counter()
        combine_step()
        torch.npu.synchronize()
        combine_ms = (time.perf_counter() - combine_start) * 1000.0

    mode_label = "torch-ffn" if use_torch_ffn else "shmem-fused"
    print(f"[Rank {rank}] Running fused combine+ffn ({mode_label})...")
    torch.npu.synchronize()
    fused_start = time.perf_counter()
    fused_step()
    torch.npu.synchronize()
    fused_ms = (time.perf_counter() - fused_start) * 1000.0

    if use_torch_ffn and os.environ.get("DEBUG_FFN", "0") == "1":
        local_ref = torch_ffn_only(combine_out_data.float(), gate_w, up_w, down_w).to(torch.bfloat16)
        local_diff = (fused_out_data - local_ref).abs().max().item()
        print(f"[Rank {rank}] Local FFN diff: {local_diff}")

    gathered_expand: List[torch.Tensor] = [torch.empty_like(expand_x_data) for _ in range(world_size)]
    gathered_info: List[torch.Tensor] = [torch.empty_like(combine_info_data) for _ in range(world_size)]
    torch.npu.synchronize()
    gather_start = time.perf_counter()
    dist.all_gather(gathered_expand, expand_x_data)
    dist.all_gather(gathered_info, combine_info_data)
    torch.npu.synchronize()
    gather_ms = (time.perf_counter() - gather_start) * 1000.0

    expand_all = torch.cat(gathered_expand, dim=0)
    info_all = torch.cat(gathered_info, dim=0)

    force_scatter = os.environ.get("FORCE_SCATTER", "1") == "1"
    use_scatter = force_scatter
    if not force_scatter:
        try:
            tmp_out = torch.zeros((2, 4), device="npu", dtype=torch.float32)
            tmp_idx = torch.tensor([0, 1], device="npu", dtype=torch.int64)
            tmp_src = torch.ones((2, 4), device="npu", dtype=torch.float32)
            tmp_out.index_add_(0, tmp_idx, tmp_src)
        except RuntimeError:
            use_scatter = True

    torch.npu.synchronize()
    baseline_start = time.perf_counter()
    torch_combine_fp32 = torch_combine_only(
        expand_all, info_all, scale, rank, batch_size, top_k, use_scatter
    )
    torch_combine_bf16 = torch_combine_fp32.to(torch.bfloat16)
    torch_out_baseline = torch_ffn_only(torch_combine_fp32, gate_w, up_w, down_w).to(torch.bfloat16)
    torch.npu.synchronize()
    torch_compute_ms = (time.perf_counter() - baseline_start) * 1000.0
    if use_torch_ffn:
        if torch.allclose(combine_out_data, torch_combine_bf16, rtol=2e-2, atol=2e-2):
            print(f"[Rank {rank}] Combine output matches torch baseline.")
        else:
            combine_diff = (combine_out_data - torch_combine_bf16).abs().max().item()
            print(f"[Rank {rank}] Combine output mismatch. Max diff: {combine_diff}")
    else:
        print(f"[Rank {rank}] Combine-only check skipped for shmem-fused path.")

    if use_torch_ffn:
        torch_out = torch_ffn_only(combine_out_data.float(), gate_w, up_w, down_w).to(torch.bfloat16)
    else:
        torch_out = torch_ffn_only(torch_combine_bf16.float(), gate_w, up_w, down_w).to(torch.bfloat16)

    if torch.allclose(fused_out_data, torch_out, rtol=2e-2, atol=2e-2):
        print(f"[Rank {rank}] SUCCESS! Fused output matches torch baseline.")
    else:
        diff = (fused_out_data - torch_out).abs().max().item()
        print(f"[Rank {rank}] FAILED! Max diff: {diff}")

    if use_torch_ffn:
        combine_ms = combine_ms if combine_ms is not None else 0.0
        fused_total_ms = combine_ms + fused_ms
    else:
        fused_total_ms = fused_ms

    torch_ms = gather_ms + torch_compute_ms

    fused_avg = fused_total_ms
    torch_avg = torch_ms

    if rank == 0:
        speedup = torch_avg / fused_avg if fused_avg > 0 else 0.0
        if use_torch_ffn:
            combine_avg = combine_ms
            ffn_avg = fused_ms
            print(f"[Rank 0] Combine: {combine_avg:.3f} ms, FFN: {ffn_avg:.3f} ms, "
                  f"Total: {fused_avg:.3f} ms, Torch: {torch_avg:.3f} ms, Speedup: {speedup:.2f}x")
            print("[Rank 0] Note: timings are single-run and include first-run overhead; torch baseline includes all_gather.")
        else:
            print(f"[Rank 0] Fused: {fused_avg:.3f} ms, Torch: {torch_avg:.3f} ms, Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "4"))
    if not torch.npu.is_available():
        print("Error: NPU not available.")
        raise SystemExit(1)

    if torch.npu.device_count() < WORLD_SIZE:
        print(f"Error: Need at least {WORLD_SIZE} NPUs.")
        raise SystemExit(1)

    print(f"Running Moe Combine+FFN fused test with {WORLD_SIZE} processes...")
    mp.spawn(run_moe_combine_ffn_fused_test, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
