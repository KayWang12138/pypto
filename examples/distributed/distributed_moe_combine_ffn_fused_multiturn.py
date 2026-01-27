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


def _align_up(value: int, align: int = 16) -> int:
    return ((value + align - 1) // align) * align


def _get_chunk_size(batch_size: int) -> int:
    env_val = os.environ.get("PYPTO_MOE_FFN_CHUNK_SIZE", "").strip()
    if env_val:
        try:
            parsed = int(env_val)
            if parsed > 0:
                chunk = min(batch_size, parsed)
            else:
                chunk = batch_size
        except ValueError:
            chunk = batch_size
    else:
        chunk = min(batch_size, 128)
    if batch_size % chunk != 0:
        return batch_size
    return chunk


def setup_distributed(rank: int, world_size: int) -> Tuple[int, str]:
    global hccl_comm, hccl_comm_name, distributed_options, hccl_comm_dict
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
        distributed_options["hccl_handle"] = [hccl_comm]
        distributed_options["hccl_group_name"] = [hccl_comm_name]
        hccl_comm_dict[hccl_comm_name] = hccl_comm
        print(f"[Rank {rank}] using shmem_group: {hccl_comm_name}")
        return hccl_comm, hccl_comm_name

    hccl_comm, hccl_comm_name = pypto.distributed.init_hccl_comm_from_torch_pg(pg)

    distributed_options["hccl_handle"].append(hccl_comm)
    distributed_options["hccl_group_name"].append(hccl_comm_name)
    hccl_comm_dict[hccl_comm_name] = hccl_comm

    print(f"[Rank {rank}] hccl_handle: {hccl_comm}, hccl_group_name: {hccl_comm_name}")
    return hccl_comm, hccl_comm_name


@pypto.jit(distributed_options=distributed_options)
def dispatch_kernel(token_tensor: pypto.Tensor, token_expert_table: pypto.Tensor,
                    group_name: str, moe_config: pypto.distributed.MoeConfig,
                    expand_x: pypto.Tensor, valid_cnt: pypto.Tensor, combine_info: pypto.Tensor) -> None:
    pypto.distributed.shmem_moe_dispatch(
        token_tensor, token_expert_table, expand_x, valid_cnt, combine_info, group_name, moe_config
    )


@pypto.jit(distributed_options=distributed_options)
def combine_kernel(expert_out: pypto.Tensor, combine_info: pypto.Tensor, recv_counts: pypto.Tensor,
                   scale: pypto.Tensor, group_name: str, moe_config: pypto.distributed.MoeConfig,
                   combine_out: pypto.Tensor) -> None:
    pypto.distributed.shmem_moe_combine(
        expert_out, combine_info, recv_counts, scale, group_name,
        moe_config.rankNum, moe_config.routedExpertNum, combine_out
    )


@pypto.jit(distributed_options=distributed_options)
def ffn_kernel(combine_out: pypto.Tensor, ffn_weight: pypto.Tensor, out: pypto.Tensor) -> None:
    hidden_size = combine_out.shape[1]
    batch_size = combine_out.shape[0]
    weight_elements = ffn_weight.shape[0]
    intermediate_size = weight_elements // (hidden_size * 3)

    weight_stride = hidden_size * intermediate_size
    pypto.set_vec_tile_shapes(_align_up(min(int(weight_stride), 1024)))
    gate_weight_flat = pypto.view(ffn_weight, [weight_stride], [0])
    up_weight_flat = pypto.view(ffn_weight, [weight_stride], [weight_stride])
    down_weight_flat = pypto.view(ffn_weight, [weight_stride], [weight_stride * 2])

    gate_weight = pypto.reshape(gate_weight_flat, [hidden_size, intermediate_size])
    up_weight = pypto.reshape(up_weight_flat, [hidden_size, intermediate_size])
    down_weight = pypto.reshape(down_weight_flat, [intermediate_size, hidden_size])

    pypto.set_vec_tile_shapes(1, _align_up(max(int(hidden_size), int(intermediate_size))))
    gate_weight_fp32 = pypto.cast(gate_weight, pypto.DT_FP32)
    up_weight_fp32 = pypto.cast(up_weight, pypto.DT_FP32)
    down_weight_fp32 = pypto.cast(down_weight, pypto.DT_FP32)

    def run_chunk_ffn(combine_chunk: pypto.Tensor) -> pypto.Tensor:
        pypto.set_vec_tile_shapes(1, _align_up(max(int(hidden_size), int(intermediate_size))))
        combine_fp32 = pypto.cast(combine_chunk, pypto.DT_FP32)
        m_tile = _align_up(min(int(batch_size), 16))
        k_tile = _align_up(min(int(hidden_size), 128))
        n_tile = _align_up(min(int(intermediate_size), 128))
        pypto.set_cube_tile_shapes([m_tile, m_tile], [k_tile, k_tile], [n_tile, n_tile])
        gate_fp32 = pypto.matmul(combine_fp32, gate_weight_fp32, pypto.DT_FP32)
        up_fp32 = pypto.matmul(combine_fp32, up_weight_fp32, pypto.DT_FP32)
        neg_gate = pypto.neg(gate_fp32)
        exp_neg_gate = pypto.exp(neg_gate)
        denom = pypto.add(exp_neg_gate, 1.0)
        silu = pypto.div(gate_fp32, denom)
        inter_fp32 = pypto.mul(silu, up_fp32)
        m_tile = _align_up(min(int(batch_size), 16))
        k_tile = _align_up(min(int(intermediate_size), 128))
        n_tile = _align_up(min(int(hidden_size), 128))
        pypto.set_cube_tile_shapes([m_tile, m_tile], [k_tile, k_tile], [n_tile, n_tile])
        out_fp32 = pypto.matmul(inter_fp32, down_weight_fp32, pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, _align_up(int(hidden_size)))
        return pypto.cast(out_fp32, combine_out.dtype)

    chunk_size = _get_chunk_size(int(batch_size))
    if chunk_size >= int(batch_size):
        combine_chunk = pypto.view(combine_out, [batch_size, hidden_size], [0, 0])
        out.move(run_chunk_ffn(combine_chunk))
        return

    loop_times = int(batch_size) // chunk_size
    for chunk_idx in pypto.loop(loop_times, name="ffn_chunk"):
        offset = chunk_idx * chunk_size
        combine_chunk = pypto.view(combine_out, [chunk_size, hidden_size], [offset, 0])
        out_chunk = run_chunk_ffn(combine_chunk)
        pypto.assemble(out_chunk, [offset, 0], out)


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


def validate_combine_info(combine_info: torch.Tensor, batch_size: int, top_k: int, world_size: int,
                          local_rank: int | None = None) -> Tuple[int, int, int, int]:
    combine_info_cpu = combine_info.cpu().numpy()
    bad_rank = 0
    bad_token = 0
    bad_k = 0
    mismatch_rank = 0
    for rank_id, token_id, k_offset in combine_info_cpu:
        if rank_id < 0:
            continue
        if rank_id >= world_size:
            bad_rank += 1
        elif local_rank is not None and rank_id != local_rank:
            mismatch_rank += 1
        if token_id < 0 or token_id >= batch_size:
            bad_token += 1
        if k_offset < 0 or k_offset >= top_k:
            bad_k += 1
    return bad_rank, bad_token, bad_k, mismatch_rank


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


def torch_combine_cpu_loop(expand_all: torch.Tensor, info_all: torch.Tensor, scale: torch.Tensor,
                           rank: int, batch_size: int, top_k: int) -> torch.Tensor:
    info_cpu = info_all.numpy()
    out_fp32 = torch.zeros((batch_size, expand_all.shape[1]), dtype=torch.float32)
    for row_idx, (rank_id, token_id, k_offset) in enumerate(info_cpu):
        if rank_id != rank:
            continue
        if token_id < 0 or token_id >= batch_size or k_offset < 0 or k_offset >= top_k:
            continue
        out_fp32[token_id] += expand_all[row_idx].float() * scale[token_id, k_offset]
    return out_fp32


def run_moe_combine_ffn_multiturn(rank: int, world_size: int) -> None:
    setup_distributed(rank, world_size)
    torch.manual_seed(0)
    torch.npu.manual_seed_all(0)

    use_torch_ffn = os.environ.get("USE_TORCH_FFN", "1") == "1"
    debug_validate = os.environ.get("DEBUG_VALIDATE_COMBINE_INFO", "0") == "1"
    use_cpu_baseline = os.environ.get("BASELINE_CPU", "0") == "1"

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
    token_tensor = (torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="npu") * token_scale)
    token_expert_table = torch.randint(0, routing_expert_num, (batch_size, top_k), dtype=torch.int32, device="npu")
    scale = torch.ones(batch_size, top_k, dtype=torch.float32, device="npu")

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
    ffn_weight_flat = torch.cat(
        [gate_w.view(-1), up_w.view(-1), down_w.view(-1)], dim=0
    ).contiguous()
    ffn_weight_pto = pypto.from_torch(ffn_weight_flat)

    combine_out_data = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")
    fused_out_data = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")

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

    def wrap_inputs():
        return (
            pypto.from_torch(token_tensor),
            pypto.from_torch(token_expert_table),
            pypto.from_torch(scale),
            pypto.from_torch(expand_x_data),
            pypto.from_torch(valid_cnt_data),
            pypto.from_torch(combine_info_data),
        )

    def wrap_outputs():
        return (
            pypto.from_torch(combine_out_data),
            pypto.from_torch(fused_out_data),
        )

    def run_dispatch(token_pto, table_pto, expand_x_pto, valid_cnt_pto, combine_info_pto) -> None:
        dispatch_kernel(token_pto, table_pto, hccl_comm_name, moe_config, expand_x_pto, valid_cnt_pto, combine_info_pto)

    def run_combine_ffn(combine_out_pto, fused_out_pto, recv_counts_pto, scale_pto, expand_x_pto, combine_info_pto) -> None:
        combine_kernel(
            expand_x_pto, combine_info_pto, recv_counts_pto, scale_pto, hccl_comm_name, moe_config,
            combine_out_pto
        )
        torch.npu.synchronize()
        if use_torch_ffn:
            torch_out_fp32 = torch_ffn_only(combine_out_data.float(), gate_w, up_w, down_w)
            fused_out_data.copy_(torch_out_fp32.to(torch.bfloat16))
        else:
            ffn_kernel(combine_out_pto, ffn_weight_pto, fused_out_pto)
        torch.npu.synchronize()

    def gather_inputs() -> Tuple[torch.Tensor, torch.Tensor]:
        gathered_expand: List[torch.Tensor] = [torch.empty_like(expand_x_data) for _ in range(world_size)]
        gathered_info: List[torch.Tensor] = [torch.empty_like(combine_info_data) for _ in range(world_size)]
        dist.all_gather(gathered_expand, expand_x_data)
        dist.all_gather(gathered_info, combine_info_data)
        expand_all = torch.cat(gathered_expand, dim=0)
        info_all = torch.cat(gathered_info, dim=0)
        return expand_all, info_all

    def run_torch_baseline(expand_all: torch.Tensor, info_all: torch.Tensor) -> None:
        combine_fp32 = torch_combine_only(expand_all, info_all, scale, rank, batch_size, top_k, use_scatter)
        _ = torch_ffn_only(combine_fp32, gate_w, up_w, down_w)

    if rank == 0:
        print(f"[Rank 0] Warmup rounds: {warmup_rounds}, Test rounds: {test_rounds}")

    for _ in range(warmup_rounds):
        token_expert_table.random_(0, routing_expert_num)
        reset_buffers()
        torch.npu.synchronize()

        token_pto, table_pto, scale_pto, expand_x_pto, valid_cnt_pto, combine_info_pto = wrap_inputs()
        run_dispatch(token_pto, table_pto, expand_x_pto, valid_cnt_pto, combine_info_pto)
        torch.npu.synchronize()
        if debug_validate:
            bad_rank, bad_token, bad_k, mismatch_rank = validate_combine_info(
                combine_info_data, batch_size, top_k, world_size, rank
            )
            if rank == 0:
                print(f"[Debug] Warmup invalid combine_info -> rank:{bad_rank} token:{bad_token} k:{bad_k} "
                      f"rank_mismatch:{mismatch_rank}")

        recv_counts = build_recv_counts(combine_info_data, batch_size, top_k, world_size, rank)
        if debug_validate and rank == 0:
            print(f"[Debug] Warmup recv_counts min:{int(recv_counts.min().item())} max:{int(recv_counts.max().item())}")
        recv_counts_pto = pypto.from_torch(recv_counts)
        combine_out_pto, fused_out_pto = wrap_outputs()
        run_combine_ffn(combine_out_pto, fused_out_pto, recv_counts_pto, scale_pto, expand_x_pto, combine_info_pto)

    pypto_times: List[float] = []
    torch_times: List[float] = []
    for _ in range(test_rounds):
        token_expert_table.random_(0, routing_expert_num)
        reset_buffers()
        torch.npu.synchronize()

        token_pto, table_pto, scale_pto, expand_x_pto, valid_cnt_pto, combine_info_pto = wrap_inputs()
        run_dispatch(token_pto, table_pto, expand_x_pto, valid_cnt_pto, combine_info_pto)
        torch.npu.synchronize()
        if debug_validate:
            bad_rank, bad_token, bad_k, mismatch_rank = validate_combine_info(
                combine_info_data, batch_size, top_k, world_size, rank
            )
            if rank == 0:
                print(f"[Debug] Test invalid combine_info -> rank:{bad_rank} token:{bad_token} k:{bad_k} "
                      f"rank_mismatch:{mismatch_rank}")

        recv_counts = build_recv_counts(combine_info_data, batch_size, top_k, world_size, rank)
        if debug_validate and rank == 0:
            print(f"[Debug] Test recv_counts min:{int(recv_counts.min().item())} max:{int(recv_counts.max().item())}")
        recv_counts_pto = pypto.from_torch(recv_counts)
        combine_out_pto, fused_out_pto = wrap_outputs()

        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        run_combine_ffn(combine_out_pto, fused_out_pto, recv_counts_pto, scale_pto, expand_x_pto, combine_info_pto)
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

    expand_all, info_all = gather_inputs()

    if use_cpu_baseline:
        expand_all_cpu = expand_all.cpu()
        info_all_cpu = info_all.cpu()
        scale_cpu = scale.cpu()
        torch_combine_fp32 = torch_combine_cpu_loop(
            expand_all_cpu, info_all_cpu, scale_cpu, rank, batch_size, top_k
        )
        torch_combine_bf16 = torch_combine_fp32.to(torch.bfloat16)
        combine_out_ref = combine_out_data.cpu()
    else:
        torch_combine_fp32 = torch_combine_only(
            expand_all, info_all, scale, rank, batch_size, top_k, use_scatter
        )
        torch_combine_bf16 = torch_combine_fp32.to(torch.bfloat16)
        combine_out_ref = combine_out_data

    combine_diff = (combine_out_ref - torch_combine_bf16).abs().max().item()
    if torch.allclose(combine_out_ref, torch_combine_bf16, rtol=2e-2, atol=2e-2):
        print(f"[Rank {rank}] Combine output matches torch baseline. Max diff: {combine_diff}")
    else:
        print(f"[Rank {rank}] Combine output mismatch. Max diff: {combine_diff}")

    torch_combine_for_ffn = torch_combine_bf16
    if use_cpu_baseline:
        torch_combine_for_ffn = torch_combine_bf16.to("npu")

    if use_torch_ffn:
        torch_out = torch_ffn_only(combine_out_data.float(), gate_w, up_w, down_w).to(torch.bfloat16)
    else:
        torch_out = torch_ffn_only(torch_combine_for_ffn.float(), gate_w, up_w, down_w).to(torch.bfloat16)

    if torch.allclose(fused_out_data, torch_out, rtol=2e-2, atol=2e-2):
        print(f"[Rank {rank}] SUCCESS! Fused output matches torch baseline.")
    else:
        diff = (fused_out_data - torch_out).abs().max().item()
        print(f"[Rank {rank}] FAILED! Max diff: {diff}")

    if rank == 0:
        pypto_avg = sum(pypto_times) / len(pypto_times)
        torch_avg = sum(torch_times) / len(torch_times)
        speedup = torch_avg / pypto_avg if pypto_avg > 0 else 0.0
        print(f"[Rank 0] PyPTO avg: {pypto_avg:.3f} ms, Torch avg: {torch_avg:.3f} ms, "
              f"Speedup: {speedup:.2f}x")


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
