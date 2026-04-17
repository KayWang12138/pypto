#!/usr/bin/env python3
# coding: utf-8
#
# Perf comparison for three GLM FFN router expert implementations:
# 1) PyPTO latest path: base.moe_router_expert_main + pypto.from_torch
# 2) Torch NPU grouped path: base.ffn_router_torch_npu
# 3) Torch naive path: pure torch per-expert loop implementation

import os
import time
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import torch
import torch_npu
from numpy.testing import assert_allclose

import test_glm_ffn_router_expert_quant as base


@dataclass
class BenchConfig:
    dtype: torch.dtype = torch.bfloat16
    b: int = 1
    s: int = 4096
    topk: int = 8
    per_expert_num: int = 16
    hidden_size: int = 5120
    intermediate_size: int = 1536
    warmup: int = 2
    iters: int = 10
    seed: int = 42
    uniform_distribution: bool = False

    @staticmethod
    def from_env() -> "BenchConfig":
        cfg = BenchConfig()
        cfg.b = int(os.environ.get("GLM_B", str(cfg.b)))
        cfg.s = int(os.environ.get("GLM_S", str(cfg.s)))
        cfg.topk = int(os.environ.get("GLM_TOPK", str(cfg.topk)))
        cfg.per_expert_num = int(os.environ.get("GLM_NUM_EXPERTS", str(cfg.per_expert_num)))
        cfg.hidden_size = int(os.environ.get("GLM_HIDDEN_SIZE", str(cfg.hidden_size)))
        cfg.intermediate_size = int(os.environ.get("GLM_INTERMEDIATE_SIZE", str(cfg.intermediate_size)))
        cfg.warmup = int(os.environ.get("GLM_WARMUP", str(cfg.warmup)))
        cfg.iters = int(os.environ.get("GLM_ITERS", str(cfg.iters)))
        cfg.seed = int(os.environ.get("GLM_GROUP_SEED", str(cfg.seed)))
        cfg.uniform_distribution = os.environ.get("GLM_GROUP_UNIFORM", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        # Keep dtype simple and explicit.
        dtype_str = os.environ.get("GLM_DTYPE", "bf16").strip().lower()
        if dtype_str in {"bf16", "bfloat16"}:
            cfg.dtype = torch.bfloat16
        elif dtype_str in {"fp16", "float16"}:
            cfg.dtype = torch.float16
        else:
            raise ValueError(f"Unsupported GLM_DTYPE={dtype_str}, only bf16/fp16 are supported")
        return cfg


def _validate_cfg_for_latest_impl(cfg: BenchConfig) -> None:
    # The latest PyPTO implementation in base.check_args has hard constraints.
    if cfg.hidden_size != 5120:
        raise ValueError("Latest PyPTO path requires hidden_size=5120")
    if cfg.intermediate_size != 1536:
        raise ValueError("Latest PyPTO path requires intermediate_size=1536")
    if cfg.dtype != torch.bfloat16:
        raise ValueError("Latest PyPTO path currently expects bf16 test flow")


def _reset_peak_memory() -> None:
    if hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats()
    if hasattr(torch.npu, "reset_max_memory_allocated"):
        torch.npu.reset_max_memory_allocated()


def _get_peak_memory_bytes() -> int:
    if hasattr(torch.npu, "max_memory_allocated"):
        return int(torch.npu.max_memory_allocated())
    return -1


def _run_bench(name: str, fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        _ = fn()
        torch.npu.synchronize()

    latencies_ms: List[float] = []
    peak_mem_bytes: int = 0
    for _ in range(iters):
        _reset_peak_memory()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = fn()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        peak_mem_bytes = max(peak_mem_bytes, _get_peak_memory_bytes())

    return {
        "name": name,
        "avg_ms": float(sum(latencies_ms) / len(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p90_ms": float(np.percentile(latencies_ms, 90)),
        "std_ms": float(statistics.pstdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0,
        "peak_mem_mb": peak_mem_bytes / (1024.0 * 1024.0) if peak_mem_bytes >= 0 else -1.0,
    }


def _build_group_list(total_tokens: int, local_expert_num: int, device_id: int, uniform: bool, seed: int) -> torch.Tensor:
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if local_expert_num <= 0:
        raise ValueError("local_expert_num must be positive")
    if total_tokens == 0:
        return torch.zeros(local_expert_num, dtype=torch.int32, device=f"npu:{device_id}")

    if uniform:
        base_cnt = total_tokens // local_expert_num
        remainder = total_tokens % local_expert_num
        group_list_np = np.full(local_expert_num, base_cnt, dtype=np.int32)
        if remainder > 0:
            group_list_np[:remainder] += 1
    else:
        # Skewed realistic routing.
        rng = np.random.default_rng(seed)
        alpha = np.ones(local_expert_num, dtype=np.float64) * 0.7
        probs = rng.dirichlet(alpha)
        group_list_np = rng.multinomial(total_tokens, probs).astype(np.int32)

    return torch.from_numpy(group_list_np).to(device=f"npu:{device_id}", dtype=torch.int32)


def _prepare_input(cfg: BenchConfig, device_id: int):
    torch.manual_seed(cfg.seed)
    total_tokens = cfg.b * cfg.s * cfg.topk
    hidden_states = (
        torch.randn((total_tokens, cfg.hidden_size), dtype=cfg.dtype, device=f"npu:{device_id}") * 0.01 * 2 - 0.01
    )
    hidden_states, hidden_states_scale = base.ffn_golden_quan_per_token(hidden_states)
    hidden_states_scale = hidden_states_scale.reshape(-1).to(torch.float32)

    group_list = _build_group_list(
        total_tokens=total_tokens,
        local_expert_num=cfg.per_expert_num,
        device_id=device_id,
        uniform=cfg.uniform_distribution,
        seed=cfg.seed,
    )
    group_list_cumsum = base.get_token_acc_table(group_list).to(torch.int32)

    w13 = (
        torch.randn(
            (cfg.per_expert_num, cfg.hidden_size, cfg.intermediate_size * 2),
            dtype=cfg.dtype,
            device=f"npu:{device_id}",
        )
        * 0.01
        * 2
        - 0.01
    )
    w13, w13_scale = base.ffn_golden_quan_per_channel_3d(w13)
    w13_scale = w13_scale.squeeze(1).to(torch.float32)

    w2 = (
        torch.randn(
            (cfg.per_expert_num, cfg.intermediate_size, cfg.hidden_size),
            dtype=cfg.dtype,
            device=f"npu:{device_id}",
        )
        * 0.01
        * 2
        - 0.01
    )
    w2, w2_scale = base.ffn_golden_quan_per_channel_3d(w2)
    w2_scale = w2_scale.squeeze(1).to(cfg.dtype)

    ffn_res = torch.empty(hidden_states.shape, dtype=w2_scale.dtype, device=f"npu:{device_id}")
    return hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res


def _swiglu_torch(x: torch.Tensor) -> torch.Tensor:
    mid = x.shape[-1] // 2
    left = x[:, :mid]
    right = x[:, mid:]
    return left * torch.sigmoid(left) * right


def ffn_router_torch_naive(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    group_list: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Naive reference (pure torch):
    - Dequant x per token
    - Per expert loop: x @ w13 -> swiglu -> per-token quant -> @ w2 -> dequant
    """
    total_tokens, hidden_size = hidden_states.shape
    out_dtype = w2_scale.dtype
    out = torch.zeros((total_tokens, hidden_size), device=hidden_states.device, dtype=out_dtype)

    group_list_i64 = group_list.to(torch.int64)
    group_cumsum = torch.cumsum(group_list_i64, dim=0)
    token_starts = torch.cat(
        [torch.tensor([0], dtype=torch.int64, device=group_list.device), group_cumsum[:-1]],
        dim=0,
    )

    for exp_idx in range(group_list.shape[0]):
        token_num = int(group_list_i64[exp_idx].item())
        if token_num <= 0:
            continue
        start = int(token_starts[exp_idx].item())
        end = start + token_num

        x_i8 = hidden_states[start:end]
        x_scale = hidden_states_scale[start:end].reshape(-1, 1).to(torch.float32)
        x_fp32 = x_i8.to(torch.float32) * x_scale

        w13_fp32 = w13[exp_idx].to(torch.float32) * w13_scale[exp_idx].to(torch.float32).reshape(1, -1)
        up_proj = torch.matmul(x_fp32, w13_fp32)
        swiglu_out = _swiglu_torch(up_proj)

        # quant per-token
        swiglu_abs_max = swiglu_out.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        q_scale = 127.0 / swiglu_abs_max
        down_q = torch.trunc(torch.round(swiglu_out * q_scale)).to(torch.int8)
        down_deq_scale = 1.0 / q_scale

        w2_fp32 = w2[exp_idx].to(torch.float32) * w2_scale[exp_idx].to(torch.float32).reshape(1, -1)
        down_i32 = torch.matmul(down_q.to(torch.float32), w2[exp_idx].to(torch.float32))
        down_fp32 = down_i32 * down_deq_scale * w2_scale[exp_idx].to(torch.float32).reshape(1, -1)
        out[start:end] = down_fp32.to(out_dtype)

    return out


def _print_stats(title: str, stats: Dict[str, float]) -> None:
    print(
        f"{title}: avg={stats['avg_ms']:.3f}ms, p50={stats['p50_ms']:.3f}ms, "
        f"p90={stats['p90_ms']:.3f}ms, std={stats['std_ms']:.3f}ms, peak_mem={stats['peak_mem_mb']:.2f}MB"
    )


def test_ffn_router_impl_perf_compare() -> None:
    cfg = BenchConfig.from_env()
    _validate_cfg_for_latest_impl(cfg)

    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    (
        hidden_states,
        hidden_states_scale,
        group_list,
        group_list_cumsum,
        w13,
        w13_scale,
        w2,
        w2_scale,
        _,
    ) = _prepare_input(cfg, device_id)

    # PyPTO latest calling style (keep aligned with latest file).
    ffn_res_pypto = torch.empty_like(hidden_states, dtype=w2_scale.dtype)
    inputs = {
        hidden_states: [0],
        hidden_states_scale: [0],
        group_list: [],
        group_list_cumsum: [],
        w13: [],
        w13_scale: [],
        w2: [],
        w2_scale: [],
    }
    outputs = {ffn_res_pypto: [0]}
    pto_inputs = [base.pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [base.pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    def run_pypto() -> torch.Tensor:
        base.moe_router_expert_main(*pto_inputs, *pto_outputs)
        base.pypto.runtime._device_synchronize()
        return ffn_res_pypto

    def run_torch_npu() -> torch.Tensor:
        return base.ffn_router_torch_npu(
            hidden_states, hidden_states_scale, group_list, w13, w13_scale, w2, w2_scale
        )

    def run_torch_naive() -> torch.Tensor:
        return ffn_router_torch_naive(
            hidden_states, hidden_states_scale, group_list, w13, w13_scale, w2, w2_scale
        )

    # correctness checks
    pypto_out = run_pypto()
    torch_npu_out = run_torch_npu()
    torch_naive_out = run_torch_naive()
    valid_size = int(group_list.sum().item() * cfg.hidden_size)
    assert_allclose(
        np.array(pypto_out.cpu().flatten().tolist()[:valid_size]),
        np.array(torch_npu_out.cpu().flatten().tolist()[:valid_size]),
        rtol=0.0078125,
        atol=0.0001,
    )
    assert_allclose(
        np.array(pypto_out.cpu().flatten().tolist()[:valid_size]),
        np.array(torch_naive_out.cpu().flatten().tolist()[:valid_size]),
        rtol=0.03,
        atol=0.002,
    )

    # perf
    pypto_stats = _run_bench("pypto_latest", run_pypto, warmup=cfg.warmup, iters=cfg.iters)
    torch_npu_stats = _run_bench("torch_npu_grouped", run_torch_npu, warmup=cfg.warmup, iters=cfg.iters)
    torch_naive_stats = _run_bench("torch_naive", run_torch_naive, warmup=cfg.warmup, iters=cfg.iters)

    print("\n[PerfCompare][GLM FFN Router Expert Quant]")
    print(
        f"cfg: b={cfg.b}, s={cfg.s}, topk={cfg.topk}, experts={cfg.per_expert_num}, "
        f"hidden={cfg.hidden_size}, inter={cfg.intermediate_size}, dtype={cfg.dtype}, "
        f"warmup={cfg.warmup}, iters={cfg.iters}, uniform={cfg.uniform_distribution}, seed={cfg.seed}"
    )
    print(f"group_list={group_list.cpu().tolist()}")
    print(f"valid_tokens={int(group_list.sum().item())}")
    _print_stats("PyPTO(latest)", pypto_stats)
    _print_stats("TorchNPU(grouped)", torch_npu_stats)
    _print_stats("Torch(naive)", torch_naive_stats)
    print(
        f"speedup(TorchNPU/PyPTO)={torch_npu_stats['avg_ms'] / pypto_stats['avg_ms']:.3f}x, "
        f"speedup(Naive/PyPTO)={torch_naive_stats['avg_ms'] / pypto_stats['avg_ms']:.3f}x"
    )

