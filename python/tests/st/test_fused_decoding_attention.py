#!/usr/bin/env python3
# coding: utf-8

import os
import time
import numpy as np
import torch
import torch_npu
import pytest

import sys
from pathlib import Path
# repo root: .../zimo_pypto_0323
REPO_ROOT = Path(__file__).resolve().parents[3]
GLM_DIR = REPO_ROOT / "models" / "glm_v4_5"
# 让 glm_attention_fusion.py 里的 `from utils...` 能找到 models/glm_v4_5/utils
sys.path.insert(0, str(GLM_DIR))
# 同时保留 repo 根路径，便于其它相对导包
sys.path.insert(0, str(REPO_ROOT))

from models.glm_v4_5.utils.np_compare import detailed_allclose_manual as compare
import models.glm_v4_5.utils.golden.attn_golden as attn_golden
from models.glm_v4_5.glm_attention_fusion import attention, get_qwen_common_config


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _build_attention_inputs(attn_cfg, device: str, device_id: int, seed: int = 0):
    """
    生成一轮 attention 输入，返回：
    - inputs_pypto: 给 attention(...) 用
    - inputs_torch: 给 attn_golden.attention_golden(...) 用（cache 已 clone）
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch_dtype = torch.bfloat16
    b = attn_cfg.b
    s1 = attn_cfg.s1
    d = attn_cfg.q_d
    n1 = attn_cfg.n1
    n2 = attn_cfg.n2
    bs = b * s1
    hidden_size = attn_cfg.hidden_size
    q_size = n1 * d
    total_head_size = q_size + 2 * d
    rotary_dim = d // 2
    half_rotary_dim = rotary_dim // 2

    block_num = attn_cfg.kv_num_blocks
    block_size = attn_cfg.block_size
    max_num_blocks_per_query = attn_cfg.max_num_blocks_per_query

    actual_seq_lens = attn_cfg.actual_seq.to(dtype=torch.int32, device=device)
    kv_cache_shape = [attn_cfg.kv_num_blocks, block_size, n2, d]
    block_table_shape = [attn_cfg.block_table_batch, max_num_blocks_per_query]

    slot_mapping = torch.randperm(block_num * block_size, dtype=torch.int32, device=device)[:b]
    key_cache = torch.empty(kv_cache_shape, dtype=torch_dtype, device=device).uniform_(-1, 1) * 0
    value_cache = torch.empty(kv_cache_shape, dtype=torch_dtype, device=device).uniform_(-1, 1) * 0
    block_tables = attn_golden.gen_block_table(actual_seq_lens, block_size, block_table_shape)

    # Torch baseline 用 clone，确保两路输入一致
    key_cache_clone = key_cache.clone()
    value_cache_clone = value_cache.clone()

    hidden_states = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=device)
    input_layernorm_weight = torch.rand(hidden_size, dtype=torch.bfloat16, device=device)
    input_layernorm_bias = torch.rand(hidden_size, dtype=torch.bfloat16, device=device)
    qkv_proj_scale = torch.rand(hidden_size, dtype=torch.bfloat16, device=device)
    qkv_proj_offset = torch.rand(hidden_size, dtype=torch.bfloat16, device=device)

    qkv_proj_weight = torch.randint(
        0, 128, size=(hidden_size, total_head_size), dtype=torch.int8, device=f"npu:{device_id}"
    )
    qkv_proj_weight = torch_npu.npu_format_cast(qkv_proj_weight, 29)
    qkv_proj_quant_bias = torch.randint(
        0, 128, size=(total_head_size,), dtype=torch.int32, device=f"npu:{device_id}"
    )
    qkv_proj_deq_scale = torch.rand(total_head_size, dtype=torch.float32, device=device)
    q_norm_weight = torch.rand(d, dtype=torch.bfloat16, device=device)
    q_norm_bias = torch.rand(d, dtype=torch.bfloat16, device=device)
    k_norm_weight = torch.rand(d, dtype=torch.bfloat16, device=device)
    k_norm_bias = torch.rand(d, dtype=torch.bfloat16, device=device)
    cos = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=device)
    sin = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=device)

    common_kwargs = dict(
        hidden_states=hidden_states,
        residual=residual,
        input_layernorm_weight=input_layernorm_weight,
        input_layernorm_bias=input_layernorm_bias,
        qkv_proj_scale=qkv_proj_scale,
        qkv_proj_offset=qkv_proj_offset,
        qkv_proj_weight=qkv_proj_weight,
        qkv_proj_quant_bias=qkv_proj_quant_bias,
        qkv_proj_deq_scale=qkv_proj_deq_scale,
        q_norm_weight=q_norm_weight,
        q_norm_bias=q_norm_bias,
        k_norm_weight=k_norm_weight,
        k_norm_bias=k_norm_bias,
        cos=cos,
        sin=sin,
        block_tables=block_tables,
        actual_seq_lens=actual_seq_lens,
        slot_mapping=slot_mapping,
        eps=attn_cfg.eps,
        enable_residual=True,
        num_decode_tokens=0,
    )

    inputs_pypto = dict(
        **common_kwargs,
        key_cache=key_cache,
        value_cache=value_cache,
    )
    inputs_torch = dict(
        **common_kwargs,
        key_cache=key_cache_clone,
        value_cache=value_cache_clone,
    )
    return inputs_pypto, inputs_torch


@pytest.mark.soc("950", "910")
def test_fused_attention_pypto_vs_torch_perf():
    """
    单 case（官方默认 size）：
    - warmup + 正式轮次性能对比
    - PyPTO attention vs torch golden attention
    """
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch_npu.npu.config.allow_internal_format = True
    device = f"npu:{device_id}"

    warmup_rounds = int(os.environ.get("ATTN_WARMUP_ROUNDS", 2))
    num_rounds = int(os.environ.get("ATTN_NUM_ROUNDS", 5))
    total_rounds = warmup_rounds + num_rounds

    # 保持和官方 attention 配置一致
    attn_cfg, _ = get_qwen_common_config(device=device)

    # compare 阈值与官方 test_attention 一致
    residual_rtol, residual_atol = 0.1, 0.1
    out_rtol, out_atol = 0.3, 0.3

    pypto_times, pypto_mems = [], []
    torch_times, torch_mems = [], []

    print(
        f"\n🚀 FusedAttention PyPTO vs Torch 多轮测试 "
        f"(warmup={warmup_rounds}, rounds={num_rounds}) on {device}"
    )
    print(
        f"Config: b={attn_cfg.b}, s1={attn_cfg.s1}, s2={attn_cfg.s2}, "
        f"n1={attn_cfg.n1}, n2={attn_cfg.n2}, q_d={attn_cfg.q_d}, hidden={attn_cfg.hidden_size}"
    )

    for r in range(total_rounds):
        inputs_pypto, inputs_torch = _build_attention_inputs(
            attn_cfg=attn_cfg, device=device, device_id=device_id, seed=2026 + r
        )

        # PyPTO
        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out_pypto, residual_pypto = attention(**inputs_pypto)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        pypto_t = t1 - t0
        pypto_m = torch.npu.max_memory_allocated(device) / (1024 ** 2)

        # Torch golden
        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out_torch, residual_torch = attn_golden.attention_golden(**inputs_torch)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        torch_t = t1 - t0
        torch_m = torch.npu.max_memory_allocated(device) / (1024 ** 2)

        # 正确性校验
        compare(
            np.array(residual_torch.detach().cpu().flatten().tolist()),
            np.array(residual_pypto.detach().cpu().flatten().tolist()),
            f"Round{r+1} residual_golden_vs_pypto",
            rtol=residual_rtol,
            atol=residual_atol,
        )
        compare(
            np.array(out_torch.detach().cpu().flatten().tolist()),
            np.array(out_pypto.detach().cpu().flatten().tolist()),
            f"Round{r+1} attention_golden_vs_pypto",
            rtol=out_rtol,
            atol=out_atol,
        )

        if r >= warmup_rounds:
            pypto_times.append(pypto_t)
            pypto_mems.append(pypto_m)
            torch_times.append(torch_t)
            torch_mems.append(torch_m)
            print(
                f"Round {r+1:02d} | "
                f"PyPTO: {pypto_t*1000:.2f} ms / {pypto_m:.1f} MB | "
                f"Torch: {torch_t*1000:.2f} ms / {torch_m:.1f} MB"
            )

        del inputs_pypto, inputs_torch, out_pypto, residual_pypto, out_torch, residual_torch
        torch.npu.empty_cache()

    avg_pypto_t = _mean(pypto_times)
    avg_pypto_m = _mean(pypto_mems)
    avg_torch_t = _mean(torch_times)
    avg_torch_m = _mean(torch_mems)

    print("\n" + "=" * 90)
    print("📈 [SUMMARY] FusedAttention PyPTO vs Torch (official single-size case)")
    print(f"Torch: {avg_torch_t*1000:.2f} ms | {avg_torch_m:.1f} MB")
    print(f"PyPTO: {avg_pypto_t*1000:.2f} ms | {avg_pypto_m:.1f} MB")
    if avg_pypto_t > 0:
        print(f"Speedup(Torch/PyPTO): {avg_torch_t / avg_pypto_t:.2f}x")
    print("=" * 90)


if __name__ == "__main__":
    test_fused_attention_pypto_vs_torch_perf()