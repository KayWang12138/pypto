#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Flash Attention Score with Online Softmax

This module implements Flash Attention using online softmax algorithm,
which avoids storing the full attention matrix and provides better
numerical stability through block-wise computation.
"""

import argparse
import logging
import math
import os
import time
from typing import Optional

import numpy as np
import torch
import torch_npu
from numpy.testing import assert_allclose

try:
    from torch._dynamo import allow_in_graph
    from torch._subclasses.fake_tensor import FakeTensor
except Exception:
    def allow_in_graph(fn):
        return fn

    class FakeTensor:  # type: ignore[no-redef]
        pass

from flash_attention_score_impl import (
    BLOCK_SIZE_KV,
    BLOCK_SIZE_Q,
    HEAD_DIM,
    KV_UNROLL_LIST,
    MODEL_PRESET,
    NUM_HEADS,
    PV_CUBE_TILE_SHAPES,
    QK_CUBE_TILE_SHAPES,
    VEC_TILE_SHAPES,
    flash_attention_score_backward_kv_kernel_with_mask,
    flash_attention_score_backward_kernel_with_mask,
    flash_attention_score_kernel_with_mask,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


MODEL_SHAPE_PRESETS = {
    "aigcode_8b_jamba_gdn_moe": {
        "batch": 1,
        "seq_q": 8192,
        "seq_kv": 8192,
        "mask": "causal",
        "source": "/sharedata/llx/pto/Mindspeed-LLM/examples/mcore/qwen2/pretrain_aigcode_8b_4k_jamba_gdn_moe_cann850_tpe.sh",
        "note": "Q/K/V=[B,32,S,128], no GQA",
    },
}

PRESET_SHAPE = MODEL_SHAPE_PRESETS.get(MODEL_PRESET, {})
DEFAULT_BATCH_SIZE = PRESET_SHAPE.get("batch", 4)
DEFAULT_SEQ_LEN_Q = PRESET_SHAPE.get("seq_q", 64)
DEFAULT_SEQ_LEN_KV = PRESET_SHAPE.get("seq_kv", 128)
DEFAULT_MASK_TYPE = PRESET_SHAPE.get("mask", "causal")
DEFAULT_BENCHMARK_WARMUP = 1
DEFAULT_BENCHMARK_ITERS = 3
BACKWARD_SEQ_LEN_Q = min(DEFAULT_SEQ_LEN_Q, BLOCK_SIZE_Q)
BACKWARD_SEQ_LEN_KV = min(DEFAULT_SEQ_LEN_KV, BLOCK_SIZE_KV)
DEFAULT_GOLDEN_BLOCK_Q = 128
FORWARD_COMPARE_RTOL = 0.03
FORWARD_COMPARE_ATOL = 0.02


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logging.info("Please set the environment variable TILE_FWK_DEVICE_ID before running:")
        logging.info("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        logging.info(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def flash_attention_score_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Flash Attention reference implemented with dense matmul + softmax.

    Args:
        query: Query tensor, shape [b, n, sq, d], dtype bfloat16
        key: Key tensor, shape [b, n, skv, d], dtype bfloat16
        value: Value tensor, shape [b, n, skv, d], dtype bfloat16
        atten_mask: Attention mask tensor, shape [sq, skv], dtype uint8
                   值为 1 表示不参与计算，值为 0 表示参与计算

    Returns:
        attention_out: Output tensor, shape [b, n, sq, d], dtype bfloat16
    """
    d = query.shape[-1]
    scale = 1.0 / math.sqrt(d)
    query_fp32 = query.float()
    key_t_fp32 = key.float().transpose(-2, -1)
    value_fp32 = value.float()
    output_fp32 = torch.empty_like(query_fp32)
    mask = None
    if atten_mask is not None:
        mask = atten_mask.bool().unsqueeze(0).unsqueeze(0)

    for q_start in range(0, query.shape[2], DEFAULT_GOLDEN_BLOCK_Q):
        q_end = min(q_start + DEFAULT_GOLDEN_BLOCK_Q, query.shape[2])
        q_block = query_fp32[:, :, q_start:q_end, :]
        scores = torch.matmul(q_block, key_t_fp32) * scale
        if mask is not None:
            scores = scores.masked_fill(mask[:, :, q_start:q_end, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        output_fp32[:, :, q_start:q_end, :] = torch.matmul(weights, value_fp32)

    return output_fp32.to(torch.bfloat16)


def make_mask(mask_type: str, seq_len_q: int, seq_len_kv: int, device: str) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if mask_type == "none":
        return None, None
    if mask_type == "causal":
        mask_bool = torch.triu(torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool, device=device), diagonal=1)
    elif mask_type == "half":
        mask_bool = torch.zeros(seq_len_q, seq_len_kv, dtype=torch.bool, device=device)
        mask_bool[:, seq_len_kv // 2:] = True
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")
    return mask_bool.float(), mask_bool


def pypto_flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask_fp32: Optional[torch.Tensor],
) -> torch.Tensor:
    atten_mask_fp32 = _normalize_mask_fp32(query, key, atten_mask_fp32)
    output = torch.empty_like(query)
    for q_start in range(0, query.shape[2], BLOCK_SIZE_Q):
        q_end = min(q_start + BLOCK_SIZE_Q, query.shape[2])
        query_block = query[:, :, q_start:q_end, :].contiguous()
        mask_block = atten_mask_fp32[q_start:q_end, :].contiguous()
        output_block = torch.empty_like(query_block)
        flash_attention_score_kernel_with_mask(query_block, key, value, mask_block, output_block)
        output[:, :, q_start:q_end, :] = output_block
    return output


def _normalize_mask_fp32(
    query: torch.Tensor,
    key: torch.Tensor,
    atten_mask_fp32: Optional[torch.Tensor],
) -> torch.Tensor:
    if atten_mask_fp32 is None:
        return torch.zeros(query.shape[2], key.shape[2], dtype=torch.float32, device=query.device)
    return atten_mask_fp32


def _bnsd_to_sbh(x: torch.Tensor) -> torch.Tensor:
    seq_len, batch_size, num_heads, head_dim = x.shape[2], x.shape[0], x.shape[1], x.shape[3]
    return x.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, num_heads * head_dim)


def _sbh_to_bnsd(x: torch.Tensor, batch_size: int, num_heads: int, head_dim: int) -> torch.Tensor:
    seq_len = x.shape[0]
    return x.view(seq_len, batch_size, num_heads, head_dim).permute(1, 2, 0, 3).contiguous()


def cann_flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask_bool: Optional[torch.Tensor],
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    query_sbh = _bnsd_to_sbh(query)
    key_sbh = _bnsd_to_sbh(key)
    value_sbh = _bnsd_to_sbh(value)
    output_sbh = torch_npu.npu_fusion_attention(
        query_sbh,
        key_sbh,
        value_sbh,
        NUM_HEADS,
        "SBH",
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask_bool,
        scale=scale,
        pre_tockens=65536,
        next_tockens=0,
        keep_prob=1.0,
        inner_precise=0,
        sparse_mode=0,
    )[0]
    return _sbh_to_bnsd(output_sbh, query.shape[0], NUM_HEADS, query.shape[-1])


def _sync_device():
    torch.npu.synchronize()


def _format_cube_tile_shapes(tile_shapes: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> str:
    return ";".join(f"{m},{n}" for m, n in tile_shapes)


def _format_vec_tile_shapes(tile_shapes: tuple[int, ...]) -> str:
    return ",".join(str(v) for v in tile_shapes)


def _measure_ms(fn) -> float:
    _sync_device()
    start = time.perf_counter()
    fn()
    _sync_device()
    return (time.perf_counter() - start) * 1e3


def _summarize_ms(samples: list[float]) -> dict[str, float]:
    arr = np.array(samples, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _measure_many_ms(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync_device()
    samples = [_measure_ms(fn) for _ in range(iters)]
    return _summarize_ms(samples)


def _measure_backward_only_many_ms(forward_builder, grad_output: torch.Tensor, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        y = forward_builder()
        y.backward(grad_output)
    _sync_device()
    samples = []
    for _ in range(iters):
        y = forward_builder()
        samples.append(_measure_ms(lambda: y.backward(grad_output)))
    return _summarize_ms(samples)


def _print_stats(title: str, pypto_stats: dict[str, float], cann_stats: dict[str, float]) -> None:
    ratio = pypto_stats["mean_ms"] / cann_stats["mean_ms"]
    logging.info(
        "%-18s PyPTO mean/p50/min/max = %.3f/%.3f/%.3f/%.3f ms, "
        "CANN mean/p50/min/max = %.3f/%.3f/%.3f/%.3f ms, PyPTO/CANN = %.3fx",
        title,
        pypto_stats["mean_ms"],
        pypto_stats["p50_ms"],
        pypto_stats["min_ms"],
        pypto_stats["max_ms"],
        cann_stats["mean_ms"],
        cann_stats["p50_ms"],
        cann_stats["min_ms"],
        cann_stats["max_ms"],
        ratio,
    )


def pypto_flash_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask_fp32: Optional[torch.Tensor],
    output: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query.shape[2] > BLOCK_SIZE_Q or key.shape[2] > BLOCK_SIZE_KV:
        raise NotImplementedError(
            f"Current PyPTO flash-attention backward is only verified for single-block shapes "
            f"(seq_q <= {BLOCK_SIZE_Q}, seq_kv <= {BLOCK_SIZE_KV}). "
            f"Got seq_q={query.shape[2]}, seq_kv={key.shape[2]}."
        )
    atten_mask_fp32 = _normalize_mask_fp32(query, key, atten_mask_fp32)
    grad_query = torch.zeros_like(query)
    grad_key = torch.zeros_like(key)
    grad_value = torch.zeros_like(value)
    flash_attention_score_backward_kernel_with_mask(
        query,
        key,
        value,
        atten_mask_fp32,
        output,
        grad_output,
        grad_query,
    )
    flash_attention_score_backward_kv_kernel_with_mask(
        query,
        key,
        value,
        atten_mask_fp32,
        output,
        grad_output,
        grad_key,
        grad_value,
    )
    return grad_query, grad_key, grad_value


class PyPTOFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        atten_mask_fp32: Optional[torch.Tensor],
    ) -> torch.Tensor:
        atten_mask_fp32 = _normalize_mask_fp32(query, key, atten_mask_fp32)
        output = pypto_flash_attention_forward(query, key, value, atten_mask_fp32)
        ctx.save_for_backward(query, key, value, atten_mask_fp32, output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value, atten_mask_fp32, output = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_query, grad_key, grad_value = pypto_flash_attention_backward(
            query,
            key,
            value,
            atten_mask_fp32,
            output,
            grad_output,
        )
        return grad_query, grad_key, grad_value, None


@allow_in_graph
def pypto_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask_fp32: Optional[torch.Tensor],
) -> torch.Tensor:
    if isinstance(query, FakeTensor):
        return torch.zeros_like(query)
    return PyPTOFlashAttentionFunction.apply(query, key, value, atten_mask_fp32)


def test_flash_attention_score(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    mask_type: str,
    device_id=None,
):
    """Test Flash Attention Score"""
    logging.info("=" * 60)
    logging.info("Test: Flash Attention Score with Online Softmax")
    logging.info("=" * 60)

    device = f"npu:{device_id}"

    query = torch.randn(batch_size, NUM_HEADS, seq_len_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    key = torch.randn(batch_size, NUM_HEADS, seq_len_kv, HEAD_DIM, dtype=torch.bfloat16, device=device)
    value = torch.randn(batch_size, NUM_HEADS, seq_len_kv, HEAD_DIM, dtype=torch.bfloat16, device=device)

    atten_mask_fp32, atten_mask_bool = make_mask(mask_type, seq_len_q, seq_len_kv, device)
    output = pypto_flash_attention_forward(query, key, value, atten_mask_fp32)
    logging.info("Reference mode: cann")
    golden = cann_flash_attention_forward(query, key, value, atten_mask_bool)

    logging.info(f"Input shape: query={query.shape}, key={key.shape}, value={value.shape}")
    logging.info(f"Output shape: {output.shape}")
    logging.info(
        "Kernel config: heads=%d, head_dim=%d, q_block=%d, kv_block=%d, mask=%s, qk_cube=%s, pv_cube=%s, vec=%s",
        NUM_HEADS,
        HEAD_DIM,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        mask_type,
        _format_cube_tile_shapes(QK_CUBE_TILE_SHAPES),
        _format_cube_tile_shapes(PV_CUBE_TILE_SHAPES),
        _format_vec_tile_shapes(VEC_TILE_SHAPES),
    )
    logging.info("CANN compare mode: training (SBH, sparse_mode=0)")
    if MODEL_PRESET in MODEL_SHAPE_PRESETS:
        logging.info(
            "Model preset: %s (%s)",
            MODEL_PRESET,
            MODEL_SHAPE_PRESETS[MODEL_PRESET]["note"],
        )

    output_fp32 = output.float()
    golden_fp32 = golden.float()
    max_diff = (output_fp32 - golden_fp32).abs().max().item()
    mean_diff = (output_fp32 - golden_fp32).abs().mean().item()

    logging.info(f"Max difference: {max_diff:.6f}")
    logging.info(f"Mean difference: {mean_diff:.6f}")
    logging.info("Forward compare tolerance: rtol=%.4f, atol=%.4f", FORWARD_COMPARE_RTOL, FORWARD_COMPARE_ATOL)

    assert_allclose(
        output_fp32.cpu().numpy().flatten(),
        golden_fp32.cpu().numpy().flatten(),
        rtol=FORWARD_COMPARE_RTOL,
        atol=FORWARD_COMPARE_ATOL,
    )
    logging.info("✓ Flash Attention Score test passed!")


def _run_reference_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask_bool: Optional[torch.Tensor],
    grad_output: torch.Tensor,
    reference_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if reference_mode == "golden_cpu":
        query_ref = query.detach().clone().to("cpu").requires_grad_(True)
        key_ref = key.detach().clone().to("cpu").requires_grad_(True)
        value_ref = value.detach().clone().to("cpu").requires_grad_(True)
        grad_output_ref = grad_output.detach().clone().to("cpu")
        atten_mask_ref = None if atten_mask_bool is None else atten_mask_bool.detach().clone().to("cpu")
        output_ref = flash_attention_score_golden(query_ref, key_ref, value_ref, atten_mask_ref)
        output_ref.backward(grad_output_ref)
        return output_ref.detach(), query_ref.grad.detach(), key_ref.grad.detach(), value_ref.grad.detach()

    query_ref = query.detach().clone().requires_grad_(True)
    key_ref = key.detach().clone().requires_grad_(True)
    value_ref = value.detach().clone().requires_grad_(True)
    if reference_mode == "cann":
        output_ref = cann_flash_attention_forward(query_ref, key_ref, value_ref, atten_mask_bool)
    else:
        output_ref = flash_attention_score_golden(query_ref, key_ref, value_ref, atten_mask_bool)
    output_ref.backward(grad_output)
    return output_ref.detach(), query_ref.grad.detach(), key_ref.grad.detach(), value_ref.grad.detach()


def test_flash_attention_backward(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    mask_type: str,
    device_id=None,
):
    logging.info("=" * 60)
    logging.info("Test: Flash Attention Score Backward")
    logging.info("=" * 60)

    device = f"npu:{device_id}"
    query = torch.randn(batch_size, NUM_HEADS, seq_len_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    key = torch.randn(batch_size, NUM_HEADS, seq_len_kv, HEAD_DIM, dtype=torch.bfloat16, device=device)
    value = torch.randn(batch_size, NUM_HEADS, seq_len_kv, HEAD_DIM, dtype=torch.bfloat16, device=device)
    grad_output = torch.randn(batch_size, NUM_HEADS, seq_len_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    atten_mask_fp32, atten_mask_bool = make_mask(mask_type, seq_len_q, seq_len_kv, device)

    query_py = query.detach().clone().requires_grad_(True)
    key_py = key.detach().clone().requires_grad_(True)
    value_py = value.detach().clone().requires_grad_(True)
    output_py = pypto_flash_attention(query_py, key_py, value_py, atten_mask_fp32)
    output_py.backward(grad_output)

    reference_mode = "golden_cpu"
    logging.info("Backward reference mode: %s", reference_mode)
    output_ref, grad_query_ref, grad_key_ref, grad_value_ref = _run_reference_backward(
        query,
        key,
        value,
        atten_mask_bool,
        grad_output,
        reference_mode,
    )

    output_py_fp32 = output_py.detach().float().cpu()
    output_ref_fp32 = output_ref.float().cpu()
    grad_query_py_fp32 = query_py.grad.detach().float().cpu()
    grad_key_py_fp32 = key_py.grad.detach().float().cpu()
    grad_value_py_fp32 = value_py.grad.detach().float().cpu()
    grad_query_ref_fp32 = grad_query_ref.float().cpu()
    grad_key_ref_fp32 = grad_key_ref.float().cpu()
    grad_value_ref_fp32 = grad_value_ref.float().cpu()

    logging.info("Forward max difference: %.6f", float(torch.max(torch.abs(output_py_fp32 - output_ref_fp32))))
    logging.info("dQ max difference: %.6f", float(torch.max(torch.abs(grad_query_py_fp32 - grad_query_ref_fp32))))
    logging.info("dK max difference: %.6f", float(torch.max(torch.abs(grad_key_py_fp32 - grad_key_ref_fp32))))
    logging.info("dV max difference: %.6f", float(torch.max(torch.abs(grad_value_py_fp32 - grad_value_ref_fp32))))

    assert_allclose(
        grad_query_py_fp32.cpu().numpy().flatten(),
        grad_query_ref_fp32.cpu().numpy().flatten(),
        rtol=0.03,
        atol=0.02,
    )
    assert_allclose(
        grad_key_py_fp32.cpu().numpy().flatten(),
        grad_key_ref_fp32.cpu().numpy().flatten(),
        rtol=0.03,
        atol=0.02,
    )
    assert_allclose(
        grad_value_py_fp32.cpu().numpy().flatten(),
        grad_value_ref_fp32.cpu().numpy().flatten(),
        rtol=0.03,
        atol=0.02,
    )
    logging.info("✓ Flash Attention Score backward test passed!")


def benchmark_flash_attention_score(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    mask_type: str,
    warmup: int,
    iters: int,
    device_id: int,
):
    logging.info("=" * 72)
    logging.info("Flash Attention Score Benchmark Summary")
    logging.info("=" * 72)

    device = f"npu:{device_id}"
    query = torch.randn(batch_size, NUM_HEADS, seq_len_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    key = torch.randn(batch_size, NUM_HEADS, seq_len_kv, HEAD_DIM, dtype=torch.bfloat16, device=device)
    value = torch.randn(batch_size, NUM_HEADS, seq_len_kv, HEAD_DIM, dtype=torch.bfloat16, device=device)
    atten_mask_fp32, atten_mask_bool = make_mask(mask_type, seq_len_q, seq_len_kv, device)
    cann_mask_supported = True

    pypto_first_call_ms = _measure_ms(lambda: pypto_flash_attention_forward(query, key, value, atten_mask_fp32))
    cann_first_call_ms = _measure_ms(lambda: cann_flash_attention_forward(query, key, value, atten_mask_bool))

    pypto_stats = _measure_many_ms(
        lambda: pypto_flash_attention_forward(query, key, value, atten_mask_fp32),
        warmup,
        iters,
    )
    cann_stats = _measure_many_ms(
        lambda: cann_flash_attention_forward(query, key, value, atten_mask_bool),
        warmup,
        iters,
    )

    logging.info(
        "Shape: batch=%d, heads=%d, seq_q=%d, seq_kv=%d, head_dim=%d, mask=%s",
        batch_size,
        NUM_HEADS,
        seq_len_q,
        seq_len_kv,
        HEAD_DIM,
        mask_type,
    )
    if MODEL_PRESET in MODEL_SHAPE_PRESETS:
        preset = MODEL_SHAPE_PRESETS[MODEL_PRESET]
        logging.info("Model preset: %s", MODEL_PRESET)
        logging.info("Preset source: %s", preset["source"])
        logging.info("Preset note: %s", preset["note"])
    logging.info(
        "Kernel config: q_block=%d, kv_block=%d, kv_unroll=%s, qk_cube=%s, pv_cube=%s, vec=%s",
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        ",".join(str(v) for v in KV_UNROLL_LIST),
        _format_cube_tile_shapes(QK_CUBE_TILE_SHAPES),
        _format_cube_tile_shapes(PV_CUBE_TILE_SHAPES),
        _format_vec_tile_shapes(VEC_TILE_SHAPES),
    )
    logging.info("CANN compare mode: training (SBH, sparse_mode=0)")
    logging.info("Warmup: %d, iters: %d", warmup, iters)
    logging.info("PyPTO first forward call (compile + run): %.3f ms", pypto_first_call_ms)
    logging.info("CANN first forward call: %.3f ms", cann_first_call_ms)
    logging.info("Steady-state results below exclude the first JIT compile path.")
    _print_stats("forward", pypto_stats, cann_stats)


def benchmark_flash_attention_score_training(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    mask_type: str,
    warmup: int,
    iters: int,
    device_id: int,
):
    logging.info("=" * 72)
    logging.info("Flash Attention Score Training Benchmark Summary")
    logging.info("=" * 72)

    device = f"npu:{device_id}"
    base_query = torch.randn(batch_size, NUM_HEADS, seq_len_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    base_key = torch.randn(batch_size, NUM_HEADS, seq_len_kv, HEAD_DIM, dtype=torch.bfloat16, device=device)
    base_value = torch.randn(batch_size, NUM_HEADS, seq_len_kv, HEAD_DIM, dtype=torch.bfloat16, device=device)
    grad_output = torch.randn(batch_size, NUM_HEADS, seq_len_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    atten_mask_fp32, atten_mask_bool = make_mask(mask_type, seq_len_q, seq_len_kv, device)
    pypto_first_forward_ms = _measure_ms(lambda: pypto_flash_attention_forward(base_query, base_key, base_value, atten_mask_fp32))
    cann_first_forward_ms = _measure_ms(lambda: cann_flash_attention_forward(base_query, base_key, base_value, atten_mask_bool))

    q_first = base_query.detach().clone().requires_grad_(True)
    k_first = base_key.detach().clone().requires_grad_(True)
    v_first = base_value.detach().clone().requires_grad_(True)
    y_first = pypto_flash_attention(q_first, k_first, v_first, atten_mask_fp32)
    pypto_first_backward_ms = _measure_ms(lambda: y_first.backward(grad_output))

    q_first_cann = base_query.detach().clone().requires_grad_(True)
    k_first_cann = base_key.detach().clone().requires_grad_(True)
    v_first_cann = base_value.detach().clone().requires_grad_(True)
    y_first_cann = cann_flash_attention_forward(q_first_cann, k_first_cann, v_first_cann, atten_mask_bool)
    cann_first_backward_ms = _measure_ms(lambda: y_first_cann.backward(grad_output))

    pypto_forward_stats = _measure_many_ms(
        lambda: pypto_flash_attention_forward(base_query, base_key, base_value, atten_mask_fp32),
        warmup,
        iters,
    )
    cann_forward_stats = _measure_many_ms(
        lambda: cann_flash_attention_forward(base_query, base_key, base_value, atten_mask_bool),
        warmup,
        iters,
    )

    def build_pypto_backward_graph():
        q = base_query.detach().clone().requires_grad_(True)
        k = base_key.detach().clone().requires_grad_(True)
        v = base_value.detach().clone().requires_grad_(True)
        return pypto_flash_attention(q, k, v, atten_mask_fp32)

    def build_cann_backward_graph():
        q = base_query.detach().clone().requires_grad_(True)
        k = base_key.detach().clone().requires_grad_(True)
        v = base_value.detach().clone().requires_grad_(True)
        return cann_flash_attention_forward(q, k, v, atten_mask_bool)

    def pypto_forward_backward():
        q = base_query.detach().clone().requires_grad_(True)
        k = base_key.detach().clone().requires_grad_(True)
        v = base_value.detach().clone().requires_grad_(True)
        y = pypto_flash_attention(q, k, v, atten_mask_fp32)
        y.backward(grad_output)

    def cann_forward_backward():
        q = base_query.detach().clone().requires_grad_(True)
        k = base_key.detach().clone().requires_grad_(True)
        v = base_value.detach().clone().requires_grad_(True)
        y = cann_flash_attention_forward(q, k, v, atten_mask_bool)
        y.backward(grad_output)

    pypto_backward_stats = _measure_backward_only_many_ms(build_pypto_backward_graph, grad_output, warmup, iters)
    cann_backward_stats = _measure_backward_only_many_ms(build_cann_backward_graph, grad_output, warmup, iters)
    pypto_fwdbwd_stats = _measure_many_ms(pypto_forward_backward, warmup, iters)
    cann_fwdbwd_stats = _measure_many_ms(cann_forward_backward, warmup, iters)

    logging.info(
        "Shape: batch=%d, heads=%d, seq_q=%d, seq_kv=%d, head_dim=%d, mask=%s",
        batch_size,
        NUM_HEADS,
        seq_len_q,
        seq_len_kv,
        HEAD_DIM,
        mask_type,
    )
    logging.info(
        "Kernel config: q_block=%d, kv_block=%d, kv_unroll=%s, qk_cube=%s, pv_cube=%s, vec=%s",
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        ",".join(str(v) for v in KV_UNROLL_LIST),
        _format_cube_tile_shapes(QK_CUBE_TILE_SHAPES),
        _format_cube_tile_shapes(PV_CUBE_TILE_SHAPES),
        _format_vec_tile_shapes(VEC_TILE_SHAPES),
    )
    logging.info("CANN compare mode: training (SBH, sparse_mode=0)")
    logging.info("Warmup: %d, iters: %d", warmup, iters)
    logging.info("PyPTO first forward call (compile + run): %.3f ms", pypto_first_forward_ms)
    logging.info("CANN first forward call: %.3f ms", cann_first_forward_ms)
    logging.info("PyPTO first backward call (compile + run): %.3f ms", pypto_first_backward_ms)
    logging.info("CANN first backward call: %.3f ms", cann_first_backward_ms)
    logging.info("Steady-state results below exclude the first JIT compile path.")
    _print_stats("forward", pypto_forward_stats, cann_forward_stats)
    _print_stats("backward", pypto_backward_stats, cann_backward_stats)
    _print_stats("forward+backward", pypto_fwdbwd_stats, cann_fwdbwd_stats)


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Flash Attention Score Example",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "example_id",
        type=str,
        nargs="?",
        default="flash_attention::test_forward",
        help="Example ID to run.",
    )
    args = parser.parse_args()

    logging.info("\n" + "=" * 60)
    logging.info("PyPTO Flash Attention Score Example")
    logging.info("=" * 60 + "\n")

    device_id = get_device_id()
    if device_id is None:
        return
    import torch_npu
    torch.npu.set_device(device_id)
    logging.info(f"Running on NPU device {device_id}\n")

    try:
        examples = {
            "flash_attention::test_forward": lambda: test_flash_attention_score(
                DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN_Q, DEFAULT_SEQ_LEN_KV, DEFAULT_MASK_TYPE, device_id
            ),
            "flash_attention::test_backward": lambda: test_flash_attention_backward(
                DEFAULT_BATCH_SIZE, BACKWARD_SEQ_LEN_Q, BACKWARD_SEQ_LEN_KV, DEFAULT_MASK_TYPE, device_id
            ),
            "flash_attention::benchmark": lambda: benchmark_flash_attention_score(
                DEFAULT_BATCH_SIZE,
                DEFAULT_SEQ_LEN_Q,
                DEFAULT_SEQ_LEN_KV,
                DEFAULT_MASK_TYPE,
                DEFAULT_BENCHMARK_WARMUP,
                DEFAULT_BENCHMARK_ITERS,
                device_id,
            ),
            "flash_attention::benchmark_train": lambda: benchmark_flash_attention_score_training(
                DEFAULT_BATCH_SIZE,
                BACKWARD_SEQ_LEN_Q,
                BACKWARD_SEQ_LEN_KV,
                DEFAULT_MASK_TYPE,
                DEFAULT_BENCHMARK_WARMUP,
                DEFAULT_BENCHMARK_ITERS,
                device_id,
            ),
        }
        if args.example_id not in examples:
            raise ValueError(f"Unsupported example_id: {args.example_id}")
        examples[args.example_id]()

        logging.info("=" * 60)
        logging.info("All tests passed!")
        logging.info("=" * 60)
    except Exception as e:
        logging.info(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
