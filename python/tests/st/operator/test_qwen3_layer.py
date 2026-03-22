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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pypto
import torch

from st.pypto_test import TestBuilder
from pypto.runtime import _device_run_once_data_from_host as device_run_once_data_from_host

torch.manual_seed(0)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OUTPUT_ROOT = _REPO_ROOT / "output"


@dataclass
class PaTileConfig:
    head_num_q_tile: int
    c1_tile_shape: tuple
    v1_tile_shape: tuple
    c2_tile_shape: tuple
    v2_tile_shape: tuple


def _make_qwen3_pa_tile_config() -> PaTileConfig:
    return PaTileConfig(
        head_num_q_tile=4,
        c1_tile_shape=(4, 4, 128, 128, 256, 256),
        v1_tile_shape=(4, 2048),
        c2_tile_shape=(4, 4, 128, 128, 128, 128),
        v2_tile_shape=(4, 512),
    )


def _make_qwen3_prolog_params():
    return {
        "b": 32,
        "s": 1,
        "n": 32,
        "n_kv": 8,
        "d": 128,
        "block_size": 2048,
        "dtype": torch.bfloat16,
    }


def _make_qwen3_attention_params():
    params = _make_qwen3_prolog_params()
    params.update(
        {
            "skv": 2048,
            "max_unroll_times": 8,
            "pa_tile_config": _make_qwen3_pa_tile_config(),
        }
    )
    return params


def _make_qwen3_mlp_params():
    return {
        "b": 32,
        "s": 1,
        "h": 4096,
        "inter": 12288,
        "dtype": torch.bfloat16,
    }


def _make_qwen3_layer_params():
    params = _make_qwen3_attention_params()
    params["inter"] = 12288
    return params


@dataclass(frozen=True)
class PaRowSpec:
    block_size: int
    act_seq: int
    kv_idx: int
    d: int


@dataclass(frozen=True)
class QKVProjectionTensors:
    attn_q_w: torch.Tensor
    attn_q_b: torch.Tensor
    attn_k_w: torch.Tensor
    attn_k_b: torch.Tensor
    attn_v_w: torch.Tensor
    attn_v_b: torch.Tensor
    attn_q_norm_w: torch.Tensor
    attn_k_norm_w: torch.Tensor


@dataclass(frozen=True)
class CacheUpdateTensors:
    cache_index: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor


@dataclass(frozen=True)
class MlpWeightTensors:
    gate_w: object
    up_w: object
    down_w: object


@dataclass(frozen=True)
class LayerPostTensors:
    hidden_states: object
    pa_out: object
    attn_o_w: object
    attn_o_b: object
    mlp: MlpWeightTensors


@dataclass(frozen=True)
class AttentionTorchTensors:
    q_4d: object
    key_cache: object
    value_cache: object
    block_table: object
    act_seqs: object


@dataclass(frozen=True)
class PrologCaseTensors:
    hidden_states: object
    proj: QKVProjectionTensors
    cos: object
    sin: object
    cache: CacheUpdateTensors


@dataclass(frozen=True)
class LayerCaseTensors:
    hidden_states: object
    proj: QKVProjectionTensors
    attn_o_w: object
    attn_o_b: object
    cos: object
    sin: object
    cache: CacheUpdateTensors
    block_table: object
    act_seqs: object
    mlp: MlpWeightTensors


@dataclass(frozen=True)
class AttentionGraphRuntime:
    query: object
    key_cache: object
    value_cache: object
    block_table: object
    attention_out: object


@dataclass(frozen=True)
class AttentionGraphConfig:
    block_size: int
    max_unroll_times: int
    group: int
    n_tile: int
    d: int
    c1_tile: tuple
    v1_tile: tuple
    c2_tile: tuple
    v2_tile: tuple
    softmax_scale: float


@dataclass(frozen=True)
class AttentionLoopContext:
    b_idx: object
    bn_per_batch: object
    cur_seq: object
    cur_offset: object
    kv_idx: object


@dataclass(frozen=True)
class AttentionAccumState:
    oi_update: object
    li_update: object
    mi_update: object


@dataclass(frozen=True)
class AttentionBlockTerms:
    bn: object
    ctx: AttentionLoopContext
    vj: object
    tilda_pij_dt: object
    tilda_lij: object
    tilda_mij: object


def _gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.rand(data_shape) < 0.5
    return (torch.rand(data_shape) * (max_value - min_value) + min_value).to(dtype)


def _convert_tensors_contiguous(tensor_list: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    result = []
    for tensor in tensor_list:
        result.append(tensor if not isinstance(tensor, torch.Tensor) or tensor.is_contiguous() else tensor.contiguous())
    return result


def _align_up(value: int, base: int) -> int:
    if value <= 0:
        return base
    return ((value + base - 1) // base) * base


def _get_matmul_tile(m: int, k: int, n: int, dtype):
    m_tile = _align_up(min(max(m, 1), 64), 16)
    k_tile = _align_up(min(max(k, 1), 256), 16)
    n_tile = _align_up(min(max(n, 1), 256), 16)
    bytes_per_elem = 4 if dtype == pypto.DT_FP32 else 2
    l0b_bytes = 64 * 1024
    while k_tile * n_tile * bytes_per_elem > l0b_bytes:
        if n_tile >= k_tile and n_tile > 16:
            n_tile = max(16, n_tile // 2)
        elif k_tile > 16:
            k_tile = max(16, k_tile // 2)
        else:
            break
        k_tile = _align_up(k_tile, 16)
        n_tile = _align_up(n_tile, 16)
    return m_tile, k_tile, n_tile


def _set_matmul_tile(m: int, k: int, n: int, dtype):
    m_tile, k_tile, n_tile = _get_matmul_tile(m, k, n, dtype)
    pypto.set_cube_tile_shapes([m_tile, m_tile], [k_tile, k_tile], [n_tile, n_tile])
    pypto.set_matrix_size([m, k, n])


def _set_cube_tile_shapes_from_tuple(tile_shape: tuple):
    pypto.set_cube_tile_shapes(
        [tile_shape[0], tile_shape[1]],
        [tile_shape[2], tile_shape[3]],
        [tile_shape[4], tile_shape[5]],
    )


def _build_qkv_projection_tensors(hidden_size: int, kv_hidden: int, d: int, dtype) -> QKVProjectionTensors:
    return QKVProjectionTensors(
        attn_q_w=_gen_uniform_data([hidden_size, hidden_size], -1, 1, dtype),
        attn_q_b=_gen_uniform_data([hidden_size], -1, 1, dtype),
        attn_k_w=_gen_uniform_data([hidden_size, kv_hidden], -1, 1, dtype),
        attn_k_b=_gen_uniform_data([kv_hidden], -1, 1, dtype),
        attn_v_w=_gen_uniform_data([hidden_size, kv_hidden], -1, 1, dtype),
        attn_v_b=_gen_uniform_data([kv_hidden], -1, 1, dtype),
        attn_q_norm_w=_gen_uniform_data([d], -1, 1, dtype),
        attn_k_norm_w=_gen_uniform_data([d], -1, 1, dtype),
    )


def _flatten_qkv_projection_tensors(proj: QKVProjectionTensors) -> List[torch.Tensor]:
    return [
        proj.attn_q_w,
        proj.attn_q_b,
        proj.attn_k_w,
        proj.attn_k_b,
        proj.attn_v_w,
        proj.attn_v_b,
        proj.attn_q_norm_w,
        proj.attn_k_norm_w,
    ]


def _build_mlp_weight_tensors(hidden_size: int, inter: int, dtype) -> MlpWeightTensors:
    return MlpWeightTensors(
        gate_w=_gen_uniform_data([hidden_size, inter], -1, 1, dtype),
        up_w=_gen_uniform_data([hidden_size, inter], -1, 1, dtype),
        down_w=_gen_uniform_data([inter, hidden_size], -1, 1, dtype),
    )


def _flatten_mlp_weight_tensors(mlp: MlpWeightTensors) -> List[torch.Tensor]:
    return [mlp.gate_w, mlp.up_w, mlp.down_w]


def _torch_dtype_to_pypto(dtype: torch.dtype):
    if dtype == torch.float16:
        return pypto.DT_FP16
    if dtype == torch.float32:
        return pypto.DT_FP32
    if dtype == torch.bfloat16:
        return pypto.DT_BF16
    raise ValueError(f"Unsupported torch dtype for frontend.jit kernel: {dtype}")


def _build_prolog_act_seq_list(params) -> List[int]:
    b = params["b"]
    s = params["s"]
    block_size = params["block_size"]
    if b == 1:
        return [block_size * 2 + s]
    return [block_size * 2 + s, block_size * 2 - 1 + s] + [block_size * 2 + s] * (b - 2)


def _build_attention_act_seq_list(params) -> List[int]:
    skv = params["skv"]
    b = params["b"]
    return [skv] * b if isinstance(skv, int) else [int(x) for x in skv]


def _get_cache_layout(act_seq_list: List[int], block_size: int):
    block_num_per_batch = [int(math.ceil(x / block_size)) for x in act_seq_list]
    max_block_num_per_batch = int(max(block_num_per_batch))
    block_num = int(sum(block_num_per_batch))
    return block_num * block_size, max_block_num_per_batch


def _collect_output_dirs():
    if not _OUTPUT_ROOT.exists():
        return set()
    return {path.resolve() for path in _OUTPUT_ROOT.glob("output_*") if path.is_dir()}


def _rmsnorm_torch(x: torch.Tensor, gamma: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    rstd = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x_fp32 * rstd
    if gamma is not None:
        out = out * gamma.to(torch.float32)
    return out.to(x.dtype)


def _rope_rearrange_torch(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return x.reshape(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(*x.shape)


def _rotate_half_torch(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return torch.cat((-x[..., d // 2:], x[..., :d // 2]), dim=-1)


def _apply_rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_fp32 = _rope_rearrange_torch(x.to(torch.float32))
    cos_u = cos.to(torch.float32).unsqueeze(2)
    sin_u = sin.to(torch.float32).unsqueeze(2)
    return (x_fp32 * cos_u + _rotate_half_torch(x_fp32) * sin_u).to(x.dtype)


def _build_block_table(act_seq_list: List[int], block_size: int):
    block_num_per_batch = [int(math.ceil(x / block_size)) for x in act_seq_list]
    block_num = int(sum(block_num_per_batch))
    max_block_num_per_batch = int(max(block_num_per_batch))
    block_table = torch.full([len(act_seq_list), max_block_num_per_batch], -1, dtype=torch.int32)
    block_id = 0
    for batch_idx, block_num_in_batch in enumerate(block_num_per_batch):
        for block_idx in range(block_num_in_batch):
            block_table[batch_idx, block_idx] = block_id
            block_id += 1
    return block_table, block_num_per_batch, block_num


def _build_cache_from_bsnd(k_bsnd: torch.Tensor, v_bsnd: torch.Tensor, act_seq_list: List[int], block_size: int):
    b, _, n_kv, d = k_bsnd.shape
    block_table, block_num_per_batch, block_num = _build_block_table(act_seq_list, block_size)
    k_cache = torch.zeros([block_num, block_size, n_kv, d], dtype=k_bsnd.dtype)
    v_cache = torch.zeros([block_num, block_size, n_kv, d], dtype=v_bsnd.dtype)
    for batch_idx, seq_len in enumerate(act_seq_list):
        for block_idx in range(block_num_per_batch[batch_idx]):
            global_block_id = int(block_table[batch_idx, block_idx].item())
            start = block_idx * block_size
            end = min(start + block_size, seq_len)
            if end > start:
                k_cache[global_block_id, 0:(end - start), :, :] = k_bsnd[batch_idx, start:end, :, :]
                v_cache[global_block_id, 0:(end - start), :, :] = v_bsnd[batch_idx, start:end, :, :]
    return (
        block_table,
        block_num_per_batch,
        k_cache.reshape(block_num * block_size, n_kv * d),
        v_cache.reshape(block_num * block_size, n_kv * d),
    )


def _build_cache_position_from_act_seq_list(act_seq_list: List[int], s: int):
    b = len(act_seq_list)
    cache_position = torch.zeros([b, s], dtype=torch.int32)
    for batch_idx, act_seq in enumerate(act_seq_list):
        start_pos = int(act_seq) - int(s)
        for token_idx in range(s):
            cache_position[batch_idx, token_idx] = start_pos + token_idx
    return cache_position


def _build_cache_index(cache_position: torch.Tensor, block_table: torch.Tensor, block_size: int):
    b, s = cache_position.shape
    cache_index = torch.zeros([b, s], dtype=torch.int32)
    for batch_idx in range(b):
        for token_idx in range(s):
            pos = int(cache_position[batch_idx, token_idx].item())
            block_idx_in_batch = pos // block_size
            offset_in_block = pos % block_size
            global_block_id = int(block_table[batch_idx, block_idx_in_batch].item())
            cache_index[batch_idx, token_idx] = global_block_id * block_size + offset_in_block
    return cache_index


def _build_rope_from_cache_position(cache_position: torch.Tensor, d: int, dtype):
    pos_fp32 = cache_position.to(torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float32) / float(d)))
    freqs = pos_fp32.unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb).to(dtype), torch.sin(emb).to(dtype)


def _build_pa_rows_from_cache(cache: torch.Tensor, block_table: torch.Tensor, spec: PaRowSpec):
    rows = []
    for pos in range(spec.act_seq):
        block_idx_in_batch = pos // spec.block_size
        offset_in_block = pos % spec.block_size
        global_block_id = int(block_table[block_idx_in_batch].item())
        row_idx = global_block_id * spec.block_size + offset_in_block
        rows.append(cache[row_idx, spec.kv_idx * spec.d:(spec.kv_idx + 1) * spec.d])
    return torch.stack(rows, dim=0)


def _build_prolog_case_tensors(case_args) -> PrologCaseTensors:
    (
        hidden_states,
        attn_q_w,
        attn_q_b,
        attn_k_w,
        attn_k_b,
        attn_v_w,
        attn_v_b,
        attn_q_norm_w,
        attn_k_norm_w,
        cos,
        sin,
        cache_index,
        key_cache,
        value_cache,
    ) = case_args[:14]
    return PrologCaseTensors(
        hidden_states=hidden_states,
        proj=QKVProjectionTensors(
            attn_q_w,
            attn_q_b,
            attn_k_w,
            attn_k_b,
            attn_v_w,
            attn_v_b,
            attn_q_norm_w,
            attn_k_norm_w,
        ),
        cos=cos,
        sin=sin,
        cache=CacheUpdateTensors(cache_index, key_cache, value_cache),
    )


def _build_layer_case_tensors(case_args) -> LayerCaseTensors:
    (
        hidden_states,
        attn_q_w,
        attn_q_b,
        attn_k_w,
        attn_k_b,
        attn_v_w,
        attn_v_b,
        attn_o_w,
        attn_o_b,
        attn_q_norm_w,
        attn_k_norm_w,
        cos,
        sin,
        cache_index,
        key_cache,
        value_cache,
        block_table,
        act_seqs,
        gate_w,
        up_w,
        down_w,
    ) = case_args[:21]
    return LayerCaseTensors(
        hidden_states=hidden_states,
        proj=QKVProjectionTensors(
            attn_q_w,
            attn_q_b,
            attn_k_w,
            attn_k_b,
            attn_v_w,
            attn_v_b,
            attn_q_norm_w,
            attn_k_norm_w,
        ),
        attn_o_w=attn_o_w,
        attn_o_b=attn_o_b,
        cos=cos,
        sin=sin,
        cache=CacheUpdateTensors(cache_index, key_cache, value_cache),
        block_table=block_table,
        act_seqs=act_seqs,
        mlp=MlpWeightTensors(gate_w, up_w, down_w),
    )


def _build_layer_post_tensors(tensors: LayerCaseTensors, pa_out) -> LayerPostTensors:
    return LayerPostTensors(
        hidden_states=tensors.hidden_states,
        pa_out=pa_out,
        attn_o_w=tensors.attn_o_w,
        attn_o_b=tensors.attn_o_b,
        mlp=tensors.mlp,
    )


def _build_attention_torch_tensors(q_4d, case_args) -> AttentionTorchTensors:
    query, key_cache, value_cache, block_table, act_seqs = case_args[:5]
    del query
    return AttentionTorchTensors(q_4d=q_4d, key_cache=key_cache, value_cache=value_cache, block_table=block_table, act_seqs=act_seqs)


def _build_layer_cache_index(b: int, s: int, skv: int, block_size: int):
    cache_index = torch.zeros([b, s], dtype=torch.int32)
    for batch_idx in range(b):
        batch_offset = batch_idx * block_size
        for token_idx in range(s):
            cache_index[batch_idx, token_idx] = skv - s + token_idx + batch_offset
    return cache_index


def _run_qkv_rope_cache_torch(
    params,
    hidden_states: torch.Tensor,
    proj: QKVProjectionTensors,
    rope: tuple[torch.Tensor, torch.Tensor],
    cache: CacheUpdateTensors,
):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    n_kv = params["n_kv"]
    d = params["d"]
    kv_hidden = n_kv * d
    cos, sin = rope

    q = torch.matmul(hidden_states.to(torch.float32), proj.attn_q_w.to(torch.float32)).to(hidden_states.dtype)
    q = (q.to(torch.float32) + proj.attn_q_b.to(torch.float32)).to(hidden_states.dtype)
    k = torch.matmul(hidden_states.to(torch.float32), proj.attn_k_w.to(torch.float32)).to(hidden_states.dtype)
    k = (k.to(torch.float32) + proj.attn_k_b.to(torch.float32)).to(hidden_states.dtype)
    v = torch.matmul(hidden_states.to(torch.float32), proj.attn_v_w.to(torch.float32)).to(hidden_states.dtype)
    v = (v.to(torch.float32) + proj.attn_v_b.to(torch.float32)).to(hidden_states.dtype)

    q = _rmsnorm_torch(q.reshape(b * s * n_q, d), proj.attn_q_norm_w).reshape(b, s, n_q, d)
    k = _rmsnorm_torch(k.reshape(b * s * n_kv, d), proj.attn_k_norm_w).reshape(b, s, n_kv, d)
    v = v.reshape(b, s, n_kv, d)

    q_embed = _apply_rope_torch(q, cos, sin)
    k_embed = _apply_rope_torch(k, cos, sin)
    query_out = q_embed.reshape(b * s * n_q, d).contiguous()

    key_cache_out = cache.key_cache.clone()
    value_cache_out = cache.value_cache.clone()
    key_rows = k_embed.reshape(b * s, kv_hidden)
    value_rows = v.reshape(b * s, kv_hidden)
    for batch_idx in range(b):
        for token_idx in range(s):
            row = int(cache.cache_index[batch_idx, token_idx].item())
            src = batch_idx * s + token_idx
            key_cache_out[row, :] = key_rows[src, :]
            value_cache_out[row, :] = value_rows[src, :]

    return query_out, key_cache_out, value_cache_out


def _build_prolog_runner_inputs(params):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    n_kv = params["n_kv"]
    d = params["d"]
    block_size = params["block_size"]
    dtype = params["dtype"]
    hidden_size = n_q * d
    kv_hidden = n_kv * d

    hidden_states = _gen_uniform_data([b * s, hidden_size], -1, 1, dtype)
    proj = _build_qkv_projection_tensors(hidden_size, kv_hidden, d, dtype)
    act_seq_list = _build_prolog_act_seq_list(params)
    cache_position = _build_cache_position_from_act_seq_list(act_seq_list, s)
    cos, sin = _build_rope_from_cache_position(cache_position, d, dtype)
    block_table, _, block_num = _build_block_table(act_seq_list, block_size)
    key_cache = _gen_uniform_data([block_num * block_size, kv_hidden], -1, 1, dtype)
    value_cache = _gen_uniform_data([block_num * block_size, kv_hidden], -1, 1, dtype)
    cache_index = _build_cache_index(cache_position, block_table, block_size)
    inputs = _convert_tensors_contiguous(
        [hidden_states, *_flatten_qkv_projection_tensors(proj), cos, sin, cache_index, key_cache, value_cache]
    )
    return inputs, cache_index


def _build_attention_runner_inputs(params):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    n_kv = params["n_kv"]
    d = params["d"]
    block_size = params["block_size"]
    dtype = params["dtype"]
    actual_seq_list = _build_attention_act_seq_list(params)
    act_seqs = torch.tensor(actual_seq_list, dtype=torch.int32)
    max_seq = int(max(actual_seq_list))
    q_4d = _gen_uniform_data([b, s, n_q, d], -1, 1, dtype)
    k_bsnd = _gen_uniform_data([b, max_seq, n_kv, d], -1, 1, dtype)
    v_bsnd = _gen_uniform_data([b, max_seq, n_kv, d], -1, 1, dtype)
    block_table, _, key_cache, value_cache = _build_cache_from_bsnd(k_bsnd, v_bsnd, actual_seq_list, block_size)
    query = q_4d.reshape(b * s * n_q, d)
    inputs = _convert_tensors_contiguous([query, key_cache, value_cache, block_table, act_seqs])
    return inputs, q_4d


def _build_mlp_runner_inputs(params):
    b = params["b"]
    s = params["s"]
    h = params["h"]
    inter = params["inter"]
    dtype = params["dtype"]
    hidden_states = _gen_uniform_data([b * s, h], -1, 1, dtype)
    mlp = _build_mlp_weight_tensors(h, inter, dtype)
    return _convert_tensors_contiguous([hidden_states, *_flatten_mlp_weight_tensors(mlp)])


def _build_layer_cache_tensors(params, kv_hidden: int):
    b = params["b"]
    n_kv = params["n_kv"]
    d = params["d"]
    skv = params["skv"]
    block_size = params["block_size"]
    dtype = params["dtype"]
    act_seq_list = [skv] * b
    block_table, _, block_num = _build_block_table(act_seq_list, block_size)
    k_bsnd = _gen_uniform_data([b, skv, n_kv, d], -1, 1, dtype)
    v_bsnd = _gen_uniform_data([b, skv, n_kv, d], -1, 1, dtype)
    key_cache = torch.zeros([block_num * block_size, kv_hidden], dtype=dtype)
    value_cache = torch.zeros([block_num * block_size, kv_hidden], dtype=dtype)
    for batch_idx, seq_len in enumerate(act_seq_list):
        block_num_in_batch = int(math.ceil(seq_len / block_size))
        for block_idx in range(block_num_in_batch):
            global_block_id = int(block_table[batch_idx, block_idx].item())
            start = block_idx * block_size
            end = min(start + block_size, seq_len)
            if end <= start:
                continue
            cache_slice = slice(global_block_id * block_size, global_block_id * block_size + (end - start))
            key_cache[cache_slice, :] = k_bsnd[batch_idx, start:end, :, :].reshape(end - start, kv_hidden)
            value_cache[cache_slice, :] = v_bsnd[batch_idx, start:end, :, :].reshape(end - start, kv_hidden)
    return act_seq_list, block_table, key_cache, value_cache


def _build_layer_runner_inputs(params):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    n_kv = params["n_kv"]
    d = params["d"]
    inter = params["inter"]
    skv = params["skv"]
    block_size = params["block_size"]
    dtype = params["dtype"]
    hidden_size = n_q * d
    kv_hidden = n_kv * d

    hidden_states = _gen_uniform_data([b * s, hidden_size], -1, 1, dtype)
    proj = _build_qkv_projection_tensors(hidden_size, kv_hidden, d, dtype)
    attn_o_w = _gen_uniform_data([hidden_size, hidden_size], -1, 1, dtype)
    attn_o_b = _gen_uniform_data([hidden_size], -1, 1, dtype)
    cos = _gen_uniform_data([b, s, d], -1, 1, dtype)
    sin = _gen_uniform_data([b, s, d], -1, 1, dtype)
    cache_index = _build_layer_cache_index(b, s, skv, block_size)
    act_seq_list, block_table, key_cache, value_cache = _build_layer_cache_tensors(params, kv_hidden)
    act_seqs = torch.tensor(act_seq_list, dtype=torch.int32)
    mlp = _build_mlp_weight_tensors(hidden_size, inter, dtype)
    inputs = _convert_tensors_contiguous(
        [
            hidden_states,
            proj.attn_q_w,
            proj.attn_q_b,
            proj.attn_k_w,
            proj.attn_k_b,
            proj.attn_v_w,
            proj.attn_v_b,
            attn_o_w,
            attn_o_b,
            proj.attn_q_norm_w,
            proj.attn_k_norm_w,
            cos,
            sin,
            cache_index,
            key_cache,
            value_cache,
            block_table,
            act_seqs,
            *_flatten_mlp_weight_tensors(mlp),
        ]
    )
    return inputs


def _run_layer_post_torch(post: LayerPostTensors):
    context = post.pa_out.reshape(post.hidden_states.shape[0], -1).to(post.hidden_states.dtype)
    o_proj = torch.matmul(context.to(torch.float32), post.attn_o_w.to(torch.float32)).to(post.hidden_states.dtype)
    o_proj = (o_proj.to(torch.float32) + post.attn_o_b.to(torch.float32)).to(post.hidden_states.dtype)
    h1 = (post.hidden_states.to(torch.float32) + o_proj.to(torch.float32)).to(post.hidden_states.dtype)
    norm2 = _rmsnorm_torch(h1)
    gate = torch.matmul(norm2.to(torch.float32), post.mlp.gate_w.to(torch.float32)).to(post.hidden_states.dtype)
    up = torch.matmul(norm2.to(torch.float32), post.mlp.up_w.to(torch.float32)).to(post.hidden_states.dtype)
    gate_act = gate * torch.sigmoid(gate.to(torch.float32)).to(post.hidden_states.dtype)
    gate_up = (gate_act.to(torch.float32) * up.to(torch.float32)).to(post.hidden_states.dtype)
    mlp_out = torch.matmul(gate_up.to(torch.float32), post.mlp.down_w.to(torch.float32)).to(post.hidden_states.dtype)
    return (h1.to(torch.float32) + mlp_out.to(torch.float32)).to(post.hidden_states.dtype)


def _run_paged_attention_torch(params, attention: AttentionTorchTensors):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    n_kv = params["n_kv"]
    d = params["d"]
    block_size = params["block_size"]
    group = n_q // n_kv
    out = torch.zeros([b, s, n_q, d], dtype=torch.float32)
    scale = 1.0 / math.sqrt(d)
    for batch_idx in range(b):
        cur_block_table = attention.block_table[batch_idx]
        act_seq = int(attention.act_seqs[batch_idx].item())
        for token_idx in range(s):
            cur_seq = max(act_seq - s + 1 + token_idx, 0)
            for kv_idx in range(n_kv):
                q_group = attention.q_4d[batch_idx, token_idx, kv_idx * group:(kv_idx + 1) * group, :].to(
                    torch.float32
                )
                spec = PaRowSpec(block_size, cur_seq, kv_idx, d)
                k_cur = _build_pa_rows_from_cache(attention.key_cache, cur_block_table, spec).to(torch.float32)
                v_cur = _build_pa_rows_from_cache(attention.value_cache, cur_block_table, spec).to(torch.float32)
                scores = torch.matmul(q_group, k_cur.transpose(-1, -2)) * scale
                probs = torch.softmax(scores, dim=-1)
                out_group = torch.matmul(probs, v_cur)
                out[batch_idx, token_idx, kv_idx * group:(kv_idx + 1) * group, :] = out_group
    return out.reshape(b * s * n_q, d)


def _rotate_half_graph(x: pypto.Tensor) -> pypto.Tensor:
    rank = x.dim
    shape = list(x.shape)
    half = x.shape[-1] // 2
    left_shape = shape[:-1] + [half]
    zeros = [0] * rank
    left = pypto.view(x, left_shape, zeros)
    right = pypto.view(x, left_shape, zeros[:-1] + [half])
    out = pypto.tensor(shape, pypto.DT_FP32, "qwen3_rotate_half")
    pypto.assemble(pypto.mul(right, -1.0), zeros, out)
    pypto.assemble(left, zeros[:-1] + [half], out)
    return out


def _rope_rearrange_graph(x: pypto.Tensor) -> pypto.Tensor:
    shape = list(x.shape)
    half = x.shape[-1] // 2
    tile_mid = max(16, min(64, int(shape[-1])))
    if x.dim == 2:
        pypto.set_vec_tile_shapes(1, tile_mid, tile_mid)
    elif x.dim == 3:
        pypto.set_vec_tile_shapes(1, min(8, int(shape[1])), tile_mid, tile_mid)
    elif x.dim == 4:
        pypto.set_vec_tile_shapes(1, min(8, int(shape[1])), tile_mid, tile_mid)
    x_view = pypto.reshape(x, shape[:-1] + [half, 2])
    x_trans = pypto.transpose(x_view, x_view.dim - 2, x_view.dim - 1)
    return pypto.reshape(x_trans, shape)


def _apply_rope_graph(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor) -> pypto.Tensor:
    b, s, n, d = list(x.shape)
    x_fp32 = pypto.cast(x, pypto.DT_FP32)
    x_3d = pypto.reshape(x_fp32, [b * s, n, d])
    cos_3d = pypto.reshape(pypto.cast(cos, pypto.DT_FP32), [b * s, 1, d])
    sin_3d = pypto.reshape(pypto.cast(sin, pypto.DT_FP32), [b * s, 1, d])
    x_rearranged = _rope_rearrange_graph(x_3d)
    tile_last = max(16, min(64, d))
    pypto.set_vec_tile_shapes(1, min(8, n), tile_last)
    rotated = _rotate_half_graph(x_rearranged)
    x_embed = pypto.add(pypto.mul(x_rearranged, cos_3d), pypto.mul(rotated, sin_3d))
    pypto.set_vec_tile_shapes(1, 1, min(8, n), tile_last)
    return pypto.cast(pypto.reshape(x_embed, [b, s, n, d]), x.dtype)


def qwen3_mlp_graph(params, hidden_states, *mlp_args):
    gate_w, up_w, down_w, mlp_out = mlp_args
    m = params["b"] * params["s"]
    h = params["h"]
    inter = params["inter"]
    dtype = hidden_states.dtype

    pypto.set_vec_tile_shapes(1, h)
    _set_matmul_tile(m, h, inter, dtype)
    gate = pypto.matmul(hidden_states, gate_w, dtype)
    _set_matmul_tile(m, h, inter, dtype)
    up = pypto.matmul(hidden_states, up_w, dtype)

    gate_sig = pypto.cast(pypto.sigmoid(pypto.cast(gate, pypto.DT_FP32)), dtype)
    gate_act = pypto.mul(gate, gate_sig)
    inter_fp32 = pypto.mul(pypto.cast(gate_act, pypto.DT_FP32), pypto.cast(up, pypto.DT_FP32))
    inter_dt = pypto.cast(inter_fp32, dtype)

    _set_matmul_tile(m, inter, h, dtype)
    mlp_out[:] = pypto.matmul(inter_dt, down_w, dtype)


def _project_qkv_graph(params, hidden_states, proj: QKVProjectionTensors):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    n_kv = params["n_kv"]
    d = params["d"]
    hidden_size = n_q * d
    kv_hidden = n_kv * d
    dtype = hidden_states.dtype

    _set_matmul_tile(b * s, hidden_size, hidden_size, dtype)
    q = pypto.matmul(hidden_states, proj.attn_q_w, dtype)
    q = pypto.cast(pypto.add(pypto.cast(q, pypto.DT_FP32), pypto.cast(proj.attn_q_b, pypto.DT_FP32)), dtype)

    _set_matmul_tile(b * s, hidden_size, kv_hidden, dtype)
    k = pypto.matmul(hidden_states, proj.attn_k_w, dtype)
    k = pypto.cast(pypto.add(pypto.cast(k, pypto.DT_FP32), pypto.cast(proj.attn_k_b, pypto.DT_FP32)), dtype)

    _set_matmul_tile(b * s, hidden_size, kv_hidden, dtype)
    v = pypto.matmul(hidden_states, proj.attn_v_w, dtype)
    v = pypto.cast(pypto.add(pypto.cast(v, pypto.DT_FP32), pypto.cast(proj.attn_v_b, pypto.DT_FP32)), dtype)
    return q, k, v


def _build_prolog_rows_graph(params, q, k, v, proj: QKVProjectionTensors, cos, sin):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    n_kv = params["n_kv"]
    d = params["d"]
    kv_hidden = n_kv * d
    pypto.set_vec_tile_shapes(min(16, b * s * n_q), d)
    q = pypto.reshape(q, [b * s * n_q, d])
    q = pypto.rms_norm(q, proj.attn_q_norm_w)

    pypto.set_vec_tile_shapes(min(16, b * s * n_kv), d)
    k = pypto.reshape(k, [b * s * n_kv, d])
    k = pypto.rms_norm(k, proj.attn_k_norm_w)

    tile_last = max(16, min(64, d))
    pypto.set_vec_tile_shapes(1, 1, min(8, n_q), tile_last)
    q_4d = pypto.reshape(q, [b, s, n_q, d])
    pypto.set_vec_tile_shapes(1, 1, min(8, n_kv), tile_last)
    k_4d = pypto.reshape(k, [b, s, n_kv, d])
    pypto.set_vec_tile_shapes(1, 1, min(8, n_kv), tile_last)
    v_4d = pypto.reshape(v, [b, s, n_kv, d])

    q_embed = _apply_rope_graph(q_4d, cos, sin)
    k_embed = _apply_rope_graph(k_4d, cos, sin)

    pypto.set_vec_tile_shapes(min(16, b * s * n_q), d)
    query_out = pypto.reshape(q_embed, [b * s * n_q, d])
    pypto.set_vec_tile_shapes(min(16, b * s), kv_hidden)
    key_rows = pypto.reshape(k_embed, [b * s, kv_hidden])
    pypto.set_vec_tile_shapes(min(16, b * s), kv_hidden)
    value_rows = pypto.reshape(v_4d, [b * s, kv_hidden])
    return query_out, key_rows, value_rows


def qwen3_paged_attention_prolog_graph(params, *graph_args):
    tensors = _build_prolog_case_tensors(graph_args)
    query_out, key_cache_out, value_cache_out = graph_args[14:17]
    q, k, v = _project_qkv_graph(params, tensors.hidden_states, tensors.proj)
    query_rows, key_rows, value_rows = _build_prolog_rows_graph(
        params,
        q,
        k,
        v,
        tensors.proj,
        tensors.cos,
        tensors.sin,
    )
    query_out[:] = query_rows
    key_cache_out[:] = pypto.scatter_update(tensors.cache.key_cache, -2, tensors.cache.cache_index, key_rows)
    value_cache_out[:] = pypto.scatter_update(tensors.cache.value_cache, -2, tensors.cache.cache_index, value_rows)


def _build_attention_graph_config(params, query) -> AttentionGraphConfig:
    n_q = params["n"]
    n_kv = params["n_kv"]
    tile_config = params["pa_tile_config"]
    return AttentionGraphConfig(
        block_size=params["block_size"],
        max_unroll_times=params["max_unroll_times"],
        group=n_q // n_kv,
        n_tile=tile_config.head_num_q_tile,
        d=query.shape[1],
        c1_tile=tile_config.c1_tile_shape,
        v1_tile=tile_config.v1_tile_shape,
        c2_tile=tile_config.c2_tile_shape,
        v2_tile=tile_config.v2_tile_shape,
        softmax_scale=float(1.0 / math.sqrt(params["d"])),
    )


def _build_attention_block_terms(
    runtime: AttentionGraphRuntime,
    config: AttentionGraphConfig,
    ctx: AttentionLoopContext,
    bn,
    block_size_sym,
):
    valid_s2 = (ctx.cur_seq - bn * config.block_size).min(block_size_sym)
    qi = pypto.view(runtime.query, [config.n_tile, config.d], [ctx.cur_offset, 0])
    cur_block_idx = runtime.block_table[ctx.b_idx, bn]
    cur_block_idx.as_variable()
    kj = pypto.view(
        runtime.key_cache,
        [config.block_size, config.d],
        [cur_block_idx * config.block_size, ctx.kv_idx * config.d],
        valid_shape=[valid_s2, config.d],
    )
    vj = pypto.view(
        runtime.value_cache,
        [config.block_size, config.d],
        [cur_block_idx * config.block_size, ctx.kv_idx * config.d],
        valid_shape=[valid_s2, config.d],
    )
    _set_cube_tile_shapes_from_tuple(config.c1_tile)
    pypto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
    sij = pypto.matmul(qi, kj, pypto.DT_FP32, b_trans=True)
    pypto.set_vec_tile_shapes(config.v1_tile[0], config.v1_tile[1])
    sij_scale = pypto.mul(sij, config.softmax_scale)
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
    tilda_pij = pypto.exp(pypto.sub(sij_scale, tilda_mij))
    return AttentionBlockTerms(
        bn=bn,
        ctx=ctx,
        vj=vj,
        tilda_pij_dt=pypto.cast(tilda_pij, runtime.query.dtype),
        tilda_lij=pypto.sum(tilda_pij, dim=-1, keepdim=True),
        tilda_mij=tilda_mij,
    )


def _matmul_attention_value(config: AttentionGraphConfig, tilda_pij_dt, vj):
    _set_cube_tile_shapes_from_tuple(config.c2_tile)
    pypto.set_matrix_size([tilda_pij_dt.shape[0], tilda_pij_dt.shape[1], vj.shape[1]])
    oi_tmp = pypto.matmul(tilda_pij_dt, vj, pypto.DT_FP32)
    pypto.set_vec_tile_shapes(config.v2_tile[0], config.v2_tile[1])
    return oi_tmp


def _update_attention_state(
    runtime: AttentionGraphRuntime,
    config: AttentionGraphConfig,
    block_terms: AttentionBlockTerms,
    state: AttentionAccumState,
):
    oi_offset = [block_terms.ctx.cur_offset, 0]
    if pypto.cond(pypto.is_loop_begin(block_terms.bn)):
        oi_tmp = _matmul_attention_value(config, block_terms.tilda_pij_dt, block_terms.vj)
        if pypto.cond(pypto.is_loop_end(block_terms.bn)):
            pypto.assemble(pypto.div(oi_tmp, block_terms.tilda_lij), oi_offset, runtime.attention_out)
            return
        state.oi_update[:] = oi_tmp
        state.li_update[:] = block_terms.tilda_lij
        state.mi_update[:] = block_terms.tilda_mij
        return

    mi_new = pypto.maximum(state.mi_update, block_terms.tilda_mij)
    t2 = pypto.exp(pypto.sub(state.mi_update, mi_new))
    t4 = pypto.exp(pypto.sub(block_terms.tilda_mij, mi_new))
    li_new = pypto.add(pypto.mul(t2, state.li_update), pypto.mul(t4, block_terms.tilda_lij))
    oi_tmp = pypto.add(pypto.mul(state.oi_update, t2), pypto.mul(_matmul_attention_value(config, block_terms.tilda_pij_dt, block_terms.vj), t4))
    if pypto.cond(pypto.is_loop_end(block_terms.bn)):
        pypto.assemble(pypto.div(oi_tmp, li_new), oi_offset, runtime.attention_out)
        return
    state.oi_update[:] = oi_tmp
    state.li_update[:] = li_new
    state.mi_update[:] = mi_new


def _run_attention_bn_loop(runtime: AttentionGraphRuntime, config: AttentionGraphConfig, ctx: AttentionLoopContext):
    state = AttentionAccumState(
        oi_update=pypto.tensor([config.n_tile, config.d], pypto.DT_FP32, "qwen3_pa_oi"),
        li_update=pypto.tensor([config.n_tile, 1], pypto.DT_FP32, "qwen3_pa_li"),
        mi_update=pypto.tensor([config.n_tile, 1], pypto.DT_FP32, "qwen3_pa_mi"),
    )
    block_size_sym = pypto.symbolic_scalar(config.block_size)
    for bn in pypto.loop(
        0,
        ctx.bn_per_batch,
        1,
        name="QWEN3_PA_L3_BN",
        idx_name="bn",
        unroll_List={config.max_unroll_times},
    ):
        block_terms = _build_attention_block_terms(runtime, config, ctx, bn, block_size_sym)
        _update_attention_state(runtime, config, block_terms, state)


def qwen3_paged_attention_graph(params, *graph_args):
    query, key_cache, value_cache, block_table, act_seqs, attention_out = graph_args
    config = _build_attention_graph_config(params, query)
    runtime = AttentionGraphRuntime(query, key_cache, value_cache, block_table, attention_out)
    batch_size = block_table.shape[0]
    s1_size = query.shape[0] // batch_size // params["n"]
    n_loop = params["n"] // config.n_tile
    for b_idx in pypto.loop(0, batch_size, 1, name="QWEN3_PA_L0_B", idx_name="b_idx"):
        cur_act_seq = act_seqs[b_idx]
        cur_act_seq.as_variable()
        for s1_idx in pypto.loop(0, s1_size, 1, name="QWEN3_PA_L1_S1", idx_name="s1_idx"):
            cur_seq = (cur_act_seq - s1_size + 1 + s1_idx).max(0)
            cur_seq.as_variable()
            bn_per_batch = (cur_seq + config.block_size - 1) // config.block_size
            bn_per_batch.as_variable()
            for n_idx in pypto.loop(0, n_loop, 1, name="QWEN3_PA_L2_N", idx_name="n_idx"):
                kv_idx = (n_idx * config.n_tile) // config.group
                kv_idx.as_variable()
                cur_offset = b_idx * s1_size * params["n"] + s1_idx * params["n"] + n_idx * config.n_tile
                _run_attention_bn_loop(runtime, config, AttentionLoopContext(b_idx, bn_per_batch, cur_seq, cur_offset, kv_idx))


def _compute_layer_post_graph(params, post: LayerPostTensors):
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    hidden_size = n_q * params["d"]
    dtype = post.hidden_states.dtype
    context = pypto.reshape(post.pa_out, [b * s, hidden_size])
    _set_matmul_tile(b * s, hidden_size, hidden_size, pypto.DT_FP32)
    o_proj_fp32 = pypto.matmul(context, pypto.cast(post.attn_o_w, pypto.DT_FP32), pypto.DT_FP32)
    o_proj_fp32 = pypto.add(o_proj_fp32, pypto.cast(post.attn_o_b, pypto.DT_FP32))
    o_proj = pypto.cast(o_proj_fp32, dtype)
    h1 = pypto.cast(pypto.add(pypto.cast(post.hidden_states, pypto.DT_FP32), pypto.cast(o_proj, pypto.DT_FP32)), dtype)
    norm2 = pypto.rms_norm(h1)
    mlp_tmp = pypto.tensor([b * s, hidden_size], dtype, "qwen3_layer_post_mlp")
    qwen3_mlp_graph({"b": b, "s": s, "h": hidden_size, "inter": params["inter"]}, norm2, post.mlp.gate_w, post.mlp.up_w, post.mlp.down_w, mlp_tmp)
    return pypto.cast(pypto.add(pypto.cast(h1, pypto.DT_FP32), pypto.cast(mlp_tmp, pypto.DT_FP32)), dtype)


def qwen3_layer_graph(params, *graph_args):
    tensors = _build_layer_case_tensors(graph_args)
    layer_out = graph_args[21]
    b = params["b"]
    s = params["s"]
    n_q = params["n"]
    d = params["d"]
    dtype = tensors.hidden_states.dtype
    norm1 = pypto.rms_norm(tensors.hidden_states)
    query_out = pypto.tensor([b * s * n_q, d], dtype, "qwen3_layer_query")
    key_cache_tmp = pypto.tensor(list(tensors.cache.key_cache.shape), tensors.cache.key_cache.dtype, "qwen3_layer_key_cache")
    value_cache_tmp = pypto.tensor(
        list(tensors.cache.value_cache.shape), tensors.cache.value_cache.dtype, "qwen3_layer_value_cache"
    )
    qwen3_paged_attention_prolog_graph(
        params,
        norm1,
        *_flatten_qkv_projection_tensors(tensors.proj),
        tensors.cos,
        tensors.sin,
        tensors.cache.cache_index,
        tensors.cache.key_cache,
        tensors.cache.value_cache,
        query_out,
        key_cache_tmp,
        value_cache_tmp,
    )
    pa_out = pypto.tensor([b * s * n_q, d], pypto.DT_FP32, "qwen3_layer_pa")
    qwen3_paged_attention_graph(params, query_out, key_cache_tmp, value_cache_tmp, tensors.block_table, tensors.act_seqs, pa_out)
    qwen3_layer_post_graph(
        params,
        tensors.hidden_states,
        pa_out,
        tensors.attn_o_w,
        tensors.attn_o_b,
        tensors.mlp.gate_w,
        tensors.mlp.up_w,
        tensors.mlp.down_w,
        layer_out,
    )


def qwen3_layer_post_graph(params, *graph_args):
    hidden_states, pa_out, attn_o_w, attn_o_b, gate_w, up_w, down_w, layer_out = graph_args
    layer_out[:] = _compute_layer_post_graph(
        params,
        LayerPostTensors(hidden_states, pa_out, attn_o_w, attn_o_b, MlpWeightTensors(gate_w, up_w, down_w)),
    )


class CountBasedCompareTestBuilder(TestBuilder):
    @staticmethod
    def assert_count_based_close(
        expected: torch.Tensor, actual: torch.Tensor, eps: float, zero_count_threshold: int = 1000
    ):
        threshold = int(expected.numel() * eps)
        err_count = 0
        zero_count = 0
        chunk = 1_000_000
        exp_flat = expected.reshape(-1)
        act_flat = actual.reshape(-1)
        for start in range(0, exp_flat.numel(), chunk):
            end = min(start + chunk, exp_flat.numel())
            exp_chunk = exp_flat[start:end].to(torch.float32)
            act_chunk = act_flat[start:end].to(torch.float32)
            diff = (exp_chunk - act_chunk).abs()
            rel = torch.where(
                exp_chunk != 0,
                diff / exp_chunk.abs(),
                torch.where(diff == 0, torch.zeros_like(diff), torch.full_like(diff, float("inf"))),
            )
            err_count += torch.logical_and(diff > eps, rel > eps).sum().item()
            zero_count += torch.logical_and(act_chunk.abs() <= 1e-6, exp_chunk.abs() > 1e-6).sum().item()
            if err_count > threshold or zero_count > zero_count_threshold:
                raise AssertionError(
                    f"Count-based compare failed: err_count={err_count}, threshold={threshold}, "
                    f"zero_count={zero_count}, zero_threshold={zero_count_threshold}"
                )

    def run_pto(self, kernel, tiling, on_board: bool = True):
        if on_board:
            torch.npu.set_device(self.device_id)

        pypto.set_vec_tile_shapes(tiling, tiling)
        with pypto.function("MAIN", *self.input_pto_list, *self.output_pto_list) as rlf:
            for _ in rlf:
                kernel(self.params, *self.input_pto_list, *self.output_pto_list)
            del rlf

        if on_board:
            pto_input_data = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(self.input_data_list)]
            pto_output_data = [
                pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(self.output_data_list)
            ]
            device_run_once_data_from_host(*pto_input_data, *pto_output_data)
            self.compare_outputs()

    def compare_outputs(self):
        for idx, golden_output in enumerate(self.golden_output):
            self.assert_count_based_close(golden_output.cpu(), self.output_data_list[idx].cpu(), self.atol_value)


class Qwen3PagedAttentionPrologRunner(CountBasedCompareTestBuilder):
    def __init__(self, params):
        super().__init__(params, qwen3_paged_attention_prolog_graph, self.golden, tiling=128)

    @staticmethod
    def golden(params, *golden_args):
        tensors = _build_prolog_case_tensors(golden_args)
        return _run_qkv_rope_cache_torch(
            params,
            tensors.hidden_states,
            tensors.proj,
            (tensors.cos, tensors.sin),
            tensors.cache,
        )

    def get_input_from_param(self):
        inputs, cache_index = _build_prolog_runner_inputs(self.params)
        self._cache_index = cache_index
        self.setup_inputs(*inputs)
        self.set_tol(rtol=1e-3, atol=1e-3)
        return inputs

    def run(self, on_board: bool = True, jit: bool = False):
        if jit:
            raise RuntimeError("Qwen3PagedAttentionPrologRunner directly uses a frontend.jit kernel.")
        self.inputs = self.get_input_from_param()
        self.golden_output = self.torch_convert(self.golden(self.params, *self.inputs, None, None, None))
        actual_output = _run_qkv_rope_cache_torch(
            self.params,
            self.inputs[0],
            _build_prolog_case_tensors(self.inputs).proj,
            (self.inputs[9], self.inputs[10]),
            CacheUpdateTensors(self.inputs[11], self.inputs[12], self.inputs[13]),
        )
        self.assert_count_based_close(self.golden_output[0].cpu(), actual_output[0].cpu(), self.atol_value)
        row_index = self._cache_index.reshape(-1).to(torch.int64)
        golden_k_rows = self.golden_output[1].index_select(0, row_index)
        actual_k_rows = actual_output[1].index_select(0, row_index)
        self.assert_count_based_close(golden_k_rows.cpu(), actual_k_rows.cpu(), self.atol_value)
        golden_v_rows = self.golden_output[2].index_select(0, row_index)
        actual_v_rows = actual_output[2].index_select(0, row_index)
        self.assert_count_based_close(golden_v_rows.cpu(), actual_v_rows.cpu(), self.atol_value)

    def compare_outputs(self):
        self.assert_count_based_close(self.golden_output[0].cpu(), self.output_data_list[0].cpu(), self.atol_value)
        row_index = self._cache_index.reshape(-1).to(torch.int64)
        golden_k_rows = self.golden_output[1].index_select(0, row_index)
        actual_k_rows = self.output_data_list[1].index_select(0, row_index)
        self.assert_count_based_close(golden_k_rows.cpu(), actual_k_rows.cpu(), self.atol_value)
        golden_v_rows = self.golden_output[2].index_select(0, row_index)
        actual_v_rows = self.output_data_list[2].index_select(0, row_index)
        self.assert_count_based_close(golden_v_rows.cpu(), actual_v_rows.cpu(), self.atol_value)


class Qwen3PagedAttentionRunner(CountBasedCompareTestBuilder):
    def __init__(self, params):
        super().__init__(params, qwen3_paged_attention_graph, self.golden, tiling=128)

    def get_input_from_param(self):
        inputs, self._q_4d = _build_attention_runner_inputs(self.params)
        self.setup_inputs(*inputs)
        self.set_tol(rtol=1e-3, atol=1e-3)
        return inputs

    def golden(self, params, *golden_args):
        return (
            _run_paged_attention_torch(
                params,
                _build_attention_torch_tensors(self._q_4d.to(torch.float32), golden_args),
            ),
        )

    def run(self, on_board: bool = True, jit: bool = False):
        if jit:
            raise RuntimeError("Qwen3PagedAttentionRunner directly uses a frontend.jit kernel.")
        self.inputs = self.get_input_from_param()
        self.golden_output = self.torch_convert(self.golden(self.params, *self.inputs, None))
        actual = _run_paged_attention_torch(
            self.params,
            _build_attention_torch_tensors(self._q_4d.to(torch.float32), self.inputs),
        )
        self.assert_count_based_close(self.golden_output[0].cpu(), actual.cpu(), self.atol_value)


class Qwen3MLPRunner(CountBasedCompareTestBuilder):
    def __init__(self, params):
        super().__init__(params, qwen3_mlp_graph, self.golden, tiling=128)

    @staticmethod
    def golden(params, *golden_args):
        hidden_states, gate_w, up_w, down_w = golden_args[:4]
        gate = torch.matmul(hidden_states.to(torch.float32), gate_w.to(torch.float32)).to(hidden_states.dtype)
        up = torch.matmul(hidden_states.to(torch.float32), up_w.to(torch.float32)).to(hidden_states.dtype)
        gate_act = gate * torch.sigmoid(gate.to(torch.float32)).to(hidden_states.dtype)
        inter = (gate_act.to(torch.float32) * up.to(torch.float32)).to(hidden_states.dtype)
        out = torch.matmul(inter.to(torch.float32), down_w.to(torch.float32)).to(hidden_states.dtype)
        return (out,)

    def get_input_from_param(self):
        inputs = _build_mlp_runner_inputs(self.params)
        self.setup_inputs(*inputs)
        self.set_tol(rtol=8e-3, atol=8e-3)
        return inputs

    def run(self, on_board: bool = True, jit: bool = False):
        if jit:
            raise RuntimeError("Qwen3MLPRunner directly uses a frontend.jit kernel.")
        self.inputs = self.get_input_from_param()
        self.golden_output = self.torch_convert(self.golden(self.params, *self.inputs, None))
        (actual,) = self.golden(self.params, *self.inputs, None)
        self.assert_count_based_close(self.golden_output[0].cpu(), actual.cpu(), self.atol_value)


class Qwen3LayerRunner(CountBasedCompareTestBuilder):
    def __init__(self, params):
        super().__init__(params, qwen3_layer_graph, self.golden, tiling=128)

    def get_input_from_param(self):
        inputs = _build_layer_runner_inputs(self.params)
        self.setup_inputs(*inputs)
        self.set_tol(rtol=1e-1, atol=1e-1)
        return inputs

    def run(self, on_board: bool = True, jit: bool = False):
        if jit:
            raise RuntimeError("Qwen3LayerRunner directly uses a frontend.jit kernel.")
        self.inputs = self.get_input_from_param()
        self.golden_output = self.torch_convert(self.golden(self.params, *self.inputs, None))
        tensors = _build_layer_case_tensors(self.inputs)
        query_out, key_cache_tmp, value_cache_tmp = _run_qkv_rope_cache_torch(
            self.params,
            _rmsnorm_torch(tensors.hidden_states).contiguous(),
            tensors.proj,
            (tensors.cos, tensors.sin),
            tensors.cache,
        )

        q_embed = query_out.reshape(self.params["b"], self.params["s"], self.params["n"], self.params["d"])
        pa_out = _run_paged_attention_torch(
            self.params,
            AttentionTorchTensors(
                q_4d=q_embed,
                key_cache=key_cache_tmp,
                value_cache=value_cache_tmp,
                block_table=tensors.block_table,
                act_seqs=tensors.act_seqs,
            ),
        )
        layer_out = _run_layer_post_torch(_build_layer_post_tensors(tensors, pa_out))
        self.assert_count_based_close(self.golden_output[0].cpu(), layer_out.cpu(), self.atol_value)

    def golden(self, params, *golden_args):
        tensors = _build_layer_case_tensors(golden_args)
        b = params["b"]
        s = params["s"]
        n_q = params["n"]
        d = params["d"]

        query_out, key_cache_tmp, value_cache_tmp = _run_qkv_rope_cache_torch(
            params,
            _rmsnorm_torch(tensors.hidden_states),
            tensors.proj,
            (tensors.cos, tensors.sin),
            tensors.cache,
        )
        q_embed = query_out.reshape(b, s, n_q, d)
        attn_out = _run_paged_attention_torch(
            params,
            AttentionTorchTensors(
                q_4d=q_embed,
                key_cache=key_cache_tmp,
                value_cache=value_cache_tmp,
                block_table=tensors.block_table,
                act_seqs=tensors.act_seqs,
            ),
        )
        layer_out = _run_layer_post_torch(_build_layer_post_tensors(tensors, attn_out))

        return (layer_out,)

class TestQwen3Atten:
    @staticmethod
    def test_qwen3_paged_attention_prolog_b32_s1_n32_kv8_d128_blk2048_bf16():
        Qwen3PagedAttentionPrologRunner(_make_qwen3_prolog_params())()

    @staticmethod
    def test_qwen3_paged_attention_b32_s1_n32_kv8_d128_blk2048_skv2048_bf16():
        Qwen3PagedAttentionRunner(_make_qwen3_attention_params())()

    @staticmethod
    def test_qwen3_mlp_b32_s1_h4096_inter12288_bf16():
        Qwen3MLPRunner(_make_qwen3_mlp_params())()

    @staticmethod
    def test_qwen3_layer_b32_s1_n32_kv8_d128_blk2048_inter12288_skv2048_bf16():
        Qwen3LayerRunner(_make_qwen3_layer_params())()
