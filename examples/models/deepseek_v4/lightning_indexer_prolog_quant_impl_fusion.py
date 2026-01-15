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
Lightning Indexer Prolog Quantization Module

This module implements the Lightning Indexer Prolog quantization computation
for DeepSeek V32 model. It handles:
- Query computation with dynamic quantization
- Key computation with LayerNorm and RoPE
- Weight computation for indexer attention

Main Functions:
    - lightning_indexer_prolog_compute: Main computation function
    - quant_layer_norm: Quantized LayerNorm implementation
    - prolog_quant: Per-token quantization function
    - quant_rope_2d: 2D RoPE (Rotary Position Embedding) computation
    - rope_3d: 3D RoPE computation

Example:
    See testdsv32_lightning_indexer_prolog_quant.py for usage examples.
"""
from dataclasses import dataclass
from typing import List, Tuple
import pypto
import math
import torch
import torch_npu
from typing import List

from pypto import pypto_impl
from pypto.operation import op_wrapper
from lightning_indexer_prolog_quant_impl import rope_3d, prolog_quant


L0M_INDEX = 0
L1M_INDEX = 1
L0K_INDEX = 2
L1K_INDEX = 3
L0N_INDEX = 4
L1N_INDEX = 5


@dataclass
class IndexerPrologQuantConfigs:
    q_linear: List[int]
    q_hd: List[int]
    # k_linear: List[int]
    w_linear: List[int]
    unroll_list: List[int]

    cube_l1_reuse_setting: dict[int, int]
    mg_copyin_upper_bound: int
    pg_upper_bound: int
    block_size: int
    t_sub_tile: int
    chunk_size: int
    vec_nbuffer_mode: int


def lightning_indexer_prolog_quant_fusion_compute(
    x_in,
    q_norm_in,
    q_norm_scale_in,
    w_qb_in,
    w_qb_scale_in,
    w_proj_in,
    cos_idx_rope_in,
    sin_idx_rope_in,
    hadamard_q_in,
    wkv,
    wgate,
    kv_state,
    score_state,
    ape,
    weight_rms,
    cos_kv,
    sin_kv,
    hadamard_kv,
    kv_cache,
    slots,
    q_int8_out,
    q_scale_out,
    weights_out,
    key,
    key_scale,
    ratio,
    start_pos,
    eps_rms,
    configs
):
    """Compute Lightning Indexer Prolog with quantization.

    Main computation function for Lightning Indexer Prolog quantization.
    This function processes input tokens to generate quantized query, key, and weights
    for the indexer attention mechanism. The computation includes:

    1. Query Path:
       - Dequantize q_norm (INT8) to FP32
       - Apply linear transformation with w_qb
       - Apply RoPE (Rotary Position Embedding)
       - Apply Hadamard transformation
       - Quantize to INT8 with per-token-head scale

    2. Key Path:
       - Linear transformation with wk
       - LayerNorm normalization
       - Apply RoPE
       - Apply Hadamard transformation
       - Quantize to INT8 with per-token-head scale
       - Update key cache using scatter_update

    3. Weights Path:
       - Linear transformation with w_proj
       - Normalize by sqrt(head_num * head_dim)
       - Convert to FP16

    Args:
        x_in: Input hidden states tensor, shape (t, h), dtype BF16
        q_norm_in: Quantized query norm tensor, shape (t, q_lora_rank), dtype INT8
        q_norm_scale_in: Query norm dequantization scale, shape (t, 1), dtype FP32
        w_qb_in: Query projection weight matrix, INT8 format with NZ layout
        w_qb_scale_in: Query weight dequantization scale, shape (head_num * head_dim, 1), dtype FP32
        wk_in: Key projection weight matrix, BF16 format with NZ layout
        w_proj_in: Weight projection matrix, BF16 format with NZ layout
        ln_gamma_k_in: LayerNorm scale parameter for key, shape (head_dim,), dtype BF16
        ln_beta_k_in: LayerNorm shift parameter for key, shape (head_dim,), dtype BF16
        cos_idx_rope_in: Cosine values for RoPE, shape (t, rope_head_dim), dtype BF16
        sin_idx_rope_in: Sine values for RoPE, shape (t, rope_head_dim), dtype BF16
        hadamard_q_in: Hadamard transformation matrix for query, shape (head_dim, head_dim), dtype BF16
        hadamard_k_in: Hadamard transformation matrix for key, shape (head_dim, head_dim), dtype BF16
        k_int8_in: Input key cache, shape (block_num, block_size, n_kv, head_dim), dtype INT8
        k_scale_in: Key cache scale, shape (block_num, block_size, n_kv, 1), dtype FP16
        k_cache_index_in: Cache index for scatter update, shape (t,), dtype INT64
        q_int8_out: Output quantized query tensor, shape (t, head_num, head_dim), dtype INT8
        q_scale_out: Output query quantization scale, shape (t, head_num, 1), dtype FP16
        k_int8_out: Output key cache (updated in-place), shape (block_num, block_size, n_kv, head_dim), dtype INT8
        k_scale_out: Output key cache scale (updated in-place), shape (block_num, block_size, n_kv, 1), dtype FP16
        weights_out: Output weights tensor, shape (t, head_num), dtype FP16
        attrs: IndexerPrologQuantAttr object containing:
            - eps: LayerNorm epsilon value
            - layerout_query: Query layout format (e.g., "TND")
            - layerout_key: Key layout format (e.g., "PA_BSND")
        configs: IndexerPrologQuantConfigs object containing tiling and optimization parameters

    Note:
        - The function processes tokens in tiles using loop_unroll for optimization
        - All outputs are written in-place using pypto.assemble or scatter_update
        - The computation uses dynamic tiling based on configs.unroll_list
    """
    x_dtype = x_in.dtype
    # 动态轴
    t = x_in.shape[0]
    h = x_in.shape[1]
    q_lora_rank = q_norm_in.shape[1]
    head_num = w_proj_in.shape[1]
    head_dim = hadamard_q_in.shape[0]
    rope_head_dim = cos_idx_rope_in.shape[1]
    w_qb_scale = pypto.reshape(w_qb_scale_in, [1, head_num * head_dim], inplace=True)

    unroll_list = configs.unroll_list
    for tIdx, unrollLength in pypto.loop_unroll(0, t, 1, name="IndexerPrologQuantQuantLoop", idx_name="tIdx",
                                                unroll_list=unroll_list, ):
        t_tile = unrollLength
        # 获取query计算的各阶段Tile参数
        q_linear = configs.q_linear
        q_hd = configs.q_hd
        # 多分档内会将t_tile作为档位，offset无需乘t_tile
        q_norm = pypto.view(q_norm_in, [t_tile, q_lora_rank], [tIdx, 0], valid_shape=[t_tile, q_lora_rank])
        q_norm_scale = pypto.view(q_norm_scale_in, [t_tile, 1], [tIdx, 0], valid_shape=[t_tile, 1])
        pypto.set_semantic_label("Query-Linear")
        pypto.set_cube_tile_shapes([q_linear[L0M_INDEX], q_linear[L1M_INDEX]],
                                   [q_linear[L0K_INDEX], q_linear[L1K_INDEX]],
                                   [q_linear[L0N_INDEX], q_linear[L1N_INDEX]], True)
        q_s32 = pypto.matmul(q_norm, w_qb_in, pypto.DT_INT32)  # (t_tile, head_num * head_dim)

        pypto.set_semantic_label("Query-Dequant")

        pypto.set_vec_tile_shapes(configs.t_sub_tile, head_num * head_dim // configs.chunk_size)
        q_f32 = pypto.cast(q_s32, pypto.DT_FP32)  # (t_tile, head_num * head_dim), fp32
        q_f32 = q_f32 * q_norm_scale  # (t_tile, head_num * head_dim), fp32
        q_f32 = q_f32 * w_qb_scale

        q_cast = pypto.cast(q_f32, x_dtype)

        q_bf16 = pypto.reshape(q_cast, [t_tile, head_num, head_dim], valid_shape=[t_tile, head_num, head_dim])
        
        # UB view
        q_rope = pypto.view(q_bf16, [t_tile, head_num, rope_head_dim], [0, 0, 0],
                            valid_shape=[t_tile, head_num, rope_head_dim])
        q_nope = pypto.view(q_bf16, [t_tile, head_num, head_dim - rope_head_dim], [0, 0, rope_head_dim],
                            valid_shape=[t_tile, head_num, head_dim - rope_head_dim])
        rope_cos = pypto.view(cos_idx_rope_in, [t_tile, rope_head_dim], [tIdx, 0],
                              valid_shape=[t_tile, rope_head_dim])
        rope_sin = pypto.view(sin_idx_rope_in, [t_tile, rope_head_dim], [tIdx, 0],
                              valid_shape=[t_tile, rope_head_dim])

        q_roped = rope_3d(q_rope, rope_cos, rope_sin, configs)  # [t_tile, head_num, rope_head_dim]
        pypto.set_vec_tile_shapes(configs.t_sub_tile, head_num // configs.chunk_size, head_dim)
        q_nope = pypto.cast(pypto.cast(q_nope, pypto.DT_FP32), q_bf16.dtype)
        q_cat = pypto.concat([q_roped, q_nope], -1)  # [t_tile, head_num, head_dim]

        hadamard_q = pypto.reshape(hadamard_q_in, [1, head_dim, head_dim], valid_shape=[1, head_dim, head_dim])

        pypto.set_semantic_label("Query-Hadamard")
        cur_max_unroll = 32
        qHdMTile = cur_max_unroll if t_tile < cur_max_unroll else q_hd[L0M_INDEX]
        pypto.set_cube_tile_shapes([qHdMTile, qHdMTile], [q_hd[L0K_INDEX], q_hd[L1K_INDEX]],
                                   [q_hd[L0N_INDEX], q_hd[L1N_INDEX]])
        q_hadamard = pypto.matmul(q_cat, hadamard_q, x_dtype)  # (t_tile, head_num, head_dim)
        pypto.set_vec_tile_shapes(configs.t_sub_tile, head_num // configs.chunk_size, head_dim)
        
        pypto.set_semantic_label("Query-Quant")
        pypto.set_vec_tile_shapes(configs.t_sub_tile, head_num // configs.chunk_size, head_dim)
        q_res = prolog_quant(q_hadamard)
        q_scale = pypto.cast(q_res[1], pypto.DT_FP16)
        
        pypto.assemble(q_res[0], [tIdx, 0, 0], q_int8_out)
        pypto.assemble(q_scale, [tIdx, 0, 0], q_scale_out)

        x = pypto.view(x_in, [t_tile, h], [tIdx, 0], valid_shape=[t_tile, h])  # 这里将t_tile分档，offset不需要乘t_tile
        pypto.set_semantic_label("Weight-Linear")
        w_linear = configs.w_linear
        pypto.set_cube_tile_shapes([w_linear[L0M_INDEX], w_linear[L1M_INDEX]],
                                   [w_linear[L0K_INDEX], w_linear[L1K_INDEX]],
                                   [w_linear[L0N_INDEX], w_linear[L1N_INDEX]])
        pypto.set_vec_tile_shapes(t_tile, head_num)
        weights = pypto.cast(pypto.matmul(x, w_proj_in, x_dtype), pypto.DT_FP32)
        weights = pypto.mul(weights, 1.0 / (math.sqrt(head_num) * math.sqrt(head_dim)))
        weights_bf16 = pypto.cast(weights, pypto.DT_BF16)
        pypto.assemble(weights_bf16, [tIdx, 0], weights_out)


@pypto.jit
def lightning_indexer_quant_prolog_fusion(
    x_in,
    q_norm_in,
    q_norm_scale_in,
    w_qb_in,
    w_qb_scale_in,
    w_proj_in,
    cos_idx_rope_in,
    sin_idx_rope_in,
    hadamard_q_in,
    wkv,
    wgate,
    kv_state,
    score_state,
    ape,
    weight_rms,
    cos_kv,
    sin_kv,
    hadamard_kv,
    kv_cache,
    slots,
    q_int8_out,
    q_scale_out,
    weights_out,
    key,
    key_scale,
    ratio,
    start_pos,
    eps_rms,
    configs
):
    """JIT-compiled wrapper for Lightning Indexer Prolog quantization computation.

    This is the main entry point for the Lightning Indexer Prolog quantization operator.
    It sets up optimization passes and runtime options before calling the core
    computation function.

    Args:
        x_in: Input hidden states tensor, shape (t, h), dtype BF16
        q_norm_in: Quantized query norm tensor, shape (t, q_lora_rank), dtype INT8
        q_norm_scale_in: Query norm dequantization scale, shape (t, 1), dtype FP32
        w_qb_in: Query projection weight matrix, INT8 format with NZ layout
        w_qb_scale_in: Query weight dequantization scale, shape (head_num * head_dim, 1), dtype FP32
        wk_in: Key projection weight matrix, BF16 format with NZ layout
        w_proj_in: Weight projection matrix, BF16 format with NZ layout
        ln_gamma_k_in: LayerNorm scale parameter for key, shape (head_dim,), dtype BF16
        ln_beta_k_in: LayerNorm shift parameter for key, shape (head_dim,), dtype BF16
        cos_idx_rope_in: Cosine values for RoPE, shape (t, rope_head_dim), dtype BF16
        sin_idx_rope_in: Sine values for RoPE, shape (t, rope_head_dim), dtype BF16
        hadamard_q_in: Hadamard transformation matrix for query, shape (head_dim, head_dim), dtype BF16
        hadamard_k_in: Hadamard transformation matrix for key, shape (head_dim, head_dim), dtype BF16
        k_int8_in: Input key cache, shape (block_num, block_size, n_kv, head_dim), dtype INT8
        k_scale_in: Key cache scale, shape (block_num, block_size, n_kv, 1), dtype FP16
        k_cache_index_in: Cache index for scatter update, shape (t,), dtype INT64
        q_int8_out: Output quantized query tensor, shape (t, head_num, head_dim), dtype INT8
        q_scale_out: Output query quantization scale, shape (t, head_num, 1), dtype FP16
        k_int8_out: Output key cache (updated in-place), shape (block_num, block_size, n_kv, head_dim), dtype INT8
        k_scale_out: Output key cache scale (updated in-place), shape (block_num, block_size, n_kv, 1), dtype FP16
        weights_out: Output weights tensor, shape (t, head_num), dtype FP16
        attrs: IndexerPrologQuantAttr object containing operator attributes
        configs: IndexerPrologQuantConfigs object containing optimization configurations

    Note:
        This function is decorated with @pypto.jit for JIT compilation.
        It configures pass options for memory optimization and calls the core
        computation function.
    """

    pypto.set_runtime_options(device_sched_mode=1)

    lightning_indexer_prolog_quant_fusion_compute(
        x_in,
        q_norm_in,
        q_norm_scale_in,
        w_qb_in,
        w_qb_scale_in,
        w_proj_in,
        cos_idx_rope_in,
        sin_idx_rope_in,
        hadamard_q_in,
        wkv,
        wgate,
        kv_state,
        score_state,
        ape,
        weight_rms,
        cos_kv,
        sin_kv,
        hadamard_kv,
        kv_cache,
        slots,
        q_int8_out,
        q_scale_out,
        weights_out,
        key,
        key_scale,
        ratio,
        start_pos,
        eps_rms,
        configs
    )
