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
from typing import List
import pypto
import math
from typing import List

from common import quant_tensor


L0M_INDEX = 0
L1M_INDEX = 1
L0K_INDEX = 2
L1K_INDEX = 3
L0N_INDEX = 4
L1N_INDEX = 5


@dataclass
class Rope3dTileConfig:
    # two_dim_tile: List[int]
    three_dim_tile: List[int]
    four_dim_tile: List[int]


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


def scatter_update_3d(input, index, src):
    input_shape = input.shape
    d = src.shape[2]
    pypto.set_vec_tile_shapes(1, 4, d)
    src = pypto.reshape(src, [src.shape[0] * src.shape[1], src.shape[2]])
    input = pypto.reshape(input, [input.shape[0] * input.shape[1], input.shape[2]])
    pypto.set_vec_tile_shapes(64, 64)
    index = pypto.reshape(index, [1, index.shape[0] * index.shape[1]])
    pypto.set_vec_tile_shapes(1, d)
    if (index.shape[0] * index.shape[1]) % 4 == 0:
        pypto.set_vec_tile_shapes(4, d)
    output = pypto.scatter_update(input, -2, index, src)
    return pypto.reshape(output, input_shape)


def scatter_update_4d(input, index, src):
    input_shape = input.shape
    d = src.shape[2]
    pypto.set_vec_tile_shapes(1, 4, d)
    src = pypto.reshape(src, [src.shape[0] * src.shape[1], src.shape[2]])
    input = pypto.reshape(
        input, [input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]]
    )
    pypto.set_vec_tile_shapes(64, 64)
    index = pypto.reshape(index, [index.shape[0], 1])
    pypto.set_vec_tile_shapes(1, d)
    if (index.shape[0] * index.shape[1]) % 4 == 0:
        pypto.set_vec_tile_shapes(4, d)
    output = pypto.scatter_update(input, -2, index, src)
    return pypto.reshape(output, input_shape)


def rope_3d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor) -> pypto.Tensor:
    """Apply 3D Rotary Position Embedding (RoPE) to input tensor.

    Implements RoPE transformation for 3D tensors with shape (t_tile, head_num, rope_dim).
    The RoPE is applied independently to each head using the provided cosine and sine values.

    Args:
        x: Input tensor of shape (t_tile, head_num, rope_dim)
        cos: Cosine values for RoPE, shape (t_tile, rope_dim)
        sin: Sine values for RoPE, shape (t_tile, rope_dim)
        configs: Configuration object containing tiling parameters:
            - t_sub_tile: Sub-tile size for t dimension
            - chunk_size: Chunk size for head dimension processing

    Returns:
        Tensor with RoPE applied, same shape as input x

    Note:
        The function broadcasts cos and sin to match the head dimension,
        then applies rotation: x_rotated = x * cos + rotate_half(x) * sin
    """
    head_num_axis = 1
    head_dim_axis = 2

    x_dtype = x.dtype
    t_tile = x.shape[0]
    head_num = x.shape[head_num_axis]
    rope_dim = x.shape[head_dim_axis]

    pypto.set_vec_tile_shapes(1, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(1, head_num // 2, rope_dim)
    x_view = pypto.cast(x, pypto.DT_FP32)
    cast_cos = pypto.reshape(cast_cos, [t_tile, 1, rope_dim])
    cast_sin = pypto.reshape(cast_sin, [t_tile, 1, rope_dim])

    x_embed = (x_view * cast_cos) + ((rotate_half(x_view)) * cast_sin)
    res = pypto.cast(x_embed, x_dtype)
    return res


def softmax(x: pypto.Tensor, dim) -> pypto.Tensor:
    xmax = pypto.amax(x, dim, keepdim=True)
    xsub = pypto.sub(x, xmax)
    xexp = pypto.exp(xsub)
    xsum = pypto.sum(xexp, dim, keepdim=True)
    xdiv = pypto.div(xexp, xsum)
    return xdiv


def rms_norm(
    input_tensor: pypto.Tensor, gamma: pypto.Tensor, epsilon=1e-6
) -> pypto.Tensor:
    input_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    dim = len(input_tensor.shape)
    shape = [1] * dim
    shape[dim - 1] = gamma.shape[0]
    gamma_cast = pypto.reshape(gamma, shape)
    gamma_fp32 = pypto.cast(gamma_cast, pypto.DT_FP32)
    y = pypto.mul(input_fp32, input_fp32)
    y = pypto.mul(y, 1.0 / input_tensor.shape[dim - 1])
    y = pypto.sum(y, -1, keepdim=True)
    y = pypto.add(y, epsilon)
    y = pypto.sqrt(y)
    ones_vector = pypto.full(y.shape, 1.0, pypto.DT_FP32)
    y = pypto.div(ones_vector, y)
    y = pypto.mul(input_fp32, y)
    y = pypto.mul(gamma_fp32, y)
    y = pypto.cast(y, input_tensor.dtype)
    return y


def rotate_half(input_tensor: pypto.Tensor) -> pypto.Tensor:
    chunk_size = 2
    shape = input_tensor.shape
    shape_size = len(shape)
    shape[shape_size - 1] //= chunk_size
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    x1 = pypto.view(input_tensor, shape, offset1)
    x2 = pypto.view(input_tensor, shape, offset2)
    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def interleaved_rope_3d(
    x: pypto.Tensor,
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    rope_3d_config: Rope3dTileConfig,
) -> pypto.Tensor:
    pypto.set_vec_tile_shapes(*rope_3d_config.three_dim_tile)  # (1, 64, 64)
    cast_x = pypto.cast(x, pypto.DataType.DT_FP32)
    cast_cos = pypto.cast(cos, pypto.DataType.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DataType.DT_FP32)
    pypto.set_vec_tile_shapes(*rope_3d_config.four_dim_tile)  # (1, 64, 128, 128)

    x_view = pypto.reshape(cast_x, [x.shape[0], x.shape[1], x.shape[2] // 2, 2])
    x_trans = pypto.transpose(x_view, 2, 3)
    x_re_second = pypto.reshape(x_trans, x.shape)
    x_embed = x_re_second * cast_cos + rotate_half(x_re_second) * cast_sin

    return pypto.cast(x_embed, x.dtype)


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
    cache_index_2d,
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
    kv_state_out,
    score_state_out,
    key,
    key_scale,
    ratio,
    start_pos,
    eps_rms,
    configs,
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

    overlap = ratio == 4
    coff = 1 + overlap
    d = wkv.shape[1] // coff
    should_compress = (start_pos + 1) % ratio == 0
    pos = start_pos % ratio

    unroll_list = configs.unroll_list
    pypto.set_vec_tile_shapes(64, 64)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], True)
    t_tile = 4
    x_in_tmp = pypto.view(x_in, [t_tile, h], [0, 0])
    x_kv = pypto.cast(x_in_tmp, pypto.DT_FP32)  ## b,s,h
    kv = pypto.matmul(x_kv, wkv, pypto.DT_FP32)  ## b*s,2d
    kv = pypto.reshape(kv, [t_tile, 1, wkv.shape[1]])  ## b,s,2d
    score = pypto.matmul(x_kv, wgate, pypto.DT_FP32)
    score = pypto.reshape(score, [t_tile, 1, wgate.shape[1]])  ## b,s,2d

    pypto.set_vec_tile_shapes(16, 1, 2 * d)
    score = pypto.add(score, ape[pos, :])  ## b,1,2d
    if overlap:
        index = pypto.view(cache_index_2d, [t_tile, 1], [0, ratio + pos])
        kv_state = scatter_update_3d(kv_state, index, kv)
        score_state = scatter_update_3d(score_state, index, score)
        if should_compress:
            pypto.set_vec_tile_shapes(1, 8, 2 * d)
            kv_state_tmp = pypto.concat(
                [
                    kv_state[:t_tile, :ratio, :d],
                    kv_state[:t_tile, ratio : ratio + ratio, d:],
                ],
                1,
            )  ## b,8,d
            score_state_tmp = pypto.concat(
                [
                    score_state[:t_tile, :ratio, :d],
                    score_state[:t_tile, ratio : ratio + ratio, d:],
                ],
                1,
            )  ## b,8,d
            kv = kv_state_tmp * softmax(score_state_tmp, 1)  ## b,8,d
            kv = pypto.sum(kv, 1, keepdim=True)  ## b,1,d

            index = cache_index_2d[:t_tile, :ratio]
            kv_state_view = kv_state[:t_tile, ratio : ratio + ratio, :]
            score_state_view = score_state[:t_tile, ratio : ratio + ratio, :]
            kv_state = scatter_update_3d(kv_state, index, kv_state_view)
            score_state = scatter_update_3d(score_state, index, score_state_view)

    else:
        index = pypto.view(cache_index_2d, [t_tile, 1], [0, pos])
        kv_state = scatter_update_3d(kv_state, index, kv)
        score_state = scatter_update_3d(score_state, index, score)
        if should_compress:
            pypto.set_vec_tile_shapes(1, 8, 2 * d)
            kv = kv_state[:t_tile, :, :] * softmax(
                score_state[:t_tile, :, :], 1
            )  ## b,8,d
            kv = pypto.sum(kv, 1, keepdim=True)  ## b,1,d

    if should_compress:
        pypto.set_vec_tile_shapes(1, 8, d)
        kv = rms_norm(pypto.cast(kv, x_dtype), weight_rms, eps_rms)  ## b,cut,d

        kv_nope = pypto.view(
            kv, [kv.shape[0], kv.shape[1], d - rope_head_dim], [0, 0, 0]
        )
        kv_rope = pypto.view(
            kv, [kv.shape[0], kv.shape[1], rope_head_dim], [0, 0, d - rope_head_dim]
        )
        sin_kv = pypto.view(sin_kv, kv_rope.shape, [0, 0, 0])  ## b, cut, 64
        cos_kv = pypto.view(cos_kv, kv_rope.shape, [0, 0, 0])  ## b, cut, 64
        rope3d_tile_config = Rope3dTileConfig([1, 64, 64], [1, 64, 128, 128])
        kv_rope = interleaved_rope_3d(kv_rope, cos_kv, sin_kv, rope3d_tile_config)
        pypto.set_vec_tile_shapes(1, 8, d)
        kv = pypto.concat([kv_nope, kv_rope], dim=-1)  ## b,cut,d

        kv = pypto.reshape(kv, [kv.shape[0] * kv.shape[1], d])
        kv = pypto.matmul(kv, hadamard_kv, pypto.DT_FP32)  ## b*cut,d
        kv = pypto.reshape(kv, [t_tile, kv.shape[0] // t_tile, d])  ## b,cut,d

        pypto.assemble(kv, [0, 0, 0], key)

    for tIdx, unrollLength in pypto.loop_unroll(
        0,
        t,
        1,
        name="IndexerPrologQuantQuantLoop",
        idx_name="tIdx",
        unroll_list=unroll_list,
    ):
        t_tile = unrollLength
        # 获取query计算的各阶段Tile参数
        q_linear = configs.q_linear
        q_hd = configs.q_hd
        # 多分档内会将t_tile作为档位，offset无需乘t_tile
        q_norm = pypto.view(
            q_norm_in,
            [t_tile, q_lora_rank],
            [tIdx, 0],
            valid_shape=[t_tile, q_lora_rank],
        )
        q_norm_scale = pypto.view(
            q_norm_scale_in, [t_tile, 1], [tIdx, 0], valid_shape=[t_tile, 1]
        )
        pypto.set_semantic_label("Query-Linear")
        pypto.set_cube_tile_shapes(
            [q_linear[L0M_INDEX], q_linear[L1M_INDEX]],
            [q_linear[L0K_INDEX], q_linear[L1K_INDEX]],
            [q_linear[L0N_INDEX], q_linear[L1N_INDEX]],
            True,
        )
        q_s32 = pypto.matmul(
            q_norm, w_qb_in, pypto.DT_INT32
        )  # (t_tile, head_num * head_dim)

        pypto.set_semantic_label("Query-Dequant")

        pypto.set_vec_tile_shapes(
            configs.t_sub_tile, head_num * head_dim // configs.chunk_size
        )
        q_f32 = pypto.cast(q_s32, pypto.DT_FP32)  # (t_tile, head_num * head_dim), fp32
        q_f32 = q_f32 * q_norm_scale  # (t_tile, head_num * head_dim), fp32
        q_f32 = q_f32 * w_qb_scale

        q_cast = pypto.cast(q_f32, x_dtype)

        q_bf16 = pypto.reshape(
            q_cast,
            [t_tile, head_num, head_dim],
            valid_shape=[t_tile, head_num, head_dim],
        )

        # UB view
        q_rope = pypto.view(
            q_bf16,
            [t_tile, head_num, rope_head_dim],
            [0, 0, 0],
            valid_shape=[t_tile, head_num, rope_head_dim],
        )
        q_nope = pypto.view(
            q_bf16,
            [t_tile, head_num, head_dim - rope_head_dim],
            [0, 0, rope_head_dim],
            valid_shape=[t_tile, head_num, head_dim - rope_head_dim],
        )
        rope_cos = pypto.view(
            cos_idx_rope_in,
            [t_tile, rope_head_dim],
            [tIdx, 0],
            valid_shape=[t_tile, rope_head_dim],
        )
        rope_sin = pypto.view(
            sin_idx_rope_in,
            [t_tile, rope_head_dim],
            [tIdx, 0],
            valid_shape=[t_tile, rope_head_dim],
        )

        q_roped = rope_3d(
            q_rope, rope_cos, rope_sin
        )  # [t_tile, head_num, rope_head_dim]
        pypto.set_vec_tile_shapes(
            configs.t_sub_tile, head_num // configs.chunk_size, head_dim
        )
        q_nope = pypto.cast(pypto.cast(q_nope, pypto.DT_FP32), q_bf16.dtype)
        q_cat = pypto.concat([q_roped, q_nope], -1)  # [t_tile, head_num, head_dim]

        hadamard_q = pypto.reshape(
            hadamard_q_in, [1, head_dim, head_dim], valid_shape=[1, head_dim, head_dim]
        )

        pypto.set_semantic_label("Query-Hadamard")
        cur_max_unroll = 32
        qHdMTile = cur_max_unroll if t_tile < cur_max_unroll else q_hd[L0M_INDEX]
        pypto.set_cube_tile_shapes(
            [qHdMTile, qHdMTile],
            [q_hd[L0K_INDEX], q_hd[L1K_INDEX]],
            [q_hd[L0N_INDEX], q_hd[L1N_INDEX]],
        )
        q_hadamard = pypto.matmul(
            q_cat, hadamard_q, x_dtype
        )  # (t_tile, head_num, head_dim)
        pypto.set_vec_tile_shapes(
            configs.t_sub_tile, head_num // configs.chunk_size, head_dim
        )

        pypto.set_semantic_label("Query-Quant")
        pypto.set_vec_tile_shapes(
            configs.t_sub_tile, head_num // configs.chunk_size, head_dim
        )
        q_res = quant_tensor(q_hadamard)
        q_scale = pypto.cast(q_res[1], pypto.DT_FP16)

        pypto.assemble(q_res[0], [tIdx, 0, 0], q_int8_out)
        pypto.assemble(q_scale, [tIdx, 0, 0], q_scale_out)

        x = pypto.view(
            x_in, [t_tile, h], [tIdx, 0], valid_shape=[t_tile, h]
        )  # 这里将t_tile分档，offset不需要乘t_tile
        pypto.set_semantic_label("Weight-Linear")
        w_linear = configs.w_linear
        pypto.set_cube_tile_shapes(
            [w_linear[L0M_INDEX], w_linear[L1M_INDEX]],
            [w_linear[L0K_INDEX], w_linear[L1K_INDEX]],
            [w_linear[L0N_INDEX], w_linear[L1N_INDEX]],
        )
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
    cache_index_2d,
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
    kv_state_out,
    score_state_out,
    key,
    key_scale,
    ratio,
    start_pos,
    eps_rms,
    configs,
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
        cache_index_2d,
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
        kv_state_out,
        score_state_out,
        key,
        key_scale,
        ratio,
        start_pos,
        eps_rms,
        configs,
    )
