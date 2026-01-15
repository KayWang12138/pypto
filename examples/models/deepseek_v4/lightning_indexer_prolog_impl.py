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
Lightning Indexer Prolog Module

This module implements the Lightning Indexer Prolog computation
for DeepSeek V32 model. It handles:
- Query computation
- Weight computation for indexer attention

Main Functions:
    - lightning_indexer_prolog_compute: Main computation function
    - rope_3d: 3D RoPE computation

Example:
    See testdsv32_lightning_indexer_prolog.py for usage examples.
"""
from dataclasses import dataclass
import pypto
import math
import torch
import torch_npu
from typing import List

SHAPE_DIM_2 = 2
SHAPE_DIM_3 = 3

NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_7168 = 7168

TILE_CUBE_DIM = 6
Q_PARAM_DIM = 2
NZ_DIM = 4
COS_SIN_DIM = 2
L0M_INDEX = 0
L1M_INDEX = 1
L0K_INDEX = 2
L1K_INDEX = 3
L0N_INDEX = 4
L1N_INDEX = 5
SCATTER_DIM = -2
NZ_FIRST_DIM = 16
NZ_B8_C0 = 32
NZ_B16_C0 = 16

VEC_TILE_256 = 256
VEC_TILE_128 = 128
VEC_TILE_64 = 64
VEC_TILE_8 = 8
VEC_TILE_4 = 4
VEC_TILE_32 = 32


@dataclass
class IndexerPrologInput:
    x: torch.tensor  # BF16, (t, h)
    q_norm: torch.tensor  # BF16, (t, qLoraRank)
    w_qb: torch.tensor  # INT8, (headNum * headDim // NZ_B8_C0, qLoraRank // NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B8_C0), NZ
    w_proj: torch.tensor  # BF16, (headNum // NZ_B16_C0, h // NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B16_C0), NZ
    cos_idx_rope: torch.tensor  # BF16, (t, ropeHeadDim)
    sin_idx_rope: torch.tensor  # BF16, (t, ropeHeadDim)
    hadamard_q: torch.tensor  # BF16, (headDim, headDim)


@dataclass
class IndexerPrologOutput:
    q_bf16: torch.tensor
    weights: torch.tensor


@dataclass
class IndexerPrologAttr:
    eps: float
    layerout_query: str
    layerout_key: str


@dataclass
class IndexerPrologConfigs:
    q_linear: List[int]
    q_hd: List[int]
    w_linear: List[int]
    unroll_list: List[int]

    cube_l1_reuse_setting: dict[int, int]
    mg_copyin_upper_bound: int
    pg_upper_bound: int
    block_size: int
    t_sub_tile: int
    chunk_size: int
    vec_nbuffer_mode: int


def rotate_half(input_tensor: pypto.tensor) -> pypto.tensor:
    """Rotate half of the tensor dimensions for RoPE computation.

    Splits the last dimension in half and applies rotation transformation:
    [-x2, x1] where x1 is the first half and x2 is the second half.
    This is a key component of RoPE (Rotary Position Embedding).

    Args:
        input_tensor: Input tensor with last dimension divisible by 2

    Returns:
        Rotated tensor with same shape as input, where the first half of
        the last dimension is negated and swapped with the second half

    Raises:
        AssertionError: If the last dimension is not divisible by 2

    Example:
        If input is [a, b, c, d] along last dim, output is [-c, -d, a, b]
    """
    chunk_size = 2
    shape = input_tensor.shape
    shape_size = len(shape)
    assert shape_size >= 1
    assert shape[shape_size - 1] % chunk_size == 0
    shape[shape_size - 1] //= chunk_size
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    x1 = pypto.view(input_tensor, shape, offset1)
    x2 = pypto.view(input_tensor, shape, offset2)
    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rope_3d(x: pypto.tensor, cos: pypto.tensor, sin: pypto.tensor, configs: IndexerPrologConfigs) -> pypto.tensor:
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
    assert (len(x.shape) == SHAPE_DIM_3 and len(cos.shape) == SHAPE_DIM_2 and len(sin.shape) == SHAPE_DIM_2)

    x_dtype = x.dtype
    t_tile = x.shape[0]
    head_num = x.shape[head_num_axis]
    rope_dim = x.shape[head_dim_axis]

    pypto.set_vec_tile_shapes(1, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(configs.t_sub_tile, head_num // configs.chunk_size, rope_dim)
    x_view = pypto.cast(x, pypto.DT_FP32)
    cast_cos = pypto.reshape(cast_cos, [t_tile, 1, rope_dim])
    cast_sin = pypto.reshape(cast_sin, [t_tile, 1, rope_dim])

    x_embed = (x_view * cast_cos) + ((rotate_half(x_view)) * cast_sin)
    res = pypto.cast(x_embed, x_dtype)
    return res

def lightning_indexer_prolog_compute(x_in,
                                   q_norm_in,
                                   w_qb_in,
                                   w_proj_in,
                                   cos_idx_rope_in,
                                   sin_idx_rope_in,
                                   hadamard_q_in,
                                   q_bf16_out,
                                   weights_out, 
                                   configs):
    """Compute Lightning Indexer Prolog.

    Main computation function for Lightning Indexer Prolog.
    This function processes input tokens to generate query and weights
    for the indexer attention mechanism. The computation includes:

    1. Query Path:
       - Apply linear transformation with w_qb
       - Apply RoPE (Rotary Position Embedding)
       - Apply Hadamard transformation

    3. Weights Path:
       - Linear transformation with w_proj
       - Normalize by sqrt(head_num * head_dim)
       - Convert to BF16

    Args:
        x_in: Input hidden states tensor, shape (t, h), dtype BF16
        q_norm_in: query norm tensor, shape (t, q_lora_rank), dtype BF16
        w_qb_in: Query projection weight matrix, BF16 format with NZ layout
        w_proj_in: Weight projection matrix, BF16 format with NZ layout
        cos_idx_rope_in: Cosine values for RoPE, shape (t, rope_head_dim), dtype BF16
        sin_idx_rope_in: Sine values for RoPE, shape (t, rope_head_dim), dtype BF16
        hadamard_q_in: Hadamard transformation matrix for query, shape (head_dim, head_dim), dtype BF16
        q_bf16_out: Output query tensor, shape (t, head_num, head_dim), dtype BF16
        weights_out: Output weights tensor, shape (t, head_num), dtype BF16
        attrs: IndexerPrologAttr object containing:
            - eps: LayerNorm epsilon value
            - layerout_query: Query layout format (e.g., "TND")
        configs: IndexerPrologConfigs object containing tiling and optimization parameters

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

    unroll_list = configs.unroll_list
    for tIdx, unrollLength in pypto.loop_unroll(0, t, 1, name="IndexerPrologLoop", idx_name="tIdx",
                                                unroll_list=unroll_list, ):
        t_tile = unrollLength
        # 获取query计算的各阶段Tile参数
        q_linear = configs.q_linear
        q_hd = configs.q_hd
        # 多分档内会将t_tile作为档位，offset无需乘t_tile
        q_norm = pypto.view(q_norm_in, [t_tile, q_lora_rank], [tIdx, 0], valid_shape=[t_tile, q_lora_rank])
        pypto.set_semantic_label("Query-Linear")
        pypto.set_cube_tile_shapes([q_linear[L0M_INDEX], q_linear[L1M_INDEX]],
                                   [q_linear[L0K_INDEX], q_linear[L1K_INDEX]],
                                   [q_linear[L0N_INDEX], q_linear[L1N_INDEX]], True)
        q_f32 = pypto.matmul(q_norm, w_qb_in, pypto.DT_FP32)  # (t_tile, head_num * head_dim)

        pypto.set_vec_tile_shapes(configs.t_sub_tile, head_num * head_dim // configs.chunk_size)
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
        q_res = pypto.matmul(q_cat, hadamard_q, x_dtype)  # (t_tile, head_num, head_dim)

        pypto.set_vec_tile_shapes(configs.t_sub_tile, head_num // configs.chunk_size, head_dim)

        pypto.assemble(q_res, [tIdx, 0, 0], q_bf16_out)

        x = pypto.view(x_in, [t_tile, h], [tIdx, 0], valid_shape=[t_tile, h])  # 这里将t_tile分档，offset不需要乘t_tile
        pypto.set_semantic_label("Weight-Linear")
        w_linear = configs.w_linear
        pypto.set_cube_tile_shapes([w_linear[L0M_INDEX], w_linear[L1M_INDEX]],
                                   [w_linear[L0K_INDEX], w_linear[L1K_INDEX]],
                                   [w_linear[L0N_INDEX], w_linear[L1N_INDEX]])
        pypto.set_vec_tile_shapes(t_tile, head_num)
        weights = pypto.cast(pypto.matmul(x, w_proj_in, x_dtype), pypto.DT_FP32)
        weights = pypto.mul(weights, 1.0 / (math.sqrt(head_num) * math.sqrt(head_dim)))
        weights_f16 = pypto.cast(weights, pypto.DT_BF16)
        pypto.assemble(weights_f16, [tIdx, 0], weights_out)


@pypto.jit
def lightning_indexer_prolog(x_in,
                            q_norm_in,
                            w_qb_in,
                            w_proj_in,
                            cos_idx_rope_in,
                            sin_idx_rope_in,
                            hadamard_q_in,
                            q_bf16_out,
                            weights_out,
                            configs):
    """JIT-compiled wrapper for Lightning Indexer Prolog computation.

    This is the main entry point for the Lightning Indexer Prolog operator.
    It sets up optimization passes and runtime options before calling the core
    computation function.

    Args:
        x_in: Input hidden states tensor, shape (t, h), dtype BF16
        q_norm_in: query norm tensor, shape (t, q_lora_rank), dtype BF16
        w_qb_in: Query projection weight matrix, BF16 format with NZ layout
        w_proj_in: Weight projection matrix, BF16 format with NZ layout
        cos_idx_rope_in: Cosine values for RoPE, shape (t, rope_head_dim), dtype BF16
        sin_idx_rope_in: Sine values for RoPE, shape (t, rope_head_dim), dtype BF16
        hadamard_q_in: Hadamard transformation matrix for query, shape (head_dim, head_dim), dtype BF16
        q_bf16_out: Output query tensor, shape (t, head_num, head_dim), dtype BF16
        weights_out: Output weights tensor, shape (t, head_num), dtype FP16
        configs: IndexerPrologConfigs object containing optimization configurations

    Note:
        This function is decorated with @pypto.jit for JIT compilation.
        It configures pass options for memory optimization and calls the core
        computation function.
    """
    pypto.set_pass_options(vec_nbuffer_mode=configs.vec_nbuffer_mode)
    pypto.set_pass_options(cube_l1_reuse_setting=configs.cube_l1_reuse_setting)
    pypto.set_pass_options(mg_copyin_upper_bound=configs.mg_copyin_upper_bound)
    pypto.set_pass_options(pg_upper_bound=configs.pg_upper_bound)

    pypto.set_runtime_options(device_sched_mode=1)

    lightning_indexer_prolog_compute(x_in,
                                   q_norm_in,
                                   w_qb_in,
                                   w_proj_in,
                                   cos_idx_rope_in,
                                   sin_idx_rope_in,
                                   hadamard_q_in,
                                   q_bf16_out,
                                   weights_out,
                                   configs)