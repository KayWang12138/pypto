#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
FlashAttentionScoreGrad PyPTO Kernel 实现 (性能优化版)

优化项:
  1. S_TILE 64 → 128，减少循环迭代
  2. cube_tile_shapes 增大至 [128,128],[64,256],[128,128]
  3. pass_options: cube_l1_reuse_setting Q 常驻 L1
  4. runtime_options: stitch_function_max_num=128, device_sched_mode=1
  5. 内层 unroll_list=[8, 4, 2, 1]
"""

from dataclasses import dataclass

import pypto
import torch


# ============================================================
# 模块级常量
# ============================================================
NUM_HEADS = 8
HEAD_DIM = 64
S_TILE = 128  # 优化: 64 → 128


@dataclass
class TileConfig:
    """Tile configuration for compute_tile."""

    c_tile: list
    v_tile_s: list
    v_tile_d: list
    s_tile_size: int


@dataclass
class ComputeTileInputs:
    """Inputs for compute_tile function."""

    q_i: object
    k_j: object
    v_j: object
    dy_i: object
    smax_i: object
    ssum_i: object
    d_i: object
    actual_s1: int
    actual_s2: int
    scale_value: float
    cfg: TileConfig


@dataclass
class S1Inputs:
    """Container for s1 direction inputs."""

    q_i: object
    dy_i: object
    ao_i: object
    smax_i: object
    ssum_i: object
    d_i: object
    actual_s1: int


def compute_tile(inp: ComputeTileInputs):
    """计算一个 (s1_tile, s2_tile) 块的 P_ij 和 ds_ij。"""
    pypto.set_vec_tile_shapes(inp.cfg.v_tile_s[0], inp.cfg.v_tile_s[1])
    pypto.set_cube_tile_shapes(inp.cfg.c_tile[0], inp.cfg.c_tile[1], inp.cfg.c_tile[2])
    s_ij = pypto.matmul(inp.q_i, inp.k_j, pypto.DT_FP32, b_trans=True)
    s_ij = pypto.view(s_ij, [inp.cfg.s_tile_size, inp.cfg.s_tile_size], [0, 0],
                      valid_shape=[inp.actual_s1, inp.actual_s2])

    pypto.set_vec_tile_shapes(inp.cfg.v_tile_s[0], inp.cfg.v_tile_s[1])
    s_ij = pypto.mul(s_ij, inp.scale_value)
    p_ij = pypto.exp(pypto.sub(s_ij, inp.smax_i))
    p_ij = pypto.div(p_ij, inp.ssum_i)

    pypto.set_vec_tile_shapes(inp.cfg.v_tile_s[0], inp.cfg.v_tile_s[1])
    pypto.set_cube_tile_shapes(inp.cfg.c_tile[0], inp.cfg.c_tile[1], inp.cfg.c_tile[2])
    dp_ij = pypto.matmul(inp.dy_i, inp.v_j, pypto.DT_FP32, b_trans=True)
    dp_ij = pypto.view(dp_ij, [inp.cfg.s_tile_size, inp.cfg.s_tile_size], [0, 0],
                       valid_shape=[inp.actual_s1, inp.actual_s2])

    pypto.set_vec_tile_shapes(inp.cfg.v_tile_s[0], inp.cfg.v_tile_s[1])
    ds_ij = pypto.mul(p_ij, pypto.sub(dp_ij, inp.d_i))

    return p_ij, ds_ij


@dataclass
class KernelInputs:
    """Container for kernel input tensors."""

    q: object
    k: object
    v: object
    dy: object
    softmax_max: object
    softmax_sum: object
    attention_out: object
    dq: object
    dk: object
    dv: object
    batch_size: object
    scale_value: float


@dataclass
class PassInputs:
    """Container for pass function inputs."""

    cfg: TileConfig
    c_tile: list
    v_tile_d: list
    b_idx: int
    n_idx: int
    s: int
    s_loop: int
    bn_base: int


def _prepare_s1_inputs(ki: KernelInputs, s1_idx, bn_base, s, v_tile_d) -> S1Inputs:
    """Prepare inputs for s1 direction."""
    s1_off = bn_base + s1_idx * S_TILE
    actual_s1 = (s - s1_idx * S_TILE).min(S_TILE)

    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
    q_i = pypto.view(ki.q, [S_TILE, HEAD_DIM], [s1_off, 0],
                     valid_shape=[actual_s1, HEAD_DIM])
    dy_i = pypto.view(ki.dy, [S_TILE, HEAD_DIM], [s1_off, 0],
                      valid_shape=[actual_s1, HEAD_DIM])
    ao_i = pypto.view(ki.attention_out, [S_TILE, HEAD_DIM], [s1_off, 0],
                      valid_shape=[actual_s1, HEAD_DIM])

    sm_i_8 = pypto.view(ki.softmax_max, [S_TILE, 8], [s1_off, 0],
                        valid_shape=[actual_s1, 8])
    ss_i_8 = pypto.view(ki.softmax_sum, [S_TILE, 8], [s1_off, 0],
                        valid_shape=[actual_s1, 8])
    pypto.set_vec_tile_shapes(S_TILE, 8)
    smax_i = pypto.view(sm_i_8, [S_TILE, 1], [0, 0],
                        valid_shape=[actual_s1, 1])
    ssum_i = pypto.view(ss_i_8, [S_TILE, 1], [0, 0],
                        valid_shape=[actual_s1, 1])

    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
    dy_ao_fp32 = pypto.cast(pypto.mul(dy_i, ao_i), pypto.DT_FP32)
    d_i = pypto.sum(dy_ao_fp32, -1, keepdim=True)

    return S1Inputs(q_i, dy_i, ao_i, smax_i, ssum_i, d_i, actual_s1)


def _compute_dq_pass(ki: KernelInputs, pi: PassInputs):
    """趟1: 计算 dQ."""
    for s1_idx in pypto.loop(pi.s_loop, name="LOOP_s1_dq", idx_name="s1_idx"):
        s1_inputs = _prepare_s1_inputs(ki, s1_idx, pi.bn_base, pi.s, pi.v_tile_d)
        s1_off = pi.bn_base + s1_idx * S_TILE

        dq_acc = pypto.tensor([S_TILE, HEAD_DIM], pypto.DT_FP32, "dq_acc")

        for s2_idx in pypto.loop(pi.s_loop, name="LOOP_s2_dq",
                                 idx_name="s2_idx",
                                 unroll_list=[8, 4, 2, 1]):
            s2_off = pi.bn_base + s2_idx * S_TILE
            actual_s2 = (pi.s - s2_idx * S_TILE).min(S_TILE)

            pypto.set_vec_tile_shapes(pi.v_tile_d[0], pi.v_tile_d[1])
            k_j = pypto.view(ki.k, [S_TILE, HEAD_DIM], [s2_off, 0],
                             valid_shape=[actual_s2, HEAD_DIM])
            v_j = pypto.view(ki.v, [S_TILE, HEAD_DIM], [s2_off, 0],
                             valid_shape=[actual_s2, HEAD_DIM])

            ct_inp = ComputeTileInputs(
                s1_inputs.q_i, k_j, v_j, s1_inputs.dy_i, s1_inputs.smax_i,
                s1_inputs.ssum_i, s1_inputs.d_i, s1_inputs.actual_s1, actual_s2,
                ki.scale_value, pi.cfg)
            _, ds_ij = compute_tile(ct_inp)

            ds_bf16 = pypto.cast(ds_ij, pypto.DT_BF16)
            pypto.set_cube_tile_shapes(pi.c_tile[0], pi.c_tile[1], pi.c_tile[2])
            pypto.set_vec_tile_shapes(pi.v_tile_d[0], pi.v_tile_d[1])
            dq_tile = pypto.matmul(ds_bf16, k_j, pypto.DT_FP32)

            pypto.set_vec_tile_shapes(pi.v_tile_d[0], pi.v_tile_d[1])
            if pypto.is_loop_begin(s2_idx):
                dq_acc[:] = dq_tile
            else:
                dq_acc[:] = dq_acc + dq_tile

            if pypto.is_loop_end(s2_idx):
                pypto.set_vec_tile_shapes(pi.v_tile_d[0], pi.v_tile_d[1])
                dq_final = pypto.cast(
                    pypto.mul(dq_acc, ki.scale_value), pypto.DT_BF16)
                pypto.assemble(dq_final, [s1_off, 0], ki.dq)


def _compute_dk_dv_pass(ki: KernelInputs, pi: PassInputs):
    """趟2: 计算 dK, dV."""
    for s2_idx in pypto.loop(pi.s_loop, name="LOOP_s2_dkv",
                             idx_name="s2_idx"):
        s2_off = pi.bn_base + s2_idx * S_TILE
        actual_s2 = (pi.s - s2_idx * S_TILE).min(S_TILE)

        pypto.set_vec_tile_shapes(pi.v_tile_d[0], pi.v_tile_d[1])
        k_j = pypto.view(ki.k, [S_TILE, HEAD_DIM], [s2_off, 0],
                         valid_shape=[actual_s2, HEAD_DIM])
        v_j = pypto.view(ki.v, [S_TILE, HEAD_DIM], [s2_off, 0],
                         valid_shape=[actual_s2, HEAD_DIM])

        dk_acc = pypto.tensor([S_TILE, HEAD_DIM], pypto.DT_FP32, "dk_acc")
        dv_acc = pypto.tensor([S_TILE, HEAD_DIM], pypto.DT_FP32, "dv_acc")

        for s1_idx in pypto.loop(pi.s_loop, name="LOOP_s1_dkv",
                                 idx_name="s1_idx",
                                 unroll_list=[8, 4, 2, 1]):
            s1_inputs = _prepare_s1_inputs(ki, s1_idx, pi.bn_base, pi.s, pi.v_tile_d)

            ct_inp = ComputeTileInputs(
                s1_inputs.q_i, k_j, v_j, s1_inputs.dy_i, s1_inputs.smax_i,
                s1_inputs.ssum_i, s1_inputs.d_i, s1_inputs.actual_s1, actual_s2,
                ki.scale_value, pi.cfg)
            p_ij, ds_ij = compute_tile(ct_inp)

            ds_bf16 = pypto.cast(ds_ij, pypto.DT_BF16)
            p_bf16 = pypto.cast(p_ij, pypto.DT_BF16)
            pypto.set_cube_tile_shapes(pi.c_tile[0], pi.c_tile[1], pi.c_tile[2])
            pypto.set_vec_tile_shapes(pi.v_tile_d[0], pi.v_tile_d[1])
            dk_tile = pypto.matmul(ds_bf16, s1_inputs.q_i, pypto.DT_FP32, a_trans=True)
            dv_tile = pypto.matmul(p_bf16, s1_inputs.dy_i, pypto.DT_FP32, a_trans=True)

            pypto.set_vec_tile_shapes(pi.v_tile_d[0], pi.v_tile_d[1])
            if pypto.is_loop_begin(s1_idx):
                dk_acc[:] = dk_tile
                dv_acc[:] = dv_tile
            else:
                dk_acc[:] = dk_acc + dk_tile
                dv_acc[:] = dv_acc + dv_tile

            if pypto.is_loop_end(s1_idx):
                pypto.set_vec_tile_shapes(pi.v_tile_d[0], pi.v_tile_d[1])
                dk_final = pypto.cast(
                    pypto.mul(dk_acc, ki.scale_value), pypto.DT_BF16)
                dv_final = pypto.cast(dv_acc, pypto.DT_BF16)
                pypto.assemble(dk_final, [s2_off, 0], ki.dk)
                pypto.assemble(dv_final, [s2_off, 0], ki.dv)


def _run_kernel_body(ki: KernelInputs, b, s, s_loop):
    """Run the kernel body loops."""
    c_tile = [[S_TILE, S_TILE], [HEAD_DIM, 256], [S_TILE, S_TILE]]
    v_tile_s = [S_TILE, S_TILE]
    v_tile_d = [S_TILE, HEAD_DIM]
    cfg = TileConfig(c_tile, v_tile_s, v_tile_d, S_TILE)

    for b_idx in pypto.loop(b, name="LOOP_b", idx_name="b_idx"):
        for n_idx in pypto.loop(NUM_HEADS, name="LOOP_n", idx_name="n_idx"):
            bn_base = (b_idx * NUM_HEADS + n_idx) * s
            pi = PassInputs(cfg, c_tile, v_tile_d, b_idx, n_idx, s, s_loop, bn_base)
            _compute_dq_pass(ki, pi)
            _compute_dk_dv_pass(ki, pi)


def _prepare_and_run_kernel(q, k, v, dy, softmax_max, softmax_sum, attention_out, dq, dk, dv, batch_size, scale_value):
    """Prepare kernel inputs and run kernel body."""
    b = batch_size.shape[0]
    total = q.shape[0]
    s = total // b // NUM_HEADS
    s_loop = s // S_TILE

    ki = KernelInputs(q, k, v, dy, softmax_max, softmax_sum,
                      attention_out, dq, dk, dv, batch_size, scale_value)
    _run_kernel_body(ki, b, s, s_loop)


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 8},
        "cube_nbuffer_setting": {0: 4},
    }
)
def flash_attention_score_grad_kernel(
    q: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dy: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    softmax_max: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    softmax_sum: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    attention_out: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dq: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dk: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dv: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    batch_size: pypto.Tensor([pypto.DYN], pypto.DT_INT32),
    scale_value: float,
):
    _prepare_and_run_kernel(q, k, v, dy, softmax_max, softmax_sum, attention_out, dq, dk, dv, batch_size, scale_value)


@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1},
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 4},
    }
)
def flash_attention_score_grad_kernel_profile(
    q: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dy: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    softmax_max: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    softmax_sum: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    attention_out: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dq: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dk: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dv: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    batch_size: pypto.Tensor([pypto.DYN], pypto.DT_INT32),
    scale_value: float,
):
    """Profile 版本 kernel，带有 debug_options 用于生成泳道图数据。"""
    _prepare_and_run_kernel(q, k, v, dy, softmax_max, softmax_sum, attention_out, dq, dk, dv, batch_size, scale_value)


@dataclass
class WrapperInputs:
    """Container for wrapper inputs."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    dy: torch.Tensor
    softmax_max: torch.Tensor
    softmax_sum: torch.Tensor
    attention_out: torch.Tensor
    scale_value: float
    num_heads: int = NUM_HEADS
    head_dim: int = HEAD_DIM


def flash_attention_score_grad_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    attention_out: torch.Tensor,
    scale_value: float,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
):
    """算子 wrapper，供测试调用。"""
    wi = WrapperInputs(query, key, value, dy, softmax_max, softmax_sum,
                       attention_out, scale_value, num_heads, head_dim)
    batch_size, num_heads, seq_len, head_dim = wi.query.shape
    if num_heads != wi.num_heads or head_dim != wi.head_dim:
        raise ValueError(f"Shape mismatch: num_heads={num_heads}, head_dim={head_dim}, "
                         f"expected num_heads={wi.num_heads}, "
                         f"head_dim={wi.head_dim}")
    if seq_len % S_TILE != 0:
        raise ValueError(f"seq_len={seq_len} must be multiple of S_TILE={S_TILE}")

    q_flat = wi.query.reshape(-1, head_dim).contiguous()
    k_flat = wi.key.reshape(-1, head_dim).contiguous()
    v_flat = wi.value.reshape(-1, head_dim).contiguous()
    dy_flat = wi.dy.reshape(-1, head_dim).contiguous()
    ao_flat = wi.attention_out.reshape(-1, head_dim).contiguous()
    sm_flat = wi.softmax_max.reshape(-1, 8).contiguous()
    ss_flat = wi.softmax_sum.reshape(-1, 8).contiguous()

    dq_flat = torch.empty_like(q_flat)
    dk_flat = torch.empty_like(k_flat)
    dv_flat = torch.empty_like(v_flat)

    batch_tensor = torch.zeros(batch_size, dtype=torch.int32, device=wi.query.device)

    flash_attention_score_grad_kernel(
        q_flat, k_flat, v_flat, dy_flat,
        sm_flat, ss_flat, ao_flat,
        dq_flat, dk_flat, dv_flat,
        batch_tensor,
        wi.scale_value,
    )

    return (dq_flat.reshape(batch_size, num_heads, seq_len, head_dim),
            dk_flat.reshape(batch_size, num_heads, seq_len, head_dim),
            dv_flat.reshape(batch_size, num_heads, seq_len, head_dim))
