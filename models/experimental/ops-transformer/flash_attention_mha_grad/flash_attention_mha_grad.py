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
Flash Attention Backward with Dynamic Variable Length Sequences

语义约定:
  - Q 侧: s1_size (Q seqlen), 张量包括 Q/O/dO/L/M/dQ
  - KV 侧: s2_size (KV seqlen), 张量包括 K/V/dK/dV
  - S2_TILE: KV 序列维度的分块大小 (将 s2_size 切分为多个 tile 迭代)

3 loops: batch + head + kv_tile (KV sequence tiling).
Tiles KV sequence dimension by S2_TILE to reduce intermediate attention matrix
from [s1_size, s2_size] to [s1_size, S2_TILE] per iteration.
dK and dV are accumulated across kv tiles.
"""

import os
import logging
from dataclasses import dataclass

import torch
import numpy as np
import pytest
from numpy.testing import assert_allclose

import pypto


logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM

# KV 序列维度的分块大小 (全局配置常量)
S2_TILE = 320


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logging.info("Please set TILE_FWK_DEVICE_ID before running:")
        logging.info("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        return None


@pypto.frontend.jit(
    pass_options={
        "cube_nbuffer_setting": {-1: 8},
        "vec_nbuffer_setting": {-1: 16},
        "cube_l1_reuse_setting": {-1: 8},
        "pg_upper_bound": 500000000,
    },
    runtime_options={
        "stitch_function_max_num": 256,
        "device_sched_mode": 0,
    },
    debug_options={
        "runtime_debug_mode": 1
    }
)
def flash_attention_varlen_backward_kernel(
    # Q 侧输入: shape=[bs, N, D], bs=DYNAMIC, N=num_heads, D=head_dim
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # KV 侧输入: shape=[bs, N, D]
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # Q 侧: 前向输出 O, shape=[bs, N, D]
    o: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # Q 侧: dO, shape=[bs, N, D]
    do: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # Q 侧: softmax 中间量 L, shape=[bs, N, 1]
    l_input: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    # Q 侧: softmax 中间量 M, shape=[bs, N, 1]
    m_input: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    # Q 侧输出: dQ, shape=[bs, N*D] (二维)
    dq: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # KV 侧输出: dK, shape=[bs, N*D] (二维)
    dk: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # KV 侧输出: dV, shape=[bs, N*D] (二维)
    dv: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # actual_q: shape=[batch_size], actual_q[i]=第 i 个 batch 的 Q seqlen (s1_size)
    actual_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    # actual_kv: shape=[batch_size], actual_kv[i]=第 i 个 batch 的 KV seqlen (s2_size)
    actual_kv: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
):
    """
    Flash Attention Backward - 4 loops (batch + head + q_tile + kv_tile).

    输入张量为三维 [bs, N, D] (bs=DYNAMIC, N=num_heads, D=head_dim),
    进入循环前 reshape inplace 为 [bs, N*D] 以便按 head 做 view 切片。

    张量布局 (Q: s1_size, KV: s2_size):
      Q 侧: Q/O/dO/L/M/dQ — 每个 batch 占用 s1 行 (实际 Q seqlen, 从 actual_q 获取)
      KV 侧: K/V/dK/dV    — 每个 batch 占用 s2_total 行 (实际 KV seqlen, 从 actual_kv 获取)

    actual_q.shape[0]  = batch_size
    actual_q[i]  = 第 i 个 batch 的 Q seqlen
    actual_kv[i] = 第 i 个 batch 的 KV seqlen

    计算流程 (per batch, per head, per q_tile, per kv_tile):
      对 Q seqlen 按 S2_TILE 分块, 对 KV seqlen 按 S2_TILE 分块:
        S_tile = Q_tile @ K_tile^T * scale         [sq, s2]       BF16→FP32
        P_tile = exp(S*scale - M) / L               [sq, s2]       FP32
        dP_tile = dO_tile @ V_tile^T                [sq, s2]       BF16→FP32
        D = sum(O_tile * dO_tile, dim=-1)           [sq, 1]        BF16→FP32
        dS_tile = P * (dP - D)                      [sq, s2]       FP32→BF16
        dK_tile += dS^T @ Q_tile * scale            [s2, D]        BF16→FP32→BF16
        dV_tile += P^T @ dO_tile                    [s2, D]        BF16→BF16
        dQ_partial = dS @ K_tile * scale            [sq, D]        BF16→FP32
      dQ 在 kv_tile 循环中累加 (FP32), 最终 cast 为 BF16 写回。
    """
    pypto.experimental.set_operation_options(combine_axis=True)

    # ---- 从三维输入获取 N(num_heads) 和 D(head_dim), 然后 reshape 为二维 ----
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    hidden_dim = num_heads * head_dim
    bs = q.shape[0]
    scale = 1.0 / (head_dim ** 0.5)

    # reshape inplace: q/k/v/o/do [bs, N, D] → [bs, N*D]
    q_2d = pypto.reshape(q, [bs, hidden_dim], inplace=True)
    k_2d = pypto.reshape(k, [bs, hidden_dim], inplace=True)
    v_2d = pypto.reshape(v, [bs, hidden_dim], inplace=True)
    o_2d = pypto.reshape(o, [bs, hidden_dim], inplace=True)
    do_2d = pypto.reshape(do, [bs, hidden_dim], inplace=True)
    # reshape inplace: l_input/m_input [bs, N, 1] → [bs, N]
    l_input_2d = pypto.reshape(l_input, [bs, num_heads], inplace=True)
    m_input_2d = pypto.reshape(m_input, [bs, num_heads], inplace=True)
    # dq/dk/dv 输入时已经是 [bs, N*D], 无需 reshape

    # bs = actual_q 的 shape (即 batch_size)
    bs = actual_q.shape[0]
    for b_idx in pypto.loop(bs, name="batch_loop", parallel=True):
        # s1: 当前 batch 的 Q seqlen (从 actual_q 获取, 不使用全局变量)
        s1 = actual_q[b_idx]
        s1.as_variable()
        # s2_total: 当前 batch 的 KV seqlen (从 actual_kv 获取, 不使用全局变量)
        s2_total = actual_kv[b_idx]
        s2_total.as_variable()

        # Q 侧偏移: 每个 batch 占 s1 行
        q_start = b_idx * s1
        # KV 侧偏移: 每个 batch 占 s2_total 行
        kv_start = b_idx * s2_total

        # Q seqlen 按 S2_TILE 分块的 tile 数量
        num_q_tiles = (s1 + S2_TILE - 1) // S2_TILE
        num_q_tiles.as_variable()
        # KV seqlen 按 S2_TILE 分块的 tile 数量 (S2_TILE 为全局配置常量)
        num_kv_tiles = (s2_total + S2_TILE - 1) // S2_TILE
        num_kv_tiles.as_variable()

        head_num_loop = num_heads // 2
        for h_idx in pypto.loop(head_num_loop, name="head_loop"):

            for q_tile_idx in pypto.loop(num_q_tiles, name="q_tile_loop"):
                # ---- Q 侧: 当前 Q tile 的偏移和有效长度 ----
                q_ofs = q_tile_idx * S2_TILE
                sq = (s1 - q_ofs).min(S2_TILE)
                sq.as_variable()

                # dQ 累加器 (FP32), shape=[S2_TILE, head_dim], 跨 kv_tile 累加
                dq_update = pypto.tensor([S2_TILE, head_dim], pypto.DT_FP32, "dq_update")

                for kv_tile_idx in pypto.loop(num_kv_tiles, name="kv_tile_loop"):

                    for h_s_idx in range(2):
                        h_ofs = (h_idx * 2 + h_s_idx) * head_dim
                        # L/M reshape 后为 [bs, N], head 索引不乘 head_dim
                        h_idx_lm = h_idx * 2 + h_s_idx

                        # ---- Q 侧: 加载当前 Q tile 的 Q/O/dO/M/L ----
                        # q/o/do reshape 后为 [bs, N*head_dim], 静态 shape=[S2_TILE, head_dim]
                        qi = pypto.view(q_2d, [S2_TILE, head_dim], [q_start + q_ofs, h_ofs],
                                        valid_shape=[sq, head_dim])
                        oi = pypto.view(o_2d, [S2_TILE, head_dim], [q_start + q_ofs, h_ofs],
                                        valid_shape=[sq, head_dim])
                        doi = pypto.view(do_2d, [S2_TILE, head_dim], [q_start + q_ofs, h_ofs],
                                        valid_shape=[sq, head_dim])
                        # l_input/m_input reshape 后为 [bs, N], 每个 head 1 个值
                        m_i = pypto.view(m_input_2d, [S2_TILE, 1], [q_start + q_ofs, h_idx_lm],
                                        valid_shape=[sq, 1])
                        l_i = pypto.view(l_input_2d, [S2_TILE, 1], [q_start + q_ofs, h_idx_lm],
                                        valid_shape=[sq, 1])

                        # ---- KV 侧: 当前 KV tile 的偏移和有效长度 ----
                        s2_ofs = kv_tile_idx * S2_TILE
                        s2 = (s2_total - s2_ofs).min(S2_TILE)
                        s2.as_variable()

                        pypto.set_pass_options(sg_set_scope=1)
                        # KV 侧: 加载 K_tile, V_tile (静态 S2_TILE, 有效 s2)
                        ki_tile = pypto.view(k_2d, [S2_TILE, head_dim], [kv_start + s2_ofs, h_ofs],
                                             valid_shape=[s2, head_dim])
                        vi_tile = pypto.view(v_2d, [S2_TILE, head_dim], [kv_start + s2_ofs, h_ofs],
                                             valid_shape=[s2, head_dim])

                        # 计算公式： head_dim = sum(O * dO, dim=-1, keepdim=True) -> [sq, 1]
                        # dtype: BF16 → cast → FP32, mul, sum
                        pypto.set_vec_tile_shapes(512, 64)
                        oi_fp32 = pypto.cast(oi, pypto.DT_FP32)
                        doi_fp32 = pypto.cast(doi, pypto.DT_FP32)
                        do_o = pypto.mul(oi_fp32, doi_fp32)
                        d_tile = pypto.sum(do_o, dim=-1, keepdim=True)
                        pypto.set_pass_options(sg_set_scope=-1)

                        # 计算公式： dP_tile = dO_tile @ V_tile^T -> [sq, s2]
                        # 数据类型转换： dtype: BF16 matmul → FP32
                        pypto.set_cube_tile_shapes([128, 512], [64, 64], [256, 512])
                        dp = pypto.matmul(doi, vi_tile, out_dtype=pypto.DT_FP32, b_trans=True)
                        pypto.set_vec_tile_shapes(64, 512)
                        dp = pypto.view(dp, [S2_TILE, S2_TILE], [0, 0], valid_shape=[sq, s2])

                        # 计算公式：S_tile = Q_tile @ K_tile^T -> [sq, s2]
                        # 数据类型转换：dtype: BF16 matmul → FP32
                        pypto.set_cube_tile_shapes([128, 512], [64, 64], [256, 512])
                        scores = pypto.matmul(qi, ki_tile, out_dtype=pypto.DT_FP32, b_trans=True)

                        pypto.set_vec_tile_shapes(64, 512)
                        scores = pypto.view(scores, [S2_TILE, S2_TILE], [0, 0], valid_shape=[sq, s2])

                        # 计算公式： P_tile = exp(S * scale - M) / L  (softmax)
                        # 数据类型转换：dtype: FP32 全程
                        pypto.set_pass_options(sg_set_scope=2)
                        p = pypto.div(pypto.exp(pypto.sub(pypto.mul(scores, scale), m_i)), l_i)
                        # 计算公式： dS_tile = P * (dP - head_dim)
                        # 数据类型转换：dtype: FP32
                        ds = pypto.mul(p, pypto.sub(dp, d_tile))
                        # 数据类型转换：dtype: FP32 → cast → BF16 (用于后续 matmul 输入)
                        ds_half = pypto.cast(ds, pypto.DT_BF16)
                        p_half = pypto.cast(p, pypto.DT_BF16)
                        pypto.set_pass_options(sg_set_scope=-1)

                        # 计算公式： dK_tile += dS^T @ Q_tile * scale -> [s2, head_dim]
                        # 数据类型转换：dtype: BF16 matmul → FP32, mul(scale), cast → BF16
                        pypto.set_cube_tile_shapes([128, 512], [256, 512], [64, 64])
                        dk_tile_mm = pypto.matmul(ds_half, qi, out_dtype=pypto.DT_FP32, a_trans=True)
                        pypto.set_vec_tile_shapes(512, 64)
                        dk_tile = pypto.mul(dk_tile_mm, scale)
                        # KV 侧写回: dK[kv_start + s2_ofs]
                        pypto.assemble(pypto.cast(dk_tile, pypto.DT_BF16), [kv_start + s2_ofs, h_ofs], dk)

                        # 计算公式： dV_tile += P^T @ dO_tile -> [s2, head_dim]
                        # 数据类型转换：dtype: BF16 matmul → BF16 (直接输出 BF16)
                        pypto.set_cube_tile_shapes([128, 512], [256, 512], [64, 64])
                        dv_tile = pypto.matmul(p_half, doi, out_dtype=pypto.DT_BF16, a_trans=True)
                        # KV 侧写回: dV[kv_start + s2_ofs]
                        pypto.assemble(dv_tile, [kv_start + s2_ofs, h_ofs], dv)

                        # 计算公式： dQ_partial = dS @ K_tile * scale -> [sq, head_dim]
                        # 数据类型转换：dtype: BF16 matmul → FP32, mul(scale)
                        pypto.set_cube_tile_shapes([128, 512], [256, 512], [64, 64])
                        dq_partial_mm = pypto.matmul(ds_half, ki_tile, out_dtype=pypto.DT_FP32)
                        dq_partial = pypto.mul(dq_partial_mm, scale)

                        # dQ 在 kv_tile 循环中累加 (FP32)
                        if pypto.is_loop_begin(kv_tile_idx):
                            if pypto.is_loop_end(kv_tile_idx):
                                # 单个 KV tile: 直接 cast → BF16 写回 Q 侧对应 tile 位置
                                pypto.assemble(pypto.cast(dq_partial, pypto.DT_BF16), [q_start + q_ofs, h_ofs], dq)
                            else:
                                dq_update[:] = dq_partial
                        else:
                            dq_new = pypto.add(dq_update, dq_partial)
                            if pypto.is_loop_end(kv_tile_idx):
                                # 最后一个 KV tile: cast → BF16 写回 Q 侧对应 tile 位置
                                pypto.assemble(pypto.cast(dq_new, pypto.DT_BF16), [q_start + q_ofs, h_ofs], dq)
                            else:
                                dq_update[:] = dq_new


########################################################################
# 公共工具函数
########################################################################


@dataclass
class TileConfig:
    """
    分块配置结构体，封装 kernel 所需的 tile 参数。

    将 tile 信息集中管理，避免在 run_test 中直接引用全局变量。

    Attributes:
        s2_tile:     KV 序列维度的分块大小 (kernel 中 S2_TILE,
                     用于将 KV seqlen 切分为多个 tile 迭代)
    """
    s2_tile: int = S2_TILE


def create_inputs(batch_size, s1_size, s2_size, num_heads, head_dim, device):
    """
    创建 padded 布局的输入张量。

    布局说明 (Q: s1_size, KV: s2_size):
      - Q/O/dO/L/M/dQ: 每个 batch 占用 s1_size 行 (Q seqlen)
      - K/V/dK/dV:      每个 batch 占用 s2_size 行 (KV seqlen)

    数据类型严格对应 kernel 签名:
      - q/k/v:       BF16  (kernel: pypto.DT_BF16)
      - actual_q:    INT32 (kernel: pypto.DT_INT32) — 每个 batch 的 Q seqlen
      - actual_kv:   INT32 (kernel: pypto.DT_INT32) — 每个 batch 的 KV seqlen

    Args:
        batch_size: 批次数量
        s1_size:    Q 序列长度 (每个 batch 的 Q seqlen)
        s2_size:    KV 序列长度 (每个 batch 的 KV seqlen)
        num_heads:  注意力头数
        head_dim:   每个头的维度
        device:     计算设备
    Returns:
        q:          shape=[batch_size * s1_size, num_heads, head_dim], dtype=bfloat16
        k, v:       shape=[batch_size * s2_size, num_heads, head_dim], dtype=bfloat16
        actual_q:   shape=[batch_size], dtype=int32
                    actual_q[i]=第 i 个 batch 的 s1_size
        actual_kv:  shape=[batch_size], dtype=int32
                    actual_kv[i]=第 i 个 batch 的 s2_size
    """
    total_q = batch_size * s1_size
    total_kv = batch_size * s2_size

    torch.manual_seed(42)
    # Q 张量: shape=[batch * s1_size, num_heads, head_dim], dtype=BF16
    # kernel 签名为 [DYNAMIC, N, D], 三维输入
    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1 + 0.5
    # K/V 张量: shape=[batch * s2_size, num_heads, head_dim], dtype=BF16
    k = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1 + 0.5
    v = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1 + 0.5

    # 计算流：actual_q: [s1_size, s1_size, ...] — shape=[batch_size], Q 每个 batch 的 seqlen
    q_seqlens = [s1_size] * batch_size
    actual_q = torch.tensor(
        q_seqlens, dtype=torch.int32, device=device
    )
    # 计算流：actual_kv: [s2_size, s2_size, ...] — shape=[batch_size], KV 每个 batch 的 seqlen
    kv_seqlens = [s2_size] * batch_size
    actual_kv = torch.tensor(
        kv_seqlens, dtype=torch.int32, device=device
    )
    return q, k, v, actual_q, actual_kv, q_seqlens, kv_seqlens


def attention_backward_golden(q, k, v, o_input, do_t, scale):
    """
    Golden reference: 严格模拟 kernel 内部的 dtype 转换流程。

    Q: [s1_size, head_dim],  KV: [s2_size, head_dim]
    输入 q/k/v/o_input/do_t 均为 BF16, 与 kernel 签名一致。

    Kernel dtype 转换对照:
      1. O, dO: BF16 → cast → FP32               (pypto.cast → DT_FP32)
      2. scores = Q(BF16) @ K^T(BF16) → FP32      (matmul out_dtype=FP32)
      3. P = softmax(scores) → FP32                (FP32 全程)
      4. dP = dO(BF16) @ V^T(BF16) → FP32          (matmul out_dtype=FP32)
      5. D = sum(O_fp32 * dO_fp32) → FP32
      6. dS = P * (dP - D) → FP32
      7. ds_half = cast(dS, BF16)                  (pypto.cast → DT_BF16)
         p_half  = cast(P, BF16)                   (pypto.cast → DT_BF16)
      8. dK = ds_half^T(BF16) @ Q(BF16) → FP32 * scale → cast BF16
      9. dV = p_half^T(BF16) @ dO(BF16) → BF16       (matmul out_dtype=BF16)
     10. dQ_partial = ds_half(BF16) @ K(BF16) → FP32 * scale
         dQ 跨 tile 累加 (FP32), 最终 cast BF16

    Args:
        q:       [s1_size, head_dim] BF16 — Q 切片
        k, v:    [s2_size, head_dim] BF16 — KV 切片
        o_input: [s1_size, head_dim] BF16 — 前向输出 O (预计算, 与 kernel 输入一致)
        do_t:    [s1_size, head_dim] BF16 — dO 切片
        scale:   attention scale factor (1/sqrt(head_dim))
    Returns:
        dq: [s1_size, head_dim] BF16, dk: [s2_size, head_dim] BF16, dv: [s2_size, head_dim] BF16
    """
    # 计算流：scores = Q(BF16) @ K^T(BF16) → FP32 (kernel: matmul out_dtype=FP32)
    scores = torch.matmul(q.float(), k.float().T) * scale
    # 计算流： P = softmax (kernel: exp(S*scale - M) / L, 全程 FP32)
    p = torch.softmax(scores, dim=-1)

    #  计算流：---- D = sum(O_fp32 * dO_fp32, dim=-1) ----
    # kernel: O 来自输入张量 (BF16) → cast(O, FP32) * cast(dO, FP32) → sum
    # 使用传入的 o_input (BF16) 而非重算, 与 kernel 严格一致
    d = (o_input.float() * do_t.float()).sum(dim=-1, keepdim=True)

    #  计算流： ---- dP = dO(BF16) @ V^T(BF16) → FP32 ----
    dp = torch.matmul(do_t.float(), v.float().T)

    #  计算流：---- dS = P * (dP - D) → FP32 ----
    ds = p * (dp - d)

    # ---- 中间 cast: FP32 → BF16 (kernel: pypto.cast → DT_BF16) ----
    ds_half = ds.to(torch.bfloat16)
    p_half = p.to(torch.bfloat16)

    #  计算流：---- dK = ds_half^T(BF16) @ Q(BF16) → FP32 * scale → BF16 ----
    #  计算流：kernel: matmul(ds_half, qi, out_dtype=FP32, a_trans=True) 即 ds_half^T @ Q
    #  计算流：ds_half: [s1, s2], Q: [s1, D] → ds_half^T: [s2, s1] @ Q: [s1, D] → [s2, D]
    dk_fp32 = torch.matmul(ds_half.float().T, q.float()) * scale
    dk = dk_fp32.to(torch.bfloat16)

    #  计算流：---- dV = p_half^T(BF16) @ dO(BF16) → BF16 ----
    #  计算流：kernel: matmul(p_half, doi, out_dtype=BF16, a_trans=True) 即 p_half^T @ dO
    #  计算流：p_half: [s1, s2], dO: [s1, D] → p_half^T: [s2, s1] @ dO: [s1, D] → [s2, D]
    # 注: 模拟 BF16 matmul — 先 FP32 计算再 cast BF16
    dv = torch.matmul(p_half.float().T, do_t.float()).to(torch.bfloat16)

    #  计算流：---- dQ = ds_half(BF16) @ K(BF16) → FP32 * scale → BF16 ----
    #  计算流：kernel: matmul(ds_half, ki_tile, out_dtype=FP32) → mul(scale) → 累加(FP32) → cast(BF16)
    dq_fp32 = torch.matmul(ds_half.float(), k.float()) * scale
    dq = dq_fp32.to(torch.bfloat16)

    return dq, dk, dv


def compute_l_m_o(q, k, v, scale, head_dim):
    """
    预计算 softmax 中间量 L, M 和前向输出 O。

    Q: [s1_size, head_dim],  KV: [s2_size, head_dim]

    对应 kernel 输入:
      - l_input: pypto.DT_FP32 — softmax 分母 L
      - m_input: pypto.DT_FP32 — softmax 最大值 M
      - o:       pypto.DT_BF16 — 前向注意力输出 O

    计算过程:
      scores = Q @ K^T * scale        [s1_size, s2_size]  (FP32)
      M = max(scores, dim=-1)         [s1_size, 1]        (FP32, 数值稳定)
      P_unnorm = exp(scores - M)      [s1_size, s2_size]  (FP32)
      L = sum(P_unnorm, dim=-1)       [s1_size, 1]        (FP32, softmax 分母)
      O = (P_unnorm / L) @ V          [s1_size, head_dim] (FP32 → BF16)

    Args:
        q:        [s1_size, head_dim] — Q 切片
        k, v:     [s2_size, head_dim] — KV 切片
        scale:    attention scale factor
        head_dim: 每个头的维度 (用于 expand L/M)
    Returns:
        l: [s1_size, 1], dtype=float32
        m: [s1_size, 1], dtype=float32
        o: [s1_size, head_dim], dtype=bfloat16
    """
    scores = torch.matmul(q.float(), k.float().T) * scale
    m = scores.max(dim=-1, keepdim=True)[0]
    p = torch.exp(scores - m)
    l = p.sum(dim=-1, keepdim=True)
    o = torch.matmul(p / l, v.float())
    # L, M 保持 [sq, 1], 不再 expand
    return l, m, o.to(torch.bfloat16)


def run_test(device, batch_size=None, num_heads=None, s1_size=None,
             s2_size=None, dim=None, tile_config=None):
    """
    运行单个测试用例: 构造输入 → 调用 kernel → 与 golden 对比。

    参数语义 (Q: s1_size, KV: s2_size):
      - batch_size: 批次数量              (默认 1)
      - num_heads:  注意力头数            (默认 NUM_HEADS=8)
      - s1_size:    Q 序列长度            (默认 320)
      - s2_size:    KV 序列长度           (默认 = s1_size, 自注意力时相等)
      - dim:        每个头的维度 head_dim (默认 HEAD_DIM=64)
      - tile_config: TileConfig 分块配置  (默认使用全局 S2_TILE)

    输入张量布局:
      - Q/O/dO/L/M/dQ: [batch_size * s1_size, hidden_dim]  — Q 侧
      - K/V/dK/dV:      [batch_size * s2_size, hidden_dim]  — KV 侧
      - actual_q:        [batch_size] — 每个 batch 的 Q seqlen (s1_size)
      - actual_kv:       [batch_size] — 每个 batch 的 KV seqlen (s2_size)

    数据类型严格对应 kernel 签名:
      ┌────────────┬──────────────────────────────────────────┬─────────────┐
      │ 张量       │ kernel 签名                              │ host dtype  │
      ├────────────┼──────────────────────────────────────────┼─────────────┤
      │ q/k/v/o/do │ pypto.Tensor([DYN, HIDDEN_DIM], DT_BF16) │ bfloat16    │
      │ l/m        │ pypto.Tensor([DYN, HIDDEN_DIM], DT_FP32) │ float32     │
      │ dq/dk/dv   │ pypto.Tensor([DYN, HIDDEN_DIM], DT_BF16) │ bfloat16    │
      │ actual_seq │ pypto.Tensor([DYN],            DT_INT32) │ int32       │
      └────────────┴──────────────────────────────────────────┴─────────────┘

    Kernel 内部 dtype 转换流程:
      1. O, dO: BF16 → cast → FP32 (计算 D = sum(O*dO))
      2. matmul 输出: 大部分 out_dtype=FP32，仅 dV 的 matmul 直接输出 BF16
      3. P, dS: FP32 → cast → BF16 (作为后续 matmul 的输入，减少带宽)
      4. dQ/dK 输出: FP32 → cast → BF16 后 assemble 写回

    Args:
        device:      计算设备
        batch_size:  批次数量 (可选)
        num_heads:   注意力头数 (可选)
        s1_size:     Q 序列长度 (可选)
        s2_size:     KV 序列长度 (可选, 默认 = s1_size)
        dim:         每个头的维度 head_dim (可选)
        tile_config: TileConfig 分块配置 (可选)
    Returns:
        passed: 是否通过精度校验
    """
    # ---- 参数默认值，未传则使用全局常量 ----
    if batch_size is None:
        batch_size = 1
    if num_heads is None:
        num_heads = NUM_HEADS
    if s1_size is None:
        s1_size = 320
    if s2_size is None:
        # 默认自注意力: KV seqlen = Q seqlen
        s2_size = s1_size
    if dim is None:
        dim = HEAD_DIM
    if tile_config is None:
        tile_config = TileConfig()

    # 派生常量 (全部从输入参数计算，不直接引用全局变量)
    hidden_dim = num_heads * dim
    scale = 1.0 / (dim ** 0.5)

    logging.info("=" * 60)
    logging.info(f"Test Case: batch={batch_size}, heads={num_heads}, "
          f"Q:s1_size={s1_size}, KV:s2_size={s2_size}, dim={dim}")
    logging.info(f"  hidden_dim={hidden_dim}, scale={scale:.6f}, "
          f"s2_tile={tile_config.s2_tile}")
    logging.info("=" * 60)

    # ---- 构造输入张量，dtype 严格匹配 kernel 签名 ----
    # Q: [batch * s1_size, num_heads, dim], KV: [batch * s2_size, num_heads, dim]
    # kernel 签名为三维 [DYNAMIC, N, D]
    q, k, v, actual_q, actual_kv, q_seqlens, kv_seqlens = create_inputs(
        batch_size, s1_size, s2_size, num_heads, dim, device)

    total_q = batch_size * s1_size
    total_kv = batch_size * s2_size

    torch.manual_seed(42)
    # dO: Q 侧, shape=[batch * s1_size, num_heads, dim], dtype=BF16
    do_t = torch.randn(total_q, num_heads, dim, dtype=torch.bfloat16, device=device) * 0.1 + 0.5

    # L, M: Q 侧, shape=[batch * s1_size, num_heads, 1], dtype=FP32
    l_out = torch.empty(total_q, num_heads, 1, dtype=torch.float32, device=device)
    m_out = torch.empty(total_q, num_heads, 1, dtype=torch.float32, device=device)
    # O: Q 侧, shape=[batch * s1_size, num_heads, dim], dtype=BF16
    o_out = torch.empty(total_q, num_heads, dim, dtype=torch.bfloat16, device=device)

    # 预计算每个 (batch, head) 的 L, M, O
    # 三维张量按 [seq, head, dim] 索引: tensor[q_off:q_off+sq, h, :]
    for b in range(batch_size):
        sq = q_seqlens[b]    # Q seqlen for this batch
        skv = kv_seqlens[b]  # KV seqlen for this batch
        q_off = b * s1_size
        kv_off = b * s2_size
        for h in range(num_heads):
            l_h, m_h, o_h = compute_l_m_o(
                q[q_off: q_off + sq, h, :],
                k[kv_off: kv_off + skv, h, :],
                v[kv_off: kv_off + skv, h, :],
                scale, dim)
            # l_h, m_h: [sq, 1], o_h: [sq, dim]
            l_out[q_off: q_off + sq, h, :] = l_h
            m_out[q_off: q_off + sq, h, :] = m_h
            o_out[q_off: q_off + sq, h, :] = o_h

    # dQ: Q 侧, shape=[batch * s1_size, hidden_dim] (二维), dtype=BF16
    dq_out = torch.empty(total_q, hidden_dim, dtype=torch.bfloat16, device=device)
    # dK, dV: KV 侧, shape=[batch * s2_size, hidden_dim] (二维), dtype=BF16
    dk_out = torch.empty(total_kv, hidden_dim, dtype=torch.bfloat16, device=device)
    dv_out = torch.empty(total_kv, hidden_dim, dtype=torch.bfloat16, device=device)

    # ---- Golden 计算: 先完整计算所有 batch/head 的 golden dQ/dK/dV ----
    # golden dQ/dK/dV 与 kernel 输出同 shape: [total, hidden_dim], dtype=BF16
    dq_golden = torch.empty(total_q, hidden_dim, dtype=torch.bfloat16, device=device)
    dk_golden = torch.empty(total_kv, hidden_dim, dtype=torch.bfloat16, device=device)
    dv_golden = torch.empty(total_kv, hidden_dim, dtype=torch.bfloat16, device=device)

    for b in range(batch_size):
        sq = q_seqlens[b]
        skv = kv_seqlens[b]
        q_off = b * s1_size
        kv_off = b * s2_size
        for h in range(num_heads):
            h_off = h * dim
            dq_g, dk_g, dv_g = attention_backward_golden(
                q[q_off: q_off + sq, h, :],
                k[kv_off: kv_off + skv, h, :],
                v[kv_off: kv_off + skv, h, :],
                o_out[q_off: q_off + sq, h, :],
                do_t[q_off: q_off + sq, h, :],
                scale)
            # golden 返回 [seq, dim] BF16, 写入对应 [total, hidden_dim] 位置
            dq_golden[q_off: q_off + sq, h_off: h_off + dim] = dq_g
            dk_golden[kv_off: kv_off + skv, h_off: h_off + dim] = dk_g
            dv_golden[kv_off: kv_off + skv, h_off: h_off + dim] = dv_g

    # ---- 调用 kernel ----
    logging.info("  Running kernel...")
    flash_attention_varlen_backward_kernel(
        q, k, v, o_out, do_t, l_out, m_out,
        dq_out, dk_out, dv_out,
        actual_q, actual_kv)

    # ---- 精度校验: kernel 输出 vs golden 输出, 使用 numpy assert_allclose ----
    rtol = 0.0078125  # 1/128, 约 BF16 精度
    atol = 0.0001

    passed = True
    for name, npu_tensor, golden_tensor in [
        ("dQ", dq_out, dq_golden),
        ("dK", dk_out, dk_golden),
        ("dV", dv_out, dv_golden),
    ]:
        npu_np = npu_tensor.float().cpu().numpy()
        golden_np = golden_tensor.float().cpu().numpy()
        max_diff = np.abs(npu_np - golden_np).max()
        try:
            assert_allclose(npu_np, golden_np, rtol=rtol, atol=atol)
            logging.info(f"  {name}: PASSED (max_diff={max_diff:.6f}, rtol={rtol}, atol={atol})")
        except AssertionError as e:
            logging.info(f"  {name}: FAILED (max_diff={max_diff:.6f}, rtol={rtol}, atol={atol})")
            logging.info(f"    {e}")
            passed = False

    logging.info(f"  {'PASSED' if passed else 'FAILED'}")
    return passed


########################################################################
# 独立测试用例
#
# 每个 test_XX 函数定义一组独立的测试规格。
# run_test 支持可选参数:
#   batch_size, num_heads, s1_size(Q seqlen), s2_size(KV seqlen), dim, tile_config
# 不传的参数使用全局默认值。
# 在 main() 中选择要执行的用例，注释/取消注释即可切换。
########################################################################


def test_01(device):
    """ 用例规格信息：batch=8, heads=8, s1=320, s2=320, dim=64"""
    return run_test(device, batch_size=8, num_heads=8, s1_size=320, s2_size=320, dim=64)


@pytest.mark.skip(reason="large test case")
def test_02(device):
    """ 用例规格信息：batch=8, heads=8, s1=2432, s2=2432, dim=64"""
    return run_test(device, batch_size=8, num_heads=8, s1_size=2432, s2_size=2432, dim=64)


@pytest.mark.skip(reason="large test case")
def test_03(device):
    """ 用例规格信息：batch=8, heads=16, s1=32, s2=32, dim=32"""
    return run_test(device, batch_size=8, num_heads=16, s1_size=32, s2_size=32, dim=32)


@pytest.mark.skip(reason="large test case")
def test_04(device):
    """ 用例规格信息：batch=8, heads=16, s1=64, s2=64, dim=32"""
    return run_test(device, batch_size=8, num_heads=16, s1_size=64, s2_size=64, dim=32)


@pytest.mark.skip(reason="large test case")
def test_05(device):
    """ 用例规格信息：batch=8, heads=8, s1=32, s2=32, dim=64"""
    return run_test(device, batch_size=8, num_heads=8, s1_size=32, s2_size=32, dim=64)


@pytest.mark.skip(reason="large test case")
def test_06(device):
    """ 用例规格信息：batch=8, heads=4, s1=64, s2=64, dim=128"""
    return run_test(device, batch_size=8, num_heads=4, s1_size=64, s2_size=64, dim=128)


def main():
    """
    主入口: 在下方列表中选择要运行的测试用例。
    注释/取消注释即可控制执行哪些用例。
    """
    logging.info("Flash Attention Backward (3-loop, KV tiling)")

    device_id = get_device_id()
    if device_id is None:
        return

    import torch_npu
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'

    # ---- 选择要运行的测试用例 (注释/取消注释即可) ----
    test_funcs = [
        test_01,    # batch=8, heads=8, s1=320, s2=320, dim=64
        # 用例 test_02,    # batch=8, heads=8, s1=2432, s2=2432, dim=64
        test_03,    # batch=8, heads=16, s1=32, s2=32, dim=32
        test_04,    # batch=8, heads=16, s1=64, s2=64, dim=32
        test_05,    # batch=8, heads=8, s1=32, s2=32, dim=64
        test_06,    # batch=8, heads=4, s1=64, s2=64, dim=128
    ]

    results = []
    for _, fn in enumerate(test_funcs):
        try:
            passed = fn(device)
            results.append((fn.__name__, fn.__doc__, passed))
        except Exception as e:
            logging.info(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((fn.__name__, fn.__doc__, False))

    # ---- 汇总结果 ----
    logging.info("=" * 60)
    logging.info("Summary:")
    logging.info("-" * 60)
    all_passed = True
    for name, desc, passed in results:
        status = "PASSED" if passed else "FAILED"
        logging.info(f"  {name}: {desc}  => {status}")
        if not passed:
            all_passed = False
    logging.info(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")


if __name__ == "__main__":
    main()