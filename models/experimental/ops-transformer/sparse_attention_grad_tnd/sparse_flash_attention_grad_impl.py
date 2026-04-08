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
Sparse Flash Attention Grad - PyPTO Implementation (TND format, nope/rope split, dynamic shapes)

实现SFA反向算子：根据前向中间结果和输出梯度，计算dQ, dK, dV。

**TND 格式**
  - 输入: q_nope(T1, N1, D), q_pe(T1, N1, DR), k_nope(T2, N2, D), k_pe(T2, N2, DR), value(T2, N2, D)
  - T1 动态 (total_q tokens), T2 动态 (total_kv tokens), B 静态
  - actual_seq_qlen(B,) 和 actual_seq_kvlen(B,) 为前缀和格式
  - 主循环: for b in B -> for s in s_count，s_count = qlen[b] - qlen[b-1]*(b>0)

**静态shape规则**
  - 所有输入tensor shape必须是静态的，只有valid_shape可以用动态符号
  - KV-side tensor 使用 MAX_TOTAL_KV 作为静态上界
  - Q-side 的 nope/rope 在循环内 view 出静态 [G, D]/[G, DR]，然后 concat
  - K-side 的 nope/rope 分别 index_select 出静态 [K, D]/[K, DR]，然后 concat

**动态shape设计**
  - T1, T2 通过 pypto.frontend.dynamic 标识为动态轴
  - KV tensor 传入时 shape 为 (T2, N2, D)，内部 reshape 为 2D 后
    view 到 [MAX_TOTAL_KV*N2, D] with valid_shape=[T2*N2, D]
  - dk_nope/dv 用 MAX_TOTAL_KV 静态shape + valid_shape
  - dk_pe 暂保持动态 shape (如有报错再改为 MAX_TOTAL_KV)

**valid_shape on indices**
  - cur_kv_len = min(K, actual_kv_len_for_batch)
  - sparse_idx view带valid_shape=[1, cur_kv_len]

**关键设计：批量处理G个head**
  - 对每个(b, s)位置，同时处理所有G个query heads (M=G)
  - G个heads共享相同的sparse indices (因为N2=1)

计算公式（对每个(b, s)位置）：
  1. Gather: sel_k_nope, sel_k_pe, sel_v 分别 index_select  -> 各 (K, D)/(K, DR)/(K, D)
  2. sel_k = concat(sel_k_nope, sel_k_pe)                     -> (K, D+DR)
  3. Q_group = concat(q_nope_view, q_pe_view)                  -> (G, D+DR)
  4. S = Q_group @ sel_k^T * scale                             -> (G, K)
  5. P = exp(S - mi) / li                                      -> (G, K)
  6. dP = dO_group @ sel_v^T                                   -> (G, K)
  7. dV_local = P^T @ dO_group                                 -> (K, D)
  8. D_val = rowsum(dO * out)                                   -> (G, 1)
  9. dS = P * (dP - D_val)                                     -> (G, K)
  10. dQ_local = dS @ sel_k * scale                             -> (G, D+DR)
  11. dK_local = dS^T @ Q_group * scale                         -> (K, D+DR)
  12. view dQ_local/dK_local 的 nope/rope 部分，分别写回输出
"""
import os
import math
from dataclasses import dataclass
import numpy as np
import pypto
import torch
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import FakeTensor

MAX_TOTAL_KV = 128 * 1024


def sparse_flash_attention_grad_compute(
    q_nope, q_pe, k_nope, k_pe, value,
    sparse_idx,
    d_out, out, sm_max, sm_sum,
    actual_seq_qlen, actual_seq_kvlen,
    dq_nope_2d, dq_pe_2d, dk_nope_2d, dk_pe_2d, dv_2d, dk_nope_out, dk_pe_out, dk_2d, dk_out,
    N1, N2, D, DR, K, G, scale_value,
):
    """
    PyPTO实现SFA反向计算核心逻辑 - TND格式, nope/rope split, 动态shape。

    输入 (TND 3D):
        q_nope:          (T1, N1, D) BF16        T1 动态
        q_pe:            (T1, N1, DR) BF16
        k_nope:          (T2, N2, D) BF16        T2 动态
        k_pe:            (T2, N2, DR) BF16
        value:           (T2, N2, D) BF16
        sparse_idx:      (T1, N2, K) INT32       indices into T2
        d_out:           (T1, N1, D) BF16
        out:             (T1, N1, D) BF16
        sm_max:          (N2, T1, G) FP32
        sm_sum:          (N2, T1, G) FP32
        actual_seq_qlen: (B,) INT32              前缀和
        actual_seq_kvlen:(B,) INT32              前缀和

    Returns:
        dq_nope, dq_pe, dk_nope, dk_pe, dv (all TND 3D)
    """
    dtype = q_nope.dtype  # BF16
    D_full = D + DR

    T1_sym = q_nope.shape[0]       # dynamic
    T2_sym = k_nope.shape[0]       # dynamic (actual total_kv)
    B_sym = actual_seq_qlen.shape[0]  # dynamic

    # ======== Reshape 3D -> 2D in init loop ========
    for _ in pypto.loop(0, 1, 1, name="LOOP_reshape"):
        pypto.set_vec_tile_shapes(128, 576)
        # Q-side: dynamic T1
        q_nope_2d = pypto.reshape(q_nope, [T1_sym * N1, D], inplace=True)      # (T1*N1, D)
        q_pe_2d = pypto.reshape(q_pe, [T1_sym * N1, DR], inplace=True)         # (T1*N1, DR)
        d_out_2d = pypto.reshape(d_out, [T1_sym * N1, D], inplace=True)        # (T1*N1, D)
        out_2d = pypto.reshape(out, [T1_sym * N1, D], inplace=True)            # (T1*N1, D)
        sparse_idx_2d = pypto.reshape(sparse_idx, [T1_sym * N2, K], inplace=True)  # (T1*N2, K)
        sparse_idx_1d = pypto.reshape(sparse_idx, [T1_sym * N2 * K], inplace=True)  # (T1*N2*K)

        # KV-side: T2 dynamic, reshape to 2D then view to static shape with valid_shape
        k_nope_2d = pypto.reshape(k_nope, [T2_sym * N2, D], inplace=True)  # dynamic (T2*N2, D)
        k_pe_2d = pypto.reshape(k_pe, [T2_sym * N2, DR], inplace=True)     # dynamic (T2*N2, DR)
        value_2d = pypto.reshape(value, [T2_sym * N2, D], inplace=True)    # dynamic (T2*N2, D)

        # sm_max/sm_sum: (N2, T1, G) -> reshape to (N2*T1, G)
        sm_max_2d = pypto.reshape(sm_max, [N2 * T1_sym, G], inplace=True)      # (N2*T1, G)
        sm_sum_2d = pypto.reshape(sm_sum, [N2 * T1_sym, G], inplace=True)      # (N2*T1, G)

    # ======== Main compute: for B -> for S ========
    for b_idx in pypto.loop(0, B_sym, 1, name="LOOP_B", idx_name="bIdx"):
        # 计算当前batch的q/kv offset和s_count
        # actual_seq_qlen 是前缀和: q_offset = qlen[b-1] if b>0 else 0, s_count = qlen[b] - q_offset
        cur_qlen = actual_seq_qlen[b_idx]         # qlen prefix sum at b
        q_offset = (b_idx > 0) * actual_seq_qlen[b_idx - 1]         # 0 when b=0, prev_qlen when b>0
        s_count = cur_qlen - q_offset

        cur_kvlen = actual_seq_kvlen[b_idx]
        kv_offset = (b_idx > 0) * actual_seq_kvlen[b_idx - 1]
        actual_kv_len = cur_kvlen - kv_offset       # actual kv length for this batch

        for s_idx in pypto.loop(0, s_count, 1, name="LOOP_S", idx_name="sIdx"):
            t_idx = q_offset + s_idx  # global token index in T1
            # casual
            cur_kv_valid = (actual_kv_len - s_count + 1 + s_idx).max(0).min(K)

            for kv_head_idx in range(1):
                # ======== Step 1: Get sparse indices with valid_shape ========
                topk_row = t_idx
                pypto.set_vec_tile_shapes(1, K)
                cur_indices_2d = pypto.view(sparse_idx_2d, [1, K], [topk_row, 0],
                                            valid_shape=[1, cur_kv_valid])
                cur_indices_1d = pypto.view(sparse_idx_1d, [K], [topk_row * K],
                                               valid_shape=[cur_kv_valid])

                # ======== Step 2: Gather sel_k_nope, sel_k_pe, sel_v (static shapes) ========
                # index_select produces static shapes: (K, D), (K, DR), (K, D)
                pypto.set_vec_tile_shapes(128, D)
                k_nope_2d_view = pypto.view(k_nope_2d, [MAX_TOTAL_KV, D], [0, 0], valid_shape=[T1_sym * N1, D])
                sel_k_nope = pypto.index_select(k_nope_2d_view, 0, cur_indices_1d)  # (K, D) BF16
                pypto.set_vec_tile_shapes(128, DR)
                k_pe_2d_view = pypto.view(k_pe_2d, [MAX_TOTAL_KV, DR], [0, 0], valid_shape=[T1_sym * N1, DR])
                sel_k_pe = pypto.index_select(k_pe_2d_view, 0, cur_indices_1d)      # (K, DR) BF16

                # Concat sel_k_nope + sel_k_pe -> sel_k (static [K, D+DR])
                pypto.set_vec_tile_shapes(16, D)
                sel_k = pypto.tensor([K, D+DR], dtype=pypto.DT_BF16)
                pypto.assemble(sel_k_nope, [0, 0], sel_k)
                pypto.assemble(sel_k_pe, [0, D], sel_k)

                # ======== Step 3: Q_group (static shape concat), dO_group, Out_group ========
                q_head_start = kv_head_idx * G
                q_offset_2d = t_idx * N1 + q_head_start
                # View static-shape slices from q_nope_2d/q_pe_2d, then concat
                qn_slice = pypto.view(q_nope_2d, [G, D], [q_offset_2d, 0])    # static (G, D)
                qr_slice = pypto.view(q_pe_2d, [G, DR], [q_offset_2d, 0])     # static (G, DR)
                pypto.set_vec_tile_shapes(16, D)
                q_group = pypto.tensor([G, D+DR], dtype=pypto.DT_BF16)
                pypto.assemble(qn_slice, [0, 0], q_group)
                pypto.assemble(qr_slice, [0, D], q_group)

                # sm_max/sm_sum: reshaped to (N2*T1, G), row = kv_head_idx*T1 + t_idx
                sm_row = kv_head_idx * T1_sym + t_idx

                # ======== Step 4: S = Q_full @ sel_k^T * scale ========
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [256, 256])
                sel_k_view = pypto.view(sel_k, [K, D_full], [0, 0], valid_shape=[cur_kv_valid, D_full])
                s_scores = pypto.matmul(q_group, sel_k_view, pypto.DT_FP32,
                                        a_trans=False, b_trans=True)  # (G, K)

                # ======== Step 5: 恢复softmax P ========
                pypto.set_vec_tile_shapes(1, 16)
                cur_mi_row = pypto.view(sm_max_2d, [1, G], [sm_row, 0])
                cur_li_row = pypto.view(sm_sum_2d, [1, G], [sm_row, 0])
                mi_col = pypto.reshape(cur_mi_row, [G, 1])
                li_col = pypto.reshape(cur_li_row, [G, 1])

                pypto.set_vec_tile_shapes(16, K)
                s_scaled = pypto.mul(s_scores, scale_value)
                s_shifted = pypto.sub(s_scaled, mi_col)
                exp_s = pypto.exp(s_shifted)
                p_mat = pypto.div(exp_s, li_col)
                p_bf16 = pypto.cast(p_mat, dtype)

                # ======== Step 6: dP = dO_group @ sel_v^T ========
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [256, 256])
                do_group = pypto.view(d_out_2d, [G, D], [q_offset_2d, 0])     # (G, D)
                sel_v_view = pypto.view(sel_k_nope, [K, D], [0, 0], valid_shape=[cur_kv_valid, D])
                dp_mat = pypto.matmul(do_group, sel_v_view, pypto.DT_FP32,
                                      a_trans=False, b_trans=True)  # (G, K)

                # ======== Step 7: dV_local = P^T @ dO_group ========
                pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128])
                dv_local = pypto.matmul(p_bf16, do_group, pypto.DT_FP32,
                                        a_trans=True, b_trans=False)  # (K, D)
                dv_local_valid = pypto.view(dv_local, [K, D], [0, 0], valid_shape=[cur_kv_valid, D])

                # ======== Step 8: D_val, dS ========
                pypto.set_vec_tile_shapes(16, D)
                do_f32 = pypto.cast(do_group, pypto.DT_FP32)
                out_group = pypto.view(out_2d, [G, D], [q_offset_2d, 0])      # (G, D)
                out_f32 = pypto.cast(out_group, pypto.DT_FP32)
                do_out_prod = pypto.mul(do_f32, out_f32)
                d_val = pypto.sum(do_out_prod, dim=-1, keepdim=True)  # (G, 1)

                pypto.set_vec_tile_shapes(16, K)
                dp_minus_d = pypto.sub(dp_mat, d_val)
                ds_mat = pypto.mul(p_mat, dp_minus_d)
                ds_bf16 = pypto.cast(ds_mat, dtype)

                # ======== Step 9: dQ_local = dS @ sel_k * scale -> (G, D+DR) ========
                pypto.set_cube_tile_shapes([128, 128], [256, 256], [128, 128])
                sel_k_view_2 = pypto.view(sel_k, [K, D_full], [0, 0], valid_shape=[cur_kv_valid, D_full])
                dq_local = pypto.matmul(ds_bf16, sel_k_view_2, pypto.DT_FP32,
                                        a_trans=False, b_trans=False)  # (G, D+DR)
                pypto.set_vec_tile_shapes(16, D_full)
                dq_scaled = pypto.mul(dq_local, scale_value)
                dq_cast = pypto.cast(dq_scaled, dtype)

                # assemble dQ nope/rope
                dq_nope_part = pypto.view(dq_cast, [G, D], [0, 0])
                dq_pe_part = pypto.view(dq_cast, [G, DR], [0, D])
                pypto.set_vec_tile_shapes(16, D)
                pypto.assemble(dq_nope_part, [q_offset_2d, 0], dq_nope_2d)
                pypto.set_vec_tile_shapes(16, DR)
                pypto.assemble(dq_pe_part, [q_offset_2d, 0], dq_pe_2d)

                # ======== Step 10: dK_local = dS^T @ Q_group * scale -> (K, D+DR) ========
                pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128])
                dk_local = pypto.matmul(ds_bf16, q_group, pypto.DT_FP32,
                                        a_trans=True, b_trans=False)  # (K, D+DR)
                pypto.set_vec_tile_shapes(16, D)
                dk_scaled = pypto.mul(dk_local, scale_value)

                # ======== Step 11: scatter-add dK (nope+rope), dV ========
                # dk_scaled: (K, D+DR) FP32, dv_local: (K, D) FP32
                dk_nope_part = pypto.view(dk_scaled, [K, D], [0, 0], valid_shape=[cur_kv_valid, D])     # (K, D) FP32
                dk_v = pypto.add(dk_nope_part, dv_local_valid)
                dk_pe_part = pypto.view(dk_scaled, [K, DR], [0, D], valid_shape=[cur_kv_valid, DR])       # (K, DR) FP32
                dk_all = pypto.concat([dk_v, dk_pe_part], dim=-1)

                pypto.set_vec_tile_shapes(16, D_full)
                dk_2d_view = pypto.view(dk_2d, [MAX_TOTAL_KV, D + DR], [0, 0], valid_shape=[T2_sym, D+DR])
                dk_out[:] = pypto.index_add_(dk_2d_view, -2, cur_indices_1d, dk_all)

    return dq_nope_2d, dq_pe_2d, dk_nope_2d, dk_pe_2d, dv_2d, dk_out


@pypto.frontend.jit(
    pass_options={
        "pg_upper_bound": 5000000,
        "vec_nbuffer_setting": {-1: 4, -2:1},
        "cube_l1_reuse_setting": {-1: 8},
    },
    debug_options=dict(runtime_debug_mode=1),
    runtime_options={
        "device_sched_mode": 3,
        "stitch_function_max_num": 256
    },
)
def sparse_flash_attention_grad(
    q_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # T1, N1, D
    q_pe: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # T1, N1, DR
    k_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # T2, N2, D
    k_pe: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # T2, N2, DR
    value: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # T2, N2, D
    sparse_idx: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT32), # T1, N2, K
    d_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # T1, N1, D
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16), # T1, N1, D
    sm_max: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # N2, T1, G
    sm_sum: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # N2, T1, G
    actual_seq_qlen: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32), # B
    actual_seq_kvlen: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32), # B
    dq_nope_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16), # T1 * N1, D
    dq_pe_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16), # T1 * N1, DR

    dk_nope_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # T2 * N2, D
    dk_pe_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # T2 * N2, DR
    dv_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # T2 * N2, D

    dk_nope_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # T2 * N2, D
    dk_pe_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16), # T1 * N1, DR

    dk_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # T2 * N2, D
    dk_2d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32), # T2 * N2, D
    N1, N2, D, DR, K, G, scale_value
):
    """JIT-compiled SFA backward kernel, TND format, nope/rope split, dynamic shapes."""
    pypto.experimental.set_operation_options(combine_axis=True)

    sparse_flash_attention_grad_compute(
        q_nope, q_pe, k_nope, k_pe, value,
        sparse_idx,
        d_out, out, sm_max, sm_sum,
        actual_seq_qlen, actual_seq_kvlen,
        dq_nope_out, dq_pe_out, dk_nope_out, dk_pe_out, dv_out, dk_nope_in, dk_pe_in, dk_out, dk_2d,
        N1, N2, D, DR, K, G, scale_value
    )


def check_input_output_shape_dtype(q_nope, q_pe, k_nope, k_pe, value, sparse_idx, d_out, out, sm_max, sm_sum,
        actual_seq_qlen, actual_seq_kvlen):
    assert actual_seq_kvlen is not None and actual_seq_kvlen.dim() == 1, f"actual_seq_kvlen dim num is {actual_seq_kvlen.dim()}, expected 1"
    assert actual_seq_qlen is not None and actual_seq_qlen.dim() == 1, f"actual_seq_qlen dim num is {actual_seq_qlen.dim()}, expected 1"
    assert sparse_idx is not None and sparse_idx.dim() == 3, f"topk_indices_npu dim num is {sparse_idx.dim()}, expected 3"

    assert q_nope.dim() == 3 and q_nope.size(2) == 512 and q_nope.dtype == torch.bfloat16, \
        f"q_nope dim num is {q_nope.dim()}, q_nope axis 2 is {q_nope.size(1)}, q_nope dtype is {q_nope.dtype}, expected 3, 512, torch.bfloat16"
    assert q_pe.dim() == 3 and q_pe.size(2) == 64 and q_pe.dtype == torch.bfloat16, \
        f"q_pe dim num is {q_pe.dim()}, q_pe axis 2 is {q_pe.size(1)}, q_pe dtype is {q_pe.dtype}, expected 3, 64, torch.bfloat16"

    assert k_nope.dim() == 3 and k_nope.size(2) == 512 and k_nope.dtype == torch.bfloat16, \
        f"k_nope dim num is {k_nope.dim()}, k_nope axis 2 is {k_nope.size(2)}, k_nope dtype is {k_nope.dtype}, expected 3, 512, torch.bfloat16"
    assert k_pe.dim() == 3 and k_pe.size(2) == 64 and k_pe.dtype == torch.bfloat16, \
        f"k_pe dim num is {k_pe.dim()}, k_pe axis 2 is {k_pe.size(2)}, k_pe dtype is {k_pe.dtype}, expected 3, 64, torch.bfloat16"

    assert value.dim() == 3 and value.size(2) == 512 and value.dtype == torch.bfloat16, \
        f"value dim num is {value.dim()}, value axis 2 is {value.size(2)}, value dtype is {value.dtype}, expected 3, 512, torch.bfloat16"

    assert d_out.dim() == 3 and d_out.size(2) == 512 and d_out.dtype == torch.bfloat16, \
        f"d_out dim num is {d_out.dim()}, d_out axis 2 is {d_out.size(2)}, d_out dtype is {d_out.dtype}, expected 3, 512, torch.bfloat16"
    assert out.dim() == 3 and out.size(2) == 512 and out.dtype == torch.bfloat16, \
        f"out dim num is {out.dim()}, d_out axis 2 is {out.size(2)}, out dtype is {out.dtype}, expected 3, 512, torch.bfloat16"

    assert sm_max.dim() == 3 and sm_max.size(0) == 1 and sm_max.dtype == torch.float32, \
        f"sm_max dim num is {sm_max.dim()}, sm_max axis 0 is {sm_max.size(0)}, sm_max dtype is {sm_max.dtype}, expected 3, 1, torch.bfloat16"
    assert sm_sum.dim() == 3 and sm_sum.size(0) == 1 and sm_sum.dtype == torch.float32, \
        f"sm_sum dim num is {sm_sum.dim()}, sm_sum axis 0 is {sm_sum.size(0)}, sm_sum dtype is {sm_sum.dtype}, expected 3, 1, torch.bfloat16"


@allow_in_graph
def npu_sfa_sparse_attention_grad(q_nope, q_pe, k_nope, k_pe, value, sparse_idx, d_out, out, sm_max, sm_sum,
        actual_seq_qlen, actual_seq_kvlen, scale_value):
    assert not isinstance(q_nope, FakeTensor), f"q_nope is FakeTensor"

    # check_input_output_shape_dtype
    check_input_output_shape_dtype(q_nope, q_pe, k_nope, k_pe, value, sparse_idx, d_out, out, sm_max, sm_sum,
        actual_seq_qlen, actual_seq_kvlen)

    print("****************** pypto npu_sfa_sparse_attention_grad *************************")

    T1, N1, D = q_nope.size(0), q_nope.size(1), q_nope.size(2)
    DR = q_pe.size(2)
    T2, N2 = k_nope.size(0), k_nope.size(1)
    dtype = q_nope.dtype
    K = sparse_idx.size(2)
    G = N1 // N2

    # 创建输出tensor
    dq_nope_out_shape = (T1 * N1, D)
    dq_pe_out_shape = (T1 * N1, DR)
    dk_nope_out_shape = (T2 * N2, D)
    dk_pe_out_shape = (T2 * N2, DR)
    dv_out_shape = (T2 * N2, D)
    dq_nope_out = torch.empty(dq_nope_out_shape, dtype=dtype, device=q_nope.device)
    dq_pe_out = torch.empty(dq_pe_out_shape, dtype=dtype, device=q_nope.device)
    dk_nope_out = torch.zeros(dk_nope_out_shape, dtype=torch.float32, device=q_nope.device)

    dk_pe_out = torch.zeros(dk_pe_out_shape, dtype=torch.float32, device=q_nope.device)
    dv_out = torch.zeros(dv_out_shape, dtype=torch.float32, device=q_nope.device)

    dk_out = torch.zeros((T2 * N2, D+DR), dtype=torch.float32, device=q_nope.device)

    pto_inputs = [q_nope, q_pe, k_nope, k_pe, value, sparse_idx, d_out, out, sm_max, sm_sum,
        actual_seq_qlen, actual_seq_kvlen, dq_nope_out, dq_pe_out, dk_nope_out, dk_pe_out, dv_out, dk_nope_out, dk_pe_out, dk_out, dk_out,
        N1, N2, D, DR, K, G, scale_value]

    sparse_flash_attention_grad(*pto_inputs)

    dq_nope_out = torch.reshape(dq_nope_out, (T1, N1, D))
    dq_pe_out = torch.reshape(dq_pe_out, (T1, N1, DR))

    dk_nope_out = torch.reshape(dk_out[:, :D], (T2, N2, D)).to(dtype)
    dk_pe_out = torch.reshape(dk_out[:, D:], (T2, N2, DR)).to(dtype)
    dv_out = torch.reshape(dv_out, (T2, N2, D)).to(dtype)

    return dq_nope_out, dq_pe_out, dk_nope_out, dk_pe_out, dv_out
