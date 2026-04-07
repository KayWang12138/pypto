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
Flash Attention Score Implementation with Online Softmax

This module implements Flash Attention using block-wise computation with
online softmax algorithm for numerical stability.
"""

import math
import os

import pypto


def _jit_debug_options_from_env():
    raw = os.environ.get("PYPTO_RUNTIME_DEBUG_MODE")
    if raw in (None, ""):
        return None
    return {"runtime_debug_mode": int(raw)}


def _jit_runtime_options():
    return {
        "stitch_function_max_num": 256,
        "device_sched_mode": 1,
    }


def _jit_pass_options():
    return {
        "pg_upper_bound": 5000000,
        "cube_l1_reuse_setting": {0: 8},
        "cube_nbuffer_setting": {0: 4},
        "vec_nbuffer_setting": {0: 4},
    }


MODEL_PRESET = "aigcode_8b_jamba_gdn_moe"

NUM_HEADS = 32
HEAD_DIM = 128
BLOCK_SIZE_KV = 128
BLOCK_SIZE_Q = 8192
ASSUME_CAUSAL = True
KV_UNROLL_LIST = (1,)
QK_CUBE_TILE_SHAPES = ((128, 128), (64, 256), (256, 256))
PV_CUBE_TILE_SHAPES = ((128, 128), (128, 512), (128, 128))
VEC_TILE_SHAPES = (128, 256)


@pypto.frontend.jit(
    pass_options=_jit_pass_options(),
    runtime_options=_jit_runtime_options(),
    debug_options=_jit_debug_options_from_env(),
)
def flash_attention_score_kernel_with_mask(
    query: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    key: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    value: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    atten_mask: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
):
    """
    Flash Attention Score kernel with online softmax on a single q-block.

    `query` and `output` represent one q-block `[B, N, Sq_block, D]`.
    `atten_mask` holds the matching q rows against the full kv sequence `[Sq_block, Skv]`.
    """
    batch_size = query.shape[0]
    seq_len_q = query.shape[2]
    seq_len_kv = key.shape[2]

    scale = 1.0 / math.sqrt(HEAD_DIM)

    pypto.set_vec_tile_shapes(*VEC_TILE_SHAPES)

    num_blocks_kv = (seq_len_kv + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV

    for b_idx in pypto.loop(0, batch_size, 1, name="LOOP_B", idx_name="b_idx"):
        for n_idx in pypto.loop(
            0,
            NUM_HEADS,
            1,
            name="LOOP_N",
            idx_name="n_idx",
            submit_before_loop=True,
        ):
            cur_q_size = pypto.min(BLOCK_SIZE_Q, seq_len_q)

            oi_update = pypto.tensor([BLOCK_SIZE_Q, HEAD_DIM], pypto.DT_FP32, "oi_update")
            li_update = pypto.tensor([BLOCK_SIZE_Q, 1], pypto.DT_FP32, "li_update")
            mi_update = pypto.tensor([BLOCK_SIZE_Q, 1], pypto.DT_FP32, "mi_update")

            q_block_4d = pypto.view(
                query,
                [1, 1, BLOCK_SIZE_Q, HEAD_DIM],
                [b_idx, n_idx, 0, 0],
                valid_shape=[1, 1, cur_q_size, HEAD_DIM],
            )
            q_block = pypto.reshape(
                q_block_4d,
                [BLOCK_SIZE_Q, HEAD_DIM],
                valid_shape=[cur_q_size, HEAD_DIM],
            )

            for kv_block_idx, _ in pypto.loop_unroll(
                0,
                num_blocks_kv,
                1,
                name="LOOP_KV_BLOCK",
                idx_name="kv_block_idx",
                unroll_list=list(KV_UNROLL_LIST),
                submit_before_loop=True,
            ):
                kv_start = kv_block_idx * BLOCK_SIZE_KV
                cur_block_size = pypto.min(BLOCK_SIZE_KV, seq_len_kv - kv_start)
                is_first_valid_block = pypto.is_loop_begin(kv_block_idx)
                is_last_valid_block = pypto.is_loop_end(kv_block_idx)

                k_block_4d = pypto.view(
                    key,
                    [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                    [b_idx, n_idx, kv_start, 0],
                    valid_shape=[1, 1, cur_block_size, HEAD_DIM],
                )
                k_block = pypto.reshape(
                    k_block_4d,
                    [BLOCK_SIZE_KV, HEAD_DIM],
                    valid_shape=[cur_block_size, HEAD_DIM],
                )

                pypto.set_cube_tile_shapes(
                    list(QK_CUBE_TILE_SHAPES[0]),
                    list(QK_CUBE_TILE_SHAPES[1]),
                    list(QK_CUBE_TILE_SHAPES[2]),
                )
                scores = pypto.matmul(
                    q_block,
                    k_block,
                    pypto.DT_FP32,
                    a_trans=False,
                    b_trans=True,
                )
                scores_scaled = pypto.mul(scores, scale)
                mask_block = pypto.view(
                    atten_mask,
                    [BLOCK_SIZE_Q, BLOCK_SIZE_KV],
                    [0, kv_start],
                    valid_shape=[cur_q_size, cur_block_size],
                )
                pypto.set_vec_tile_shapes(*VEC_TILE_SHAPES)
                valid_mask = pypto.mul(pypto.add(mask_block, -1.0), -1.0)
                masked_bias = pypto.mul(mask_block, -10000.0)
                scores_for_softmax = pypto.add(scores_scaled, masked_bias)

                v_block_4d = pypto.view(
                    value,
                    [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                    [b_idx, n_idx, kv_start, 0],
                    valid_shape=[1, 1, cur_block_size, HEAD_DIM],
                )
                v_block = pypto.reshape(
                    v_block_4d,
                    [BLOCK_SIZE_KV, HEAD_DIM],
                    valid_shape=[cur_block_size, HEAD_DIM],
                )
                v_block_fp32 = pypto.cast(v_block, pypto.DT_FP32)
                pypto.set_cube_tile_shapes(
                    list(PV_CUBE_TILE_SHAPES[0]),
                    list(PV_CUBE_TILE_SHAPES[1]),
                    list(PV_CUBE_TILE_SHAPES[2]),
                )
                if is_first_valid_block:
                    block_max = pypto.amax(scores_for_softmax, dim=-1, keepdim=True)
                    p_ij = pypto.exp(pypto.sub(scores_for_softmax, block_max))
                    p_ij = pypto.mul(p_ij, valid_mask)
                    block_sum = pypto.sum(p_ij, dim=-1, keepdim=True)
                    block_out = pypto.matmul(p_ij, v_block_fp32, pypto.DT_FP32)

                    if is_last_valid_block:
                        o_final = pypto.div(block_out, block_sum)
                        o_final_bf16 = pypto.cast(o_final, pypto.DT_BF16)
                        o_final_4d = pypto.reshape(
                            o_final_bf16,
                            [1, 1, BLOCK_SIZE_Q, HEAD_DIM],
                            valid_shape=[1, 1, cur_q_size, HEAD_DIM],
                        )
                        pypto.assemble(o_final_4d, [b_idx, n_idx, 0, 0], output)
                    else:
                        oi_update[:] = block_out
                    li_update[:] = block_sum
                    mi_update[:] = block_max
                else:
                    block_max = pypto.amax(scores_for_softmax, dim=-1, keepdim=True)
                    max_new = pypto.maximum(mi_update, block_max)
                    p_ij = pypto.exp(pypto.sub(scores_for_softmax, max_new))
                    p_ij = pypto.mul(p_ij, valid_mask)
                    block_sum = pypto.sum(p_ij, dim=-1, keepdim=True)

                    update_mul = pypto.exp(pypto.sub(mi_update, max_new))
                    li_new = pypto.add(pypto.mul(li_update, update_mul), block_sum)

                    block_out = pypto.matmul(p_ij, v_block_fp32, pypto.DT_FP32)

                    oi_new = pypto.add(pypto.mul(oi_update, update_mul), block_out)

                    if is_last_valid_block:
                        o_final = pypto.div(oi_new, li_new)
                        o_final_bf16 = pypto.cast(o_final, pypto.DT_BF16)
                        o_final_4d = pypto.reshape(
                            o_final_bf16,
                            [1, 1, BLOCK_SIZE_Q, HEAD_DIM],
                            valid_shape=[1, 1, cur_q_size, HEAD_DIM],
                        )
                        pypto.assemble(o_final_4d, [b_idx, n_idx, 0, 0], output)
                    else:
                        oi_update[:] = oi_new
                    li_update[:] = li_new
                    mi_update[:] = max_new


@pypto.frontend.jit(
    pass_options=_jit_pass_options(),
    runtime_options=_jit_runtime_options(),
    debug_options=_jit_debug_options_from_env(),
)
def flash_attention_score_backward_kernel_with_mask(
    query: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    key: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    value: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    atten_mask: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    grad_output: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    grad_query: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
):
    """
    Flash Attention backward kernel on [B, N, S, D] inputs.

    Gradient formulas:
        dP = dO @ V^T
        dS = P * (dP - sum(dO * O, dim=-1))
        dQ = dS @ K * scale

    The kernel recomputes per-q-block online softmax statistics in a first pass,
    then accumulates normalized block probabilities in a second pass for dQ.
    """
    batch_size = query.shape[0]
    seq_len_q = query.shape[2]
    seq_len_kv = key.shape[2]

    scale = 1.0 / math.sqrt(HEAD_DIM)

    pypto.set_vec_tile_shapes(*VEC_TILE_SHAPES)

    num_blocks_kv = (seq_len_kv + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    num_blocks_q = (seq_len_q + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    use_causal_block_skip = ASSUME_CAUSAL and BLOCK_SIZE_Q == BLOCK_SIZE_KV

    for b_idx in pypto.loop(0, batch_size, 1, name="BWD_LOOP_B", idx_name="b_idx"):
        for n_idx in pypto.loop(0, NUM_HEADS, 1, name="BWD_LOOP_N", idx_name="n_idx"):
            for q_block_idx in pypto.loop(0, num_blocks_q, 1, name="BWD_LOOP_Q_BLOCK", idx_name="q_block_idx"):
                q_start = q_block_idx * BLOCK_SIZE_Q
                cur_q_size = pypto.min(BLOCK_SIZE_Q, seq_len_q - q_start)

                q_block_4d = pypto.view(
                    query,
                    [1, 1, BLOCK_SIZE_Q, HEAD_DIM],
                    [b_idx, n_idx, q_start, 0],
                    valid_shape=[1, 1, cur_q_size, HEAD_DIM],
                )
                q_block = pypto.reshape(
                    q_block_4d,
                    [BLOCK_SIZE_Q, HEAD_DIM],
                    valid_shape=[cur_q_size, HEAD_DIM],
                )
                q_block_fp32 = pypto.cast(q_block, pypto.DT_FP32)
                output_block_4d = pypto.view(
                    output,
                    [1, 1, BLOCK_SIZE_Q, HEAD_DIM],
                    [b_idx, n_idx, q_start, 0],
                    valid_shape=[1, 1, cur_q_size, HEAD_DIM],
                )
                output_block = pypto.reshape(
                    output_block_4d,
                    [BLOCK_SIZE_Q, HEAD_DIM],
                    valid_shape=[cur_q_size, HEAD_DIM],
                )
                grad_output_block_4d = pypto.view(
                    grad_output,
                    [1, 1, BLOCK_SIZE_Q, HEAD_DIM],
                    [b_idx, n_idx, q_start, 0],
                    valid_shape=[1, 1, cur_q_size, HEAD_DIM],
                )
                grad_output_block = pypto.reshape(
                    grad_output_block_4d,
                    [BLOCK_SIZE_Q, HEAD_DIM],
                    valid_shape=[cur_q_size, HEAD_DIM],
                )

                output_block_fp32 = pypto.cast(output_block, pypto.DT_FP32)
                grad_output_block_fp32 = pypto.cast(grad_output_block, pypto.DT_FP32)
                row_dot = pypto.sum(
                    pypto.mul(grad_output_block_fp32, output_block_fp32),
                    dim=-1,
                    keepdim=True,
                )

                m_update = pypto.tensor([BLOCK_SIZE_Q, 1], pypto.DT_FP32, "m_update")
                l_update = pypto.tensor([BLOCK_SIZE_Q, 1], pypto.DT_FP32, "l_update")

                for kv_block_idx in pypto.loop(
                    0,
                    num_blocks_kv,
                    1,
                    name="BWD_LOOP_KV_STATS",
                    idx_name="kv_block_idx",
                ):
                    kv_start = kv_block_idx * BLOCK_SIZE_KV
                    cur_block_size = pypto.min(BLOCK_SIZE_KV, seq_len_kv - kv_start)
                    if use_causal_block_skip:
                        should_run_block = kv_block_idx <= q_block_idx
                    else:
                        should_run_block = True

                    if should_run_block:
                        if use_causal_block_skip:
                            is_first_valid_block = kv_block_idx == 0
                        else:
                            is_first_valid_block = pypto.is_loop_begin(kv_block_idx)

                        k_block_4d = pypto.view(
                            key,
                            [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                            [b_idx, n_idx, kv_start, 0],
                            valid_shape=[1, 1, cur_block_size, HEAD_DIM],
                        )
                        k_block = pypto.reshape(
                            k_block_4d,
                            [BLOCK_SIZE_KV, HEAD_DIM],
                            valid_shape=[cur_block_size, HEAD_DIM],
                        )

                        pypto.set_cube_tile_shapes(
                            list(QK_CUBE_TILE_SHAPES[0]),
                            list(QK_CUBE_TILE_SHAPES[1]),
                            list(QK_CUBE_TILE_SHAPES[2]),
                        )
                        scores = pypto.matmul(
                            q_block,
                            k_block,
                            pypto.DT_FP32,
                            a_trans=False,
                            b_trans=True,
                        )
                        scores_scaled = pypto.mul(scores, scale)
                        apply_mask = True
                        if use_causal_block_skip:
                            apply_mask = kv_block_idx == q_block_idx
                        if apply_mask:
                            mask_block = pypto.view(
                                atten_mask,
                                [BLOCK_SIZE_Q, BLOCK_SIZE_KV],
                                [q_start, kv_start],
                                valid_shape=[cur_q_size, cur_block_size],
                            )
                            valid_mask = pypto.mul(pypto.add(mask_block, -1.0), -1.0)
                            masked_bias = pypto.mul(mask_block, -10000.0)
                            scores_for_softmax = pypto.add(scores_scaled, masked_bias)
                        else:
                            scores_for_softmax = scores_scaled

                        m_ij = pypto.amax(scores_for_softmax, dim=-1, keepdim=True)
                        p_tilde = pypto.exp(pypto.sub(scores_for_softmax, m_ij))
                        if apply_mask:
                            p_tilde = pypto.mul(p_tilde, valid_mask)
                        l_ij = pypto.sum(p_tilde, dim=-1, keepdim=True)

                        if is_first_valid_block:
                            m_update[:] = m_ij
                            l_update[:] = l_ij
                        else:
                            m_new = pypto.maximum(m_update, m_ij)
                            l_new = pypto.add(
                                pypto.mul(l_update, pypto.exp(pypto.sub(m_update, m_new))),
                                pypto.mul(l_ij, pypto.exp(pypto.sub(m_ij, m_new))),
                            )
                            m_update[:] = m_new
                            l_update[:] = l_new

                dq_update = pypto.tensor([BLOCK_SIZE_Q, HEAD_DIM], pypto.DT_FP32, "dq_update")
                for kv_block_idx in pypto.loop(
                    0,
                    num_blocks_kv,
                    1,
                    name="BWD_LOOP_KV_GRAD",
                    idx_name="kv_block_idx",
                ):
                    kv_start = kv_block_idx * BLOCK_SIZE_KV
                    cur_block_size = pypto.min(BLOCK_SIZE_KV, seq_len_kv - kv_start)
                    if use_causal_block_skip:
                        should_run_block = kv_block_idx <= q_block_idx
                    else:
                        should_run_block = True

                    if should_run_block:
                        if use_causal_block_skip:
                            is_first_valid_block = kv_block_idx == 0
                            is_last_valid_block = kv_block_idx == q_block_idx
                        else:
                            is_first_valid_block = pypto.is_loop_begin(kv_block_idx)
                            is_last_valid_block = pypto.is_loop_end(kv_block_idx)

                        k_block_4d = pypto.view(
                            key,
                            [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                            [b_idx, n_idx, kv_start, 0],
                            valid_shape=[1, 1, cur_block_size, HEAD_DIM],
                        )
                        k_block = pypto.reshape(
                            k_block_4d,
                            [BLOCK_SIZE_KV, HEAD_DIM],
                            valid_shape=[cur_block_size, HEAD_DIM],
                        )
                        v_block_4d = pypto.view(
                            value,
                            [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                            [b_idx, n_idx, kv_start, 0],
                            valid_shape=[1, 1, cur_block_size, HEAD_DIM],
                        )
                        v_block = pypto.reshape(
                            v_block_4d,
                            [BLOCK_SIZE_KV, HEAD_DIM],
                            valid_shape=[cur_block_size, HEAD_DIM],
                        )
                        k_block_fp32 = pypto.cast(k_block, pypto.DT_FP32)

                        pypto.set_cube_tile_shapes(
                            list(QK_CUBE_TILE_SHAPES[0]),
                            list(QK_CUBE_TILE_SHAPES[1]),
                            list(QK_CUBE_TILE_SHAPES[2]),
                        )
                        scores = pypto.matmul(
                            q_block,
                            k_block,
                            pypto.DT_FP32,
                            a_trans=False,
                            b_trans=True,
                        )
                        scores_scaled = pypto.mul(scores, scale)
                        apply_mask = True
                        if use_causal_block_skip:
                            apply_mask = kv_block_idx == q_block_idx
                        if apply_mask:
                            mask_block = pypto.view(
                                atten_mask,
                                [BLOCK_SIZE_Q, BLOCK_SIZE_KV],
                                [q_start, kv_start],
                                valid_shape=[cur_q_size, cur_block_size],
                            )
                            valid_mask = pypto.mul(pypto.add(mask_block, -1.0), -1.0)
                            masked_bias = pypto.mul(mask_block, -10000.0)
                            scores_for_softmax = pypto.add(scores_scaled, masked_bias)
                        else:
                            scores_for_softmax = scores_scaled

                        m_ij = pypto.amax(scores_for_softmax, dim=-1, keepdim=True)
                        p_tilde = pypto.exp(pypto.sub(scores_for_softmax, m_ij))
                        if apply_mask:
                            p_tilde = pypto.mul(p_tilde, valid_mask)

                        pij_scale = pypto.div(
                            pypto.exp(pypto.sub(m_ij, m_update)),
                            l_update,
                        )
                        p_ij = pypto.mul(p_tilde, pij_scale)

                        v_block_fp32 = pypto.cast(v_block, pypto.DT_FP32)
                        pypto.set_cube_tile_shapes(
                            list(QK_CUBE_TILE_SHAPES[0]),
                            list(QK_CUBE_TILE_SHAPES[1]),
                            list(QK_CUBE_TILE_SHAPES[2]),
                        )
                        dp_ij = pypto.matmul(
                            grad_output_block_fp32,
                            v_block_fp32,
                            pypto.DT_FP32,
                            a_trans=False,
                            b_trans=True,
                        )
                        ds_ij = pypto.mul(
                            p_ij,
                            pypto.sub(dp_ij, row_dot),
                        )

                        pypto.set_cube_tile_shapes(
                            list(PV_CUBE_TILE_SHAPES[0]),
                            list(PV_CUBE_TILE_SHAPES[1]),
                            list(PV_CUBE_TILE_SHAPES[2]),
                        )
                        dq_ij = pypto.mul(
                            pypto.matmul(ds_ij, k_block_fp32, pypto.DT_FP32),
                            scale,
                        )
                        if is_first_valid_block:
                            dq_update[:] = dq_ij
                        else:
                            dq_update[:] = pypto.add(dq_update, dq_ij)

                        if is_last_valid_block:
                            grad_query_block_bf16 = pypto.cast(dq_update, pypto.DT_BF16)
                            grad_query_block_4d = pypto.reshape(
                                grad_query_block_bf16,
                                [1, 1, BLOCK_SIZE_Q, HEAD_DIM],
                                valid_shape=[1, 1, cur_q_size, HEAD_DIM],
                            )
                            pypto.assemble(grad_query_block_4d, [b_idx, n_idx, q_start, 0], grad_query)


@pypto.frontend.jit(
    pass_options=_jit_pass_options(),
    runtime_options=_jit_runtime_options(),
    debug_options=_jit_debug_options_from_env(),
)
def flash_attention_score_backward_kv_kernel_with_mask(
    query: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    key: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    value: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    atten_mask: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    grad_output: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    grad_key: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    grad_value: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
):
    """
    Flash Attention backward kernel for dK/dV.

    The kv gradients are accumulated with kv-block as the writeback unit to avoid
    reading and writing the same output slice in one graph.
    """
    batch_size = query.shape[0]
    seq_len_q = query.shape[2]
    seq_len_kv = key.shape[2]

    scale = 1.0 / math.sqrt(HEAD_DIM)

    pypto.set_vec_tile_shapes(*VEC_TILE_SHAPES)

    # The Python wrapper already constrains backward to single-block shapes.
    if True:
        cur_q_size = pypto.min(BLOCK_SIZE_Q, seq_len_q)
        cur_block_size = pypto.min(BLOCK_SIZE_KV, seq_len_kv)
        for b_idx in pypto.loop(0, batch_size, 1, name="BWD_KV_LOOP_B", idx_name="b_idx"):
            for n_idx in pypto.loop(0, NUM_HEADS, 1, name="BWD_KV_LOOP_N", idx_name="n_idx"):
                q_block_4d = pypto.view(
                    query,
                    [1, 1, BLOCK_SIZE_Q, HEAD_DIM],
                    [b_idx, n_idx, 0, 0],
                    valid_shape=[1, 1, cur_q_size, HEAD_DIM],
                )
                q_block = pypto.reshape(
                    q_block_4d,
                    [BLOCK_SIZE_Q, HEAD_DIM],
                    valid_shape=[cur_q_size, HEAD_DIM],
                )
                q_block_fp32 = pypto.cast(q_block, pypto.DT_FP32)
                k_block_4d = pypto.view(
                    key,
                    [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                    [b_idx, n_idx, 0, 0],
                    valid_shape=[1, 1, cur_block_size, HEAD_DIM],
                )
                k_block = pypto.reshape(
                    k_block_4d,
                    [BLOCK_SIZE_KV, HEAD_DIM],
                    valid_shape=[cur_block_size, HEAD_DIM],
                )
                v_block_4d = pypto.view(
                    value,
                    [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                    [b_idx, n_idx, 0, 0],
                    valid_shape=[1, 1, cur_block_size, HEAD_DIM],
                )
                v_block = pypto.reshape(
                    v_block_4d,
                    [BLOCK_SIZE_KV, HEAD_DIM],
                    valid_shape=[cur_block_size, HEAD_DIM],
                )
                output_block_4d = pypto.view(
                    output,
                    [1, 1, BLOCK_SIZE_Q, HEAD_DIM],
                    [b_idx, n_idx, 0, 0],
                    valid_shape=[1, 1, cur_q_size, HEAD_DIM],
                )
                output_block = pypto.reshape(
                    output_block_4d,
                    [BLOCK_SIZE_Q, HEAD_DIM],
                    valid_shape=[cur_q_size, HEAD_DIM],
                )
                grad_output_block_4d = pypto.view(
                    grad_output,
                    [1, 1, BLOCK_SIZE_Q, HEAD_DIM],
                    [b_idx, n_idx, 0, 0],
                    valid_shape=[1, 1, cur_q_size, HEAD_DIM],
                )
                grad_output_block = pypto.reshape(
                    grad_output_block_4d,
                    [BLOCK_SIZE_Q, HEAD_DIM],
                    valid_shape=[cur_q_size, HEAD_DIM],
                )

                output_block_fp32 = pypto.cast(output_block, pypto.DT_FP32)
                grad_output_block_fp32 = pypto.cast(grad_output_block, pypto.DT_FP32)
                k_block_fp32 = pypto.cast(k_block, pypto.DT_FP32)
                v_block_fp32 = pypto.cast(v_block, pypto.DT_FP32)

                row_dot = pypto.sum(
                    pypto.mul(grad_output_block_fp32, output_block_fp32),
                    dim=-1,
                    keepdim=True,
                )

                pypto.set_cube_tile_shapes(
                    list(QK_CUBE_TILE_SHAPES[0]),
                    list(QK_CUBE_TILE_SHAPES[1]),
                    list(QK_CUBE_TILE_SHAPES[2]),
                )
                scores = pypto.matmul(
                    q_block,
                    k_block,
                    pypto.DT_FP32,
                    a_trans=False,
                    b_trans=True,
                )
                scores_scaled = pypto.mul(scores, scale)
                if ASSUME_CAUSAL and BLOCK_SIZE_Q == BLOCK_SIZE_KV:
                    mask_block = pypto.view(
                        atten_mask,
                        [BLOCK_SIZE_Q, BLOCK_SIZE_KV],
                        [0, 0],
                        valid_shape=[cur_q_size, cur_block_size],
                    )
                    valid_mask = pypto.mul(pypto.add(mask_block, -1.0), -1.0)
                    masked_bias = pypto.mul(mask_block, -10000.0)
                    scores_for_softmax = pypto.add(scores_scaled, masked_bias)
                else:
                    scores_for_softmax = scores_scaled
                    valid_mask = None

                m_ij = pypto.amax(scores_for_softmax, dim=-1, keepdim=True)
                p_tilde = pypto.exp(pypto.sub(scores_for_softmax, m_ij))
                if valid_mask is not None:
                    p_tilde = pypto.mul(p_tilde, valid_mask)
                l_ij = pypto.sum(p_tilde, dim=-1, keepdim=True)
                p_ij = pypto.div(p_tilde, l_ij)

                dp_ij = pypto.matmul(
                    grad_output_block_fp32,
                    v_block_fp32,
                    pypto.DT_FP32,
                    a_trans=False,
                    b_trans=True,
                )
                ds_ij = pypto.mul(
                    p_ij,
                    pypto.sub(dp_ij, row_dot),
                )

                pypto.set_cube_tile_shapes(
                    list(PV_CUBE_TILE_SHAPES[0]),
                    list(PV_CUBE_TILE_SHAPES[1]),
                    list(PV_CUBE_TILE_SHAPES[2]),
                )
                dk_ij = pypto.mul(
                    pypto.matmul(ds_ij, q_block_fp32, pypto.DT_FP32, a_trans=True, b_trans=False),
                    scale,
                )
                dv_ij = pypto.matmul(
                    p_ij,
                    grad_output_block_fp32,
                    pypto.DT_FP32,
                    a_trans=True,
                    b_trans=False,
                )

                grad_key_block_bf16 = pypto.cast(dk_ij, pypto.DT_BF16)
                grad_value_block_bf16 = pypto.cast(dv_ij, pypto.DT_BF16)
                grad_key_block_4d = pypto.reshape(
                    grad_key_block_bf16,
                    [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                    valid_shape=[1, 1, cur_block_size, HEAD_DIM],
                )
                grad_value_block_4d = pypto.reshape(
                    grad_value_block_bf16,
                    [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                    valid_shape=[1, 1, cur_block_size, HEAD_DIM],
                )
                pypto.assemble(grad_key_block_4d, [b_idx, n_idx, 0, 0], grad_key)
                pypto.assemble(grad_value_block_4d, [b_idx, n_idx, 0, 0], grad_value)
