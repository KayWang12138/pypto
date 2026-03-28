#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""
Flash Attention Score Grad Implementation

This module implements Flash Attention backward gradient computation.

Based on design.md:
- API mapping: matmul, mul, add, sub, exp, amax, sum, div, cast, view, reshape
- Tiling: Cube + Vector (mixed type)
- Softmax: manually implemented (amax → sub → exp → sum → div)
- Softmax_grad: p * (dp - sum(p * dp))
- Use matmul b_trans=True/a_trans=True to avoid separate transpose
- FP32 intermediate computation for numerical stability

Inputs: query, key, value, dy (BNSD layout, FP16/BF16)
Outputs: dq_out, dk_out, dv_out (FP16/BF16)
"""

import math
import pypto
import torch

# ─────────────────────────────────────────────
# Static shape configuration (P0 test case: [2,8,128,64])
# ─────────────────────────────────────────────
NUM_HEADS = 8
HEAD_DIM = 64
SEQ_LEN = 128


# ─────────────────────────────────────────────
# Causal mask kernel (sparse_mode=3) - Dynamic axis version
# Dynamic axes: batch (axis 0), seq_len (axis 2)
# Static axes: num_heads (axis 1), head_dim (axis 3)
# ─────────────────────────────────────────────

@pypto.frontend.jit(
    pass_options={
        "pg_upper_bound": 5000000,
        "cube_l1_reuse_setting": {0: 8},
        "cube_nbuffer_setting": {},
    },
    runtime_options={
        "stitch_function_max_num": 256,
        "device_sched_mode": 1,
    },
)
def flash_attention_score_grad_kernel_causal(
    query: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP16),
    key: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP16),
    value: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP16),
    dy: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP16),
    atten_mask: pypto.Tensor([SEQ_LEN, SEQ_LEN], pypto.DT_FP32),
    dq_out: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP16),
    dk_out: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP16),
    dv_out: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP16),
):
    """
    Flash Attention Score Grad kernel with causal mask (sparse_mode=3).
    
    Dynamic axes: batch (axis 0), seq_len (axis 2)
    Static axes: num_heads (axis 1), head_dim (axis 3)
    
    Causal mask: upper triangular, positions (i, j) where j > i are masked (-inf).
    atten_mask: 1 for masked positions, 0 for valid positions.
    """
    batch_size_sym = query.shape[0]
    seq_len_sym = query.shape[2]
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    pypto.set_cube_tile_shapes([128, 256], [64, 128], [256, 512])
    pypto.set_vec_tile_shapes(1, 8, 16, 1024)
    
    for b_idx in pypto.loop(0, batch_size_sym, 1, name="LOOP_B", idx_name="b_idx"):
        for n_idx in pypto.loop(0, NUM_HEADS, 1, name="LOOP_N", idx_name="n_idx"):
            q_slice = pypto.view(query, [1, 1, SEQ_LEN, HEAD_DIM],
                                 [b_idx, n_idx, 0, 0],
                                 valid_shape=[1, 1, seq_len_sym, HEAD_DIM])
            q_2d = pypto.reshape(q_slice, [SEQ_LEN, HEAD_DIM],
                                 valid_shape=[seq_len_sym, HEAD_DIM])
            q_fp32 = pypto.cast(q_2d, pypto.DT_FP32)
            
            k_slice = pypto.view(key, [1, 1, SEQ_LEN, HEAD_DIM],
                                 [b_idx, n_idx, 0, 0],
                                 valid_shape=[1, 1, seq_len_sym, HEAD_DIM])
            k_2d = pypto.reshape(k_slice, [SEQ_LEN, HEAD_DIM],
                                 valid_shape=[seq_len_sym, HEAD_DIM])
            k_fp32 = pypto.cast(k_2d, pypto.DT_FP32)
            
            v_slice = pypto.view(value, [1, 1, SEQ_LEN, HEAD_DIM],
                                 [b_idx, n_idx, 0, 0],
                                 valid_shape=[1, 1, seq_len_sym, HEAD_DIM])
            v_2d = pypto.reshape(v_slice, [SEQ_LEN, HEAD_DIM],
                                 valid_shape=[seq_len_sym, HEAD_DIM])
            v_fp32 = pypto.cast(v_2d, pypto.DT_FP32)
            
            dy_slice = pypto.view(dy, [1, 1, SEQ_LEN, HEAD_DIM],
                                 [b_idx, n_idx, 0, 0],
                                 valid_shape=[1, 1, seq_len_sym, HEAD_DIM])
            dy_2d = pypto.reshape(dy_slice, [SEQ_LEN, HEAD_DIM],
                                 valid_shape=[seq_len_sym, HEAD_DIM])
            dy_fp32 = pypto.cast(dy_2d, pypto.DT_FP32)
            
            scores = pypto.matmul(q_fp32, k_fp32, pypto.DT_FP32,
                                 a_trans=False, b_trans=True)
            scores_scaled = pypto.mul(scores, scale)
            
            mask_view = pypto.view(atten_mask, [SEQ_LEN, SEQ_LEN], [0, 0],
                                   valid_shape=[seq_len_sym, seq_len_sym])
            mask_penalty = pypto.mul(mask_view, -10000.0)
            scores_masked = pypto.add(scores_scaled, mask_penalty)
            
            m = pypto.amax(scores_masked, dim=-1, keepdim=True)
            s_sub_m = pypto.sub(scores_masked, m)
            p_exp = pypto.exp(s_sub_m)
            l = pypto.sum(p_exp, dim=-1, keepdim=True)
            p = pypto.div(p_exp, l)
            
            dp = pypto.matmul(dy_fp32, v_fp32, pypto.DT_FP32,
                             a_trans=False, b_trans=True)
            
            p_dp = pypto.mul(p, dp)
            sum_p_dp = pypto.sum(p_dp, dim=-1, keepdim=True)
            dp_sub_sum = pypto.sub(dp, sum_p_dp)
            ds = pypto.mul(p, dp_sub_sum)
            
            dq = pypto.matmul(ds, k_fp32, pypto.DT_FP32)
            dq_scaled = pypto.mul(dq, scale)
            
            dk = pypto.matmul(ds, q_fp32, pypto.DT_FP32, a_trans=True)
            dk_scaled = pypto.mul(dk, scale)
            
            dv = pypto.matmul(p, dy_fp32, pypto.DT_FP32, a_trans=True)
            
            dq_fp16 = pypto.cast(dq_scaled, pypto.DT_FP16)
            dq_4d = pypto.reshape(dq_fp16, [1, 1, SEQ_LEN, HEAD_DIM],
                                 valid_shape=[1, 1, seq_len_sym, HEAD_DIM])
            dq_out[b_idx:b_idx+1, n_idx:n_idx+1, :, :] = dq_4d
            
            dk_fp16 = pypto.cast(dk_scaled, pypto.DT_FP16)
            dk_4d = pypto.reshape(dk_fp16, [1, 1, SEQ_LEN, HEAD_DIM],
                                 valid_shape=[1, 1, seq_len_sym, HEAD_DIM])
            dk_out[b_idx:b_idx+1, n_idx:n_idx+1, :, :] = dk_4d
            
            dv_fp16 = pypto.cast(dv, pypto.DT_FP16)
            dv_4d = pypto.reshape(dv_fp16, [1, 1, SEQ_LEN, HEAD_DIM],
                                 valid_shape=[1, 1, seq_len_sym, HEAD_DIM])
            dv_out[b_idx:b_idx+1, n_idx:n_idx+1, :, :] = dv_4d


# ─────────────────────────────────────────────
# Wrapper functions
# ─────────────────────────────────────────────

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create causal attention mask (upper triangular).
    
    Returns mask where:
    - 1.0 for masked positions (j > i, upper triangular)
    - 0.0 for valid positions (j <= i, lower triangular including diagonal)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask  # dtype=float32, 1=masked, 0=valid


def flash_attention_score_grad_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    sparse_mode: int = 0,
) -> tuple:
    """
    Wrapper function for Flash Attention Score Grad.
    
    Args:
        query: Query tensor [B, N, S, D] (BNSD layout), FP16
        key: Key tensor [B, N, S, D] (BNSD layout), FP16
        value: Value tensor [B, N, S, D] (BNSD layout), FP16
        dy: Gradient input [B, N, S, D] (BNSD layout), FP16
        sparse_mode: Sparse mode (0=default/no mask, 3=causal)
    
    Returns:
        Tuple of (dq, dk, dv) gradients, same shape and dtype as inputs
    """
    B, N, S, D = query.shape
    assert N == NUM_HEADS, f"Num heads mismatch: {N} vs {NUM_HEADS}"
    assert D == HEAD_DIM, f"Head dim mismatch: {D} vs {HEAD_DIM}"
    
    dq_out = torch.empty_like(query)
    dk_out = torch.empty_like(key)
    dv_out = torch.empty_like(value)
    
    if sparse_mode == 3:
        atten_mask = create_causal_mask(S).to(query.device)
        flash_attention_score_grad_kernel_causal(
            query, key, value, dy, atten_mask, dq_out, dk_out, dv_out
        )
    
    return dq_out, dk_out, dv_out