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
import pypto


BATCH_SIZE = 4
NUM_HEADS = 8
SEQ_LEN_Q = 64
SEQ_LEN_KV = 128
HEAD_DIM = 64
BLOCK_SIZE_KV = 16


@pypto.frontend.jit(
    pass_options={
        "pg_upper_bound": 5000000,
    },
    runtime_options={
        "stitch_function_max_num": 128,
    },
    debug_options={
        "runtime_debug_mode": 1,
    }
)
def flash_attention_score_kernel_with_mask(
    query: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    key: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    value: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    atten_mask: pypto.Tensor((SEQ_LEN_Q, SEQ_LEN_KV), pypto.DT_FP32),
    output: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
):
    """
    Flash Attention Score kernel with online softmax (with mask).
    """
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(16, 128)
    
    num_blocks_kv = (SEQ_LEN_KV + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    
    for b_idx in pypto.loop(0, BATCH_SIZE, 1, name="LOOP_B", idx_name="b_idx"):
        for n_idx in pypto.loop(0, NUM_HEADS, 1, name="LOOP_N", idx_name="n_idx"):
            for q_idx in pypto.loop(0, SEQ_LEN_Q, 1, name="LOOP_Q", idx_name="q_idx"):
                oi_update = pypto.tensor([1, HEAD_DIM], pypto.DT_FP32, "oi_update")
                li_update = pypto.tensor([1, 1], pypto.DT_FP32, "li_update")
                mi_update = pypto.tensor([1, 1], pypto.DT_FP32, "mi_update")
                
                q_vec = pypto.view(query, [1, 1, 1, HEAD_DIM], 
                                   [b_idx, n_idx, q_idx, 0],
                                   valid_shape=[1, 1, 1, HEAD_DIM])
                q_vec_2d = pypto.reshape(q_vec, [1, HEAD_DIM])
                
                for kv_block_idx, _ in pypto.loop_unroll(0, num_blocks_kv, 1,
                                                         name="LOOP_KV_BLOCK", 
                                                         idx_name="kv_block_idx",
                                                         unroll_list={1}):
                    kv_start = kv_block_idx * BLOCK_SIZE_KV
                    cur_block_size = pypto.min(BLOCK_SIZE_KV, SEQ_LEN_KV - kv_start)
                    
                    k_block = pypto.view(key, [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                                        [b_idx, n_idx, kv_start, 0],
                                        valid_shape=[1, 1, cur_block_size, HEAD_DIM])
                    k_block_2d = pypto.reshape(k_block, [BLOCK_SIZE_KV, HEAD_DIM])
                    k_block_2d_valid = pypto.view(k_block_2d, [BLOCK_SIZE_KV, HEAD_DIM],
                                                  [0, 0],
                                                  valid_shape=[cur_block_size, HEAD_DIM])
                    
                    scores = pypto.matmul(q_vec_2d, k_block_2d_valid, pypto.DT_FP32, 
                                         a_trans=False, b_trans=True)
                    scores_scaled = pypto.mul(scores, scale)
                    
                    mask_block = pypto.view(atten_mask, [1, BLOCK_SIZE_KV],
                                           [q_idx, kv_start],
                                           valid_shape=[1, cur_block_size])
                    valid_mask = pypto.add(mask_block, -1.0)
                    valid_mask = pypto.mul(valid_mask, -1.0)
                    
                    m_ij = pypto.amax(scores_scaled, dim=-1, keepdim=True)
                    
                    s_ij_sub_m = pypto.sub(scores_scaled, m_ij)
                    p_ij = pypto.exp(s_ij_sub_m)
                    p_ij = pypto.mul(p_ij, valid_mask)
                    l_ij = pypto.sum(p_ij, dim=-1, keepdim=True)
                    
                    v_block = pypto.view(value, [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                                        [b_idx, n_idx, kv_start, 0],
                                        valid_shape=[1, 1, cur_block_size, HEAD_DIM])
                    v_block_2d = pypto.reshape(v_block, [BLOCK_SIZE_KV, HEAD_DIM])
                    v_block_2d_valid = pypto.view(v_block_2d, [BLOCK_SIZE_KV, HEAD_DIM],
                                                  [0, 0],
                                                  valid_shape=[cur_block_size, HEAD_DIM])
                    v_block_fp32 = pypto.cast(v_block_2d_valid, pypto.DT_FP32)
                    
                    o_ij = pypto.matmul(p_ij, v_block_fp32, pypto.DT_FP32)
                    
                    if pypto.is_loop_begin(kv_block_idx):
                        if pypto.is_loop_end(kv_block_idx):
                            o_final = pypto.div(o_ij, l_ij)
                            o_final_bf16 = pypto.cast(o_final, pypto.DT_BF16)
                            o_final_4d = pypto.reshape(o_final_bf16, [1, 1, 1, HEAD_DIM])
                            output[b_idx: b_idx + 1, n_idx: n_idx + 1, q_idx: q_idx + 1, :] = o_final_4d
                        else:
                            oi_update[:] = o_ij
                        li_update[:] = l_ij
                        mi_update[:] = m_ij
                    else:
                        mi_new = pypto.maximum(mi_update, m_ij)
                        
                        alpha = pypto.exp(pypto.sub(mi_update, mi_new))
                        beta = pypto.exp(pypto.sub(m_ij, mi_new))
                        
                        li_new = pypto.add(
                            pypto.mul(alpha, li_update),
                            pypto.mul(beta, l_ij)
                        )
                        
                        oi_scaled = pypto.mul(oi_update, alpha)
                        o_ij_scaled = pypto.mul(o_ij, beta)
                        oi_new = pypto.add(oi_scaled, o_ij_scaled)
                        
                        if pypto.is_loop_end(kv_block_idx):
                            o_final = pypto.div(oi_new, li_new)
                            o_final_bf16 = pypto.cast(o_final, pypto.DT_BF16)
                            o_final_4d = pypto.reshape(o_final_bf16, [1, 1, 1, HEAD_DIM])
                            output[b_idx: b_idx + 1, n_idx: n_idx + 1, q_idx: q_idx + 1, :] = o_final_4d
                        else:
                            oi_update[:] = oi_new
                        li_update[:] = li_new
                        mi_update[:] = mi_new