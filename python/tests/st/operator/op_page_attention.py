#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
import pypto
import math


def op_page_attention(q_nope, k_nope_cache, v_nope_cache, q_rope, k_rope_cache,
    block_table, act_seqs, attention_out, params): 
    model_hyper_params = params[1]
    block_size = model_hyper_params["block_size"]
    tile_config = model_hyper_params["tile_config"]
    max_unroll_times = model_hyper_params["max_unroll_times"]
    is_nz_format = model_hyper_params["is_nz_format"]
    dtype = q_nope.dtype
    d_n = q_nope.shape[1]
    d_r = q_rope.shape[1]
    softmax_scale = (d_n+d_r) ** -0.5
    n_tile = tile_config.head_num_q_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    def inside_main_function():
        batch_size = block_table.shape[0]
        n_q = q_nope.shape[0] // batch_size
        n_loop = n_q // n_tile
        for b_idx in pypto.loop(0, batch_size, 1, name="LOOP_L0_bIdx", idx_name="b_idx"):
            def inside_b_idx_loop(b_idx):
                cur_seq = act_seqs[b_idx]
                bn_per_batch = (cur_seq + block_size - 1) // block_size
                bn_per_batch.as_variable()
                for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                    def inside_n_idx_loop(b_idx, n_idx, bn_per_batch):
                        nonlocal n_tile
                        cur_n_tile = n_tile
                        oi_update = pypto.tensor([n_tile, d_n], pypto.DT_FP32, "oi_update")
                        li_update = pypto.tensor([n_tile, 1], pypto.DT_FP32, "li_update")
                        mi_update = pypto.tensor([n_tile, 1], pypto.DT_FP32, "mi_update")
                        cur_offset = b_idx * n_q + n_idx * n_tile
                        oi_offset = [cur_offset, 0]  
                        
                        for bn in pypto.loop(0, bn_per_batch, 1, name="LOOP_L2_bn", 
                                           idx_name="bn", unroll_List={max_unroll_times}):
                            def inside_bn_loop(**kwargs):
                                b_idx = kwargs.get("b_idx")
                                block_table = kwargs.get("block_table")
                                cur_seq = kwargs.get("cur_seq")
                                bn = kwargs.get("bn")
                                block_size = kwargs.get("block_size")
                                nonlocal oi_update, li_update, mi_update
                                cur_s2_tile = block_size
                                qn = pypto.view(q_nope, [cur_n_tile, d_n], [cur_offset, 0])
                                qr = pypto.view(q_rope, [cur_n_tile, d_r], [cur_offset, 0])
                                qi = pypto.tensor([cur_n_tile, d_n + d_r], dtype, "qi")
                                pypto.assemble(qn, [0, 0], qi)
                                pypto.assemble(qr, [0, d_n], qi)
                                cur_block_idx = block_table[b_idx, bn]
                                cur_block_idx.as_variable()
                                kn = pypto.view(k_nope_cache, [cur_s2_tile, d_n],
                                                [cur_block_idx * block_size, 0],
                                                valid_shape=[(cur_seq - bn * block_size).min(block_size), d_n])
                                kr = pypto.view(k_rope_cache, [cur_s2_tile, d_r],
                                                [cur_block_idx * block_size, 0],
                                                valid_shape=[(cur_seq - bn * block_size).min(block_size), d_r])
                                kj_format = pypto.TileOpFormat.TILEOP_NZ if is_nz_format else (
                                    pypto.TileOpFormat.TILEOP_ND
                                )
                                kj = pypto.tensor([cur_s2_tile, d_n + d_r], dtype, "kj", kj_format)
                                pypto.assemble(kn, [0, 0], kj)
                                pypto.assemble(kr, [0, d_n], kj)
                                kj = pypto.view(kj, [cur_s2_tile, d_n + d_r], [0, 0],
                                            valid_shape=[(cur_seq - bn * block_size).min(block_size), d_r + d_n])
                                vj = pypto.view(v_nope_cache, [cur_s2_tile, d_n], [cur_block_idx * block_size, 0],
                                                valid_shape=[(cur_seq - bn * block_size).min(block_size), d_n])

                                pypto.set_semantic_label("MatMul")
                                pypto.set_cube_tile_shapes(
                                    [c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]],
                                    [c1_tile[4], c1_tile[5]])
                                pypto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
                                sij = pypto.matmul(qi, kj, pypto.DT_FP32, b_trans=True)
                                pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                pypto.set_semantic_label("SoftMax")
                                sij_scale = pypto.mul(sij, float(softmax_scale))
                                pypto.set_semantic_label("SoftMax")
                                tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                                tsub = pypto.sub(sij_scale, tilda_mij)
                                tilda_pij = pypto.exp(tsub)
                                tilda_pij_f16 = pypto.cast(tilda_pij, dtype)
                                tilda_lij = pypto.sum(tilda_pij, dim=-1, keepdim=True)

                                if pypto.cond(pypto.is_loop_begin(bn)):
                                    def inside_if_loop_begin():
                                        nonlocal oi_update, li_update, mi_update
                                        pypto.set_cube_tile_shapes(
                                            [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]],
                                            [c2_tile[4], c2_tile[5]])
                                        pypto.set_semantic_label("b1-matmul2")
                                        pypto.set_matrix_size(
                                            [tilda_pij_f16.shape[0], tilda_pij_f16.shape[1],
                                                vj.shape[1]])
                                        oi_tmp = pypto.matmul(tilda_pij_f16,
                                                            vj, pypto.DT_FP32)
                                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                        pypto.set_semantic_label("b1-after-matmul2")
                                        if pypto.cond(pypto.is_loop_end(bn)):
                                            pypto.set_semantic_label("b1-after-matmul2")
                                            oi_update[:] = (pypto.div(oi_tmp, tilda_lij))
                                            pypto.assemble(oi_update, oi_offset, attention_out)
                                        else:
                                            oi_update[:] = (oi_tmp)
                                        li_update[:] = (tilda_lij)
                                        mi_update[:] = (tilda_mij)
                                    inside_if_loop_begin()
                                else:
                                    def inside_else_loop_begin():
                                        nonlocal oi_update, li_update, mi_update
                                        oi = oi_update
                                        li = li_update
                                        mi = mi_update
                                        pypto.set_semantic_label("Softmax-acc")
                                        mi_new = pypto.maximum(mi, tilda_mij)
                                        t1 = pypto.sub(mi, mi_new)
                                        t2 = pypto.exp(t1)
                                        t3 = pypto.sub(tilda_mij, mi_new)
                                        t4 = pypto.exp(t3)
                                        t5 = pypto.mul(t4, tilda_lij)
                                        t6 = pypto.mul(t2, li)
                                        li_new = pypto.add(t6, t5)
                                        q3 = pypto.mul(oi, t2)
                                        pypto.set_semantic_label("bn-matmul2")
                                        pypto.set_cube_tile_shapes(
                                        [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4],
                                        c2_tile[5]])
                                        pypto.set_matrix_size(
                                            [tilda_pij_f16.shape[0],
                                                tilda_pij_f16.shape[1], vj.shape[1]])
                                        q1 = pypto.matmul(tilda_pij_f16, vj, pypto.DT_FP32)
                                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                        pypto.set_semantic_label("bn-after-matmul2")
                                        q2 = pypto.mul(q1, t4)
                                        oi_tmp = pypto.add(q3, q2)
                                        if pypto.cond(pypto.is_loop_end(bn)):
                                            oi_update[:] = (pypto.div(oi_tmp, li_new))
                                            pypto.assemble(oi_update, oi_offset, attention_out)
                                        else:
                                            oi_update[:] = (oi_tmp)
                                        li_update[:] = (li_new)
                                        mi_update[:] = (mi_new)
                                    inside_else_loop_begin()
                            inside_bn_loop(
                                b_idx=b_idx,
                                block_table=block_table,
                                cur_seq=cur_seq,
                                bn=bn,
                                block_size=block_size,
                                bn_per_batch=bn_per_batch)
                    inside_n_idx_loop(b_idx, n_idx, bn_per_batch)
            inside_b_idx_loop(b_idx)
    inside_main_function()


def op_page_attention_golden(q_nope, k_nope_cache, v_cache, q_rope, k_rope_cache,
    block_table, act_seqs, golden_hyper_params):
    b = golden_hyper_params["b"]
    n_q = golden_hyper_params["n_q"]
    s_q = golden_hyper_params["s_q"]
    n_kv = golden_hyper_params["n_kv"]
    kv_lora_rank = golden_hyper_params["kv_lora_rank"]
    qk_rope_dim = golden_hyper_params["qk_rope_dim"]
    d_q = kv_lora_rank + qk_rope_dim
    d_k = kv_lora_rank + qk_rope_dim
    d_v = kv_lora_rank
    n_tile = golden_hyper_params["n_tile"]
    block_size = golden_hyper_params["block_size"]
    block_num = golden_hyper_params["block_num"]
    
    q_nope = q_nope.reshape(b, n_q, s_q, kv_lora_rank)
    q_rope = q_rope.reshape(b, n_q, s_q, qk_rope_dim)
    k_nope_cache = k_nope_cache.reshape(block_num, block_size, n_kv * kv_lora_rank)
    k_rope_cache = k_rope_cache.reshape(block_num, block_size, n_kv * qk_rope_dim)
    v_cache = v_cache.reshape(block_num, block_size, n_kv * d_v)
    q_bnsd = torch.cat([q_nope, q_rope], dim=-1)
    k_cache = torch.cat([k_nope_cache, k_rope_cache], dim=-1)
    scalar = d_q ** -0.5
    tiled_out = []
    block_num_per_batch = []
    for actual_seq in act_seqs:
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
    n_loop = math.ceil(n_q / n_tile)
    for b_index in range(b):
        matmul_dtype = torch.float32
        cur_seq = act_seqs[b_index]
        bn_per_batch = math.ceil(cur_seq / block_size)
        for n_idx in range(n_loop):
            oi_update = []
            li_update = []
            mi_update = []
            qi = q_bnsd[b_index, n_idx * n_tile: (n_idx + 1) * n_tile, :, :]
            qi = qi.reshape(-1, qi.shape[-1])
            for bn in range(block_num_per_batch[b_index]):
                cur_block_idx = block_table[b_index][bn]
                s2_tile_cur = min(block_size, cur_seq - bn * block_size)
                kj = k_cache[cur_block_idx, 0:s2_tile_cur, :]
                vj = v_cache[cur_block_idx, 0:s2_tile_cur, :]
                kj = kj.reshape(s2_tile_cur, d_k)
                vj = vj.reshape(s2_tile_cur, d_v)
                
                sij = torch.matmul(
                    qi.to(matmul_dtype),
                    kj.to(matmul_dtype).mT
                )
                sij_scale = sij * scalar
                tilda_mij = sij_scale.max(dim=-1, keepdim=True).values
                t_sub = sij_scale - tilda_mij
                tilda_pij = torch.exp(t_sub)
                tilda_lij = tilda_pij.sum(dim=-1, keepdim=True)

                if bn == 0:
                    oi_tmp = torch.matmul(
                        tilda_pij.to(matmul_dtype),
                        vj.to(matmul_dtype)
                    )
                    if bn_per_batch == 1:
                        oi_update = oi_tmp / tilda_lij
                    else:
                        oi_update = oi_tmp
                    li_update = tilda_lij
                    mi_update = tilda_mij
                    continue
                oi = oi_update
                li = li_update
                mi = mi_update
                
                mi_new = torch.maximum(mi, tilda_mij)
                t1 = mi - mi_new
                t2 = torch.exp(t1)
                t3 = tilda_mij - mi_new
                t4 = torch.exp(t3)
                t5 = t4 * tilda_lij
                t6 = t2 * li
                li_new = t6 + t5
                q3 = oi * t2
                q1 = torch.matmul(tilda_pij.to(matmul_dtype), vj.to(matmul_dtype))
                q2 = q1 * t4
                oi_tmp = q3 + q2
                if bn == block_num_per_batch[b_index] - 1:
                    oi_update = oi_tmp / li_new
                else:
                    oi_update = oi_tmp
                li_update = li_new
                mi_update = mi_new
            tiled_out.append(oi_update)
    attent_out = torch.cat(tiled_out, dim=0)
    return (attent_out, )