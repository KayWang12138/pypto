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
"""
from dataclasses import dataclass
import math
import os
import pypto
import torch
from pypto import pypto_impl
from pypto.operation import op_wrapper
import numpy as np
from numpy.testing import assert_allclose
import logging


@op_wrapper
def gather_in_l1(src, offsets, size, is_b_matrix, is_trans):
    return pypto_impl.gather_in_l1(src, offsets, size, is_b_matrix, is_trans)


@op_wrapper
def gather_in_ub(
    param,
    indices,
    axis
):
    """gather_in_ub."""

    return pypto_impl.gather_in_ub(param, indices, axis)


@dataclass
class SaTileShapeConfig:
    g_tile: int
    s_kv_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


def sparse_flash_attention_quant_d_compute(query_nope, query_rope, key_nope_2d, key_rope_2d, k_nope_scales,
    offsets, kv_act_seqs, nq, n_kv, softmax_scale, topk, attention_out, tile_config):
    dtype = query_nope.dtype
    kn_dtype = key_nope_2d.dtype
    dn = query_nope.shape[1]
    dr = query_rope.shape[1]
    group = nq // n_kv
    group_tile = tile_config.g_tile
    s2_tile = tile_config.s_kv_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape
    n_kv_sym = n_kv
    batch_size_sym = kv_act_seqs.shape[0]
    s1_n1_gsym = query_nope.shape[0] // batch_size_sym
    s1_sym = s1_n1_gsym // nq
    s1_s2_sym = s1_sym * topk
    g_loop_sym = group // group_tile
    s2_sym = s1_s2_sym // s1_sym
    pypto.set_codegen_options(support_dynamic_unaligned=True)

    for batch_idx in pypto.loop(0, batch_size_sym, 1, name="LOOP_L0_idx", idx_name="bIdx"):
        cur_act_seq = kv_act_seqs[batch_idx]
        for slc_idx in pypto.loop(0, s1_sym, 1, name="LOOP_L1_s1_SA", idx_name="s1Idx"):
            cur_seq = (cur_act_seq - s1_sym + 1 + slc_idx).max(0).min(topk)
            cur_seq.as_variable()
            bn_per_batch = (cur_seq + s2_tile - 1) // s2_tile
            for n_kv_idx in pypto.loop(0, n_kv_sym, 1, name="LOOP_L2_n_kv_SA", idx_name="n_kvIdx"):
                for group_idx in pypto.loop(0, g_loop_sym, 1, name="LOOP_L3_g_SA", idx_name="gIdx"):
                    cur_group_tile = group_tile
                    oi_update = pypto.tensor([cur_group_tile, dn], pypto.DT_FP32, "oi_update")
                    li_update = pypto.tensor([1, cur_group_tile], pypto.DT_FP32, "li_update")
                    mi_update = pypto.tensor([1, cur_group_tile], pypto.DT_FP32, "mi_update")
                    cur_offset = batch_idx * s1_n1_gsym \
                        + slc_idx * nq + n_kv_idx * group + group_idx * cur_group_tile
                    oi_offset = [batch_idx, slc_idx, n_kv_idx * group + group_idx * cur_group_tile, 0]
                    for s2_idx, _ in pypto.loop_unroll(0, bn_per_batch, 1,
                        name="LOOP_L4_s2_SA", idx_name="s2_idx", unroll_list={1}):
                        cur_s2_tile = s2_tile
                        cur_kv_offset = batch_idx * s1_s2_sym \
                            + slc_idx * s2_sym + s2_idx * cur_s2_tile
                        pypto.set_semantic_label("Sa_QkMM")
                        pypto.set_vec_tile_shapes(32, 512)
                        qn = pypto.view(query_nope, [cur_group_tile, dn],
                                    [cur_offset, 0], valid_shape=[cur_group_tile, dn])
                        qr = pypto.view(query_rope, [cur_group_tile, dr],
                                    [cur_offset, 0], valid_shape=[cur_group_tile, dr])
                        qi = pypto.tensor([cur_group_tile, dn + dr], dtype, "qi")
                        pypto.assemble(qn, [0, 0], qi)
                        pypto.assemble(qr, [0, dn], qi)
                        offset_view = pypto.view(offsets, [1, cur_s2_tile],
                            [batch_idx * s1_sym + slc_idx, s2_idx * cur_s2_tile],
                            valid_shape=[1, (cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile)])
                        k_nope_2d_view = pypto.view(key_nope_2d, [cur_s2_tile, dn],
                            [0, 0], valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                        k_nope_scale_view = pypto.view(k_nope_scales, [cur_s2_tile, 4],
                            [0, 0], valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), 4])
                        kn = pypto.tensor([s2_tile, dn], dtype, "kn")
                        vj = pypto.tensor([s2_tile, dn], dtype, "vj")

                        if kn_dtype == pypto.DT_INT8:
                            pypto.set_vec_tile_shapes(32, 512)
                            #gather L1接口补充
                            kn_scale = gather_in_ub(k_nope_scale_view, offset_view, -2)
                            kn_quant = gather_in_ub(k_nope_2d_view, offset_view, -2)
                            kn_quant_fp16 = pypto.cast(kn_quant, pypto.DT_FP16)
                            kn_quant_fp32 = pypto.cast(kn_quant_fp16, pypto.DT_FP32)
                            kn_quant_fp32_tmp = pypto.reshape(kn_quant_fp32, [s2_tile * 4, 128])
                            kn_scale_tmp = pypto.reshape(kn_scale, [s2_tile * 4, 1])
                            pypto.set_vec_tile_shapes(128, 128)
                            kn_fp32 = pypto.mul(kn_quant_fp32_tmp, kn_scale_tmp)
                            kn_fp32_reshape = pypto.reshape(kn_fp32, [s2_tile, dn])
                            pypto.set_vec_tile_shapes(32, 512)
                            cur_kn_fp32 = pypto.view(kn_fp32_reshape, [cur_s2_tile, dn], [0, 0], 
                                valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                            kn = pypto.cast(cur_kn_fp32, dtype)
                            vj = pypto.cast(cur_kn_fp32, dtype)
                        else:
                            pypto.set_cube_tile_shapes([c1_tile[0], c1_tile[1]],
                                [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])
                            kn = gather_in_l1(key_nope_2d,
                                offset_view, dn, is_b_matrix=True, is_trans=True)
                        
                        pypto.set_cube_tile_shapes([c1_tile[0],
                            c1_tile[1]], [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])
                        kr = gather_in_l1(key_rope_2d,
                            offset_view, dr, is_b_matrix=True, is_trans=True)
                        q_k_n = pypto.matmul(qn, kn, pypto.DT_FP32, a_trans=False, b_trans=True)
                        q_k_r = pypto.matmul(qr, kr, pypto.DT_FP32, a_trans=False, b_trans=True)
                        
                        pypto.set_semantic_label("Sa_Qkvec1")
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij = pypto.add(q_k_n, q_k_r)
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij_reduce = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        tilda_mij = pypto.reshape(tilda_mij_reduce, [1, cur_group_tile])
                        t_sub = pypto.sub(sij_scale, tilda_mij_reduce)
                        tilda_pij = pypto.exp(t_sub)
                        tilda_pij_f16 = pypto.cast(tilda_pij, dtype)
                        tilda_lij_reduce = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                        tilda_lij = pypto.reshape(tilda_lij_reduce, [1, cur_group_tile])

                        pypto.set_semantic_label("Sa_KvMm")
                        pypto.set_cube_tile_shapes([c2_tile[0],
                            c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]])
                        pypto.set_matrix_size([tilda_pij_f16.shape[0],
                            tilda_pij_f16.shape[1], kn.shape[1]])

                        q1 = pypto.tensor([cur_group_tile, dn], dtype)
                        if kn_dtype == pypto.DT_INT8:
                            q1 = pypto.matmul(tilda_pij_f16, vj, pypto.DT_FP32)
                        else:
                            vj = gather_in_l1(key_nope_2d, offset_view,
                                dn, is_b_matrix=True, is_trans=False)
                            q1 = pypto.matmul(tilda_pij_f16, vj, pypto.DT_FP32)
                        
                        if pypto.cond(pypto.is_loop_begin(s2_idx)):
                            oi_tmp = q1
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            if pypto.cond(pypto.is_loop_end(s2_idx)):
                                pypto.set_semantic_label("Sa_KvVec2")
                                oi_update[:] = oi_tmp / tilda_lij_reduce
                                pypto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                oi_update_4_dim = pypto.cast(pypto.reshape(oi_update,
                                    [1, 1, cur_group_tile, dn]), dtype)
                                pypto.assemble(oi_update_4_dim, oi_offset, attention_out)
                            else:
                                oi_update[:] = oi_tmp
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            li_update[:] = pypto.clone(tilda_lij)
                            mi_update[:] = pypto.clone(tilda_mij)
                        else:
                            pypto.set_semantic_label("Sa_UpdateVec2")
                            oi = oi_update
                            li = li_update
                            mi = mi_update
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            mi_new = pypto.maximum(mi, tilda_mij)
                            t1 = pypto.sub(mi, mi_new)
                            t2 = pypto.exp(t1)
                            t3 = pypto.sub(tilda_mij, mi_new)
                            t4 = pypto.exp(t3)
                            t5 = pypto.mul(t4, tilda_lij)
                            t6 = pypto.mul(t2, li)
                            li_new = pypto.add(t6, t5)
                            q3 = pypto.mul(oi, pypto.reshape(t2, [cur_group_tile, 1]))
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            q2 = pypto.mul(q1, pypto.reshape(t4, [cur_group_tile, 1]))
                            oi_tmp = pypto.add(q3, q2)
                            if pypto.cond(pypto.is_loop_end(s2_idx)):
                                oi_update[:] = pypto.div(oi_tmp,
                                    pypto.reshape(li_new, [cur_group_tile, 1]))
                                pypto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                oi_update_4_dim = pypto.cast(pypto.reshape(oi_update,
                                    [1, 1, cur_group_tile, dn]), dtype)
                                pypto.assemble(oi_update_4_dim, oi_offset, attention_out)
                            else:
                                oi_update[:] = oi_tmp
                            li_update[:] = li_new
                            mi_update[:] = mi_new


@pypto.jit
def sparse_flash_attention_quant_d(in_tensors, out_tensors, n_q, n_kv, softmax_scale, topk, tile_config):
    query_nope, query_rope, key_nope_2d, key_rope_2d, k_nope_scales, offsets, kv_act_seqs = in_tensors
    attention_out, = out_tensors
    pypto.set_host_options(only_codegen=True)
    sparse_flash_attention_quant_d_compute(query_nope, query_rope, key_nope_2d, key_rope_2d, k_nope_scales, 
            offsets, kv_act_seqs, n_q, n_kv, softmax_scale, topk, attention_out, tile_config)