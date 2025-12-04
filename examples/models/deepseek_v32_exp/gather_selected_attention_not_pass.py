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


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    """
    PyTorch版本的均匀分布数据生成，与NumPy版本行为完全一致
    严格保持 [min_value, max_value) 左闭右开区间特性
    """
    # 特殊情况：全零张量
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    # 布尔类型处理：等概率生成True/False
    if dtype == torch.bool:
        # 生成[0,2)的整数，转换为bool即等概率True/False
        return torch.randint(0, 2, data_shape, dtype=dtype)
    # 浮点类型：[min_value, max_value)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        # torch.rand生成[0,1)，缩放后得到[min_value, max_value)
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    # 整数类型：[min_value, max_value)
    else:
        # torch.randint的high参数为开区间，直接对应[min_value, max_value)
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def softmax(x, input_dtype):
    """PyTorch实现的softmax函数"""
    x = x.float()
    x_max = torch.max(x, dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    y = y.to(input_dtype)
    x_sum = torch.sum(y, dim=-1, keepdim=True)
    ans = y
    return ans, x_sum, x_max


def compute_attention(input_data, params):
    """
    计算注意力机制，支持不同批次的序列长度不同
    使用PyTorch实现
    """
    q, kn, kr, kn_scales, offsets, actual_seq = input_data
    scalar, topk, d_v, is_kn_quant = params
    # 提取维度信息
    b, s1, n1, dq = q.shape
    _, dk = kn.shape
    _, dv = kr.shape

    s2_tile = 2048

    atten_out_shape = [b, s1, n1, d_v]
    input_dtype = q.dtype
    kn_dtype = kn.dtype

    # 初始化输出张量
    attention_output = torch.zeros(atten_out_shape, dtype=input_dtype)
    tmp_out = torch.zeros([b, s1, n1], dtype=input_dtype)

    for b_idx in range(b):
        cur_k_seq = actual_seq[b_idx]
        for s1_idx in range(s1):
            cur_seq = min(max(cur_k_seq - s1 + 1 + s1_idx, 0), topk)
            bn_per_batch = math.ceil(cur_seq / s2_tile)

            qi = q[b_idx, s1_idx, :, :] # (n1, dk)

            for s2_idx in range(bn_per_batch):
                s2_tile_cur = min(s2_tile, cur_seq - s2_idx * s2_tile)
                s2_start = s2_tile * s2_idx
                s2_end = s2_start + s2_tile_cur
                offset = offsets[b_idx * s1 + s1_idx, s2_start:s2_end]
                slc_kn = torch.zeros([s2_tile_cur, dk], dtype=kn_dtype)
                slc_kr = torch.zeros([s2_tile_cur, dv], dtype=input_dtype)
                slc_kn_scales = torch.zeros([s2_tile_cur, 4], dtype=torch.float32)
                for idx in range(s2_tile_cur):
                    s2_idx_tmp = s2_start + idx
                    slc_idx = offset[s2_idx_tmp]
                    slc_kn[s2_idx_tmp, :] = kn[slc_idx, :]
                    slc_kr[s2_idx_tmp, :] = kr[slc_idx, :]
                    slc_kn_scales[s2_idx_tmp, :] = kn_scales[slc_idx, :]
                
                qn_tmp = qi[..., :dk]
                qr_tmp = qi[..., dk:]
                if is_kn_quant:
                    kn_bs = slc_kn.reshape(-1, 128).to(torch.float)
                    kn_scales_tmp = slc_kn_scales.reshape(-1, 1)
                    kn_tmp = kn_bs * kn_scales_tmp
                    kn_tmp = kn_tmp.reshape(-1, 512).to(input_dtype)
                else:
                    kn_tmp = slc_kn
                kr_tmp = slc_kr
                vj = kn_tmp

                # C1
                qkn_bmm = torch.matmul(qn_tmp, kn_tmp.transpose(1, 0)).to(torch.float)
                qkr_bmm = torch.matmul(qr_tmp, kr_tmp.transpose(1, 0)).to(torch.float)

                sij = qkn_bmm + qkr_bmm
                sij_scale = sij * scalar # (n1, s2_tile)
                tilda_mij = sij_scale.amax(dim=-1, keepdims=True) # (n1, 1)
                t_sub = sij_scale - tilda_mij # (n1, s2_tile)
                tilda_pij = torch.exp(t_sub) # (n1, s2_tile)
                tilda_pij_f16 = tilda_pij.to(input_dtype)
                q1 = torch.matmul(tilda_pij_f16, vj)
                tilda_lij = tilda_pij.sum(dim=-1, keepdims=True) # (n1, 1)

                if s2_idx == 0:
                    oi_tmp = q1
                    if bn_per_batch == 1:
                        oi_update = oi_tmp / tilda_lij
                    else:
                        oi_update = oi_tmp
                    li_update = tilda_lij
                    mi_update = tilda_mij
                    tmp_out[b_idx, s1_idx, :] = tilda_lij.reshape(n1)
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
                q2 = q1 * t4
                oi_tmp = q3 + q2
                if s2_idx == bn_per_batch - 1:
                    oi_update = oi_tmp / li_new
                else:
                    oi_update = oi_tmp
                li_update = li_new
                mi_update = mi_new

            attention_output[b_idx, s1_idx, :, :] = oi_update

    return attention_output, tmp_out


def gen_gather_select_attention_golden(dtype, bn1n2s1, is_kn_quant, actual_seq):
    block_size = 128
    torch.manual_seed(42)
    b, n_q, n_kv, s_q = bn1n2s1  # 48, 128, 1, 1
    kv_lora_rank = 512
    qk_rope_dim = 64
    topk = 2048
    np.random.seed(None)
    # q head dim
    d_q = kv_lora_rank + qk_rope_dim
    # k head dim
    d_k = kv_lora_rank + qk_rope_dim
    # v head dim
    d_v = kv_lora_rank
    scalar = d_q ** -0.5
    if isinstance(actual_seq, int):
        actual_seq = [actual_seq] * b
    elif isinstance(actual_seq, list):
        if len(actual_seq) == b:
            actual_seq = actual_seq
        else:
            raise RuntimeError("unsupported actual_seq list length")
    else:
        raise RuntimeError("unsupported actual_seq data type")
    # 1. 定义shape
    shape_q = [b, s_q, n_q, d_q]

    block_num_per_batch = []
    block_num_min = 0
    block_num = 0
    for actual_seq_tmp in actual_seq:
        block_num_per_batch.append(math.ceil(actual_seq_tmp / block_size))
        block_num_min += math.ceil(actual_seq_tmp / block_size)
    block_num = block_num_min

    shape_kn = [block_num, block_size, kv_lora_rank]
    shape_kr = [block_num, block_size, qk_rope_dim]

    slc_actual_seq = []
    for i in range(b):
        slc_actual_seq.append(min(actual_seq[i], topk))
    offsets = torch.zeros(b, s_q, topk).to(torch.int32)
    for b_i in range(b):
        for s_q_i in range(s_q):
            perm = torch.randperm(block_num * block_size)
            offsets[b_i, s_q_i, :slc_actual_seq[b_i]] = perm[:slc_actual_seq[b_i]]
    offsets = offsets.reshape(b * s_q, n_kv * topk)

    q_bsnd = gen_uniform_data(shape_q, -1, 1, dtype)
    kn_bsnd_tmp = gen_uniform_data(shape_kn, -1, 1, dtype)
    
    kn_bsnd_reshape = kn_bsnd_tmp.reshape(block_num * block_size, 4, 128).to(torch.float32)
    kn_scales = kn_bsnd_reshape.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    if is_kn_quant == 1:
        kn_quant = kn_bsnd_tmp.reshape(block_num * block_size, 4, 128) / kn_scales
        kn = torch.round(kn_quant).clamp(-128, 127).to(torch.int8)
    else:
        kn = kn_bsnd_tmp
    kr = gen_uniform_data(shape_kr, -1, 1, dtype)
    # 2D
    kn = kn.reshape(block_num * block_size, kv_lora_rank)
    kn_scales = kn_scales.reshape(block_num * block_size, 4)
    kr = kr.reshape(block_num * block_size, qk_rope_dim)

    # 3. 计算attention
    params = [scalar, topk, kv_lora_rank, is_kn_quant]
    input_data = [q_bsnd, kn, kr, kn_scales, offsets, actual_seq]
    atten_out, tmp_out = compute_attention(input_data, params)

    # 4.dump 数据
    # data split to [nope + rope]
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]
    q_nope = q_nope.reshape(b * s_q * n_q, kv_lora_rank)
    q_rope = q_rope.reshape(b * s_q * n_q, qk_rope_dim)
    # input params
    input_params = [b, s_q, n_q, n_kv, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, is_kn_quant, scalar]
    kn_aux_tensor = torch.eye(512, dtype=torch.float32).to(torch.int8)
    scale_aux_tensor = torch.eye(4, dtype=torch.float32)
    input_data_map = [q_nope, q_rope, kn, kr, kn_scales, offsets, actual_seq]
    
    return input_params, kn_aux_tensor, scale_aux_tensor, input_data_map, atten_out


def select_attention_compute_v2(query_nope, query_rope, key_nope_2d, key_rope_2d, k_nope_scales,
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
        def inside_batch_idx_loop(batch_idx):
            cur_act_seq = kv_act_seqs[batch_idx]
            for slc_idx in pypto.loop(0, s1_sym, 1, name="LOOP_L1_s1_SA", idx_name="s1Idx"):
                def inside_slc_idx_loop(batch_idx, slc_idx):
                    cur_seq = (cur_act_seq - s1_sym + 1 + slc_idx).max(0).min(topk)
                    cur_seq.as_variable()
                    bn_per_batch = (cur_seq + s2_tile - 1) // s2_tile
                    for n_kv_idx in pypto.loop(0, n_kv_sym, 1, name="LOOP_L2_n_kv_SA", idx_name="n_kvIdx"):
                        def inside_n_kv_idx_loop(batch_idx, slc_idx, n_kv_idx):
                            for group_idx in pypto.loop(0, g_loop_sym, 1, name="LOOP_L3_g_SA", idx_name="gIdx"):
                                def inside_group_idx_loop(batch_idx, slc_idx, n_kv_idx, group_idx):
                                    cur_group_tile = group_tile
                                    oi_update = pypto.tensor([cur_group_tile, dn], pypto.DT_FP32, "oi_update")
                                    li_update = pypto.tensor([1, cur_group_tile], pypto.DT_FP32, "li_update")
                                    mi_update = pypto.tensor([1, cur_group_tile], pypto.DT_FP32, "mi_update")
                                    cur_offset = batch_idx * s1_n1_gsym \
                                        + slc_idx * nq + n_kv_idx * group + group_idx * cur_group_tile
                                    oi_offset = [batch_idx, slc_idx, n_kv_idx * group + group_idx * cur_group_tile, 0]
                                    for s2_idx, unroll_length in pypto.loop_unroll(0, bn_per_batch, 1,
                                        name="LOOP_L4_s2_SA", idx_name="s2_idx", unroll_list={1}):
                                        def inside_s2_idx_loop(batch_idx, slc_idx, n_kv_idx,
                                            group_idx, s2_idx, unroll_length):
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
                                                cur_kn_fp32 = pypto.view(kn_fp32_reshape, [cur_s2_tile, dn], [0,0],
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
                                                def inside_if_loop_begin():
                                                    nonlocal oi_update, li_update, mi_update
                                                    oi_tmp = q1
                                                    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                    if pypto.cond(pypto.is_loop_end(s2_idx)):
                                                        def inside_if_is_loop_end():
                                                            pypto.set_semantic_label("Sa_KvVec2")
                                                            oi_update[:] = oi_tmp / tilda_lij_reduce
                                                            pypto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                                            oi_update_4_dim = pypto.cast(pypto.reshape(oi_update,
                                                                [1, 1, cur_group_tile, dn]), dtype)
                                                            pypto.assemble(oi_update_4_dim, oi_offset, attention_out)
                                                        inside_if_is_loop_end()
                                                    else:
                                                        oi_update[:] = oi_tmp
                                                    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                    li_update[:] = pypto.clone(tilda_lij)
                                                    mi_update[:] = pypto.clone(tilda_mij)
                                                inside_if_loop_begin()
                                            else:
                                                def inside_else_loop_begin():
                                                    nonlocal oi_update, li_update, mi_update
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
                                                        def inside_if_is_loop_end():
                                                            oi_update[:] = pypto.div(oi_tmp,
                                                                pypto.reshape(li_new, [cur_group_tile, 1]))
                                                            pypto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                                            oi_update_4_dim = pypto.cast(pypto.reshape(oi_update,
                                                                [1, 1, cur_group_tile, dn]), dtype)
                                                            pypto.assemble(oi_update_4_dim, oi_offset, attention_out)
                                                        inside_if_is_loop_end()
                                                    else:
                                                        oi_update[:] = oi_tmp
                                                    li_update[:] = li_new
                                                    mi_update[:] = mi_new
                                                inside_else_loop_begin()
                                        inside_s2_idx_loop(batch_idx, slc_idx, n_kv_idx,
                                            group_idx, s2_idx, unroll_length)
                                inside_group_idx_loop(batch_idx, slc_idx, n_kv_idx, group_idx)
                        inside_n_kv_idx_loop(batch_idx, slc_idx, n_kv_idx)
                inside_slc_idx_loop(batch_idx, slc_idx)
        inside_batch_idx_loop(batch_idx)


@pypto.jit
def select_attention_v2(in_tensors, out_tensors, n_q, n_kv, softmax_scale, topk, tile_config):
    query_nope, query_rope, key_nope_2d, key_rope_2d, k_nope_scales, offsets, kv_act_seqs = in_tensors
    attention_out, = out_tensors
    pypto.set_host_options(only_codegen=True)
    
    def inside_main_function():
        select_attention_compute_v2(
            query_nope=query_nope,
            query_rope=query_rope,
            key_nope_2d=key_nope_2d,
            key_rope_2d=key_rope_2d,
            k_nope_scales=k_nope_scales,
            offsets=offsets,
            kv_act_seqs=kv_act_seqs,
            nq=n_q,
            n_kv=n_kv,
            softmax_scale=softmax_scale,
            topk=topk,
            attention_out=attention_out,
            tile_config=tile_config
        )
    inside_main_function()


def sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant, input_params,
    kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name):
    b, n1, n2, s1 = bn1n2s1
    torch.npu.set_device(5)

    tile_config = SaTileShapeConfig(
        g_tile=128,
        s_kv_tile=2048,
        c1_tile_shape=[128, 128, 64, 64, 256, 256],
        v1_tile_shape=[16, 256],
        c2_tile_shape=[128, 128, 128, 128, 128, 128],
        v2_tile_shape=[16, 128]
    )
    
    b, s1, n_q, n_kv, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, is_kn_quant, softmax_scale = input_params
    q_nope, q_rope, kn, kr, kn_scales, offsets, kv_actual_seqs = input_data

    calc_attention_out = torch.zeros([b, s1, n_q, kv_lora_rank], dtype=torch.bfloat16)
    kv_act_seqs = torch.tensor(actual_seq, dtype=torch.int32)
    input_data_npu = [tmp.npu() for tmp in [q_nope, q_rope, kn, kr, kn_scales, offsets, kv_act_seqs]]
    output_data_npu = [b.npu() for b in [calc_attention_out]]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(input_data_npu)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(output_data_npu)]
    select_attention_v2(pto_inputs, pto_outputs, n_q, n_kv, softmax_scale, topk, tile_config)
    pypto.runtime._device_synchronize()
    assert_allclose(np.array(output_data_npu[0].cpu().flatten().tolist()), np.array(atten_out.cpu().flatten().tolist()), rtol=0.005, atol=0.005)


def sparse_attention_entry(case_name: str):
    if case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s2_seqTest1_int8":
        bn1n2s1 = (4, 128, 1, 2)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900]
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511":
        # bn1n2s1数据: b, n_q, n_kv, s_q; n_kv=1
        bn1n2s1 = (32, 128, 1, 1)
        # 0为kn非量化情况，1为kn量化情况
        is_kn_quant = 0
        actual_seq = [511, 511, 511, 511]
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511_int8":
        bn1n2s1 = (32, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [511] * 32
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049":
        bn1n2s1 = (1, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [2049]
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049_int8":
        bn1n2s1 = (1, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [2049]
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047":
        bn1n2s1 = (1, 128, 1, 3)
        is_kn_quant = 0
        actual_seq = [2047]
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047_int8":
        bn1n2s1 = (1, 128, 1, 3)
        is_kn_quant = 1
        actual_seq = [2047]
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k":
        bn1n2s1 = (128, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [8096] * 128
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k_int8":
        bn1n2s1 = (128, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [8096] * 128
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [131072] * 8  # 128k
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k_int8":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [131072] * 8
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1":
        bn1n2s1 = (4, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [666, 532, 768, 900]
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1_int8":
        bn1n2s1 = (4, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900]
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2_int8":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2":
        bn1n2s1 = (8, 128, 1, 4)
        is_kn_quant = 0
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2_int8":
        bn1n2s1 = (8, 128, 1, 4)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
        input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out \
            = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)
        sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant,
            input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out, case_name)
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


"""
    "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s2_seqTest1_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2_int8",
"""
def test_sparse_attention(case_names="DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047_int8"):
    sparse_attention_entry(case_names)