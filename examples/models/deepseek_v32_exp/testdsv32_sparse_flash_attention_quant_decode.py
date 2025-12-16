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
import pytest
import torch
import pypto
import logging
import numpy as np
from numpy.testing import assert_allclose
from pypto import pypto_impl
from pypto.operation import op_wrapper
from sparse_flash_attention_quant import sparse_flash_attention_quant_d_compute, SaTileShapeConfig


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
    q, kn, kr, kn_scales, topk_indcies, block_table, actual_seq = input_data
    block_size, scalar, topk, d_v, is_kn_quant = params

    # 提取维度信息
    b, s1, n1, dq = q.shape
    _, dk = kn.shape
    _, dv = kr.shape

    s2_tile = 2048
    if topk_indcies.ndim > 2:
        topk_indcies = topk_indcies.reshape(b * s1, topk)

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

                topk_indcies_tmp = topk_indcies[b_idx * s1 + s1_idx, s2_start:s2_end]

                slc_kn = torch.zeros([s2_tile_cur, dk], dtype=kn_dtype)
                slc_kr = torch.zeros([s2_tile_cur, dv], dtype=input_dtype)
                slc_kn_scales = torch.zeros([s2_tile_cur, 4], dtype=torch.float32)

                # 当前b&s1&s2 topk_index  --->  kvCache的offset
                offset = torch.zeros([s2_tile_cur], dtype=torch.int32)
                for cur_s2_idx in range(s2_tile_cur):
                    s2_idx_tmp = s2_start + cur_s2_idx
                    topk_index = topk_indcies_tmp[s2_idx_tmp]
                    block_idx_in_batch = topk_index // block_size
                    slc_block_idx = block_table[b_idx, block_idx_in_batch]
                    tail = topk_index % block_size
                    offset[cur_s2_idx] = slc_block_idx * block_size + tail

                # 索引 kvCache
                for cur_s2_idx in range(s2_tile_cur):
                    slc_idx = offset[cur_s2_idx]
                    slc_kn[cur_s2_idx, :] = kn[slc_idx, :]
                    slc_kr[cur_s2_idx, :] = kr[slc_idx, :]
                    slc_kn_scales[cur_s2_idx, :] = kn_scales[slc_idx, :]

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


def gen_block_table(act_seq, block_size, s1, need_indices=False):
    block_num = 0
    block_num_each = []
    b = act_seq.shape[0]
    max_kv = max(act_seq)
    for cur_s in act_seq:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    block_table_shape = [b, math.ceil(max_kv / block_size)]
    block_idx_list = torch.arange(0, block_num, 1)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))].to(torch.int32)

    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

    block_table_bidx = 0
    block_idx = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx, j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    if need_indices:
        cache_index = -torch.ones((b, s1), dtype=torch.int64)
        for i in range(b):
            cur_act = act_seq[i]
            for j in range(s1):
                pos = cur_act - s1 + j
                block_idx_in_seq = pos // block_size
                global_block_id = block_table[i, block_idx_in_seq]

                offset_in_block = pos % block_size
                global_index = global_block_id * block_size + offset_in_block
                cache_index[i, j] = global_index
    else:
        cache_index = None

    return block_num, block_table, cache_index


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

    max_kv_seq = max(actual_seq)
    block_num, block_table, _ = gen_block_table(torch.tensor(actual_seq), block_size, s_q, need_indices=False)
    topk_indcies = torch.zeros(b, s_q, topk).to(torch.int32)
    slc_actual_seq = []
    for i in range(b):
        slc_actual_seq.append(min(actual_seq[i], topk))

    for b_i in range(b):
        for s_q_i in range(s_q):

            if slc_actual_seq[b_i] < topk:
                topk_indcies[b_i, s_q_i, :slc_actual_seq[b_i]] = torch.arange(0, slc_actual_seq[b_i])
            else:
                perm = torch.randperm(slc_actual_seq[b_i])
                topk_indcies[b_i, s_q_i, :] = perm[:topk]

    topk_indcies = topk_indcies.reshape(b * s_q, n_kv * topk)

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
    params = [block_size, scalar, topk, kv_lora_rank, is_kn_quant]
    input_data = [q_bsnd, kn, kr, kn_scales, topk_indcies, block_table, actual_seq]
    atten_out, tmp_out = compute_attention(input_data, params)

    # 4.dump 数据
    # data split to [nope + rope]
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]
    q_nope = q_nope.reshape(b * s_q * n_q, kv_lora_rank)
    q_rope = q_rope.reshape(b * s_q * n_q, qk_rope_dim)
    # input params
    input_params = [b, s_q, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk,
                    is_kn_quant, scalar]
    input_data_map = [q_nope, q_rope, kn, kr, kn_scales, topk_indcies, block_table, actual_seq]

    return input_params, input_data_map, atten_out


def do_test_sparse_attention_func(bn1n2s1, actual_seq, is_kn_quant, input_params, input_data, atten_out, case_name):
    b, n1, n2, s1 = bn1n2s1

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tile_config = SaTileShapeConfig(
        g_tile=128,
        s_kv_tile=2048,
        c1_tile_shape=[128, 128, 128, 128, 128, 128],
        v1_tile_shape=[8, 2048],
        c2_tile_shape=[128, 128, 128, 128, 128, 128],
        v2_tile_shape=[64, 128]
    )

    b, s1, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, \
        is_kn_quant, softmax_scale = input_params
    q_nope, q_rope, kn, kr, kn_scales, topk_indcies, block_table, kv_actual_seqs = input_data

    calc_attention_out = torch.zeros([b, s1, n_q, kv_lora_rank], dtype=torch.bfloat16)
    kv_act_seqs = torch.tensor(actual_seq, dtype=torch.int32)

    q_nope_npu = q_nope.npu()
    q_nope_pto = pypto.from_torch(q_nope_npu, dynamic_axis=[0], name="q_nope")
    q_rope_npu = q_rope.npu()
    q_rope_pto = pypto.from_torch(q_rope_npu, dynamic_axis=[0], name="q_rope")
    kn_npu = kn.npu()
    kn_pto = pypto.from_torch(kn_npu, name="kn")
    kr_npu = kr.npu()
    kr_pto = pypto.from_torch(kr_npu, name="kr")
    kn_scales_npu = kn_scales.npu()
    kn_scales_pto = pypto.from_torch(kn_scales_npu, name="kn_scales")
    topk_indcies_npu = topk_indcies.npu()
    topk_indcies_pto = pypto.from_torch(topk_indcies_npu, dynamic_axis=[0], name="topk_indcies")
    block_table_npu = block_table.npu()
    block_table_pto = pypto.from_torch(block_table_npu, dynamic_axis=[0], name="block_table")
    kv_act_seqs_npu = kv_act_seqs.npu()
    kv_act_seqs_pto = pypto.from_torch(kv_act_seqs_npu, dynamic_axis=[0], name="kv_act_seqs")

    calc_attention_out_npu = calc_attention_out.npu()
    calc_attention_out_pto = pypto.from_torch(calc_attention_out_npu, dynamic_axis=[0], name="calc_attention_out")

    pto_inputs = [q_nope_pto, q_rope_pto, kn_pto, kr_pto, kn_scales_pto, topk_indcies_pto, block_table_pto,
                  kv_act_seqs_pto]
    pto_outputs = [calc_attention_out_pto]

    max_blocknum_perbatch = math.ceil(max_kv_seq / block_size)
    sparse_flash_attention_quant_d_compute(*pto_inputs, *pto_outputs, n_q, n_kv, softmax_scale, topk,
                                           block_size, max_blocknum_perbatch, tile_config)
    pypto.runtime._device_synchronize()
    assert_allclose(np.array(calc_attention_out_npu.cpu().flatten().tolist()),
                    np.array(atten_out.cpu().flatten().tolist()), rtol=0.0005, atol=0.0005)


def get_case_config(case_name: str):
    # case参数配置字典，key为case名称，value为对应的参数元组(bn1n2s1, is_kn_quant, actual_seq)
    test_case_config = {
        "DynamicGatherSlcFlashAttnDSASTest.SFA_b4_s2_seq64K_int8_perf": (
            (4, 128, 1, 2), 1, [65536] * 4
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s2_seqTest1_int8": (
            (4, 128, 1, 2), 1, [666, 532, 768, 900]
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511": (
            (32, 128, 1, 1), 0, [511] * 32
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511_int8": (
            (32, 128, 1, 1), 1, [511] * 32
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049": (
            (1, 128, 1, 1), 0, [2049]
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049_int8": (
            (1, 128, 1, 1), 1, [2049]
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047": (
            (1, 128, 1, 3), 0, [2047]
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047_int8": (
            (1, 128, 1, 3), 1, [2047]
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k": (
            (128, 128, 1, 1), 0, [8096] * 128
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k_int8": (
            (128, 128, 1, 1), 1, [8096] * 128
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k": (
            (8, 128, 1, 1), 0, [131072] * 8
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k_int8": (
            (8, 128, 1, 1), 1, [131072] * 8
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1": (
            (4, 128, 1, 1), 0, [666, 532, 768, 900]
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1_int8": (
            (4, 128, 1, 1), 1, [666, 532, 768, 900]
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2": (
            (8, 128, 1, 1), 0, [666, 532, 768, 900, 5698, 2358, 324, 2048]
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2_int8": (
            (8, 128, 1, 1), 1, [666, 532, 768, 900, 5698, 2358, 324, 2048]
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2": (
            (8, 128, 1, 4), 0, [666, 532, 768, 900, 5698, 2358, 324, 2048]
        ),
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2_int8": (
            (8, 128, 1, 4), 1, [666, 532, 768, 900, 5698, 2358, 324, 2048]
        ),
    }
    case_config = test_case_config.get(case_name)
    return case_config


def do_test_QSFA_d_entry(case_name: str):
    case_config = get_case_config(case_name)
    if not case_config:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    bn1n2s1, is_kn_quant, actual_seq = case_config

    input_params, input_data, atten_out = gen_gather_select_attention_golden(
        torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq
    )
    do_test_sparse_attention_func(
        bn1n2s1, actual_seq, is_kn_quant,
        input_params, input_data, atten_out, case_name
    )
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


def test_SFA_b4_s2_seq64K_int8_perf():
    do_test_QSFA_d_entry("DynamicGatherSlcFlashAttnDSASTest.SFA_b4_s2_seq64K_int8_perf")


@pytest.mark.skip(reason='large case')
def test_QSFA_d_bf16_b1_s3_seq2047_int8():
    do_test_QSFA_d_entry("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047_int8")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_SFA_b4_s2_seq64K_int8_perf()
