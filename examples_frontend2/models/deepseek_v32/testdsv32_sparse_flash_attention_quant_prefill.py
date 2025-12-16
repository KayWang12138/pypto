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
import pypto
import torch
import torch_npu
import os
import math
from pathlib import Path
from sparse_flash_attention_quant_prefill import sparse_flash_attention_quant_p_compute
import pytest
import numpy as np
import logging
from numpy.testing import assert_allclose

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
    PyTorch implementation for uniform distribution data generation, behavior identical to NumPy version
    Strictly maintains the [min_value, max_value) left-closed right-open interval property
    """
    # Special case: all-zero tensor
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    # Boolean type handling: generate True/False with equal probability
    if dtype == torch.bool:
        # Generate integers in [0,2), convert to bool for equal probability True/False
        return torch.randint(0, 2, data_shape, dtype=dtype)
    # Floating-point type: [min_value, max_value)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        # torch.rand generates [0,1), scale to obtain [min_value, max_value)
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    # Integer type: [min_value, max_value)
    else:
        # torch.randint's high parameter is open interval, directly corresponding to [min_value, max_value)
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def softmax(x, input_dtype):
    """PyTorch implementation of softmax function"""
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
    Compute attention mechanism, supports varying sequence lengths across batches
    Implemented using PyTorch
    """
    q, kn, kr, kn_scales, offsets, actual_seq = input_data
    scalar, topk, d_v, is_kn_quant = params
    # Extract dimension information
    t, n1, dq = q.shape
    _, dk = kn.shape
    _, dv = kr.shape
    b = len(actual_seq)
    s1 = t // b
    s2_tile = 2048

    atten_out_shape = [t, n1, d_v]
    input_dtype = q.dtype
    kn_dtype = kn.dtype

    # Initialize output tensors
    attention_output = torch.zeros(atten_out_shape, dtype=input_dtype)
    tmp_out = torch.zeros([t, n1], dtype=input_dtype)

    for t_idx in range(t):
        b_idx = t_idx // s1
        s1_idx = t_idx % s1
        cur_k_seq = actual_seq[b_idx]
        cur_seq = min(max(cur_k_seq - s1 + 1 + s1_idx, 0), topk)
        bn_per_batch = math.ceil(cur_seq / s2_tile)

        qi = q[t_idx, :, :] # (n1, dk)

        for s2_idx in range(bn_per_batch):
            s2_tile_cur = min(s2_tile, cur_seq - s2_idx * s2_tile)
            s2_start = s2_tile * s2_idx
            s2_end = s2_start + s2_tile_cur
            offset = offsets[t_idx, s2_start:s2_end]
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
                tmp_out[t_idx, :] = tilda_lij.reshape(n1)
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

        attention_output[t_idx, :, :] = oi_update

    return attention_output, tmp_out

def gen_gather_select_attention_golden(dtype, bn1n2s1, is_kn_quant, actual_seq):
    block_size = 128
    torch.manual_seed(42)
    b, n1, nkv, s_q = bn1n2s1  # 48, 128, 1, 1
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
    # 1. define shape

    t = b * s_q
    shape_q = [t, n1, d_q]

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

    offsets = torch.zeros(t, topk).to(torch.int32)

    for t_i in range(t):
        b_i = t_i // s_q
        perm = torch.randperm(block_num * block_size)
        offsets[t_i, :slc_actual_seq[b_i]] = perm[:slc_actual_seq[b_i]]

    offsets = offsets.reshape(t, nkv * topk)

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

    # 3. compute attention
    params = [scalar, topk, kv_lora_rank, is_kn_quant]
    input_data = [q_bsnd, kn, kr, kn_scales, offsets, actual_seq]
    atten_out, tmp_out = compute_attention(input_data, params)

    # 4.dump data
    # data split to [nope + rope]
    q_nope = q_bsnd[:, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, kv_lora_rank:]
    q_nope = q_nope.reshape(t * n1, kv_lora_rank)
    q_rope = q_rope.reshape(t * n1, qk_rope_dim)

    # input params
    input_params = [b, s_q, n1, nkv, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, is_kn_quant, scalar]
    kn_aux_tensor = torch.eye(512, dtype=torch.float32).to(torch.int8)
    scale_aux_tensor = torch.eye(4, dtype=torch.float32)
    input_data_map = [q_nope, q_rope, kn, kr, kn_scales, offsets, actual_seq]

    return input_params, kn_aux_tensor, scale_aux_tensor, input_data_map, atten_out


def do_test_QSFA_p(case_name: str):
    bn1n2s1 = (4, 128, 1, 2)
    is_kn_quant = 1
    actual_seq = [666, 532, 768, 900]
    print("============bn1n2s1 is: ===================== 111")
    if case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s2_seqTest1_int8":
        bn1n2s1 = (4, 128, 1, 2)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511":
        print("============bn1n2s1 is: ===================== 222")
        # bn1n2s1数据: b, n_q, n_kv, s_q; n_kv=1
        bn1n2s1 = (32, 128, 1, 1)
        # 0为kn非量化情况，1为kn量化情况
        is_kn_quant = 0
        actual_seq = [511] * 32
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511_int8":
        bn1n2s1 = (32, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [511] * 32
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049":
        bn1n2s1 = (1, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [2049]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049_int8":
        bn1n2s1 = (1, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [2049]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047":
        bn1n2s1 = (1, 128, 1, 3)
        is_kn_quant = 0
        actual_seq = [2047]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047_int8":
        bn1n2s1 = (1, 128, 1, 3)
        is_kn_quant = 1
        actual_seq = [2047]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s256_seq2047":
        bn1n2s1 = (1, 128, 1, 256)
        is_kn_quant = 0
        actual_seq = [2047]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s256_seq2047_int8":
        bn1n2s1 = (1, 128, 1, 256)
        is_kn_quant = 1
        actual_seq = [2047]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s512_seq2047":
        bn1n2s1 = (1, 128, 1, 512)
        is_kn_quant = 0
        actual_seq = [2047]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s512_seq2047_int8":
        bn1n2s1 = (1, 128, 1, 512)
        is_kn_quant = 1
        actual_seq = [2047]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k":
        bn1n2s1 = (128, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [8096] * 128
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k_int8":
        bn1n2s1 = (128, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [8096] * 128
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [131072] * 8  # 128k
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k_int8":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [131072] * 8
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1":
        bn1n2s1 = (4, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [666, 532, 768, 900]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1_int8":
        bn1n2s1 = (4, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2_int8":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2":
        bn1n2s1 = (8, 128, 1, 4)
        is_kn_quant = 0
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2_int8":
        bn1n2s1 = (8, 128, 1, 4)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        assert False

    print("============bn1n2s1 is: =====================")
    print(bn1n2s1)
    print("============is_kn_quant is: =====================")
    print(is_kn_quant)
    print("============actual_seq is: =====================")
    print(actual_seq)

    input_params, kn_aux_tensor, scale_aux_tensor, input_data, atten_out = gen_gather_select_attention_golden(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq)

    b, n1, n2, s1 = bn1n2s1
    torch.npu.set_device(0)

    tile_config = SaTileShapeConfig(
        g_tile=128,
        s_kv_tile=2048,
        c1_tile_shape=[128, 128, 64, 64, 256, 256],
        v1_tile_shape=[16, 256],
        c2_tile_shape=[128, 128, 128, 128, 128, 128],
        v2_tile_shape=[16, 128]
    )

    # t = b * s1
    b, s1, n1, nkv, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, is_kn_quant, softmax_scale = input_params
    q_nope, q_rope, kn, kr, kn_scales, offsets, kv_actual_seqs = input_data

    t = b * s1
    #calc_attention_out = torch.zeros([t, n1, kv_lora_rank], dtype=torch.bfloat16)

    kv_act_seqs = torch.tensor(actual_seq, dtype=torch.int32)
    input_data_npu = [tmp.npu() for tmp in [q_nope, q_rope, kn, kr, kn_scales, offsets, kv_act_seqs]]
    #output_data_npu = [t.npu() for t in [calc_attention_out]]

   # pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(input_data_npu)]
    #pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(output_data_npu)]

    my_new_output = sparse_flash_attention_quant_p_compute(
        input_data_npu[0],
        input_data_npu[1],
        input_data_npu[2],
        input_data_npu[3],
        input_data_npu[4],
        input_data_npu[5],
        input_data_npu[6],
        nq=n1,
        n_kv=nkv,
        softmax_scale=softmax_scale,
        topk=topk,
        tile_config=tile_config
    )

    pypto.runtime._device_synchronize()
    diff = my_new_output.cpu().flatten() - atten_out.cpu().flatten()
    print("max_diff, mean_abs_diff:", diff.max(), torch.mean(torch.abs(diff)))


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b4_s2_seqTest1_int8():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s2_seqTest1_int8")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b32_s1_seq511():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b32_s1_seq511_int8():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511_int8")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_bf16_b1_s1_seq2049():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_bf16_b1_s1_seq2049_int8():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049_int8")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b1_s3_seq2047():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047")


def test_QSFA_p_bf16_b1_s3_seq2047_int8():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047_int8")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_bf16_b1_s256_seq2047():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s256_seq2047")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_bf16_b1_s256_seq2047_int8():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s256_seq2047_int8")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_bf16_b1_s512_seq2047():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s512_seq2047")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_bf16_b1_s512_seq2047_int8():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s512_seq2047_int8")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b128_s1_seq8k():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b128_s1_seq8k_int8():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k_int8")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b8_s1_seq128k():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b8_s1_seq128k_int8():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k_int8")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b4_s1_seqTest1():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b4_s1_seqTest1_int8():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1_int8")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b4_s1_seqTest2():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b4_s1_seqTest2_int8():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2_int8")


def test_QSFA_p_bf16_b8_s4_seqTest2():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2")


@pytest.mark.skip(reason='perf')
def test_QSFA_p_bf16_b8_s4_seqTest1_int8():
    do_test_QSFA_p("DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2_int8")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_QSFA_p_bf16_b1_s3_seq2047_int8()
    # test_QSFA_p_bf16_b8_s4_seqTest2()
