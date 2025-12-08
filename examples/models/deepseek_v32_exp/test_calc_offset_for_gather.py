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
import pypto
import torch
import torch_npu
import os
import math
from pathlib import Path
from calc_offset_for_gather import calc_offsets_for_gather_prefill_compute
import pytest
import numpy as np
import logging
from numpy.testing import assert_allclose

NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_8 = 8
NUM_16 = 16
NUM_32 = 32
NUM_64 = 64
NUM_100 = 100
NUM_128 = 128
NUM_1024 = 1024
NUM_1127 = 1127
NUM_2048 = 2048
NUM_4096 = 4096
NUM_8192 = 8192

def gen_cache_tensor(k_tensor, block_table, block_num, block_size, b):
    # logging.info("Entering into gen_cache_tensor!")
    dtype = k_tensor.dtype
    b, s, n, d = k_tensor.shape
    k_cache = torch.zeros([block_num, block_size, n * d], dtype=dtype)
    k_tensor_bsh_raw = k_tensor.reshape(b, s, n * d)

    # kv padding
    k_tensor_bsh = torch.zeros(
        (b, block_table.shape[1] * block_size, n * d), dtype=dtype)
    k_tensor_bsh[:, : k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]

    for b_idx in range(b):
        for block_idx, cache_block_idx in enumerate(block_table[b_idx]):
            block_offset = block_idx * block_size
            if cache_block_idx != -1:
                k_cache[cache_block_idx, :, :] = k_tensor_bsh[b_idx,
                                                              block_offset: (block_offset + block_size), :]

    k_cache = k_cache.reshape(block_num, block_size, n, d)
    return k_cache


def gen_block_table(b, block_size, max_kv, act_kv):
    # logging.info("Entering into gen_block_table!")
    block_num = 0
    block_num_each = []
    for cur_s in act_kv:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    shape_bt = [b, math.ceil(max_kv / block_size)]
    block_idx_list = np.arange(0, block_num, 1)
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)

    block_idx = 0
    # invalid block_id set as -1
    block_table = [-1] * shape_bt[1]
    block_table = np.tile(block_table, (shape_bt[0], 1)).astype(np.int32)

    block_table_bidx = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    return block_num, block_table


def gen_data_for_compute(params, is_quant: bool):
    b = params.get("b")
    s1 = params.get("s1")
    n1 = params.get("n1")
    d = params.get("d")
    dtype = params.get("dtype")
    s2 = params.get("s2")
    nkv = params.get("nkv")
    act_seq_len = params.get("act_seq")
    block_size = params.get("block_size")
    block_num = params.get("block_num")
    max_block_num = params.get("max_block_num")
    selected_count = params.get("selected_count")
    score_scale = params.get("score_scale")

    t = b * s1
    query = torch.randn([t, n1, d], dtype=dtype)
    weights = torch.randn([t, n1], dtype=dtype)

    k_bsnd = torch.randn([b, s2, nkv, d], dtype=dtype)
    _, block_table_list = gen_block_table(b, block_size, s2, act_seq_len)

    block_table = torch.tensor(block_table_list, dtype=torch.int32)
    act_seq = torch.tensor(act_seq_len, dtype=torch.int32)
    
    key = gen_cache_tensor(k_bsnd, block_table_list, block_num, block_size, b)
    # construct output tensor
    topk_res = torch.zeros([t, nkv, selected_count], dtype=torch.int32)

    input_data_map = {}
    if is_quant:
        # quantization
        q_scale = (query.abs().max(dim=-1, keepdim=True).values / 127).\
                    to(dtype=torch.float16).maximum(torch.tensor(1e-3))
        k_scale = (key.abs().max(dim=-1, keepdim=True).values / 127).to(dtype=torch.float16).maximum(torch.tensor(1e-3))
        query = torch.round(query / q_scale).clip(-127, 127).to(dtype=torch.int8)
        key = torch.round(key / k_scale).clip(-127, 127).to(dtype=torch.int8)
        input_data_map["query"] = query
        input_data_map["key"] = key
        input_data_map["q_scale"] = q_scale
        input_data_map["k_scale"] = k_scale
    else:
        input_data_map["query"] = query
        input_data_map["key"] = key
    input_data_map["weights"] = weights
    input_data_map["act_seq"] = act_seq
    input_data_map["block_table"] = block_table

    return input_data_map

def indexer_topk_compute(input_data_map, params, is_quant: bool):
    block_size = params.get("block_size")  # 128
    selected_count = params.get("selected_count")
    b = params.get("b")
    s1 = params.get("s1")
    n1 = params.get("n1")
    d = params.get("d")
    nkv = params.get("nkv")
    block_num = params.get("block_num")
    max_block_num = params.get("max_block_num")
    score_scale = params.get("score_scale")
    dtype = params.get("dtype")
    t = b * s1

    # get input tensors
    query = input_data_map.get("query")
    key = input_data_map.get("key")
    if is_quant:
        q_scale = input_data_map.get("q_scale")
        k_scale = input_data_map.get("k_scale")
    weights = input_data_map.get("weights")
    act_seq = input_data_map.get("act_seq")
    block_table = input_data_map.get("block_table")

    topk_value = torch.zeros([t, nkv, selected_count], dtype=torch.float32)
    topk_res = torch.zeros([t, nkv, selected_count], dtype=torch.int32)
    tmp_out = torch.zeros([t * nkv, max_block_num * block_size], dtype=torch.float32)

    g = n1 // nkv
    query = query.reshape(t * n1, d)
    key = key.reshape(block_num * block_size, nkv * d)
    weights = weights.reshape(t * n1, 1)
    if is_quant:
        q_scale = q_scale.reshape(t * n1, 1)
        k_scale = k_scale.reshape(block_num * block_size, nkv)

    avoid_fp32_to_fp16_overflow_scale = 1 / 2048

    for t_idx in range(t):
        b_idx = t_idx // s1
        s_idx = t_idx % s1
        cur_seq = act_seq[b_idx]
        
        casual_offset = s1 - s_idx - 1
        eff_seq = cur_seq - casual_offset
        actual_block = (eff_seq + block_size - 1) // block_size
        for nkv_idx in range(nkv):
            local_sum = torch.zeros(
                [1, max_block_num * block_size], dtype=torch.float32)
            for block_idx in range(actual_block):
                remain_s2 = min(block_size, eff_seq -
                                block_size * block_idx)
                cur_block_idx = block_table[b_idx][block_idx]
                q_offset = b_idx * s1 * n1 + s_idx * n1 + nkv_idx * g
                cur_q = query[q_offset: (q_offset + g), :]
                cur_k = key[cur_block_idx * block_size: (
                    cur_block_idx * block_size + remain_s2), nkv_idx * d: ((nkv_idx + 1) * d)]
                cur_w = weights[q_offset: (q_offset + g), :]

                if is_quant:
                    cur_qs = q_scale[q_offset: (q_offset + g), :]
                    cur_ks = k_scale[cur_block_idx * block_size:(cur_block_idx * block_size + remain_s2),
                        nkv_idx:(nkv_idx + 1)]
                    mm_res_i32 = torch.matmul(cur_q.to(torch.int32), cur_k.t().to(torch.int32)).to(torch.int32)
                    mm_res_fp32 = mm_res_i32.to(torch.float32) * avoid_fp32_to_fp16_overflow_scale
                    mm_res_fp16 = mm_res_fp32.to(torch.float16)
                    mm_res = mm_res_fp16 * cur_qs * cur_ks.t()
                else:
                    mm_res = torch.matmul(cur_q.to(torch.float32), cur_k.t().to(torch.float32)).to(torch.float32)

                zero_tensor = torch.zeros([g, remain_s2], dtype=mm_res.dtype)
                relu_res = torch.maximum(mm_res, zero_tensor)
                mul_res = relu_res * cur_w
                sum_res = mul_res.to(torch.float32).sum(dim=0, keepdim=True)
                local_sum[:, block_idx * block_size: (block_idx * block_size + remain_s2)] = sum_res
                cur_nkv_idx = b_idx * s1 * nkv + s_idx * nkv + nkv_idx
                tmp_out[cur_nkv_idx:(cur_nkv_idx + 1),
                    block_idx * block_size: (block_idx * block_size + remain_s2)] = sum_res

            eff_sum_res = local_sum[:, :eff_seq]
            k_num = selected_count
            if eff_seq < selected_count:
                k_num = eff_seq
            cur_value, cur_index = torch.topk(eff_sum_res, k=k_num, dim=1)
            topk_value[t_idx, nkv_idx, :eff_seq] = cur_value.reshape(1, 1, 1, k_num)
            topk_res[t_idx, nkv_idx, :eff_seq] = cur_index.reshape(1, 1, 1, k_num)

            if eff_seq < selected_count:
                topk_value[t_idx, nkv_idx, eff_seq:] = (-float(3.40282347e38)) * \
                    torch.ones([1, 1, selected_count -
                               eff_seq], dtype=torch.float32)
                topk_res[t_idx, nkv_idx, eff_seq:] = -1 * \
                    torch.ones([1, 1, selected_count -
                               eff_seq], dtype=torch.int32)

    return topk_value, topk_res, tmp_out

def gen_calc_offsets_for_gather_golden(params, topk_indcies, block_table, kv_act_seqs):
    b = params.get("b")
    s1 = params.get("s1")
    topk = params.get("selected_count")
    block_size = params.get("block_size")

    t = b * s1
    offsets = torch.zeros(t, topk).to(torch.int32)

    for t_i in range(t):
        b_i = t_i // s1
        s_i = t_i % s1
        kv_seq_len = kv_act_seqs[b_i]
        act_slc_count = min(max(kv_seq_len - s1 + 1 + s_i, 0), topk)
        topk_indcies_tmp = topk_indcies[t_i, :act_slc_count]
        for idx in range(act_slc_count):
            topk_index = topk_indcies_tmp[idx]
            block_idx_in_batch = topk_index // block_size
            slc_block_idx = block_table[b_i, block_idx_in_batch]
            tail = topk_index % block_size
            offsets[t_i, idx] = slc_block_idx * block_size + tail
    offsets = offsets.reshape([b * s1, topk])
    return offsets

def do_test_calc_offset_4_gather(case_name):
    bn1n2s1 = (4, 128, 1, 1)
    act_seq = []
    if case_name == "CalcOffsetForGather.b4_s1_2_s2_64k":
        bn1n2s1 = (4, 128, 1, 2)
        act_seq = [131072] * 4
    elif case_name == "CalcOffsetForGather.b8_s1_1_s2_64k":
        bn1n2s1 = (8, 128, 1, 1)
        act_seq = [131072] * 8
    elif case_name == "CalcOffsetForGather.b1_s1_256_s2_64k":
        bn1n2s1 = (1, 128, 1, 256)
        act_seq = [131072]
    elif case_name == "CalcOffsetForGather.b1_s1_512_s2_64k":
        bn1n2s1 = (1, 128, 1, 512)
        act_seq = [131072]
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    
    torch.npu.set_device(0)
    b, n1, nkv, s1 = bn1n2s1
    is_quant = 1
    d = 128
    block_size = 128
    dtype = torch.float16
    s2 = max(act_seq) # s2 means max act_seq
    block_num = sum([(s + block_size - 1) // block_size for s in act_seq])
    max_block_num = (s2 + block_size - 1) // block_size
    selected_count = 2048
    kv_lora_rank = 512
    
    n1_scale = 1.0 / np.sqrt(n1)
    softmax_scale = 1.0 / np.sqrt(d)
    score_scale = n1_scale * softmax_scale

    params = {}
    params["b"] = b
    params["s1"] = s1
    params["n1"] = n1
    params["d"] = d
    params["dtype"] = dtype
    params["s2"] = s2
    params["nkv"] = nkv
    params["block_size"] = block_size
    params["block_num"] = block_num
    params["max_block_num"] = max_block_num
    params["selected_count"] = selected_count
    params["score_scale"] = score_scale
    params["act_seq"] = act_seq

    t = b * s1

    input_data_map = gen_data_for_compute(params, is_quant=True)
    topk_value, topk_res, tmp_out = indexer_topk_compute(input_data_map, params, is_quant=is_quant)
    topk_res_calc = topk_res.reshape(t, nkv * selected_count)

    offsets = torch.zeros([t, nkv * selected_count], dtype=torch.int32)
    topk_res_calc_npu = topk_res_calc.npu()
    topk_res_calc_pto = pypto.from_torch(topk_res_calc_npu, dynamic_axis=[0], name="topk_res_calc")
    block_table_npu = input_data_map["block_table"].npu()
    block_table_pto = pypto.from_torch(block_table_npu, dynamic_axis=[0,1], name="block_table")
    act_seq_npu = input_data_map["act_seq"].npu()
    act_seq_pto = pypto.from_torch(act_seq_npu, dynamic_axis=[0], name="act_seq")
    offsets_npu = offsets.npu()
    offsets_pto = pypto.from_torch(offsets_npu, dynamic_axis=[0], name="offsets")

    pto_inputs = [topk_res_calc_pto, block_table_pto, act_seq_pto]
    pto_outputs = [offsets_pto]

    calc_offsets_for_gather_prefill_compute(pto_inputs, pto_outputs, block_size=block_size, topk=selected_count, batch_value=b, seq_value=s1)

    pypto.runtime._device_synchronize()
    topk_res_golden = topk_res.reshape(t, selected_count)

    offsets_golden = gen_calc_offsets_for_gather_golden(params,
        topk_res_golden, input_data_map["block_table"], input_data_map["act_seq"])

    assert_allclose(np.array(offsets_npu.cpu().flatten().tolist()),
        np.array(offsets_golden.cpu().flatten().tolist()), rtol=0.005, atol=0.005)

def test_calc_offset_4_gather_b4_s1_2_s2_64k():
    do_test_calc_offset_4_gather("CalcOffsetForGather.b4_s1_2_s2_64k")

def test_calc_offset_4_gather_b8_s1_1_s2_64k():
    do_test_calc_offset_4_gather("CalcOffsetForGather.b8_s1_1_s2_64k")

def test_calc_offset_4_gather_b1_s1_256_s2_64k():
    do_test_calc_offset_4_gather("CalcOffsetForGather.b1_s1_256_s2_64k")

def test_calc_offset_4_gather_b1_s1_512_s2_64k():
    do_test_calc_offset_4_gather("CalcOffsetForGather.b1_s1_512_s2_64k")

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    
    test_calc_offset_4_gather_b4_s1_2_s2_64k()
    test_calc_offset_4_gather_b8_s1_1_s2_64k()
    test_calc_offset_4_gather_b1_s1_256_s2_64k()
    test_calc_offset_4_gather_b1_s1_512_s2_64k()
