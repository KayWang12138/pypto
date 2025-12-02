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
from pypto import pypto_impl
from pypto.operation import op_wrapper
import torch
import numpy as np
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


@op_wrapper
def load(src, dst):
    return pypto_impl.Load(src, dst)


def gen_cache_tensor(k_tensor, block_table, block_num, block_size, b):
    logging.info("Entering into gen_cache_tensor!")
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
    logging.info("Entering into gen_block_table!")
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
    n_q = params.get("n_q")
    d = params.get("d")
    dtype = params.get("dtype")
    s2 = params.get("s2")
    n_kv = params.get("n_kv")
    act_seq_len = params.get("act_seq")
    block_size = params.get("block_size")
    block_num = params.get("block_num")
    max_block_num = params.get("max_block_num")
    selected_count = params.get("selected_count")
    score_scale = params.get("score_scale")

    query = torch.randn([b, s1, n_q, d], dtype=dtype)
    weights = torch.randn([b, s1, n_q], dtype=dtype)

    k_bsnd = torch.randn([b, s2, n_kv, d], dtype=dtype)
    _, block_table_list = gen_block_table(b, block_size, s2, act_seq_len)

    block_table = torch.tensor(block_table_list, dtype=torch.int32)
    act_seq = torch.tensor(act_seq_len, dtype=torch.int32)
    
    key = gen_cache_tensor(k_bsnd, block_table_list, block_num, block_size, b)
    # construct output tensor
    topk_res = torch.zeros([b, s1, n_kv, selected_count], dtype=torch.int32)

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
    n_q = params.get("n_q")
    d = params.get("d")
    n_kv = params.get("n_kv")
    block_num = params.get("block_num")
    max_block_num = params.get("max_block_num")
    score_scale = params.get("score_scale")
    dtype = params.get("dtype")

    # get input tensors
    query = input_data_map.get("query")
    key = input_data_map.get("key")
    if is_quant:
        q_scale = input_data_map.get("q_scale")
        k_scale = input_data_map.get("k_scale")
    weights = input_data_map.get("weights")
    act_seq = input_data_map.get("act_seq")
    block_table = input_data_map.get("block_table")

    topk_value = torch.zeros([b, s1, n_kv, selected_count], dtype=torch.float32)
    topk_res = torch.zeros([b, s1, n_kv, selected_count], dtype=torch.int32)
    tmp_out = torch.zeros([b * s1 * n_kv, max_block_num * block_size], dtype=torch.float32)

    g = n_q // n_kv
    query = query.reshape(b * s1 * n_q, d)
    key = key.reshape(block_num * block_size, n_kv * d)
    weights = weights.reshape(b * s1 * n_q, 1)
    if is_quant:
        q_scale = q_scale.reshape(b * s1 * n_q, 1)
        k_scale = k_scale.reshape(block_num * block_size, n_kv)

    avoid_fp32_to_fp16_overflow_scale = 1 / 2048

    for b_idx in range(b):
        cur_seq = act_seq[b_idx]
        for s_idx in range(s1):
            casual_offset = s1 - s_idx - 1
            eff_seq = cur_seq - casual_offset
            actual_block = (eff_seq + block_size - 1) // block_size
            for n_kv_idx in range(n_kv):
                local_sum = torch.zeros(
                    [1, max_block_num * block_size], dtype=torch.float32)
                for block_idx in range(actual_block):
                    remain_s2 = min(block_size, eff_seq -
                                    block_size * block_idx)
                    cur_block_idx = block_table[b_idx][block_idx]
                    q_offset = b_idx * s1 * n_q + s_idx * n_q + n_kv_idx * g
                    cur_q = query[q_offset: (q_offset + g), :]
                    cur_k = key[cur_block_idx * block_size: (
                        cur_block_idx * block_size + remain_s2), n_kv_idx * d: ((n_kv_idx + 1) * d)]
                    cur_w = weights[q_offset: (q_offset + g), :]

                    if is_quant:
                        cur_qs = q_scale[q_offset: (q_offset + g), :]
                        cur_ks = k_scale[cur_block_idx * block_size:(cur_block_idx * block_size + remain_s2),
                            n_kv_idx:(n_kv_idx + 1)]
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
                    cur_n_kv_idx = b_idx * s1 * n_kv + s_idx * n_kv + n_kv_idx
                    tmp_out[cur_n_kv_idx:(cur_n_kv_idx + 1),
                        block_idx * block_size: (block_idx * block_size + remain_s2)] = sum_res

                eff_sum_res = local_sum[:, :eff_seq]
                k_num = selected_count
                if eff_seq < selected_count:
                    k_num = eff_seq
                cur_value, cur_index = torch.topk(eff_sum_res, k=k_num, dim=1)
                topk_value[b_idx, s_idx, n_kv_idx, :eff_seq] = cur_value.reshape(1, 1, 1, k_num)
                topk_res[b_idx, s_idx, n_kv_idx, :eff_seq] = cur_index.reshape(1, 1, 1, k_num)

                if eff_seq < selected_count:
                    topk_value[b_idx, s_idx, n_kv_idx, eff_seq:] = (-float(3.40282347e38)) * \
                        torch.ones([1, 1, 1, selected_count -
                                   eff_seq], dtype=torch.float32)
                    topk_res[b_idx, s_idx, n_kv_idx, eff_seq:] = -1 * \
                        torch.ones([1, 1, 1, selected_count -
                                   eff_seq], dtype=torch.int32)

    return topk_value, topk_res, tmp_out


def gen_calc_offsets_for_gather_golden(params, topk_indcies, block_table, kv_act_seqs):
    b = params.get("b")
    s1 = params.get("s1")
    topk = params.get("selected_count")
    block_size = params.get("block_size")

    offsets = torch.zeros(b, s1, topk).to(torch.int32)
    for b_i in range(b):
        for s_i in range(s1):
            kv_seq_len = kv_act_seqs[b_i]
            act_slc_count = min(max(kv_seq_len - s1 + 1 + s_i, 0), topk)
            topk_indcies_tmp = topk_indcies[b_i, s_i, :act_slc_count]
            for idx in range(act_slc_count):
                topk_index = topk_indcies_tmp[idx]
                block_idx_in_batch = topk_index // block_size
                slc_block_idx = block_table[b_i, block_idx_in_batch]
                tail = topk_index % block_size
                offsets[b_i, s_i, idx] = slc_block_idx * block_size + tail
    offsets = offsets.reshape([b * s1, topk])
    return offsets


def calc_offsets_for_gather(topk_indecies, block_table, kv_act_seqs, offsets, block_size, topk, batch_value, seq_value):
    n_kv = 1
    max_block_num_per_batch = block_table.shape[1]
    b = batch_value
    s1 = seq_value

    pypto.set_host_options(only_codegen=True)
    
    pypto.set_pass_options(copyin_threshold=NUM_100 * NUM_1024 * NUM_1024,
                         cycle_lower_bound=NUM_1024,
                         cycle_upper_bound=NUM_1024 * NUM_1024,
                         l1_reuse=NUM_32,
                         parallel_threshold=NUM_2,
                         nbuffer_merge_mode=1)
    pypto.set_runtime_options(machine_sched_mode=NUM_3,
                            workspace_recycle_period=NUM_128,
                            estimated_stitch_task_max_loop_num=NUM_128)
    pypto.set_codegen_options(support_dynamic_unaligned=True)

    with pypto.function("main", [topk_indecies, block_table, kv_act_seqs], [offsets]):
        for idx in pypto.loop(0, b * s1, 1, name="LOOP_L0_idx", idx_name="idx"):
            def inside_idx_loop(idx):
                batch_idx = idx // s1
                slc_idx = idx % s1
                pypto.set_semantic_label("calc_offset")
                topk_loop = (kv_act_seqs[batch_idx, ] - s1 + 1 + slc_idx).max(0).min(topk)
                tile_0 = 1
                tile_1 = 256
                pypto.set_vec_tile_shapes(tile_0, tile_1)
                topk_indcies_reshape = pypto.view(topk_indecies, [1, n_kv * topk],
                                                            [idx, 0], valid_shape=[1, topk_loop])

                topk_indcies_reshape_fp32 = pypto.cast(topk_indcies_reshape, pypto.DataType.DT_FP32)
                topk_indcies_reshape_fp32 = pypto.add(topk_indcies_reshape_fp32, 0.5)
                block_idx_in_batchs_fp32 = pypto.div(topk_indcies_reshape_fp32, float(block_size))
                block_idx_in_batchs = pypto.cast(block_idx_in_batchs_fp32,
                    pypto.DataType.DT_INT32, pypto.CastMode.CAST_FLOOR)

                tails = pypto.sub(topk_indcies_reshape, pypto.mul(block_idx_in_batchs, block_size))

                block_table_raw_offsets = pypto.full([1, n_kv * topk], batch_idx * max_block_num_per_batch,
                    pypto.DataType.DT_INT32, valid_shape=[1, topk_loop])
                add_res = pypto.add(block_table_raw_offsets, block_idx_in_batchs)
                slc_block_idxs = load(block_table, add_res)
                block_offsets = pypto.mul(slc_block_idxs, block_size)
                offset = pypto.add(block_offsets, tails)
                pypto.assemble(offset, [idx, 0], offsets)
            inside_idx_loop(idx=idx)


def test_calc_offset_4_gather():
    torch.npu.set_device(0)
    bn1n2s1 = (4, 128, 1, 1)
    is_quant = 1
    b, n_q, n_kv, s1 = bn1n2s1

    d = 128
    block_size = 128
    dtype = torch.float16
    act_seq = [131072] * 4
    s2 = max(act_seq) # s2 means max act_seq
    block_num = sum([(s + block_size - 1) // block_size for s in act_seq])
    max_block_num = (s2 + block_size - 1) // block_size
    selected_count = 2048
    kv_lora_rank = 512

    n1_scale = 1.0 / np.sqrt(n_q)
    softmax_scale = 1.0 / np.sqrt(d)
    score_scale = n1_scale * softmax_scale

    params = {}
    params["b"] = b
    params["s1"] = s1
    params["n_q"] = n_q
    params["d"] = d
    params["dtype"] = dtype
    params["s2"] = s2
    params["n_kv"] = n_kv
    params["block_size"] = block_size
    params["block_num"] = block_num
    params["max_block_num"] = max_block_num
    params["selected_count"] = selected_count
    params["score_scale"] = score_scale
    params["act_seq"] = act_seq

    input_data_map = gen_data_for_compute(params, is_quant=True)
    topk_value, topk_res, tmp_out = indexer_topk_compute(input_data_map, params, is_quant=is_quant)
    topk_res_calc = topk_res.reshape(b * s1, n_kv * selected_count)
    
    
    @pypto.jit
    def cust_dyn_func(in_tensors, out_tensors, block_size, topk, b, s1):
        topk_indecies, block_table, kv_act_seqs = in_tensors
        offsets, = out_tensors
        calc_offsets_for_gather(topk_indecies, block_table,
            kv_act_seqs, offsets, block_size=block_size, topk=topk, batch_value=b, seq_value=s1)
    
    offsets = torch.zeros([b * s1, n_kv * selected_count], dtype=torch.int32)
    input_data = [a.npu() for a in [topk_res_calc, input_data_map["block_table"], input_data_map["act_seq"]]]
    output_data = [a.npu() for a in [offsets]]
    cust_dyn_func(input_data, output_data, block_size, selected_count, b, s1)

    pypto.runtime._device_synchronize()
    
    topk_res_golden = topk_res.reshape(b, s1, selected_count)

    offsets_golden = gen_calc_offsets_for_gather_golden(params,
        topk_res_golden, input_data_map["block_table"], input_data_map["act_seq"])
    pypto.runtime._device_synchronize()

    assert_allclose(np.array(output_data[0].cpu().flatten().tolist()),
        np.array(offsets_golden.cpu().flatten().tolist()), rtol=0.005, atol=0.005)