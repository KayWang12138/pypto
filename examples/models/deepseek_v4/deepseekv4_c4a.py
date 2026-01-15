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
import torch_npu
import pypto
import logging
import numpy as np
from c4a_impl \
    import c4a_d, SaTileShapeConfig
from utils.compare import compare

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

    return block_num, block_table, block_num_each, cache_index

def compute_c4a_with_flash(q, ori_kv, cmp_kv, ori_block_table, cmp_block_table, atten_sinks, seqused_kv, cmp_sparse_indices, softmax_scale, cmp_ratio, win_size):
    """
    计算注意力机制，支持不同批次的序列长度不同
    使用PyTorch实现
    no flash 版本
    """
    # 提取维度信息
    b, s1, n1, dq = q.shape
    dk = dq
    topk = 512
    block_size = cmp_kv.shape[1]
    cmp_kv_block_num = cmp_kv.shape[0]

    if cmp_sparse_indices.ndim > 2:
        cmp_sparse_indices = cmp_sparse_indices.reshape(b * s1, topk)

    atten_out_shape = [b, s1, n1, dk]
    input_dtype = q.dtype
    
    s2_tile = 512

    # 初始化输出张量
    attention_output = torch.zeros(atten_out_shape, dtype=input_dtype)
    
    atten_sinks_2d = atten_sinks.unsqueeze(-1)
    cmp_kv_2d = torch.reshape(cmp_kv, [cmp_kv_block_num * block_size, dk])

    for b_idx in range(b):
        act_seq = seqused_kv[b_idx]
        valid_data_len = min(win_size + s1 - 1, act_seq)  
        for s1_idx in range(s1):
            cur_seq_c4a = min(max(act_seq - s1 + 1 + s1_idx, 0), topk)
            valid_end_pos = valid_data_len - s1 + s1_idx
            valid_start_pos = valid_end_pos - min(win_size - 1, valid_end_pos)
            valid_win_len = valid_end_pos - valid_start_pos + 1

            bn_per_batch = math.ceil(cur_seq_c4a / s2_tile) + 1 # 1为wfa的循环

            qi = q[b_idx, s1_idx, :, :]

            sum_update = torch.zeros([n1, 1], dtype=torch.float32)
            oi_update = torch.zeros([n1, dq], dtype=torch.float32)
            max_update = torch.full((n1, 1), float('-inf'))
            for s2_idx in range(bn_per_batch):
                if s2_idx == 0:
                    start_block = valid_start_pos // block_size
                    end_block = valid_end_pos // block_size
                    kv_list = []
                    for block_idx in range(start_block, end_block + 1):
                        physical_block_id = ori_block_table[b_idx, block_idx]
                        kv_block = ori_kv[physical_block_id, :, 0, :]
                        kv_list.append(kv_block)

                    kv_cur = torch.cat(kv_list, axis=0)
                    kv_cur = kv_cur[valid_start_pos : valid_start_pos + valid_win_len, :]

                    sij = torch.matmul(qi.to(torch.float32), kv_cur.to(torch.float32).transpose(1, 0)) # [n1, win_size]
                    sij_scale = sij * softmax_scale  # [n1, win_size]
                    tilda_mij = torch.max(sij_scale, dim=-1, keepdims=True)[0] # [n1, 1]
                    tilda_pij = torch.exp(sij_scale - tilda_mij) # [n1, win_size]

                    sum_update = torch.sum(tilda_pij, dim=-1, keepdims=True) # [n1, 1]
                    max_update = tilda_mij
                    
                    mm2_res = torch.matmul(tilda_pij, kv_cur.to(torch.float32)) #[n1, d]
                    oi_update = mm2_res
                else:
                    cmp_s2_idx = s2_idx - 1
                    s2_tile_cur = min(s2_tile, cur_seq_c4a - cmp_s2_idx * s2_tile)
                    s2_start = s2_tile * cmp_s2_idx
                    s2_end = s2_start + s2_tile_cur

                    topk_indices_tmp = cmp_sparse_indices[b_idx * s1 + s1_idx, s2_start:s2_end]

                    kv_cur = torch.zeros([s2_tile_cur, dk], dtype=cmp_kv_2d.dtype)

                    # 当前b&s1&s2 topk_index  --->  kvCache的offset
                    for cur_s2_idx in range(s2_tile_cur):
                        topk_index = topk_indices_tmp[cur_s2_idx]
                        block_idx_in_batch = topk_index // block_size
                        slc_block_idx = cmp_block_table[b_idx, block_idx_in_batch]
                        tail = topk_index % block_size
                        slc_idx = slc_block_idx * block_size + tail
                        kv_cur[cur_s2_idx, :] = cmp_kv_2d[slc_idx, :]

                    # C1
                    sij = torch.matmul(qi.to(torch.float32), kv_cur.transpose(1, 0).to(torch.float32)).to(torch.float32)

                    # V1
                    sij_scale = sij * softmax_scale # (n1, s2_tile)
                    tilda_mij = sij_scale.amax(dim=-1, keepdims=True) # (n1, 1)
                    max_new = torch.maximum(max_update,tilda_mij)
                    t_sub = sij_scale - max_new # (n1, s2_tile)
                    tilda_pij = torch.exp(t_sub) # (n1, s2_tile)
                    sum_local = tilda_pij.sum(dim=-1, keepdims=True)# (n1, 1)
                    
                    t_sub2 = max_update - max_new
                    max_update = max_new
                    update_mul = torch.exp(t_sub2)
                    sum_update = sum_update * update_mul + sum_local
                    
                    # C2
                    mm2_res = torch.matmul(tilda_pij.to(torch.float32), kv_cur.to(torch.float32)).to(torch.float32)
                    
                    oi_update = oi_update * update_mul + mm2_res

            sub_res = atten_sinks_2d - max_update
            exp_res = torch.exp(sub_res)
            sum_total = sum_update + exp_res
            atten_out_part = (oi_update / sum_total).to(input_dtype)
            attention_output[b_idx, s1_idx, :, :] = atten_out_part
    return attention_output

def gen_c4a_golden(dtype, bn1n2s1, actual_seq):
    block_size = 128
    torch.manual_seed(42)
    b, n_q, n_kv, s_q = bn1n2s1  # 48, 128, 1, 1
    d = 512
    # qk_rope_dim = 64
    topk = 512
    np.random.seed(None)
    # q head dim
    softmax_scale = d ** -0.5

    s_kv_max = max(actual_seq)

    # 1. 定义shape
    shape_q = [b, s_q, n_q, d]

    shape_atten_sink = [n_q]

    block_num, cmp_block_table, block_num_per_batch, _ = gen_block_table(torch.tensor(actual_seq), block_size, s_q, need_indices=False)
    topk_indices = torch.zeros(b, s_q, topk).to(torch.int32)
    slc_actual_seq = []
    for i in range(b):
        slc_actual_seq.append(min(actual_seq[i], topk))

    for b_i in range(b):
        for s_q_i in range(s_q):

            if slc_actual_seq[b_i] < topk:
                topk_indices[b_i, s_q_i, :slc_actual_seq[b_i]] = torch.arange(0, slc_actual_seq[b_i])
            else:
                perm = torch.randperm(slc_actual_seq[b_i])
                topk_indices[b_i, s_q_i, :] = perm[:topk]

    topk_indices = topk_indices.reshape(b * s_q, n_kv * topk)

    q_bsnd = gen_uniform_data(shape_q, -1, 1, dtype)
    cmp_kv = gen_uniform_data([block_num, block_size, d], -1, 1, dtype)

    # gen ori_block_table
    ori_block_table_shape = [b, math.ceil(s_kv_max / block_size)]
    block_idx_list = torch.randperm(block_num, dtype=torch.int32)
    ori_block_table = [-1] * ori_block_table_shape[1]

    ori_block_table = torch.tile(torch.tensor(ori_block_table).to(torch.int32), (ori_block_table_shape[0], 1))
    block_table_batch_idx = 0
    block_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            ori_block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1

    # gen ori_kv
    ori_kv = torch.zeros([block_num, block_size, n_kv, d], dtype=dtype)
    kv_bsnd = gen_uniform_data([b, s_kv_max, n_kv, d], -1, 1, dtype)
    for b_idx in range(b):
        for block_i, kv_cache_blk_id in enumerate(ori_block_table[b_idx]):
            block_offset = block_i * block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                kv_valid = kv_bsnd[b_idx, block_offset:(block_offset + block_size), :, :]
                ori_kv[kv_cache_blk_id, 0: kv_valid.shape[0], :, :] = kv_bsnd[b_idx, block_offset:(block_offset + block_size), :, :]

    atten_sinks = gen_uniform_data(shape_atten_sink, -1, 1, torch.float32)

    win_size = 128
    cmp_ratio = 4
    # 计算attention
    atten_out = compute_c4a_with_flash(q_bsnd, ori_kv, cmp_kv, ori_block_table, cmp_block_table, atten_sinks, actual_seq, topk_indices, softmax_scale, cmp_ratio, win_size)

    # input params
    input_data_map = [q_bsnd, ori_kv, cmp_kv, ori_block_table, cmp_block_table, atten_sinks, actual_seq, topk_indices, softmax_scale, cmp_ratio, win_size]

    return input_data_map, atten_out


def do_test_c4a_func(bn1n2s1, actual_seq, input_data, atten_out, is_p):
    b, n1, n2, s1 = bn1n2s1

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    if is_p:
        tile_config = SaTileShapeConfig(
            g_tile=64,
            s_kv_tile=2048,
            c1_tile_shape=[128, 128, 128, 512, 128, 128],
            v1_tile_shape=[32, 512],
            c2_tile_shape=[128, 128, 128, 512, 128, 128], # C1的N轴与C2的K轴一致
            v2_tile_shape=[64, 128]
        )
    else:
        tile_config = SaTileShapeConfig(
            g_tile=64,
            s_kv_tile=2048,
            c1_tile_shape=[128, 128, 128, 512, 128, 128],
            v1_tile_shape=[32, 512],
            c2_tile_shape=[128, 128, 128, 512, 128, 128],
            v2_tile_shape=[64, 128]
        )

    q_bsnd, ori_kv, cmp_kv, ori_block_table, cmp_block_table, atten_sinks, actual_seq, topk_indices, softmax_scale, cmp_ratio, win_size = input_data
    kv_act_seqs = torch.tensor(actual_seq, dtype=torch.int32)

    
    q_npu = q_bsnd.npu()
    q_pto = pypto.from_torch(q_npu, dynamic_axis=[0], name="q")
    ori_kv_npu = ori_kv.npu()
    ori_kv_pto = pypto.from_torch(ori_kv_npu, name="ori_kv")
    cmp_kv_npu = cmp_kv.npu()
    cmp_kv_pto = pypto.from_torch(cmp_kv_npu, name="cmp_kv")
    ori_block_table_npu = ori_block_table.npu()
    ori_block_table_pto = pypto.from_torch(ori_block_table_npu, name="ori_block_table")
    cmp_block_table_npu = cmp_block_table.npu()
    cmp_block_table_pto = pypto.from_torch(cmp_block_table_npu, name="cmp_block_table")
    atten_sinks_npu = atten_sinks.npu()
    atten_sinks_pto = pypto.from_torch(atten_sinks_npu, name="atten_sink")
    cmp_sparse_indices = topk_indices.npu()
    cmp_sparse_indices_pto = pypto.from_torch(cmp_sparse_indices, dynamic_axis=[0], name="topk_indices")
    kv_act_seqs_npu = kv_act_seqs.npu()
    kv_act_seqs_pto = pypto.from_torch(kv_act_seqs_npu, dynamic_axis=[0], name="kv_act_seqs")

    
    calc_attention_out = torch.zeros(q_bsnd.shape, dtype=q_bsnd.dtype)
    calc_attention_out_npu = calc_attention_out.npu()
    calc_attention_out_pto = pypto.from_torch(calc_attention_out_npu, dynamic_axis=[0], name="calc_attention_out")

    pto_inputs = [q_pto, ori_kv_pto, cmp_kv_pto, ori_block_table_pto, cmp_block_table_pto, atten_sinks_pto, kv_act_seqs_pto, cmp_sparse_indices_pto]
    pto_outputs = [calc_attention_out_pto]

    # if is_p:
    #     sparse_flash_attention_p(*pto_inputs, *pto_outputs, n_q, n_kv, softmax_scale, topk, block_size,
    #                                        max_blocknum_perbatch, tile_config)
    # else:
    c4a_d(*pto_inputs, *pto_outputs, softmax_scale, cmp_ratio, win_size, tile_config)

    torch_npu.npu.synchronize()
    compare(calc_attention_out_npu.cpu(), atten_out, "atten_out", atol=0.0001, rtol=0.005, max_error_count=100)

def get_case_config(case_name: str):
    # case参数配置字典，key为case名称，value为对应的参数元组(bn1n2s1, is_kn_quant, actual_seq)
    test_case_config = {
        "c4a_bf16_b4_s2_seq64K_total_int8_d": (
            (4, 64, 1, 2), 0, [65536, 16381, 666, 15]
        ),
        "c4a_bf16_b4_s2_seq64K_per_int8_d": (
            (4, 64, 1, 2), 0, [65536] * 4
        ),
        "c4a_bf16_b4_s4_seq64K_per_int8_d": (
            (4, 64, 1, 4), 0, [65536] * 4
        ),
        "c4a_bf16_b1_s256_seq64K_int8_p": (
            (1, 64, 1, 256), 0, [65536]
        ),
        "c4a_bf16_b1_s1_seq64K_int8_d": (
            (1, 64, 1, 1), 0, [65536]
        )
    }
    case_config = test_case_config.get(case_name)
    return case_config


def do_test_c4a_entry(case_name: str, is_p: bool,  is_acl_graph: bool = False):
    case_config = get_case_config(case_name)
    if not case_config:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    bn1n2s1, is_kn_quant, actual_seq = case_config

    input_data, atten_out = gen_c4a_golden(
        torch.bfloat16, bn1n2s1, actual_seq
    )
    # if is_acl_graph:
    #     do_test_c4a_func_acl_graph(
    #         bn1n2s1, actual_seq, input_params, input_data, atten_out, is_p
    #     )
    # else:
    do_test_c4a_func(
        bn1n2s1, actual_seq, input_data, atten_out, is_p
    )
    return True


def test_c4a_bf16_b4_s2_seq64k_total_d():
    '''
    sfa decode测试函数
    '''
    do_test_c4a_entry("c4a_bf16_b4_s2_seq64K_total_int8_d", is_p=False)


# @pytest.mark.skip(reason="perf")
def test_c4a_bf16_b4_s2_seq64k_per_d():
    '''
    sfa decode测试函数
    '''
    do_test_c4a_entry("c4a_bf16_b4_s2_seq64K_per_int8_d", is_p=False)


# @pytest.mark.skip(reason="perf")
def test_c4a_bf16_b4_s4_seq64k_per_d():
    '''
    sfa decode测试函数
    '''
    do_test_c4a_entry("c4a_bf16_b4_s4_seq64K_per_int8_d", is_p=False)


# @pytest.mark.skip(reason="acl graph perf")
def test_c4a_bf16_b4_s4_seq64k_per_graph_d():
    '''
    sfa decode测试函数
    '''
    do_test_c4a_entry("c4a_bf16_b4_s4_seq64K_per_int8_d", is_p=False, is_acl_graph=True)


# @pytest.mark.skip(reason="large test case")
def test_c4a_bf16_b1_s256_seq64k_p():
    '''
    sfa prefill测试函数
    '''
    do_test_c4a_entry("c4a_bf16_b1_s256_seq64K_int8_p", is_p=True)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    # test_c4a_bf16_b4_s2_seq64k_total_d()
    # test_c4a_bf16_b4_s2_seq64k_per_d()
    # test_c4a_bf16_b1_s256_seq64k_p()
    # test_c4a_bf16_b4_s4_seq64k_per_d()
    do_test_c4a_entry("c4a_bf16_b1_s1_seq64K_int8_d", is_p=False)
