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
import math
import os
import torch
import pypto

from utils.compare import compare
from win_attention_impl import deepseekv4_win_atten


class MM(torch.nn.Module):
    def forward(
        self,  
        q_tnd: torch.Tensor,
        block_table: torch.Tensor,
        kv_cache: torch.Tensor,
        start_pos_list: torch.Tensor,
        atten_sink: torch.Tensor,
        atten_out: torch.Tensor,
        win_size: int
    ):
        deepseekv4_win_atten(q_tnd, block_table, kv_cache, start_pos_list, atten_sink, atten_out, win_size)
        return atten_out


def gen_uniform_data(data_shape, min_value, max_value, dtypes, device_id):
    if isinstance(data_shape, list):
        data_shape = tuple(data_shape)
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtypes, device=f'npu:{device_id}')
    if dtypes == torch.bool:
        return torch.randint(0, 2, size=data_shape, dtype=torch.bool, device=f'npu:{device_id}')
    return torch.rand(data_shape, dtype=dtypes, device=f'npu:{device_id}').uniform_(min_value, max_value)


def softmax(x):
    x = x.to(torch.float32)
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = torch.sum(y, dim=-1, keepdims=True)
    ans = y
    return ans / x_sum


def softmax_atten_sink(x, atten_sink):
    x = x.to(torch.float32)
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = torch.sum(y, dim=-1, keepdims=True)
    ans = y
    atten_sink = atten_sink.reshape([-1, 1])
    return ans / (x_sum + atten_sink) # [n_q] fp32


def win_atten_calc_tnd(input_params_win_attn, start_pos_list, atten_sink, q_tnd, kv_cache, block_table, device_id):

    t = input_params_win_attn[0]
    n_q = input_params_win_attn[2]
    d_q = input_params_win_attn[3]
    win = input_params_win_attn[4]
    scalar = input_params_win_attn[5]
    atten_out_shape = [t, n_q, 512]
    atten_out = torch.zeros(atten_out_shape, dtype=torch.float32, device=f'npu:{device_id}')
    block_size = kv_cache.shape[1]
    b = len(start_pos_list)
    s_q = t // b

    # q_tnd [b, s_q, n_q, d_q]   kv_cache [block_num, block_size, n_kv, d_kv]
    for t_index in range(t):
        b_index = t_index // s_q
        s1_index = t_index % s_q

        start_pos = start_pos_list[b_index]
        q_tensor_cur = q_tnd[t_index:(t_index + 1), :, :].reshape(n_q, d_q)

        cur_loc = start_pos + s1_index + 1
        valid_len = min(cur_loc, win)
        cur_start_pos = cur_loc - valid_len
        end_pos = cur_loc
        start_block = cur_start_pos // block_size
        start_offset = cur_start_pos % block_size
        end_block = (end_pos - 1) // block_size

        kv_list = []

        for block_idx in range(start_block, end_block + 1):
            physical_block_id = block_table[b_index, block_idx]
            kv_block = kv_cache[physical_block_id, :, 0, :]
            kv_list.append(kv_block)
        
        kv_cur = torch.cat(kv_list, axis=0)
        kv_cur = kv_cur[start_offset : start_offset + win, :]

        qk_mm_res = torch.matmul(q_tensor_cur.to(torch.float32), kv_cur.to(torch.float32).transpose(1, 0))
        
        qk_ele_res = qk_mm_res * scalar

        # softmax_out = softmax(qk_ele_res)
        softmax_out = softmax_atten_sink(qk_ele_res, atten_sink)

        mm2_res = torch.matmul(softmax_out, kv_cur.to(torch.float32))
        atten_out[t_index:(t_index + 1), :, :] = mm2_res

    return atten_out


def gen_win_attn_data_tnd(t, n_q, d_q, n_kv, d_kv, block_size, start_pos_list, dtypes, device_id):
    torch.manual_seed(42)
    b = len(start_pos_list)
    s_q = t // b
    new_dtype = dtypes

    actual_seq_max = max(start_pos_list)
    s_kv_max = actual_seq_max + s_q
    shape_q = [t * n_q, d_q]
    shape_kv = [b, s_kv_max, n_kv, d_kv]
    atten_out_shape = [t, n_q, d_kv]
    block_num_per_batch = []
    block_num_min = 0
    block_num = 0

    atten_sink = gen_uniform_data([n_q], -1, 1, torch.float32, device_id)

    # gen q k v data
    q = gen_uniform_data(shape_q, -1, 1, new_dtype, device_id)
    q_tnd = q.reshape(t, n_q, d_q)
    kv_bsnd = gen_uniform_data(shape_kv, -1, 1, new_dtype, device_id)

    for start_pos in start_pos_list:
        actual_seq = start_pos + s_q
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
        block_num_min += math.ceil(actual_seq / block_size)

    # gen block table
    block_table_shape = [b, math.ceil(s_kv_max / block_size)]
    block_num = block_num_min
    block_idx_list = torch.randperm(block_num, dtype=torch.int32)
    block_idx = 0
    
    # invalid block_id set as -1
    block_table = [-1] * block_table_shape[1]

    block_table = torch.tile(torch.tensor(block_table, device=f'npu:{device_id}').to(torch.int32), (block_table_shape[0], 1))
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1

    kv_cache = torch.zeros([block_num, block_size, n_kv, d_kv], dtype=new_dtype, device=f'npu:{device_id}')
    for b_idx in range(b):
        for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
            block_offset = block_i * block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                kv_valid = kv_bsnd[b_idx, block_offset:(block_offset + block_size), :, :]
                kv_cache[kv_cache_blk_id, 0: kv_valid.shape[0], :, :] = kv_bsnd[b_idx, block_offset:(block_offset + block_size), :, :]

    atten_out = torch.zeros(atten_out_shape, dtype=torch.float32, device=f'npu:{device_id}')
    return q_tnd, block_table, kv_cache, atten_sink, atten_out


def gen_win_attn_data_c128(t, n_q, d_q, n_kv, d_kv, block_size, actual_seq_list, dtypes, device_id):
    torch.manual_seed(42)
    b = len(actual_seq_list)
    new_dtype = dtypes

    actual_seq_max = max(actual_seq_list)
    s_kv_max = actual_seq_max
    shape_q = [t * n_q, d_q]
    shape_kv = [b, s_kv_max, n_kv, d_kv]
    atten_out_shape = [t, n_q, d_kv]
    block_num_per_batch = []
    block_num_min = 0
    block_num = 0

    atten_sink = gen_uniform_data([n_q], -1, 1, torch.float32, device_id)

    # gen q k v data
    q = gen_uniform_data(shape_q, -1, 1, new_dtype, device_id)
    q_tnd = q.reshape(t, n_q, d_q)
    kv_bsnd = gen_uniform_data(shape_kv, -1, 1, new_dtype, device_id)

    for actual_seq in actual_seq_list:
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
        block_num_min += math.ceil(actual_seq / block_size)

    # gen block table
    block_table_shape = [b, math.ceil(s_kv_max / block_size)]
    block_num = block_num_min
    block_idx_list = torch.randperm(block_num, dtype=torch.int32)
    block_idx = 0
    
    # invalid block_id set as -1
    block_table = [-1] * block_table_shape[1]

    block_table = torch.tile(torch.tensor(block_table, device=f'npu:{device_id}').to(torch.int32), (block_table_shape[0], 1))
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1

    kv_cache = torch.zeros([block_num, block_size, n_kv, d_kv], dtype=new_dtype, device=f'npu:{device_id}')
    for b_idx in range(b):
        for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
            block_offset = block_i * block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                kv_valid = kv_bsnd[b_idx, block_offset:(block_offset + block_size), :, :]
                kv_cache[kv_cache_blk_id, 0: kv_valid.shape[0], :, :] = kv_bsnd[b_idx, block_offset:(block_offset + block_size), :, :]

    atten_out = torch.zeros(atten_out_shape, dtype=torch.float32, device=f'npu:{device_id}')
    return q_tnd, block_table, kv_cache, atten_sink, atten_out


def win_atten_calc_tnd_decode(input_params_win_attn, start_pos_list, atten_sink, q_tnd, kv_cache, block_table, device_id):

    t = input_params_win_attn[0]
    n_q = input_params_win_attn[2]
    d_q = input_params_win_attn[3]
    win = input_params_win_attn[4]
    scalar = input_params_win_attn[5]
    atten_out_shape = [t, n_q, 512]
    atten_out = torch.zeros(atten_out_shape, dtype=torch.float32, device=f'npu:{device_id}')
    block_size = kv_cache.shape[1]
    b = len(start_pos_list)
    s_q = t // b
    
    # q_tnd [b, s_q, n_q, d_q]   kv_cache [block_num, block_size, n_kv, d_kv]
    for t_index in range(t):
        b_index = t_index // s_q
        s1_index = t_index % s_q

        start_pos = start_pos_list[b_index]
        q_tensor_cur = q_tnd[t_index:(t_index + 1), :, :].reshape(n_q, d_q)

        cur_loc = start_pos + s1_index + 1

        actual_win_size = min(start_pos + 1, win)

        # valid_len = min(cur_loc, win)

        # print("actual_win_size: ", actual_win_size)

        cur_start_pos = cur_loc - actual_win_size
        end_pos = cur_loc

        # print("cur_start_pos: ", cur_start_pos, ", end_pos: ", end_pos)

        start_block = cur_start_pos // block_size
        start_offset = s1_index
        end_block = (end_pos - 1) // block_size

        # print("start_block: ", start_block, ", start_offset: ", start_offset, ", end_block: ", end_block)

        kv_list = []

        for block_idx in range(start_block, end_block + 1):
            physical_block_id = block_table[b_index, block_idx]
            # print("b:", b_index, ", block_idx:", block_idx, ", physical_block_id:",physical_block_id)
            kv_block = kv_cache[physical_block_id, :, 0, :]
            kv_list.append(kv_block)

        kv_cur = torch.cat(kv_list, axis=0)
        kv_cur = kv_cur[start_offset : start_offset + actual_win_size, :]
        # print("kv_cur.shape:", kv_cur.shape)

        qk_mm_res = torch.matmul(q_tensor_cur.to(torch.float32), kv_cur.to(torch.float32).transpose(1, 0))
        
        qk_ele_res = qk_mm_res * scalar

        # softmax_out = softmax(qk_ele_res)
        softmax_out = softmax_atten_sink(qk_ele_res, atten_sink)

        mm2_res = torch.matmul(softmax_out, kv_cur.to(torch.float32))
        atten_out[t_index:(t_index + 1), :, :] = mm2_res
        # print(" ")

    return atten_out


def win_atten_calc_c128(input_params_win_attn, actual_seq_list, atten_sink, q_tnd, kv_cache, block_table, device_id):

    t = input_params_win_attn[0]
    n_q = input_params_win_attn[2]
    d_q = input_params_win_attn[3]
    win = input_params_win_attn[4]
    scalar = input_params_win_attn[5]
    atten_out_shape = [t, n_q, 512]
    atten_out = torch.zeros(atten_out_shape, dtype=torch.float32, device=f'npu:{device_id}')
    b = len(actual_seq_list)
    s_q = t // b
    
    # q_tnd [b, s_q, n_q, d_q]   kv_cache [block_num, block_size, n_kv, d_kv]
    for t_index in range(t):
        b_index = t_index // s_q
        s1_index = t_index % s_q

        actual_seq = actual_seq_list[b_index]
        q_tensor_cur = q_tnd[t_index:(t_index + 1), :, :].reshape(n_q, d_q)

        actual_win_size = min(actual_seq, win)
        start_offset = s1_index

        physical_block_id = block_table[b_index, 0]
        kv_cur = kv_cache[physical_block_id, :, 0, :]
        kv_cur = kv_cur[start_offset : start_offset + actual_win_size, :]

        sum_exp = torch.zeros([n_q, 1], dtype=torch.float32, device=f'npu:{device_id}')
        acc_o = torch.zeros([n_q, d_q], dtype=torch.float32, device=f'npu:{device_id}')
        scores_max = torch.full((n_q, 1), float('-inf'), device=f'npu:{device_id}')

        acc_s = torch.matmul(q_tensor_cur.to(torch.float32), kv_cur.to(torch.float32).transpose(1, 0)) # [n_q, win_size]
        acc_s = acc_s * scalar  # [n_q, win_size]
        scores_max_prev = scores_max    #[n_q, 1]
        scores_max = torch.max(acc_s, dim=-1, keepdims=True)[0] # [n_q, 1]
        scores_scale = torch.exp(scores_max_prev - scores_max)  # [n_q, 1]
        acc_s = torch.exp(acc_s - scores_max) # [n_q, win_size]
        scores_sum = torch.sum(acc_s, dim=-1, keepdims=True) # [n_q, 1]
        mul_res = sum_exp * scores_scale
        sum_exp = mul_res + scores_sum
        acc_o *= scores_scale # [n_q, d_q]
        acc_o += torch.matmul(acc_s, kv_cur.to(torch.float32)) #[n_q, d]

        sum_exp += torch.exp(atten_sink.reshape(n_q, 1) - scores_max)
        acc_o /= sum_exp
        atten_out[t_index:(t_index + 1), :, :] = acc_o

    return atten_out


def test_win_atten_c128() -> None:
    
    for b in [4]:
        for s_q in [1]:
            t = b * s_q
            win_size = 128
            n_q = 64
            block_size = 128
            n_kv = 1
            dtypes = torch.bfloat16
            head_dim = 512
            d_q = head_dim
            d_kv = head_dim
            softmax_scale = d_q ** -0.5
            input_params_win_attn = [t, n_kv, n_q, d_q, win_size, softmax_scale]

            device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
            torch.npu.set_device(device_id)

            torch.manual_seed(42)
            seqused_kv_list = torch.randint(s_q + 1, win_size + s_q, (1, ), device=f'npu:{device_id}').tolist()
            seqused_kv_list += [win_size + s_q - 1] * (b - 1)
            seqused_kv_list_tensor = torch.tensor(seqused_kv_list, dtype=torch.int32, device=f'npu:{device_id}')
            print("seqused_kv_list:", seqused_kv_list)

            q, ori_block_table, ori_kv, attn_sinks, atten_out = gen_win_attn_data_c128(t, n_q, d_q, n_kv, d_kv, block_size, seqused_kv_list_tensor, dtypes, device_id)
            
            deepseekv4_win_atten(q, ori_block_table, ori_kv, seqused_kv_list_tensor, attn_sinks, atten_out, win_size, is_decode=True, is_c128=True)

            golden = win_atten_calc_c128(input_params_win_attn, seqused_kv_list, attn_sinks, q, ori_kv, ori_block_table, device_id)
            from utils.np_compare import detailed_allclose_manual as compare
            threhold = 5e-4
            compare(golden, atten_out, "SWA decode atten_out", rtol=threhold, atol=threhold)


def test_win_atten_decode() -> None:
    
    for b in [4]:
        for s_q in [4]:
            t = b * s_q
            win = 128
            n_q = 64
            block_size = 128
            n_kv = 1
            dtypes = torch.bfloat16
            head_dim = 512
            d_q = head_dim
            d_kv = head_dim
            scalar = d_q ** -0.5
            input_params_win_attn = [t, n_kv, n_q, d_q, win, scalar]

            device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
            torch.npu.set_device(device_id)

            torch.manual_seed(42)
            actual_seq_list = torch.randint(s_q + 1, win + s_q, (b // 2, ), device=f'npu:{device_id}').tolist()
            actual_seq_list += [win + s_q - 1] * (b - b // 2)
            print("actual_seq_list:", actual_seq_list)

            start_pos_list = [item - s_q for item in actual_seq_list]
            start_pos_list_tensor = torch.tensor(start_pos_list, dtype=torch.int32, device=f'npu:{device_id}')

            q_tnd, block_table, kv_cache, atten_sink, atten_out = gen_win_attn_data_tnd(t, n_q, d_q, n_kv, d_kv, block_size, start_pos_list_tensor, dtypes, device_id)
            print("block_table:")
            print(block_table)

            deepseekv4_win_atten(q_tnd, block_table, kv_cache, start_pos_list_tensor, atten_sink, atten_out, win, is_decode=True)
            
            # model = torch.compile(MM(), backend="eager", dynamic=True)
            # g = torch.npu.NPUGraph()
            # with torch.npu.graph(g):
            #     y = model(q_tnd, block_table, kv_cache, start_pos_list_tensor, atten_sink, atten_out, win)
            # g.replay()

            golden = win_atten_calc_tnd_decode(input_params_win_attn, start_pos_list, atten_sink, q_tnd, kv_cache, block_table, device_id)
            compare(atten_out.cpu(), golden.cpu(), 'y', 0.000025, 0.005, 0.005)


def test_win_atten() -> None:
    
    for b in [4]:
        for s_q in [4]:
            t = b * s_q
            win = 128
            n_q = 64
            block_size = 128
            n_kv = 1
            dtypes = torch.bfloat16
            head_dim = 512
            d_q = head_dim
            d_kv = head_dim
            scalar = d_q ** -0.5
            input_params_win_attn = [t, n_kv, n_q, d_q, win, scalar]

            device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
            torch.npu.set_device(device_id)

            actual_seq_list = [8192] * b

            # start_pos + s1 = actual_seq
            start_pos_list = [item - s_q for item in actual_seq_list]
            start_pos_list_tensor = torch.tensor(start_pos_list, dtype=torch.int32, device=f'npu:{device_id}')

            q_tnd, block_table, kv_cache, atten_sink, atten_out = gen_win_attn_data_tnd(t, n_q, d_q, n_kv, d_kv, block_size, actual_seq_list, dtypes, device_id)
            deepseekv4_win_atten(q_tnd, block_table, kv_cache, start_pos_list_tensor, atten_sink, atten_out, win, is_decode=False)
            
            # model = torch.compile(MM(), backend="eager", dynamic=True)
            # g = torch.npu.NPUGraph()
            # with torch.npu.graph(g):
            #     y = model(q_tnd, block_table, kv_cache, start_pos_list_tensor, atten_sink, atten_out, win)
            # g.replay()

            golden = win_atten_calc_tnd(input_params_win_attn, start_pos_list, atten_sink, q_tnd, kv_cache, block_table, device_id)
            compare(atten_out.cpu(), golden.cpu(), 'y', 0.000025, 0.005, 0.005)


if __name__ == "__main__":
    test_win_atten_c128()