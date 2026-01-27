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
from win_attention_impl import deepseekv4_win_atten


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define("win_attention(Tensor q, Tensor ori_block_table, Tensor ori_kv, Tensor seqused_kv, Tensor attn_sinks, int win_size, bool is_decode, Tensor mask, Tensor actual_seq_list_q) -> (Tensor)")


@torch.library.impl(pyptolib, "win_attention", "Meta")
def win_attention(q, ori_block_table, ori_kv, seqused_kv, attn_sinks, win_size, is_decode, mask, actual_seq_list_q):
    y = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    return y


@torch.library.impl(pyptolib, "win_attention", "NPU")
def win_attention(q, ori_block_table, ori_kv, seqused_kv, attn_sinks, win_size, is_decode, mask, actual_seq_list_q):
    return deepseekv4_win_atten(q, ori_block_table, ori_kv, seqused_kv, attn_sinks, win_size, is_decode, mask, actual_seq_list_q)


class SWA(torch.nn.Module):
    def forward(self, q, ori_block_table, ori_kv, seqused_kv, attn_sinks, win_size, is_decode, mask, actual_seq_list_q):
        return torch.ops.pypto.win_attention(q, ori_block_table, ori_kv, seqused_kv, attn_sinks, win_size, is_decode, mask, actual_seq_list_q)


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


def get_mask2(s_q, n_q, device_id, block_size):
    mask2_left = torch.zeros((s_q * n_q, 128), dtype=torch.uint8, device=f'npu:{device_id}')
    mask2_tail = torch.zeros((s_q * n_q, 128), dtype=torch.uint8, device=f'npu:{device_id}')
    row_indices = torch.arange(s_q, device=f'npu:{device_id}').unsqueeze(1)
    col_indices = torch.arange(block_size * 2, device=f'npu:{device_id}').unsqueeze(0)
    mask2_right = (col_indices >= row_indices) & (col_indices < row_indices + 128).to(torch.uint8)
    mask2_right = mask2_right.unsqueeze(1).expand(-1, n_q, -1).reshape(s_q * n_q, 256)
    mask2 = torch.cat([mask2_left, mask2_right, mask2_tail], -1).to(torch.bool)
    return mask2


def gen_win_attn_data_tnd(t, n_q, d_q, n_kv, d_kv, block_size, seqused_kv_list, dtypes, device_id):
    torch.manual_seed(42)
    b = len(seqused_kv_list)
    new_dtype = dtypes
    actual_seq_max = max(seqused_kv_list)
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

    for actual_seq in seqused_kv_list:
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

    atten_out = torch.zeros(atten_out_shape, dtype=new_dtype, device=f'npu:{device_id}')
    return q_tnd, block_table, kv_cache, atten_sink, atten_out


def gen_win_attn_data_bsnd(t, n_q, d_q, n_kv, d_kv, block_size, actual_seq_list, dtypes, device_id):
    torch.manual_seed(42)
    b = len(actual_seq_list)
    new_dtype = dtypes

    actual_seq_max = max(actual_seq_list)
    s_kv_max = actual_seq_max
    shape_q = [t * n_q, d_q]
    shape_kv = [b, s_kv_max, n_kv, d_kv]
    atten_out_shape = [b, t // b, n_q, d_kv]
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

    atten_out = torch.zeros(atten_out_shape, dtype=dtypes, device=f'npu:{device_id}')
    return q_tnd.reshape(b, t // b, n_q, d_q), block_table, kv_cache, atten_sink, atten_out


def win_atten_calc_tnd_prefill(input_params_win_attn, seqused_kv_list, atten_sink, q_tnd, kv_cache, block_table, actual_seq_list_q, device_id):

    t = input_params_win_attn[0]
    n_q = input_params_win_attn[2]
    d_q = input_params_win_attn[3]
    win = input_params_win_attn[4]
    scalar = input_params_win_attn[5]
    atten_out_shape = [t, n_q, d_q]
    atten_out = torch.zeros(atten_out_shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    block_size = kv_cache.shape[1]
    b = len(seqused_kv_list)

    for b_index in range(b):
        cur_s_q = actual_seq_list_q[b_index + 1] - actual_seq_list_q[b_index]

        for s1_index in range(cur_s_q):

            t_index = actual_seq_list_q[b_index] + s1_index

            actual_seq = seqused_kv_list[b_index]
            q_tensor_cur = q_tnd[t_index:(t_index + 1), :, :].reshape(n_q, d_q)

            cur_loc = actual_seq - cur_s_q + s1_index + 1
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
            kv_cur = kv_cur[start_offset : start_offset + valid_len, :]

            sum_exp = torch.zeros([n_q, 1], dtype=torch.float32, device=f'npu:{device_id}')
            acc_s = torch.matmul(q_tensor_cur.to(torch.float32), kv_cur.to(torch.float32).transpose(1, 0)) # [n_q, win_size]
            acc_s = acc_s * scalar  # [n_q, win_size]
            scores_max = torch.max(acc_s, dim=-1, keepdims=True)[0] # [n_q, 1]
            acc_s = torch.exp(acc_s - scores_max) # [n_q, win_size]
            sum_exp = torch.sum(acc_s, dim=-1, keepdims=True) # [n_q, 1]
            sum_exp += torch.exp(atten_sink.reshape(n_q, 1) - scores_max)
            v1_res = acc_s / sum_exp
            v1_res = v1_res.to(torch.bfloat16)
            mm2_res = torch.matmul(v1_res, kv_cur) #[n_q, d]

            atten_out[t_index:(t_index + 1), :, :] = mm2_res

    return atten_out


def win_atten_calc_mtp_decode_bsnd(input_params_win_attn, actual_seq_list, atten_sink, q, kv_cache, block_table, device_id):

    t = input_params_win_attn[0]
    n_q = input_params_win_attn[2]
    d_q = input_params_win_attn[3]
    win = input_params_win_attn[4]
    scalar = input_params_win_attn[5]
    b = len(actual_seq_list)
    s_q = t // b
    atten_out_shape = [b, s_q, n_q, d_q]
    atten_out = torch.zeros(atten_out_shape, dtype=q.dtype, device=f'npu:{device_id}')
    block_size = kv_cache.shape[1]
    for b_index in range(b):
        for s1_index in range(s_q):
            actual_seq = actual_seq_list[b_index]

            valid_data_len = min(win + s_q - 1, actual_seq)
            valid_end_pos = valid_data_len - 1 - (s_q - s1_index - 1)
            valid_start_pos = valid_end_pos - min(win - 1, valid_end_pos)
            valid_win_len = valid_end_pos - valid_start_pos + 1

            q_tensor_cur = q[b_index, s1_index, :, :].reshape(n_q, d_q)

            start_block = valid_start_pos // block_size
            end_block = valid_end_pos // block_size

            kv_list = []

            for block_idx in range(start_block, end_block + 1):
                physical_block_id = block_table[b_index, block_idx]
                kv_block = kv_cache[physical_block_id, :, 0, :]
                kv_list.append(kv_block)

            kv_cur = torch.cat(kv_list, axis=0)
            kv_cur = kv_cur[valid_start_pos : valid_start_pos + valid_win_len, :]

            sum_exp = torch.zeros([n_q, 1], dtype=torch.float32, device=f'npu:{device_id}')

            acc_s = torch.matmul(q_tensor_cur.to(torch.float32), kv_cur.to(torch.float32).transpose(1, 0)) # [n_q, win_size]
            acc_s = acc_s * scalar  # [n_q, win_size]
            scores_max = torch.max(acc_s, dim=-1, keepdims=True)[0] # [n_q, 1]
            acc_s = torch.exp(acc_s - scores_max) # [n_q, win_size]
            sum_exp = torch.sum(acc_s, dim=-1, keepdims=True) # [n_q, 1]
            sum_exp += torch.exp(atten_sink.reshape(n_q, 1) - scores_max)
            v1_res = acc_s / sum_exp
            v1_res = v1_res.to(torch.bfloat16)
            mm2_res = torch.matmul(v1_res, kv_cur) #[n_q, d]
           
            atten_out[b_index, s1_index, :, :] = mm2_res

    return atten_out


def win_atten_calc_mtp_decode_tnd(input_params_win_attn, actual_seq_list, atten_sink, q, kv_cache, block_table, actual_seq_list_q, device_id):

    t = input_params_win_attn[0]
    n_q = input_params_win_attn[2]
    d_q = input_params_win_attn[3]
    win = input_params_win_attn[4]
    scalar = input_params_win_attn[5]
    b = len(actual_seq_list)
    atten_out_shape = [t, n_q, d_q]
    atten_out = torch.zeros(atten_out_shape, dtype=q.dtype, device=f'npu:{device_id}')
    block_size = kv_cache.shape[1]
    for b_index in range(b):
        s_q = actual_seq_list_q[b_index + 1] - actual_seq_list_q[b_index]
        
        for s1_index in range(s_q):
            t_index = actual_seq_list_q[b_index] + s1_index

            actual_seq = actual_seq_list[b_index]

            valid_data_len = min(win + s_q - 1, actual_seq)
            valid_end_pos = valid_data_len - 1 - (s_q - s1_index - 1)
            valid_start_pos = valid_end_pos - min(win - 1, valid_end_pos)
            valid_win_len = valid_end_pos - valid_start_pos + 1

            q_tensor_cur = q[t_index:(t_index + 1), :, :].reshape(n_q, d_q)

            start_block = valid_start_pos // block_size
            end_block = valid_end_pos // block_size

            kv_list = []

            for block_idx in range(start_block, end_block + 1):
                physical_block_id = block_table[b_index, block_idx]
                kv_block = kv_cache[physical_block_id, :, 0, :]
                kv_list.append(kv_block)

            kv_cur = torch.cat(kv_list, axis=0)
            kv_cur = kv_cur[valid_start_pos : valid_start_pos + valid_win_len, :]

            sum_exp = torch.zeros([n_q, 1], dtype=torch.float32, device=f'npu:{device_id}')

            acc_s = torch.matmul(q_tensor_cur.to(torch.float32), kv_cur.to(torch.float32).transpose(1, 0)) # [n_q, win_size]
            acc_s = acc_s * scalar  # [n_q, win_size]
            scores_max = torch.max(acc_s, dim=-1, keepdims=True)[0] # [n_q, 1]
            acc_s = torch.exp(acc_s - scores_max) # [n_q, win_size]
            sum_exp = torch.sum(acc_s, dim=-1, keepdims=True) # [n_q, 1]
            sum_exp += torch.exp(atten_sink.reshape(n_q, 1) - scores_max)
            v1_res = acc_s / sum_exp
            v1_res = v1_res.to(torch.bfloat16)
            mm2_res = torch.matmul(v1_res, kv_cur) #[n_q, d]
           
            atten_out[t_index, :, :] = mm2_res

    return atten_out


def test_win_atten_tnd_mtp_decode_mask() -> None:
    
    for b in [4]:
        # len(actual_seq_list_q) = b + 1
        # s_val ∈ [1, 4] int
        s_val = 4
        actual_seq_list_q = [i * s_val for i in range(b + 1)]
        t = actual_seq_list_q[-1]
        win_size = 128
        n_q = 64
        block_size = 128
        n_kv = 1
        dtypes = torch.bfloat16
        head_dim = 512
        d_q = head_dim
        d_kv = head_dim
        scalar = d_q ** -0.5
        input_params_win_attn = [t, n_kv, n_q, d_q, win_size, scalar]

        device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
        torch.npu.set_device(device_id)

        seqused_kv_list = [8192] * b
        seqused_kv_list_tensor = torch.tensor(seqused_kv_list, dtype=torch.int32, device=f'npu:{device_id}')
        actual_seq_list_q_tenor = torch.tensor(actual_seq_list_q, dtype=torch.int32, device=f'npu:{device_id}')
        ori_kv_len_list = [min(seqused_kv_list[i], win_size + actual_seq_list_q[i + 1]- actual_seq_list_q[i] - 1) for i in range(len(seqused_kv_list))]

        q_tnd, ori_block_table, ori_kv, atten_sink, _ = gen_win_attn_data_tnd(t, n_q, d_q, n_kv, d_kv, block_size, ori_kv_len_list, dtypes, device_id)

        mask2 = get_mask2(4, n_q, device_id, block_size)
        atten_out_2d = deepseekv4_win_atten(q_tnd, ori_block_table, ori_kv, seqused_kv_list_tensor, atten_sink, win_size, is_decode=True, mask=mask2, actual_seq_list_q=actual_seq_list_q_tenor)

        # import torchair as tng
        # from torchair.configs.compiler_config import CompilerConfig
        # compiler_config = CompilerConfig()
        # compiler_config.mode = "reduce-overhead"
        # npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
        # model = torch.compile(SWA(), dynamic=False, fullgraph=True, backend=npu_backend)
        # q_npu = q_tnd.npu()
        # ori_block_table_npu = ori_block_table.npu()
        # ori_kv_npu = ori_kv.npu()
        # seqused_kv_list_tensor_npu = seqused_kv_list_tensor.npu()
        # attn_sinks_npu = atten_sink.npu()
        # mask2_npu = mask2.npu()
        # actual_seq_list_q_tenor_npu = actual_seq_list_q_tenor.npu()
        # atten_out_2d = model(q_npu, ori_block_table_npu, ori_kv_npu, seqused_kv_list_tensor_npu, attn_sinks_npu, win_size, True, mask2_npu, actual_seq_list_q_tenor_npu)
        # pypto.runtime._device_synchronize()

        golden = win_atten_calc_mtp_decode_tnd(input_params_win_attn, seqused_kv_list, atten_sink, q_tnd, ori_kv, ori_block_table, actual_seq_list_q, device_id)
        from utils.np_compare import detailed_allclose_manual as compare
        atten_out = torch.reshape(atten_out_2d, [t, n_q, d_q])
        compare(golden, atten_out, "SWA decode tnd mask 版本", rtol=0.0078125, atol=0.0001)


def test_win_atten_bsnd_mtp_decode_mask() -> None:
    
    for b in [4]:
        for s_q in [4]: 

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

            seqused_kv_list = [8192] * b

            seqused_kv_list_tensor = torch.tensor(seqused_kv_list, dtype=torch.int32, device=f'npu:{device_id}')
            ori_kv_len_list = [min(seqused_kv, win_size + s_q - 1) for seqused_kv in seqused_kv_list]
            print("seqused_kv_list:", seqused_kv_list)

            q, ori_block_table, ori_kv, attn_sinks, atten_out = gen_win_attn_data_bsnd(t, n_q, d_q, n_kv, d_kv, block_size, ori_kv_len_list, dtypes, device_id)
            mask2 = get_mask2(s_q, n_q, device_id, block_size)
            atten_out = deepseekv4_win_atten(q, ori_block_table, ori_kv, seqused_kv_list_tensor, attn_sinks, win_size, is_decode=True, mask=mask2, actual_seq_list_q=None)

            golden = win_atten_calc_mtp_decode_bsnd(input_params_win_attn, seqused_kv_list, attn_sinks, q, ori_kv, ori_block_table, device_id)
            from utils.np_compare import detailed_allclose_manual as compare
            compare(golden, atten_out, "SWA decode bnsd mtp mask 版本", rtol=0.0078125, atol=0.0001)


def test_win_atten_bsnd_mtp_decode() -> None:
    
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

            seqused_kv_list = [8192] * b
            seqused_kv_list_tensor = torch.tensor(seqused_kv_list, dtype=torch.int32, device=f'npu:{device_id}')
            ori_kv_len_list = [min(seqused_kv, win_size + s_q - 1) for seqused_kv in seqused_kv_list]
            print("seqused_kv_list:", seqused_kv_list)

            q, ori_block_table, ori_kv, attn_sinks, _ = gen_win_attn_data_bsnd(t, n_q, d_q, n_kv, d_kv, block_size, ori_kv_len_list, dtypes, device_id)
            atten_out = deepseekv4_win_atten(q, ori_block_table, ori_kv, seqused_kv_list_tensor, attn_sinks, win_size, True, None, None)

            golden = win_atten_calc_mtp_decode_bsnd(input_params_win_attn, seqused_kv_list, attn_sinks, q, ori_kv, ori_block_table, device_id)
            from utils.np_compare import detailed_allclose_manual as compare
            compare(golden, atten_out, "SWA decode bnsd mtp 版本", rtol=0.0078125, atol=0.0001)


def test_win_atten_tnd_prefill_mask() -> None:

    for b in [1]:
        # len(actual_seq_list_q) = b + 1
        s_val = 512
        actual_seq_list_q = [i * s_val for i in range(b + 1)]
        t = actual_seq_list_q[-1]
        win_size = 128
        n_q = 64
        block_size = 128
        n_kv = 1
        dtypes = torch.bfloat16
        head_dim = 512
        d_q = head_dim
        d_kv = head_dim
        scalar = d_q ** -0.5
        input_params_win_attn = [t, n_kv, n_q, d_q, win_size, scalar]

        device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
        torch.npu.set_device(device_id)

        seqused_kv_list = [s_val] * b
        seqused_kv_list_tensor = torch.tensor(seqused_kv_list, dtype=torch.int32, device=f'npu:{device_id}')
        actual_seq_list_q_tenor = torch.tensor(actual_seq_list_q, dtype=torch.int32, device=f'npu:{device_id}')

        q_tnd, block_table, kv_cache, atten_sink, _ = gen_win_attn_data_tnd(t, n_q, d_q, n_kv, d_kv, block_size, seqused_kv_list, dtypes, device_id)
        mask2 = get_mask2(4, n_q, device_id, block_size)
        atten_out_2d = deepseekv4_win_atten(q_tnd, block_table, kv_cache, seqused_kv_list_tensor, atten_sink, win_size, is_decode=False, mask=mask2, actual_seq_list_q=actual_seq_list_q_tenor)
        
        golden = win_atten_calc_tnd_prefill(input_params_win_attn, seqused_kv_list, atten_sink, q_tnd, kv_cache, block_table, actual_seq_list_q, device_id)
        from utils.np_compare import detailed_allclose_manual as compare
        atten_out = torch.reshape(atten_out_2d, [t, n_q, d_q])
        compare(golden, atten_out, "SWA prefill tnd mask 版本", rtol=0.0078125, atol=0.0001)


def test_win_atten_tnd_prefill() -> None:

    for b in [1]:
        # len(actual_seq_list_q) = b + 1
        s_val = 512
        actual_seq_list_q = [i * s_val for i in range(b + 1)]
        t = actual_seq_list_q[-1]
        win_size = 128
        n_q = 64
        block_size = 128
        n_kv = 1
        dtypes = torch.bfloat16
        head_dim = 512
        d_q = head_dim
        d_kv = head_dim
        scalar = d_q ** -0.5
        input_params_win_attn = [t, n_kv, n_q, d_q, win_size, scalar]

        device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
        torch.npu.set_device(device_id)

        seqused_kv_list = [s_val] * b
        seqused_kv_list_tensor = torch.tensor(seqused_kv_list, dtype=torch.int32, device=f'npu:{device_id}')
        actual_seq_list_q_tenor = torch.tensor(actual_seq_list_q, dtype=torch.int32, device=f'npu:{device_id}')

        q_tnd, block_table, kv_cache, atten_sink, _ = gen_win_attn_data_tnd(t, n_q, d_q, n_kv, d_kv, block_size, seqused_kv_list, dtypes, device_id)
        atten_out_2d = deepseekv4_win_atten(q_tnd, block_table, kv_cache, seqused_kv_list_tensor, atten_sink, win_size, is_decode=False, mask=None, actual_seq_list_q=actual_seq_list_q_tenor)
        
        golden = win_atten_calc_tnd_prefill(input_params_win_attn, seqused_kv_list, atten_sink, q_tnd, kv_cache, block_table, actual_seq_list_q, device_id)
        from utils.np_compare import detailed_allclose_manual as compare
        atten_out = torch.reshape(atten_out_2d, [t, n_q, d_q])
        compare(golden, atten_out, "SWA prefill tnd 版本", rtol=0.0078125, atol=0.0001)


if __name__ == "__main__":
    test_win_atten_tnd_mtp_decode_mask()
    # test_win_atten_bsnd_mtp_decode_mask()
    # test_win_atten_bsnd_mtp_decode()
    test_win_atten_tnd_prefill_mask()
    # test_win_atten_tnd_prefill()