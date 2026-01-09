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
import numpy as np
import torch
import pypto
from numpy.testing import assert_allclose
from torch._dynamo import allow_in_graph


def main():
    test_win_atten()
    test_win_atten_allow_in_graph()


def check_args(
    q_tnd,
    block_table,
    kv_cache,
    start_pos_list,
    atten_sink,
):
    # q_tnd [b * s_q, 64, 512]
    assert q_tnd.dtype == torch.bfloat16
    assert q_tnd.ndim == 3
    assert q_tnd.shape[1] == 64
    assert q_tnd.shape[2] == 512

    # block_table [b, math.ceil(s_kv_max / block_size)]
    # start_pos_list [b]
    assert block_table.ndim == 2
    assert start_pos_list.ndim == 1
    assert block_table.shape[0] == start_pos_list.shape[0]

    # kv_cache [block_num, 128, 1, 512]
    assert kv_cache.dtype == torch.bfloat16
    assert kv_cache.ndim == 4
    assert kv_cache.shape[1] == 128
    assert kv_cache.shape[2] == 1
    assert kv_cache.shape[3] == 512

    #atten_sink [64]
    assert atten_sink.dtype == torch.float32
    assert atten_sink.ndim == 1
    assert atten_sink.shape[0] == 64


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


def softmax_pto(input: pypto.Tensor, dim: int) -> pypto.Tensor:
    dtype = input.dtype
    input = pypto.cast(input, pypto.DT_FP32)

    rowmax = pypto.amax(input, dim, True)
    sub_res = pypto.sub(input, rowmax)
    exp_res = pypto.exp(sub_res)
    esum = pypto.sum(exp_res, dim, True)
    output = pypto.div(exp_res, esum)

    if dtype != pypto.DT_FP32:
        output = pypto.cast(output, dtype)
    return output


def softmax_atten_sink_pto(input: pypto.Tensor, dim: int, atten_sink: pypto.Tensor) -> pypto.Tensor:
    input = pypto.cast(input, pypto.DT_FP32)

    rowmax = pypto.amax(input, dim, True)
    sub_res = pypto.sub(input, rowmax)
    exp_res = pypto.exp(sub_res)
    esum = pypto.sum(exp_res, dim, True)

    atten_sink = pypto.reshape(atten_sink, [atten_sink.shape[0], 1])
    esum = pypto.add(esum, atten_sink)

    output = pypto.div(exp_res, esum)
    return output


def win_atten_calc_tnd(input_params_win_attn, start_pos_list, atten_sink, q_tnd, kv_cache, block_table, device_id):

    t = input_params_win_attn[0]
    n_kv = input_params_win_attn[1]
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


def gen_win_attn_data_tnd(t, n_q, d_q, n_kv, d_kv, block_size, start_post_list, dtypes, device_id):
    torch.manual_seed(42)
    b = len(start_post_list)
    s_q = t // b
    new_dtype = dtypes

    actual_seq_max = max(start_post_list)
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

    for start_post in start_post_list:
        actual_seq = start_post + s_q
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


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"device_sched_mode": 1},
    # debug_options={"runtime_debug_mode": 1},
)
def win_atten_main_tnd(q_tnd, block_table, kv_cache, start_pos_list, atten_sink, atten_out, win):
    
    t = q_tnd.shape[0]
    n_q = q_tnd.shape[1]
    d_q = q_tnd.shape[2]
    n_kv = kv_cache.shape[2]
    scalar = d_q ** -0.5
    block_size = kv_cache.shape[1]
    d_kv = kv_cache.shape[3]
    b = start_pos_list.shape[0]
    s_q = t // b

    for t_idx in pypto.loop(t, name="LOOP_T", idx_name="t_idx"):
        b_idx = t_idx // s_q
        s1_idx = t_idx % s_q

        start_pos = start_pos_list[b_idx]
                
        pypto.set_vec_tile_shapes(128, 128, 512)
        q_tensor_cur = pypto.view(q_tnd, [1, n_q, d_q], [t_idx, 0, 0])
        q_tensor_cur = pypto.reshape(q_tensor_cur, (n_q, d_q))

        cur_loc = start_pos + s1_idx + 1
        valid_len = pypto.min(cur_loc, win)
        cur_start_pos = cur_loc - valid_len
        end_pos = cur_loc
        start_block = cur_start_pos // block_size
        start_offset = cur_start_pos % block_size
        end_block = (end_pos - 1) // block_size

        physical_block_id = block_table[b_idx, start_block]
        kv_block_0 = pypto.view(kv_cache, [1, block_size, 1, d_kv], [physical_block_id, 0, 0, 0])
        pypto.set_vec_tile_shapes(128, 256, 128, 128)
        kv_block_reshape_0 = pypto.reshape(kv_block_0, (block_size, d_kv))

        physical_block_id = block_table[b_idx, end_block]
        kv_block_1 = pypto.view(kv_cache, [1, block_size, 1, d_kv], [physical_block_id, 0, 0, 0])
        pypto.set_vec_tile_shapes(128, 256, 128, 128)
        kv_block_reshape_1 = pypto.reshape(kv_block_1, (block_size, d_kv))

        pypto.set_vec_tile_shapes(128, 256)
        kv_gather = pypto.concat([kv_block_reshape_0, kv_block_reshape_1], dim=0)

        pypto.set_vec_tile_shapes(128, 512)
        kv_cur = pypto.view(kv_gather, [win, d_kv], [start_offset, 0])

        pypto.set_cube_tile_shapes([64, 64], [256, 256 * 4], [128, 128], True, False)
        qk_mm_res = pypto.matmul(q_tensor_cur, kv_cur, pypto.DT_FP32, b_trans=True)

        pypto.set_vec_tile_shapes(32, 128)
        qk_ele_res = pypto.mul(qk_mm_res, scalar)
        # softmax_out = softmax_pto(qk_ele_res, -1)
        softmax_out = softmax_atten_sink_pto(qk_ele_res, -1, atten_sink)

        pypto.set_vec_tile_shapes(64, 512)
        kv_cur_fp32 = pypto.cast(kv_cur, pypto.DT_FP32)
        pypto.set_cube_tile_shapes([64, 64], [128, 128 * 2], [128, 128], True, False)
        mm2_res = pypto.matmul(softmax_out, kv_cur_fp32, pypto.DT_FP32)
        mm2_res_reshape = pypto.reshape(mm2_res, (1, n_q, d_kv))

        pypto.set_vec_tile_shapes(1, n_q, d_kv)
        pypto.assemble(mm2_res_reshape, [t_idx, 0, 0], atten_out)


@allow_in_graph
def deepseekv4_win_atten(q_tnd: torch.Tensor,
                        block_table: torch.Tensor,
                        kv_cache: torch.Tensor,
                        start_pos_list: torch.Tensor,
                        atten_sink: torch.Tensor,
                        atten_out: torch.Tensor,
                        win_size: int
) -> None:
    """
    """
    inputs = {
        q_tnd: [],
        block_table: [],
        kv_cache: [],
        start_pos_list: [],
        atten_sink: [],
    }
    outputs = {
        atten_out: [],
    }

    check_args(q_tnd, block_table, kv_cache, start_pos_list, atten_sink)
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    win_atten_main_tnd(*pto_inputs, *pto_outputs, win_size)
    pypto.runtime._device_synchronize()


def test_win_atten_allow_in_graph() -> None:
    
    for b in [4, 16]:
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

            torch.manual_seed(321)
            actual_seq_list = torch.randint(s_q + 1, 8193, (b, ), device=f'npu:{device_id}').tolist()
            # actual_seq_list = [8192] * b

            # start_pos + s1 = actual_seq
            start_pos_list = [item - s_q for item in actual_seq_list]
            print(f"start_pos_list: {start_pos_list}")

            start_pos_list_tensor = torch.tensor(start_pos_list, dtype=torch.int32, device=f'npu:{device_id}')

            q_tnd, block_table, kv_cache, atten_sink, atten_out = gen_win_attn_data_tnd(t, n_q, d_q, n_kv, d_kv, block_size, actual_seq_list, dtypes, device_id)

            model = torch.compile(MM(), backend="eager", dynamic=True)
            g = torch.npu.NPUGraph()
            with torch.npu.graph(g):
                y = model(q_tnd, block_table, kv_cache, start_pos_list_tensor, atten_sink, atten_out, win)
            g.replay()
            pypto.runtime._device_synchronize()

            golden = win_atten_calc_tnd(input_params_win_attn, start_pos_list, atten_sink, q_tnd, kv_cache, block_table, device_id)
            assert_allclose(np.array(y.cpu().flatten().tolist()), np.array(golden.cpu().flatten().tolist()), rtol=5e-4, atol=5e-4)


def test_win_atten() -> None:

    for b in [4, 16]:
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

            torch.manual_seed(321)
            actual_seq_list = torch.randint(s_q + 1, 8193, (b, ), device=f'npu:{device_id}').tolist()
            # actual_seq_list = [8192] * b

            # start_pos + s1 = actual_seq
            start_pos_list = [item - s_q for item in actual_seq_list]
            print(f"start_pos_list: {start_pos_list}")

            start_pos_list_tensor = torch.tensor(start_pos_list, dtype=torch.int32, device=f'npu:{device_id}')

            q_tnd, block_table, kv_cache, atten_sink, atten_out = gen_win_attn_data_tnd(t, n_q, d_q, n_kv, d_kv, block_size, actual_seq_list, dtypes, device_id)

            inputs = {
                q_tnd: [],
                block_table: [],
                kv_cache: [],
                start_pos_list_tensor: [],
                atten_sink: [],
            }
            outputs = {
                atten_out: [],
            }

            pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
            pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

            win_atten_main_tnd(*pto_inputs, *pto_outputs, win)
            pypto.runtime._device_synchronize()

            # golden
            golden = win_atten_calc_tnd(input_params_win_attn, start_pos_list, atten_sink, q_tnd, kv_cache, block_table, device_id)
            assert_allclose(np.array(atten_out.cpu().flatten().tolist()), np.array(golden.cpu().flatten().tolist()), rtol=5e-4, atol=5e-4)


if __name__ == "__main__":
    main()