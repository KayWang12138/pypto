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
import torch
import pypto
from torch._dynamo import allow_in_graph

# 暂未启用
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


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"device_sched_mode": 1},
    debug_options={"runtime_debug_mode": 1},
)
def win_atten_main_tnd_prefill(q_tnd, block_table, kv_cache, seqused_kv_list, atten_sink, atten_out, win):
    pypto.experimental.set_operation_config(combine_axis=True)
    t = q_tnd.shape[0]
    n_q = q_tnd.shape[1]
    d_q = q_tnd.shape[2]
    scalar = d_q ** -0.5
    block_size = kv_cache.shape[1]
    d_kv = kv_cache.shape[3]
    b = seqused_kv_list.shape[0]
    s_q = t // b

    for t_idx in pypto.loop(t, name="LOOP_T", idx_name="t_idx", unroll_list = [128]):
        b_idx = t_idx // s_q
        s1_idx = t_idx % s_q

        actual_seq = seqused_kv_list[b_idx]
        pypto.set_vec_tile_shapes(128, 512, 512)
        q_tensor_cur = pypto.view(q_tnd, [1, n_q, d_q], [t_idx, 0, 0])
        q_tensor_cur = pypto.reshape(q_tensor_cur, (n_q, d_q))

        cur_loc = actual_seq - s_q + s1_idx + 1
        valid_len = pypto.min(cur_loc, win)
        cur_start_pos = cur_loc - valid_len
        end_pos = cur_loc
        start_block = cur_start_pos // block_size
        start_offset = cur_start_pos % block_size
        end_block = (end_pos - 1) // block_size

        physical_block_id = block_table[b_idx, start_block]
        pypto.set_vec_tile_shapes(128, 512, 128, 512)
        kv_block_0 = pypto.view(kv_cache, [1, block_size, 1, d_kv], [physical_block_id, 0, 0, 0])
        kv_block_reshape_0 = pypto.reshape(kv_block_0, (block_size, d_kv))

        physical_block_id = block_table[b_idx, end_block]
        pypto.set_vec_tile_shapes(128, 512, 128, 512)
        kv_block_1 = pypto.view(kv_cache, [1, block_size, 1, d_kv], [physical_block_id, 0, 0, 0])
        kv_block_reshape_1 = pypto.reshape(kv_block_1, (block_size, d_kv))

        pypto.set_vec_tile_shapes(128, 512)
        kv_gather = pypto.concat([kv_block_reshape_0, kv_block_reshape_1], dim=0)

        pypto.set_vec_tile_shapes(128, 512)
        kv_cur = pypto.view(kv_gather, [win, d_kv], [start_offset, 0])

        pypto.set_cube_tile_shapes([64, 64], [256, 256 * 8], [128, 128], True, False)
        qk_mm_res = pypto.matmul(q_tensor_cur, kv_cur, pypto.DT_FP32, b_trans=True)

        pypto.set_vec_tile_shapes(32, 128)
        qk_ele_res = pypto.mul(qk_mm_res, scalar)
        # softmax_out = softmax_pto(qk_ele_res, -1)
        softmax_out = softmax_atten_sink_pto(qk_ele_res, -1, atten_sink)

        pypto.set_vec_tile_shapes(64, 512)
        kv_cur_fp32 = pypto.cast(kv_cur, pypto.DT_FP32)
        pypto.set_cube_tile_shapes([64, 64], [32, 32 * 8], [512, 512], True, False)
        mm2_res = pypto.matmul(softmax_out, kv_cur_fp32, pypto.DT_FP32)

        pypto.set_vec_tile_shapes(64, 512)
        mm2_res_reshape = pypto.reshape(mm2_res, (1, n_q, d_kv))

        pypto.set_vec_tile_shapes(1, 64, 512)
        pypto.assemble(mm2_res_reshape, [t_idx, 0, 0], atten_out)


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"device_sched_mode": 1},
    debug_options={"runtime_debug_mode": 1},
)
def win_atten_main_bsnd_mtp_decode(q, block_table, kv_cache, actual_seq_list, atten_sink, atten_out, win):
    pypto.experimental.set_operation_config(combine_axis=True)
    b = q.shape[0]
    s_q = q.shape[1]
    n_q = q.shape[2]
    d_q = q.shape[3]
    scalar = d_q ** -0.5
    block_size = kv_cache.shape[1]
    d_kv = kv_cache.shape[3]

    for b_idx in pypto.loop(b, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s_q, name="LOOP_s1", idx_name="s1_idx"):
            actual_seq = actual_seq_list[b_idx]
            
            pypto.set_vec_tile_shapes(128, 128, 512, 512)
            q_tensor_cur = pypto.view(q, [1, 1, n_q, d_q], [b_idx, s1_idx, 0, 0])
            q_tensor_cur = pypto.reshape(q_tensor_cur, (n_q, d_q))

            valid_data_len = pypto.min(win + s_q - 1, actual_seq)
            valid_end_pos = valid_data_len - s_q + s1_idx
            valid_start_pos = valid_end_pos - pypto.min(win - 1, valid_end_pos)
            valid_win_len = valid_end_pos - valid_start_pos + 1

            start_block = valid_start_pos // block_size
            end_block = valid_end_pos // block_size

            physical_block_id = block_table[b_idx, start_block]
            pypto.set_vec_tile_shapes(128, 256, 128, 256)
            kv_block_0 = pypto.view(kv_cache, [1, block_size, 1, d_kv], [physical_block_id, 0, 0, 0])
            kv_block_reshape_0 = pypto.reshape(kv_block_0, (block_size, d_kv))

            physical_block_id = block_table[b_idx, end_block]
            pypto.set_vec_tile_shapes(128, 256, 128, 256)
            kv_block_1 = pypto.view(kv_cache, [1, block_size, 1, d_kv], [physical_block_id, 0, 0, 0])
            kv_block_reshape_1 = pypto.reshape(kv_block_1, (block_size, d_kv))

            pypto.set_vec_tile_shapes(128, 256)
            kv_gather = pypto.concat([kv_block_reshape_0, kv_block_reshape_1], dim=0)

            pypto.set_vec_tile_shapes(128, 256)
            kv_cur = pypto.view(kv_gather, [win, d_kv], [valid_start_pos, 0], valid_shape=[valid_win_len, d_kv])

            sum_exp = pypto.full([64, 1], float(0), dtype=pypto.DT_FP32)
            acc_o = pypto.full([64, 512], float(0), dtype=pypto.DT_FP32)
            scores_max = pypto.full([64, 1], float("-inf"), dtype=pypto.DT_FP32)

            # 以下为block循环部分
            pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128], False, True)
            acc_s = pypto.matmul(q_tensor_cur, kv_cur, pypto.DT_FP32, b_trans=True)

            pypto.set_vec_tile_shapes(128, 128)
            acc_s = pypto.mul(acc_s, scalar)
            scores_max_prev = scores_max
            scores_max = pypto.amax(acc_s, -1, True)
            sub_res = pypto.sub(scores_max_prev, scores_max)
            scores_scale = pypto.exp(sub_res)
            sub_res = pypto.sub(acc_s, scores_max)
            acc_s = pypto.exp(sub_res)
            scores_sum = pypto.sum(acc_s, -1, True)
            mul_res = pypto.mul(sum_exp, scores_scale)
            sum_exp = pypto.add(scores_sum, mul_res)
            acc_o = pypto.mul(acc_o, scores_scale)
            kv_cur_fp32 = pypto.cast(kv_cur, pypto.DT_FP32)
            pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128], False, True)
            matmul_res = pypto.matmul(acc_s, kv_cur_fp32, pypto.DT_FP32)
            acc_o = pypto.add(acc_o, matmul_res)
            atten_sink = pypto.reshape(atten_sink, [atten_sink.shape[0], 1])
            sub_res = pypto.sub(atten_sink, scores_max)
            exp_res = pypto.exp(sub_res)
            sum_exp = pypto.add(sum_exp, exp_res)
            acc_o = pypto.div(acc_o, sum_exp)
            acc_o_reshape = pypto.reshape(acc_o, (1, 1, n_q, d_kv))

            pypto.set_vec_tile_shapes(1, 1, 64, 512)
            pypto.assemble(acc_o_reshape, [b_idx, s1_idx, 0, 0], atten_out)


@allow_in_graph
def deepseekv4_win_atten(q: torch.Tensor,
                        ori_block_table: torch.Tensor,
                        ori_kv: torch.Tensor,
                        seqused_kv: torch.Tensor,
                        attn_sinks: torch.Tensor,
                        atten_out: torch.Tensor,
                        win_size: int,
                        is_decode: bool = False,
) -> None:
    """
    """
    inputs = {
        q: [],
        ori_block_table: [],
        ori_kv: [],
        seqused_kv: [],
        attn_sinks: [],
    }
    outputs = {
        atten_out: [],
    }

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    if is_decode:
        print("Using mtp decode kernel")
        win_atten_main_bsnd_mtp_decode(*pto_inputs, *pto_outputs, win_size)
    else:
        print("Using prefill kernel")
        win_atten_main_tnd_prefill(*pto_inputs, *pto_outputs, win_size)

    pypto.runtime._device_synchronize()