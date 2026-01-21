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
    dtype = q_tnd.dtype

    q_2d = pypto.reshape(q_tnd, [b * s_q * n_q, d_q], inplace=True)
    atten_sink_2d = pypto.reshape(atten_sink, [atten_sink.shape[0], 1], inplace=True)
    kv_2d = pypto.reshape(kv_cache, [kv_cache.shape[0] *block_size * kv_cache.shape[2], d_kv], inplace=True)
    for t_idx in pypto.loop(t, name="LOOP_T", idx_name="t_idx", unroll_list = [128]):
        b_idx = t_idx // s_q
        s1_idx = t_idx % s_q
        cur_offset = t_idx * n_q

        actual_seq = seqused_kv_list[b_idx]
        pypto.set_vec_tile_shapes(128, 512)
        q_tensor_cur = pypto.view(q_2d, [n_q, d_q], [cur_offset, 0])

        cur_loc = actual_seq - s_q + s1_idx + 1
        valid_len = pypto.min(cur_loc, win)
        cur_start_pos = cur_loc - valid_len
        end_pos = cur_loc
        start_block = cur_start_pos // block_size
        start_offset = cur_start_pos % block_size
        end_block = (end_pos - 1) // block_size

        start_block_id = block_table[b_idx, start_block]
        kv_block_0 = pypto.view(kv_2d, [block_size, d_kv], [start_block_id * block_size, 0])
        end_block_id = block_table[b_idx, end_block]
        kv_block_1 = pypto.view(kv_2d, [block_size, d_kv], [end_block_id * block_size, 0])

        pypto.set_vec_tile_shapes(128, 512)
        kv_gather = pypto.concat([kv_block_0, kv_block_1], dim=0)

        pypto.set_vec_tile_shapes(128, 512)
        kv_cur = pypto.view(kv_gather, [win, d_kv], [start_offset, 0])

        pypto.set_cube_tile_shapes([64, 64], [128, 512], [128, 128], True, False)
        qk_mm_res = pypto.matmul(q_tensor_cur, kv_cur, pypto.DT_FP32, b_trans=True)

        pypto.set_vec_tile_shapes(128, 128)
        qk_ele_res = pypto.mul(qk_mm_res, scalar)
        rowmax = pypto.amax(qk_ele_res, -1, True)
        sub_res = pypto.sub(qk_ele_res, rowmax)
        exp_res = pypto.exp(sub_res)
        esum = pypto.sum(exp_res, -1, True)
        esum = pypto.add(esum, atten_sink_2d)

        softmax_out = pypto.div(exp_res, esum)
        softmax_out_b16 = pypto.cast(softmax_out, dtype)

        pypto.set_cube_tile_shapes([64, 64], [128, 128], [256, 256], False, False)
        mm2_res = pypto.matmul(softmax_out_b16, kv_cur, dtype)

        pypto.assemble(mm2_res, [cur_offset, 0], atten_out)


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
    dtype = q.dtype

    q_2d = pypto.reshape(q, [b * s_q * n_q, d_q], inplace=True)
    atten_out_2d = pypto.tensor([b * s_q * n_q, d_q], dtype, "atten_out_2d")
    atten_sink_2d = pypto.reshape(atten_sink, [atten_sink.shape[0], 1], inplace=True)
    kv_2d = pypto.reshape(kv_cache, [kv_cache.shape[0] *block_size * kv_cache.shape[2], d_kv], inplace=True)
    for b_idx in pypto.loop(b, name="LOOP_b", idx_name="b_idx"):
        actual_seq = actual_seq_list[b_idx]
        for s1_idx in pypto.loop(s_q, name="LOOP_s1", idx_name="s1_idx"):
            cur_offset = (b_idx * s_q + s1_idx) * n_q    
            pypto.set_vec_tile_shapes(128, 512)
            q_tensor_cur = pypto.view(q_2d, [n_q, d_q], [cur_offset, 0])

            valid_data_len = pypto.min(win + s_q - 1, actual_seq)
            valid_end_pos = valid_data_len - s_q + s1_idx
            valid_start_pos = valid_end_pos - pypto.min(win - 1, valid_end_pos)
            valid_win_len = valid_end_pos - valid_start_pos + 1

            pypto.set_vec_tile_shapes(128, 512)
            start_block = valid_start_pos // block_size
            end_block = valid_end_pos // block_size
            start_block_id = block_table[b_idx, start_block]
            end_block_id = block_table[b_idx, end_block]

            kv_block_0 = pypto.view(kv_2d, [block_size, d_kv], [start_block_id * block_size + valid_start_pos, 0], 
                            valid_shape=[valid_win_len - valid_start_pos, d_kv])
            kv_block_1 = pypto.view(kv_2d, [block_size, d_kv], [end_block_id * block_size, 0], 
                            valid_shape=[valid_start_pos, d_kv])

            # C1
            pypto.set_cube_tile_shapes([64, 64], [128, 512], [128, 128], True, False)
            mm1_res = pypto.tensor([n_q, win], pypto.DT_FP32, "mm1_res")
            mm1_res[0:, 0:valid_win_len - valid_start_pos] = \
                pypto.matmul(q_tensor_cur, kv_block_0, pypto.DT_FP32, b_trans=True)
            mm1_res[0:, valid_win_len - valid_start_pos: valid_win_len] = \
                pypto.matmul(q_tensor_cur, kv_block_1, pypto.DT_FP32, b_trans=True)
            acc_s = pypto.view(mm1_res, [n_q, win], [0, 0], valid_shape=[n_q, valid_win_len])

            # V1
            pypto.set_vec_tile_shapes(128, 128)
            acc_s = pypto.mul(acc_s, scalar)
            scores_max = pypto.amax(acc_s, -1, True)
            sub_res = pypto.sub(acc_s, scores_max)
            acc_s = pypto.exp(sub_res)
            
            sum_exp = pypto.sum(acc_s, -1, True)
            sub_res = pypto.sub(atten_sink_2d, scores_max)
            atten_sink_exp = pypto.exp(sub_res)
            sum_exp = pypto.add(sum_exp, atten_sink_exp)

            div_res = pypto.div(acc_s, sum_exp)
            div_res_b16 = pypto.cast(div_res, dtype)

            # C2
            pypto.set_vec_tile_shapes(128, 512)
            kv_block_0 = pypto.view(kv_2d, [block_size, d_kv], [start_block_id * block_size, 0])
            kv_block_1 = pypto.view(kv_2d, [block_size, d_kv], [end_block_id * block_size, 0])
            kv_gather = pypto.concat([kv_block_0, kv_block_1], dim=0)
            kv_cur = pypto.view(kv_gather, [win, d_kv], [valid_start_pos, 0], valid_shape=[valid_win_len, d_kv])
            pypto.set_cube_tile_shapes([64, 64], [128, 128], [256, 256], False, False)
            mm2_res = pypto.matmul(div_res_b16, kv_cur, dtype)

            pypto.assemble(mm2_res, [cur_offset, 0], atten_out_2d)
            atten_out[:] = pypto.reshape(atten_out_2d,
                                                    [atten_out.shape[0], atten_out.shape[1],
                                                    atten_out.shape[2], atten_out.shape[3]], inplace=True)


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"device_sched_mode": 1,
                    "stitch_function_inner_memory": 1024,
                    "stitch_function_outcast_memory": 1024,
                    "stitch_function_num_initial": 128},
    debug_options={"runtime_debug_mode": 1},
    pass_options={"cube_l1_reuse_mode": 2},
)
def win_atten_main_bsnd_mtp_decode_mask(q, block_table, kv_cache, actual_seq_list, atten_sink, mask2, atten_out, win):
    pypto.experimental.set_operation_config(combine_axis=True)
    b = q.shape[0]
    s_q = q.shape[1]
    n_q = q.shape[2]
    d_q = q.shape[3]
    scalar = d_q ** -0.5
    block_size = kv_cache.shape[1]
    d_kv = kv_cache.shape[3]
    dtype = q.dtype

    pypto.set_vec_tile_shapes(128, 128)
    q_2d = pypto.reshape(q, [b * s_q * n_q, d_q], inplace=True)
    kv_2d = pypto.reshape(kv_cache, [kv_cache.shape[0] * block_size * kv_cache.shape[2], d_kv], inplace=True)
    kv_block_temp = pypto.tensor([block_size * 2, d_kv], dtype, "kv_block_temp")

    for b_idx in pypto.loop(b, name="LOOP_b", idx_name="b_idx"):

        pypto.set_vec_tile_shapes(128, 128)
        atten_sink_2d = pypto.reshape(atten_sink, [atten_sink.shape[0], 1], inplace=True)
        atten_sink_2d = pypto.concat([atten_sink_2d] * s_q, dim=0)

        atten_out_2d = pypto.tensor([b * s_q * n_q, d_q], dtype, "atten_out_2d")
        cur_offset = b_idx * s_q * n_q

        actual_seq = actual_seq_list[b_idx]

        pypto.set_vec_tile_shapes(128, 512)
        q_tensor_cur = pypto.view(q_2d, [s_q * n_q, d_q], [cur_offset, 0])

        if pypto.cond(s_q > 1 and actual_seq > 128):
            # pypto.set_semantic_label("2 blocks")
            pypto.set_vec_tile_shapes(128, 512)
            physical_block_id = block_table[b_idx, 0]
            kv_block_0 = pypto.view(kv_2d, [block_size, d_kv], [physical_block_id * block_size, 0])
            physical_block_id = block_table[b_idx, 1]
            kv_block_1 = pypto.view(kv_2d, [block_size, d_kv], [physical_block_id * block_size, 0])

            # C1
            pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128], False, False)
            acc_s_0 = pypto.matmul(q_tensor_cur, kv_block_0, pypto.DT_FP32, b_trans=True)
            pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128], False, False)
            acc_s_1 = pypto.matmul(q_tensor_cur, kv_block_1, pypto.DT_FP32, b_trans=True)
            pypto.set_vec_tile_shapes(128, 256)
            acc_s = pypto.concat([acc_s_0, acc_s_1], dim=-1)
            
            start_pos = win + s_q - 1 - pypto.min(win + s_q - 1, actual_seq)
            pypto.set_vec_tile_shapes(64, 256)
            mask_block = pypto.view(mask2, [s_q * n_q, 256], [0, 128 + start_pos])
            acc_s = pypto.where(mask_block, acc_s, float("-inf"))

            # V1
            acc_s = pypto.mul(acc_s, scalar)
            scores_max = pypto.amax(acc_s, -1, True)
            sub_res = pypto.sub(acc_s, scores_max)
            acc_s = pypto.exp(sub_res)
            
            sum_exp = pypto.sum(acc_s, -1, True)
            sub_res = pypto.sub(atten_sink_2d, scores_max)
            atten_sink_exp = pypto.exp(sub_res)
            sum_exp = pypto.add(sum_exp, atten_sink_exp)

            div_res = pypto.div(acc_s, sum_exp)
            div_res_b16 = pypto.cast(div_res, dtype)

            # C2
            pypto.assemble(kv_block_0, [0, 0], kv_block_temp)
            pypto.assemble(kv_block_1, [block_size, 0], kv_block_temp)
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [256, 256], False, False) 
            mm2_res = pypto.matmul(div_res_b16, kv_block_temp, dtype)

        else:
            # pypto.set_semantic_label("1 block")
            pypto.set_vec_tile_shapes(128, 512)
            physical_block_id = block_table[b_idx, 0]
            kv_block = pypto.view(kv_2d, [block_size, d_kv], [physical_block_id * block_size, 0])

            # C1
            pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128], False, False)
            acc_s = pypto.matmul(q_tensor_cur, kv_block, pypto.DT_FP32, b_trans=True)
            
            if pypto.cond(actual_seq <= 128):
                start_pos = win + s_q - 1 - pypto.min(win + s_q - 1, actual_seq)
                pypto.set_vec_tile_shapes(128, 128)
                mask_block = pypto.view(mask2, [s_q * n_q, 128], [0, 128 + start_pos])
                acc_s = pypto.where(mask_block, acc_s, float("-inf"))

            # V1
            acc_s = pypto.mul(acc_s, scalar)
            scores_max = pypto.amax(acc_s, -1, True)
            sub_res = pypto.sub(acc_s, scores_max)
            acc_s = pypto.exp(sub_res)
            
            sum_exp = pypto.sum(acc_s, -1, True)
            sub_res = pypto.sub(atten_sink_2d, scores_max)
            atten_sink_exp = pypto.exp(sub_res)
            sum_exp = pypto.add(sum_exp, atten_sink_exp)

            div_res = pypto.div(acc_s, sum_exp)
            div_res_b16 = pypto.cast(div_res, dtype)

            # C2
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [256, 256], False, False) 
            mm2_res = pypto.matmul(div_res_b16, kv_block, dtype)

        pypto.assemble(mm2_res, [cur_offset, 0], atten_out_2d)
        atten_out[:] = pypto.reshape(atten_out_2d,
                                                [atten_out.shape[0], atten_out.shape[1],
                                                atten_out.shape[2], atten_out.shape[3]], inplace=True)


@allow_in_graph
def deepseekv4_win_atten(q: torch.Tensor,
                        ori_block_table: torch.Tensor,
                        ori_kv: torch.Tensor,
                        seqused_kv: torch.Tensor,
                        attn_sinks: torch.Tensor,
                        atten_out: torch.Tensor,
                        win_size: int,
                        is_decode: bool = False,
                        mask: torch.Tensor = None,
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

    if mask is not None:
        print("Using mask kernel")
        inputs[mask] = []
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        win_atten_main_bsnd_mtp_decode_mask(*pto_inputs, *pto_outputs, win_size)
    elif is_decode:
        print("Using mtp decode kernel")
        win_atten_main_bsnd_mtp_decode(*pto_inputs, *pto_outputs, win_size)
    else:
        print("Using prefill kernel")
        win_atten_main_tnd_prefill(*pto_inputs, *pto_outputs, win_size)

    pypto.runtime._device_synchronize()