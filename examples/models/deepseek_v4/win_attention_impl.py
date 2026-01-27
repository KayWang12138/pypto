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


def check_args(
    q_tnd,
    block_table,
    kv_cache,
    start_pos_list,
    atten_sink,
):
    assert q_tnd.dtype == torch.bfloat16
    assert q_tnd.ndim == 3
    assert q_tnd.shape[1] == 64
    assert q_tnd.shape[2] == 512

    assert block_table.ndim == 2
    assert start_pos_list.ndim == 1
    assert block_table.shape[0] == start_pos_list.shape[0]

    assert kv_cache.dtype == torch.bfloat16
    assert kv_cache.ndim == 4
    assert kv_cache.shape[1] == 128
    assert kv_cache.shape[2] == 1
    assert kv_cache.shape[3] == 512

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
    runtime_options={"device_sched_mode": 1,
                    "stitch_function_inner_memory": 1024,
                    "stitch_function_outcast_memory": 1024,
                    "stitch_function_num_initial": 128},
    debug_options={"runtime_debug_mode": 1},
)
def win_atten_main_tnd_prefill(q_tnd, block_table, kv_cache, seqused_kv_list, \
    atten_sink, actual_seq_list_q, atten_out, win):
    pypto.experimental.set_operation_config(combine_axis=True)
    t = q_tnd.shape[0]
    n_q = q_tnd.shape[1]
    d_q = q_tnd.shape[2]
    scalar = d_q ** -0.5
    block_size = kv_cache.shape[1]
    d_kv = kv_cache.shape[3]
    b = seqused_kv_list.shape[0]
    dtype = q_tnd.dtype

    q_2d = pypto.reshape(q_tnd, [t * n_q, d_q], inplace=True)
    atten_sink_2d = pypto.reshape(atten_sink, [atten_sink.shape[0], 1], inplace=True)
    kv_2d = pypto.reshape(kv_cache, [kv_cache.shape[0] * block_size * kv_cache.shape[2], d_kv], inplace=True)

    for b_idx in pypto.loop(b, name="LOOP_B", idx_name="B_idx"):
        cur_s_q = actual_seq_list_q[b_idx + 1] - actual_seq_list_q[b_idx]

        for s1_idx in pypto.loop(cur_s_q, name="LOOP_S1", idx_name="S1_idx"):

            t_idx = actual_seq_list_q[b_idx] + s1_idx

            cur_offset = t_idx * n_q

            actual_seq = seqused_kv_list[b_idx]
            pypto.set_vec_tile_shapes(128, 512)
            q_tensor_cur = pypto.view(q_2d, [n_q, d_q], [cur_offset, 0])

            cur_loc = actual_seq - cur_s_q + s1_idx + 1
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
            kv_cur = pypto.view(kv_gather, [win, d_kv], [start_offset, 0], valid_shape=[valid_len, d_kv])

            pypto.set_cube_tile_shapes([64, 64], [128, 512], [128, 128], True, False)
            acc_s = pypto.matmul(q_tensor_cur, kv_cur, pypto.DT_FP32, b_trans=True)
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

            pypto.set_cube_tile_shapes([64, 64], [128, 128], [256, 256], False, False)
            mm2_res = pypto.matmul(div_res_b16, kv_cur, dtype)
            pypto.assemble(mm2_res, [cur_offset, 0], atten_out)


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"device_sched_mode": 2,
                    "stitch_function_inner_memory": 1024,
                    "stitch_function_outcast_memory": 1024,
                    "stitch_function_num_initial": 128},
    debug_options={"runtime_debug_mode": 1},
)
def win_atten_main_tnd_prefill_mask(q_tnd, block_table, kv_cache, seqused_kv_list, atten_sink, \
    actual_seq_list_q, mask2, atten_out, win):
    pypto.experimental.set_operation_config(combine_axis=True)
    t = q_tnd.shape[0]
    n_q = q_tnd.shape[1]
    d_q = q_tnd.shape[2]
    scalar = d_q ** -0.5
    block_size = kv_cache.shape[1]
    d_kv = kv_cache.shape[3]
    b = seqused_kv_list.shape[0]
    dtype = q_tnd.dtype

    q_2d = pypto.reshape(q_tnd, [t * n_q, d_q], inplace=True)
    kv_2d = pypto.reshape(kv_cache, [kv_cache.shape[0] * block_size * kv_cache.shape[2], d_kv], inplace=True)

    for b_idx in pypto.loop(b, name="LOOP_B", idx_name="B_idx"):

        cur_s_q = actual_seq_list_q[b_idx + 1] - actual_seq_list_q[b_idx]
        # 每个batch的第一个t索引位置
        t_start_idx = actual_seq_list_q[b_idx]
        # 4个s1分为一组
        groups_num = pypto.ceil(cur_s_q, 4)

        for g_idx in pypto.loop(groups_num, name="LOOP_G", idx_name="G_idx"):

            # 当前组内 全局的t索引
            group_start_t_idx = t_start_idx + g_idx * 4 # 组内第一个t索引
            group_end_t_idx = pypto.min(t_start_idx + g_idx * 4 + 3, t_start_idx + cur_s_q - 1) # 组内最后一个t索引
            valid_group_len = group_end_t_idx - group_start_t_idx + 1 # 组内t的个数
 
            cur_offset = group_start_t_idx * n_q
            actual_seq = seqused_kv_list[b_idx]
            pypto.set_vec_tile_shapes(128, 512)
            q_tensor_cur = pypto.view(q_2d, [4 * n_q, d_q], [cur_offset, 0], \
                valid_shape=[valid_group_len * n_q, d_q])

            # 组内s1的b内索引
            group_start_s1_idx = g_idx * 4 # 当前组内的第一个s1索引
            group_end_s1_idx = g_idx * 4 + valid_group_len - 1 # 当前组的最后一个s1索引

            cur_end_s2_pos = actual_seq - (cur_s_q - 1) + group_end_s1_idx - 1 # 算末尾位置坐标
            cur_start_s2_pos = pypto.max(0, cur_end_s2_pos - win - (valid_group_len - 1) + 1) # 算开头坐标
            start_block = cur_start_s2_pos // block_size # 0
            start_block_offset = cur_start_s2_pos % block_size # 0
            end_block = cur_end_s2_pos // block_size # 0
            end_block_offset = cur_end_s2_pos % block_size # 3

            pypto.set_vec_tile_shapes(128, 128)
            atten_sink_2d = pypto.reshape(atten_sink, [atten_sink.shape[0], 1], inplace=True)
            atten_sink_2d = pypto.concat([atten_sink_2d, atten_sink_2d, atten_sink_2d, atten_sink_2d], dim=0)
            atten_sink_2d_temp = atten_sink_2d

            if pypto.cond(start_block + 1 == end_block):

                acc_s = pypto.tensor([4 * n_q, block_size * 2], pypto.DT_FP32, "acc_s")
                kv_block_temp = pypto.tensor([block_size * 2, d_kv], dtype, "kv_block_temp")

                start_block_id = block_table[b_idx, start_block]
                kv_block_0 = pypto.view(kv_2d, [block_size, d_kv], [start_block_id * block_size, 0]) # [128, 512]
                end_block_id = block_table[b_idx, end_block]
                kv_block_1 = pypto.view(kv_2d, [block_size, d_kv], [end_block_id * block_size, 0]) # [128, 512]

                pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128], False, False)
                acc_s_0 = pypto.matmul(q_tensor_cur, kv_block_0, pypto.DT_FP32, b_trans=True)
                acc_s_1 = pypto.matmul(q_tensor_cur, kv_block_1, pypto.DT_FP32, b_trans=True)

                pypto.set_vec_tile_shapes(64, 256)
                pypto.assemble(acc_s_0, [0, 0], acc_s)
                pypto.assemble(acc_s_1, [0, block_size], acc_s)

                pypto.set_vec_tile_shapes(64, 256)
                mask_block = pypto.view(mask2, [4 * n_q, block_size * 2], [0, 128 - start_block_offset], \
                    valid_shape=[valid_group_len * n_q, block_size * 2])
                acc_s = pypto.where(mask_block, acc_s, float("-inf")) # [valid_group_len * n_q, 256]
                
                pypto.set_vec_tile_shapes(64, 256)
                acc_s = pypto.mul(acc_s, scalar)
                scores_max = pypto.amax(acc_s, -1, True)
                sub_res = pypto.sub(acc_s, scores_max)
                acc_s = pypto.exp(sub_res)

                sum_exp = pypto.sum(acc_s, -1, True)
                sub_res = pypto.sub(atten_sink_2d_temp, scores_max)
                atten_sink_exp = pypto.exp(sub_res)
                sum_exp = pypto.add(sum_exp, atten_sink_exp)

                div_res = pypto.div(acc_s, sum_exp)
                div_res_b16 = pypto.cast(div_res, dtype) # [valid_group_len * n_q, 256]

                pypto.assemble(kv_block_0, [0, 0], kv_block_temp)
                pypto.assemble(kv_block_1, [block_size, 0], kv_block_temp)
                pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128], False, False) 
                mm2_res = pypto.matmul(div_res_b16, kv_block_temp, dtype)

                pypto.assemble(mm2_res, [cur_offset, 0], atten_out)
        
            elif pypto.cond(start_block == end_block):

                start_block_id = block_table[b_idx, start_block]
                kv_block = pypto.view(kv_2d, [block_size, d_kv], [start_block_id * block_size, 0])

                pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128], False, False)
                acc_s = pypto.matmul(q_tensor_cur, kv_block, pypto.DT_FP32, b_trans=True)

                pypto.set_vec_tile_shapes(64, 256)
                mask_block = pypto.view(mask2, [4 * n_q, block_size], [0, 258 - end_block_offset], \
                    valid_shape=[valid_group_len * n_q, block_size])
                acc_s = pypto.where(mask_block, acc_s, float("-inf")) # [valid_group_len * n_q, 128]
                
                pypto.set_vec_tile_shapes(64, 256)
                acc_s = pypto.mul(acc_s, scalar)
                scores_max = pypto.amax(acc_s, -1, True)
                sub_res = pypto.sub(acc_s, scores_max)
                acc_s = pypto.exp(sub_res)

                sum_exp = pypto.sum(acc_s, -1, True)
                sub_res = pypto.sub(atten_sink_2d_temp, scores_max)
                atten_sink_exp = pypto.exp(sub_res)
                sum_exp = pypto.add(sum_exp, atten_sink_exp)

                div_res = pypto.div(acc_s, sum_exp)
                div_res_b16 = pypto.cast(div_res, dtype) # [valid_group_len * n_q, 128]

                pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128], False, False) 
                mm2_res = pypto.matmul(div_res_b16, kv_block, dtype)

                pypto.assemble(mm2_res, [cur_offset, 0], atten_out)


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"device_sched_mode": 1,
                    "stitch_function_inner_memory": 1024,
                    "stitch_function_outcast_memory": 1024,
                    "stitch_function_num_initial": 128},
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
    kv_2d = pypto.reshape(kv_cache, [kv_cache.shape[0] * block_size * kv_cache.shape[2], d_kv], inplace=True)
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
            acc_s = pypto.tensor([s_q * n_q, block_size * 2], pypto.DT_FP32, "acc_s")
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
            pypto.assemble(acc_s_0, [0, 0], acc_s)
            pypto.assemble(acc_s_1, [0, block_size], acc_s)
            
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
            pypto.set_vec_tile_shapes(128, 512)
            physical_block_id = block_table[b_idx, 0]
            kv_block = pypto.view(kv_2d, [block_size, d_kv], [physical_block_id * block_size, 0])

            # C1
            pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128], False, False)
            acc_s = pypto.matmul(q_tensor_cur, kv_block, pypto.DT_FP32, b_trans=True)
            
            if pypto.cond(actual_seq <= 128):
                end_pos = pypto.max(actual_seq - 1, 0)
                pypto.set_vec_tile_shapes(128, 128)
                mask_block = pypto.view(mask2, [s_q * n_q, 128], [0, 255 + s_q - 1 - end_pos])
                acc_s = pypto.where(mask_block, acc_s, float("-inf"))

            # V1
            pypto.set_vec_tile_shapes(64, 256)
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


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"device_sched_mode": 1,
                    "stitch_function_inner_memory": 1024,
                    "stitch_function_outcast_memory": 1024,
                    "stitch_function_num_initial": 128},
    debug_options={"runtime_debug_mode": 1},
    pass_options={"cube_l1_reuse_mode": 2},
)
def win_atten_main_tnd_mtp_decode_mask(q, block_table, kv_cache, actual_seq_list, atten_sink, \
    actual_seq_list_q, mask2, atten_out, win):
    pypto.experimental.set_operation_config(combine_axis=True)
    b = actual_seq_list.shape[0]
    t = q.shape[0]
    n_q = q.shape[1]
    d_q = q.shape[2]
    scalar = d_q ** -0.5
    block_size = kv_cache.shape[1]
    d_kv = kv_cache.shape[3]
    dtype = q.dtype

    pypto.set_vec_tile_shapes(128, 128)
    q_2d = pypto.reshape(q, [t * n_q, d_q], inplace=True)
    kv_2d = pypto.reshape(kv_cache, [kv_cache.shape[0] * block_size * kv_cache.shape[2], d_kv], inplace=True)
    kv_block_temp = pypto.tensor([block_size * 2, d_kv], dtype, "kv_block_temp")

    for b_idx in pypto.loop(b, name="LOOP_b", idx_name="b_idx"):
        s_q = actual_seq_list_q[b_idx + 1] - actual_seq_list_q[b_idx]

        pypto.set_vec_tile_shapes(128, 128)
        atten_sink_2d = pypto.reshape(atten_sink, [atten_sink.shape[0], 1], inplace=True)
        atten_sink_2d = pypto.concat([atten_sink_2d] * 4, dim=0)

        cur_offset = actual_seq_list_q[b_idx] * n_q
        actual_seq = actual_seq_list[b_idx]

        pypto.set_vec_tile_shapes(128, 512)
        q_tensor_cur = pypto.view(q_2d, [4 * n_q, d_q], [cur_offset, 0], valid_shape=[s_q * n_q, d_q])

        if pypto.cond(s_q > 1) and pypto.cond(actual_seq > 128):
            acc_s = pypto.tensor([4 * n_q, block_size * 2], pypto.DT_FP32, "acc_s")

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
            pypto.assemble(acc_s_0, [0, 0], acc_s)
            pypto.assemble(acc_s_1, [0, block_size], acc_s)
            
            start_pos = win + s_q - 1 - pypto.min(win + s_q - 1, actual_seq)
            pypto.set_vec_tile_shapes(64, 256)
            mask_block = pypto.view(mask2, [4 * n_q, 256], [0, 128 + start_pos], valid_shape=[s_q * n_q, 256])
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
            pypto.set_vec_tile_shapes(128, 512)
            physical_block_id = block_table[b_idx, 0]
            kv_block = pypto.view(kv_2d, [block_size, d_kv], [physical_block_id * block_size, 0])

            # C1
            pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128], False, False)
            acc_s = pypto.matmul(q_tensor_cur, kv_block, pypto.DT_FP32, b_trans=True)
            
            if pypto.cond(actual_seq <= 128):
                end_pos = pypto.max(actual_seq - 1, 0)
                pypto.set_vec_tile_shapes(128, 128)
                mask_block = pypto.view(mask2, [4 * n_q, 128], \
                    [0, 255 + s_q - 1 - end_pos], valid_shape=[s_q * n_q, 128])
                acc_s = pypto.where(mask_block, acc_s, float("-inf"))

            # V1
            pypto.set_vec_tile_shapes(64, 256)
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

        pypto.assemble(mm2_res, [cur_offset, 0], atten_out)


@allow_in_graph
def deepseekv4_win_atten(q: torch.Tensor,
                        ori_block_table: torch.Tensor,
                        ori_kv: torch.Tensor,
                        seqused_kv: torch.Tensor,
                        attn_sinks: torch.Tensor,
                        win_size: int,
                        is_decode: bool,
                        mask: torch.Tensor,
                        actual_seq_list_q: torch.Tensor,
) -> None:
    """
    """
    if actual_seq_list_q is None: # bsnd
        atten_out = torch.zeros(q.shape, dtype=q.dtype, device=q.device)
    else: # tnd
        atten_out = torch.zeros([q.shape[0] * q.shape[1], q.shape[2]], dtype=q.dtype, device=q.device)

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

    if is_decode and mask is not None and actual_seq_list_q is None:
        print("Using mtp decode mask bsnd kernel")
        inputs[mask] = []
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        win_atten_main_bsnd_mtp_decode_mask(*pto_inputs, *pto_outputs, win_size)

    elif is_decode and mask is None and actual_seq_list_q is None:
        print("Using mtp decode bsnd kernel")
        win_atten_main_bsnd_mtp_decode(*pto_inputs, *pto_outputs, win_size)

    elif not is_decode and actual_seq_list_q is not None and mask is None:
        print("Using prefill kernel")
        inputs[actual_seq_list_q] = []
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        win_atten_main_tnd_prefill(*pto_inputs, *pto_outputs, win_size)

    elif not is_decode and actual_seq_list_q is not None and mask is not None:
        print("Using prefill mask kernel")
        inputs[actual_seq_list_q] = []
        inputs[mask] = []
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        win_atten_main_tnd_prefill_mask(*pto_inputs, *pto_outputs, win_size)

    elif is_decode and mask is not None and actual_seq_list_q is not None:
        print("Using mtp decode mask tnd kernel")
        inputs[actual_seq_list_q] = []
        inputs[mask] = []
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        win_atten_main_tnd_mtp_decode_mask(*pto_inputs, *pto_outputs, win_size)

    else:
        print("Please check your params.")
        
    pypto.runtime._device_synchronize()
    return atten_out