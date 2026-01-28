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


def check_args_tnd(
            q_tnd: torch.Tensor,
            block_table: torch.Tensor,
            kv_cache: torch.Tensor,
            seqused_kv: torch.Tensor,
            atten_sink: torch.Tensor,
            win_size: int,
            actual_seq_list_q: torch.Tensor,
):
    print("start tnd args check...")
    
    assert q_tnd != None and block_table != None and kv_cache != None and seqused_kv != None and \
        atten_sink != None and actual_seq_list_q != None

    assert q_tnd.dtype == torch.bfloat16 and q_tnd.ndim == 3 and q_tnd.shape[1] == 64 and q_tnd.shape[2] == 512, \
        f"q dtype is {q_tnd.dtype}, ndim is {q_tnd.ndim}, axis2 is {q_tnd.shape[1]}, axis3 is {q_tnd.shape[2]}"

    assert block_table.ndim == 2, f"block_table ndim is {block_table.ndim}"

    assert kv_cache.dtype == torch.bfloat16 and kv_cache.ndim == 4 and kv_cache.shape[1] == 128 and \
        kv_cache.shape[2] == 1 and kv_cache.shape[3] == 512, \
        f"kv_cache dtype is {kv_cache.dtype}, ndim is {kv_cache.ndim}, axis2 is {kv_cache.shape[1]}, \
        axis3 is {kv_cache.shape[2]}, axis4 is {kv_cache.shape[3]}"

    assert atten_sink.dtype == torch.float32 and atten_sink.ndim == 1 and atten_sink.shape[0] == 64, \
        f"atten_sink dtype is {atten_sink.dtype}, ndim is {atten_sink.ndim}, axis1 is {atten_sink.shape[0]}"

    assert seqused_kv.dtype == torch.int and seqused_kv.ndim == 1, \
        f"seqused_kv dtype is {seqused_kv.dtype}, ndim is {seqused_kv.ndim}"

    assert win_size == 128, f"win_size is {win_size}"

    assert actual_seq_list_q.dtype == torch.int and actual_seq_list_q.ndim == 1 and \
        actual_seq_list_q.shape[0] == seqused_kv.shape[0] + 1, \
        f"actual_seq_list_q dtype is {actual_seq_list_q.dtype}, ndim is {actual_seq_list_q.ndim}, \
        axis1 is {actual_seq_list_q.shape[0]}, seqused_kv axis1 is {seqused_kv.shape[0]}"


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"stitch_function_inner_memory": 1024,
                    "stitch_function_outcast_memory": 1024,
                    "stitch_function_num_initial": 128},
    pass_options={"cube_nbuffer_setting": {1: 2},
                "vec_nbuffer_mode": 2,
                "vec_nbuffer_setting": {-1: 4}}
)
def win_atten_main_tnd_mask(q_tnd, block_table, kv_cache, seqused_kv_list, atten_sink, \
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
            start_block = cur_start_s2_pos // block_size
            start_block_offset = cur_start_s2_pos % block_size
            end_block = cur_end_s2_pos // block_size
            end_block_offset = cur_end_s2_pos % block_size

            pypto.set_vec_tile_shapes(128, 128)
            atten_sink_2d = pypto.reshape(atten_sink, [atten_sink.shape[0], 1], inplace=True)
            atten_sink_2d = pypto.concat([atten_sink_2d, atten_sink_2d, atten_sink_2d, atten_sink_2d], dim=0)
            atten_sink_2d_temp = atten_sink_2d

            if pypto.cond(start_block + 2 == end_block):

                acc_s = pypto.tensor([4 * n_q, block_size * 3], pypto.DT_FP32, "acc_s")
                kv_block_temp = pypto.tensor([block_size * 3, d_kv], dtype, "kv_block_temp")

                start_block_id = block_table[b_idx, start_block]
                kv_block_0 = pypto.view(kv_2d, [block_size, d_kv], [start_block_id * block_size, 0])
                mid_block_id = block_table[b_idx, start_block + 1]
                kv_block_1 = pypto.view(kv_2d, [block_size, d_kv], [mid_block_id * block_size, 0])
                end_block_id = block_table[b_idx, end_block]
                kv_block_2 = pypto.view(kv_2d, [block_size, d_kv], [end_block_id * block_size, 0])

                pypto.set_cube_tile_shapes([256, 256], [128, 128 * 2], [128, 128], True, False)
                acc_s_0 = pypto.matmul(q_tensor_cur, kv_block_0, pypto.DT_FP32, b_trans=True)
                acc_s_1 = pypto.matmul(q_tensor_cur, kv_block_1, pypto.DT_FP32, b_trans=True)
                acc_s_2 = pypto.matmul(q_tensor_cur, kv_block_2, pypto.DT_FP32, b_trans=True)
                pypto.assemble(acc_s_0, [0, 0], acc_s)
                pypto.assemble(acc_s_1, [0, block_size], acc_s)
                pypto.assemble(acc_s_2, [0, block_size * 2], acc_s)

                pypto.set_vec_tile_shapes(64, 256)
                mask_block = pypto.view(mask2, [4 * n_q, block_size * 3], [0, 128 - start_block_offset], \
                    valid_shape=[valid_group_len * n_q, block_size * 3])
                acc_s = pypto.where(mask_block, acc_s, float("-inf"))
                
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
                div_res_b16 = pypto.cast(div_res, dtype)

                pypto.assemble(kv_block_0, [0, 0], kv_block_temp)
                pypto.assemble(kv_block_1, [block_size, 0], kv_block_temp)
                pypto.assemble(kv_block_2, [block_size * 2, 0], kv_block_temp)
                pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128], False, False) 
                mm2_res = pypto.matmul(div_res_b16, kv_block_temp, dtype)

                pypto.assemble(mm2_res, [cur_offset, 0], atten_out)

            elif pypto.cond(start_block + 1 == end_block):

                acc_s = pypto.tensor([4 * n_q, block_size * 2], pypto.DT_FP32, "acc_s")
                kv_block_temp = pypto.tensor([block_size * 2, d_kv], dtype, "kv_block_temp")

                start_block_id = block_table[b_idx, start_block]
                kv_block_0 = pypto.view(kv_2d, [block_size, d_kv], [start_block_id * block_size, 0])
                end_block_id = block_table[b_idx, end_block]
                kv_block_1 = pypto.view(kv_2d, [block_size, d_kv], [end_block_id * block_size, 0])

                pypto.set_cube_tile_shapes([256, 256], [128, 128 * 2], [128, 128], True, False)
                acc_s_0 = pypto.matmul(q_tensor_cur, kv_block_0, pypto.DT_FP32, b_trans=True)
                acc_s_1 = pypto.matmul(q_tensor_cur, kv_block_1, pypto.DT_FP32, b_trans=True)
                pypto.assemble(acc_s_0, [0, 0], acc_s)
                pypto.assemble(acc_s_1, [0, block_size], acc_s)

                pypto.set_vec_tile_shapes(64, 256)
                mask_block = pypto.view(mask2, [4 * n_q, block_size * 2], [0, 128 - start_block_offset], \
                    valid_shape=[valid_group_len * n_q, block_size * 2])
                acc_s = pypto.where(mask_block, acc_s, float("-inf"))
                
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
                div_res_b16 = pypto.cast(div_res, dtype)

                pypto.assemble(kv_block_0, [0, 0], kv_block_temp)
                pypto.assemble(kv_block_1, [block_size, 0], kv_block_temp)
                pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128], False, False) 
                mm2_res = pypto.matmul(div_res_b16, kv_block_temp, dtype)

                pypto.assemble(mm2_res, [cur_offset, 0], atten_out)
        
            elif pypto.cond(start_block == end_block):

                start_block_id = block_table[b_idx, start_block]
                kv_block = pypto.view(kv_2d, [block_size, d_kv], [start_block_id * block_size, 0])

                pypto.set_cube_tile_shapes([256, 256], [128, 128 * 2], [128, 128], True, False)
                acc_s = pypto.matmul(q_tensor_cur, kv_block, pypto.DT_FP32, b_trans=True)

                pypto.set_vec_tile_shapes(64, 256)
                mask_block = pypto.view(mask2, [4 * n_q, block_size], \
                    [0, 255 + valid_group_len - 1 - end_block_offset], valid_shape=[valid_group_len * n_q, block_size])
                acc_s = pypto.where(mask_block, acc_s, float("-inf"))
                
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
                div_res_b16 = pypto.cast(div_res, dtype)

                pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128], False, False) 
                mm2_res = pypto.matmul(div_res_b16, kv_block, dtype)

                pypto.assemble(mm2_res, [cur_offset, 0], atten_out)


@allow_in_graph
def deepseekv4_win_atten(q: torch.Tensor,
                        ori_block_table: torch.Tensor,
                        ori_kv: torch.Tensor,
                        seqused_kv: torch.Tensor,
                        attn_sinks: torch.Tensor,
                        win_size: int,
                        mask: torch.Tensor,
                        actual_seq_list_q: torch.Tensor,
) -> None:
    """
    """
    check_args_tnd(q, ori_block_table, ori_kv, seqused_kv, attn_sinks, win_size, actual_seq_list_q)
    atten_out = torch.zeros([q.shape[0] * q.shape[1], q.shape[2]], dtype=q.dtype, device=q.device)

    inputs = {
        q: [],
        ori_block_table: [],
        ori_kv: [],
        seqused_kv: [],
        attn_sinks: [],
        actual_seq_list_q: [],
        mask: []
    }
    outputs = {
        atten_out: [],
    }

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    print("Using tnd mask kernel")
    win_atten_main_tnd_mask(*pto_inputs, *pto_outputs, win_size)

    pypto.runtime._device_synchronize()
    return atten_out