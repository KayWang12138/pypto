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
Sparse Flash Attention Quantization Module

This module implements sparse flash attention with quantization support for DeepSeek V4.
It performs attention computation on top-k selected key-value pairs from cache,
supporting both standard and flash attention algorithms.

Main Functions:
    - sparse_flash_attention_compute: Standard sparse attention computation
    - sparse_flash_attention_compute_flash: Flash attention variant with online softmax
    - sparse_flash_attention_d: JIT-compiled decode version
    - sparse_flash_attention_p: JIT-compiled prefill version

Example:
    See deepseekv4_sparse_flash_attention.py for usage examples.
"""
from dataclasses import dataclass
import math
import os
import pypto
import numpy as np
import torch
import torch_npu
from pypto.experimental import gather_in_ub
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import FakeTensor

MAX_S2 = 65536

@dataclass
class SaTileShapeConfig:
    g_tile: int
    s_kv_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


def compress_sparse_flash_attention_compute(query, ori_kv, cmp_kv, ori_block_table, cmp_block_table, atten_sink,
                                   seqused_kv, cmp_sparse_indices,
                                   attention_out, nq, n_kv, softmax_scale, topk,
                                   block_size, win_size, cmp_ratio, tile_config):
    """Compute sparse compress flash attention.
    """
    dtype = query.dtype
    d = query.shape[1]
    group = nq // n_kv
    group_tile = tile_config.g_tile
    s_kv_tile = tile_config.s_kv_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    n_kv_sym = n_kv

    batch_size_sym = seqused_kv.shape[0]

    s1_n2_gsym = query.shape[0] // batch_size_sym
    s1_sym = s1_n2_gsym // nq

    g_loop_sym = group // group_tile

    atten_sink_2d = pypto.reshape(atten_sink, [atten_sink.shape[0], 1], inplace=True)
    topk_tile = topk
    s2_tile = win_size + topk_tile

    atten_out_2dim = pypto.tensor([batch_size_sym * s1_n2_gsym, d], dtype, "attenOut2Dim")

    for batch_idx in pypto.loop(0, batch_size_sym, 1, name="LOOP_L0_idx", idx_name="bIdx"):
        ori_act_seq = seqused_kv[batch_idx]
        for slc_idx in pypto.loop(0, s1_sym, 1, name="LOOP_L1_s1_SA", idx_name="s1Idx"):
            cur_win_size = (win_size - s1_sym + 1 + slc_idx).max(0).min(ori_act_seq) # for 非MTP
            cur_topk_size = (ori_act_seq // cmp_ratio - s1_sym + 1 + slc_idx).max(0).min(topk)
            cur_s2_tile = cur_win_size + cur_topk_size

            for n_kv_idx in pypto.loop(0, n_kv_sym, 1, name="LOOP_L2_n_kv_SA", idx_name="n_kvIdx"):
                for group_idx in pypto.loop(0, g_loop_sym, 1, name="LOOP_L3_g_SA", idx_name="gIdx"):
                    cur_group_tile = group_tile
                    cur_offset = batch_idx * s1_n2_gsym + slc_idx * nq + n_kv_idx * group + group_idx * cur_group_tile
                    for _, _ in pypto.loop_unroll(0, 1, 1, name="LOOP_L4_s2_SA", idx_name="s2_idx"): # 非Flash: wfa+sfa

                        pypto.set_semantic_label("Sa_V0")
                        # ---- gather: GM --> UB  [topk_tile, d]
                        pypto.set_vec_tile_shapes(64, 512)
                        cur_cmp_sparse_indices = pypto.view(cmp_sparse_indices, [1, topk_tile], [batch_idx * s1_sym + slc_idx, 0], valid_shape=[1, cur_topk_size])
                        cur_block_table = pypto.view(cmp_block_table, [1, MAX_S2 // block_size], [batch_idx, 0])
                        cmp_kv_view = pypto.view(cmp_kv, [topk_tile, d], [0, 0], valid_shape=[cur_topk_size, d])
                        compress_kv = gather_in_ub(cmp_kv_view, cur_cmp_sparse_indices, cur_block_table, block_size, -2)

                        # ---- window select: GM --> UB  [win_tile, d]
                        pypto.set_vec_tile_shapes(64, 512)
                        cur_block_idx = ori_block_table[batch_idx, 0] # for 非mtp
                        win_kv = pypto.view(ori_kv, [win_size, d], [cur_block_idx * block_size, 0], valid_shape=[cur_win_size, d])

                        kj = pypto.tensor([s2_tile, d], dtype, "kj")
                        pypto.assemble(pypto.clone(win_kv), [0, 0], kj)
                        pypto.assemble(compress_kv, [cur_win_size, 0], kj)

                        # C1
                        pypto.set_semantic_label("Sa_C1")
                        pypto.set_cube_tile_shapes([c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]], 
                                                    [c1_tile[4], c1_tile[5]], enable_multi_data_load=True)
                        qv = pypto.view(query, [cur_group_tile, d], [cur_offset, 0], valid_shape=[cur_group_tile, d])
                        cmp_kv_after_gather = pypto.view(kj, [s2_tile, d], [0, 0], valid_shape=[cur_s2_tile, d])
                        sij = pypto.matmul(qv, cmp_kv_after_gather, pypto.DT_FP32, a_trans=False, b_trans=True)

                        pypto.set_semantic_label("Sa_V1")
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij_reduce = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        t_sub = pypto.sub(sij_scale, tilda_mij_reduce)
                        tilda_pij = pypto.exp(t_sub)
                        tilda_lij_reduce = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                        sink_sub_res = pypto.sub(atten_sink_2d, tilda_mij_reduce)
                        sink_exp_res = pypto.exp(sink_sub_res)
                        tilda_lij_reduce = pypto.add(tilda_lij_reduce, sink_exp_res)
                        t_softmax = pypto.div(tilda_pij, tilda_lij_reduce)
                        tilda_pij_f16 = pypto.cast(t_softmax, dtype)

                        pypto.set_semantic_label("Sa_C2")
                        pypto.set_cube_tile_shapes([c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], 
                                                    [c2_tile[4], c2_tile[5]], enable_multi_data_load=True)
                        pypto.set_matrix_size([tilda_pij_f16.shape[0],
                                               tilda_pij_f16.shape[1], kj.shape[1]])
                        vj = pypto.view(kj, [s2_tile, d], [0, 0], valid_shape=[cur_s2_tile, d])
                        q1 = pypto.matmul(tilda_pij_f16, vj, dtype)

                        pypto.assemble(q1, [cur_offset, 0], atten_out_2dim)

                        attention_out[:] = pypto.reshape(atten_out_2dim,
                            [attention_out.shape[0], attention_out.shape[1], attention_out.shape[2], attention_out.shape[3]], inplace=True)


@pypto.jit(
    pass_options={
        "mg_copyin_upper_bound": 2 * 1024 * 1024,
        "pg_upper_bound": 50000,
        "pg_lower_bound": 512,
        "pg_parallel_lower_bound": 20,
        "vec_nbuffer_mode": 2,
        "vec_nbuffer_setting": {-1: 2},
        "cube_l1_reuse_mode": 2
    },
    runtime_options={
        "stitch_function_num_initial": 128,
        "stitch_function_inner_memory": 1024,
        "stitch_function_outcast_memory": 1024,
        "device_sched_mode": 3
    },
    host_options={"only_codegen": True}
)

def compress_sparse_flash_attention_d(query, ori_kv, cmp_kv, ori_block_table, cmp_block_table, atten_sink,
                             seqused_kv, cmp_sparse_indices,
                             attention_out, nq, n_kv, softmax_scale, topk,
                             block_size, win_size, cmp_ratio, tile_config):
    """JIT-compiled sparse compress flash attention for decode phase.
    """
    pypto.set_debug_options(runtime_debug_mode=2)

    pypto.experimental.set_operation_config(combine_axis=True)

    compress_sparse_flash_attention_compute(query, ori_kv, cmp_kv, ori_block_table, cmp_block_table, atten_sink,
                                   seqused_kv, cmp_sparse_indices,
                                   attention_out, nq, n_kv, softmax_scale, topk,
                                   block_size, win_size, cmp_ratio, tile_config)


@allow_in_graph
def npu_compress_sparse_flash_attention(query_npu, ori_kv_npu, cmp_kv_npu, ori_block_table_npu, cmp_block_table_npu, atten_sink_npu,
                                    seqused_kv_npu, cmp_sparse_indices_npu):
    tile_config = SaTileShapeConfig(
        g_tile=64,
        s_kv_tile=2048,
        c1_tile_shape=[64, 64, 128, 640, 128, 128],
        v1_tile_shape=[32, 640],
        c2_tile_shape=[64, 64, 128, 640, 128, 128],
        v2_tile_shape=[64, 128]
    )


    attention_out_npu = torch.zeros([ori_block_table_npu.size(0), cmp_sparse_indices_npu.size(0) // ori_block_table_npu.size(0), query_npu.size(0) // cmp_sparse_indices_npu.size(0), query_npu.size(1)], dtype=query_npu.dtype, device=f'{query_npu.device}')
    #attention_out_npu = torch.zeros([ori_block_table_npu.size(0), cmp_sparse_indices_npu.size(0) // ori_block_table_npu.size(0), query_npu.size(0) // cmp_sparse_indices_npu.size(0), query_npu.size(1)], dtype=query_npu.dtype, device=f'{query_npu.device}')
    if isinstance(query_npu, FakeTensor):
        return query_npu
    
    # 确定值先写死，确定整网接口有哪些传参后修改acl graph接口
    nq = 64
    n_kv = 1
    softmax_scale = 512 ** -0.5
    topk = 512
    block_size = 128
    win_size = 128
    cmp_ratio = 4


    query_pto = pypto.from_torch(query_npu, dynamic_axis=[0], name="q_nope")
    ori_kv_pto = pypto.from_torch(ori_kv_npu, dynamic_axis=[0], name="ori_kv")
    cmp_kv_pto = pypto.from_torch(cmp_kv_npu, dynamic_axis=[0], name="cmp_kv")
    ori_block_table_pto = pypto.from_torch(ori_block_table_npu, dynamic_axis=[0], name="ori_block_table")
    cmp_block_table_pto = pypto.from_torch(cmp_block_table_npu, dynamic_axis=[0], name="cmp_block_table")
    atten_sink_pto = pypto.from_torch(atten_sink_npu, name="atten_sink")
    seqused_kv_pto = pypto.from_torch(seqused_kv_npu, dynamic_axis=[0], name="seqused_kv")
    cmp_sparse_indices_pto = pypto.from_torch(cmp_sparse_indices_npu, dynamic_axis=[0], name="cmp_sparse_indices")
    
    attention_out_pto = pypto.from_torch(attention_out_npu, dynamic_axis=[0, 1], name="calc_attention_out")

    pto_inputs = [query_pto, ori_kv_pto, cmp_kv_pto, ori_block_table_pto, cmp_block_table_pto, atten_sink_pto, seqused_kv_pto, cmp_sparse_indices_pto]
    pto_outputs = [attention_out_pto]

    compress_sparse_flash_attention_d(*pto_inputs, *pto_outputs, nq, n_kv, softmax_scale, topk,
                                    block_size, win_size, cmp_ratio, tile_config)
    return attention_out_npu