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
deepseekv4 Attention Module

This module implements the Attention mechanism for deepseekv4 model, which uses
a paged memory management approach similar to operating systems to efficiently
handle variable-length sequences and dynamic batch sizes in attention computation.

Main Functions:
    - attention: Main attention function with Attention support
    - ifa_flash: JIT compiled kernel implementing Flash Attention with paged KV cache
    - gen_block_table: Generate block mapping table for Attention
    - kv_cache_concat_bsnd: Convert paged KV cache to BSND format
"""

from dataclasses import dataclass
import torch
import pypto
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph


def check_args(
        q,
        cmp_kv,
        sinks,
        cmp_block_table,
        seqused_kv,
        ori_kv,
        ori_block_table,
):
    assert q.dim() == 3 and q.size(1) == 64 and q.size(2) == 512, \
        f"q dim num is {q.dim()}, q axis1 is {q.size(1)}, q axis2 is {q.size(2)}, expected 3, 64, 512"
    assert cmp_kv.dim() == 4 and cmp_kv.size(1) == 128 and cmp_kv.size(2) == 1 and cmp_kv.size(3) == 512, \
        f"cmp_kv dim num is {cmp_kv.dim()}, cmp_kv axis1 {cmp_kv.size(1)}, cmp_kv axis2 {cmp_kv.size(2)}, \
            cmp_kv axis3 {cmp_kv.size(3)}, expected 4, 128, 1, 512"
    assert sinks.dim() == 1 and sinks.size(0) == 64, f"sinks dim num {sinks.dim()}, \
            sinks axis0 is {sinks.size(0)}, expected 1, 64"
    assert cmp_block_table.dim() == 2, f"cmp_block_table dim num {cmp_block_table.dim()}, expected 2"
    assert seqused_kv.dim() == 1, f"seqused_kv dim num {seqused_kv.dim()}, expected 1"
    assert ori_kv.dim() == 4 and ori_kv.size(1) == 128 and ori_kv.size(2) == 1 and ori_kv.size(3) == 512, \
        f"ori_kv dim num {ori_kv.dim()}, ori_kv axis1 {ori_kv.size(1)}, ori_kv axis2 {ori_kv.size(2)}, \
            ori_kv axis3 {ori_kv.size(3)}, expected 4, 128, 1, 512"
    assert ori_block_table.dim() == 2, f"ori_block_table dim num {ori_block_table.dim()}, expected 2"


@allow_in_graph
def cfa_attention(
        q: torch.Tensor,
        cmp_kv: torch.Tensor,
        sinks: torch.Tensor,
        cmp_block_table: torch.Tensor,
        seqused_kv: torch.Tensor,
        ori_kv: torch.Tensor,
        ori_block_table: torch.Tensor,
        cmp_ratio: int = 128,
) -> torch.Tensor:
    """
    Main attention function with Attention support.

    This function implements scaled dot-product attention using Attention
    mechanism, which efficiently handles variable-length sequences and dynamic
    batch sizes by managing KV cache in non-contiguous blocks.

    Args:
        q: Query tensor with shape [num_tokens, num_head, head_size]
        cmp_kv: Compressed key cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        sinks: The attention is applied to the tensor with shape is [n_q].
        cmp_block_table: Compressed block mapping table with shape [b, max_blocks]
        seqused_kv: Actual sequence lengths with shape [batch_size]
        ori_kv: Uncompressed key cache tensor with shape [block_num, ori_block_size, KV_N, D]
        ori_block_table: Uncompressed block mapping table with shape [b, num_head, head_size]
        cmp_ratio: Compression ratio of ori_kv. The data type can be `int`, and the value range is 4/128

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(q, FakeTensor):
        return
    check_args(
        q,
        cmp_kv,
        sinks,
        cmp_block_table,
        seqused_kv,
        ori_kv,
        ori_block_table,
    )
    attention_out = torch.zeros([q.size(0) * q.size(1), q.size(2)], dtype=q.dtype, device=f'{q.device}')
    unroll_list = [32]
    inputs = {
        q: [0],
        cmp_kv: [0],
        sinks: None,
        cmp_block_table: [0],
        seqused_kv: [0],
        ori_kv: [0],
        ori_block_table: [0]
    }
    outputs = {
        attention_out: [0],
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    c128_decode(*pto_inputs, *pto_outputs, cmp_ratio, unroll_list)
    attention_out = attention_out.reshape(q.shape)
    return attention_out


def kernel(q, cmp_kv, sinks, cmp_block_table, seqused_kv, ori_kv, ori_block_table, atten_out, cmp_ratio, unroll_list):
    pypto.experimental.set_operation_config(combine_axis=True)
    shape_q = q.shape
    shape_k = cmp_kv.shape
    shape_k_win = ori_kv.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_num_win_scalar = shape_k_win[0]
    blk_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    softmax_scale = dn ** -0.5
    b_scalar = seqused_kv.shape[0]
    dtype = q.dtype
    m_tile = 128
    cube_tile = 128
    k_cube_tile = 256
    cfa_s2_tile = blk_size
    combine_s2_tile = cfa_s2_tile + blk_size
    g_tile = nq
    v1_win_tile = [m_tile * 2, dn]
    c1_tile = [[m_tile, m_tile], [k_cube_tile, k_cube_tile], [cube_tile, cube_tile]]
    v1_tile = [m_tile, combine_s2_tile]
    c2_tile = [[m_tile, m_tile], [cube_tile, cube_tile], [k_cube_tile, k_cube_tile]]
    s1_s = bs_scalar // b_scalar
    g = nq // nkv
    kv_2d_shape = (block_num_scalar * blk_size, nkv * dn)
    kv_win_2d_shape = (block_num_win_scalar * blk_size, nkv * dn)
    q_2d_shape = (b_scalar * s1_s * nq, dn)
    attn_sink_2d_shape = (nq, 1)
    pypto.set_vec_tile_shapes(v1_win_tile[0], v1_win_tile[1])
    kv_2d = pypto.reshape(cmp_kv, kv_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    kv_win_2d = pypto.reshape(ori_kv, kv_win_2d_shape, inplace=True)
    win = 128
    attn_sink_2d = pypto.reshape(sinks, attn_sink_2d_shape, inplace=True)
    for bs_idx in pypto.loop(bs_scalar, name="LOOP_T", idx_name="idx", unroll_list=unroll_list):
        b_idx = bs_idx // s1_s
        s1_idx = bs_idx % s1_s
        vld_cmp_seq = (seqused_kv[b_idx] - ((s1_s - 1) - s1_idx)) // cmp_ratio
        pypto.set_pass_options(sg_set_scope=2)
        bs_ofs = b_idx * s1_s + s1_idx
        oi_ofs = [bs_ofs * g, 0]
        vld_len = seqused_kv[b_idx] - (s1_s - 1 - s1_idx)
        vld_win_len = pypto.min(vld_len, win)
        vld_start_pos = vld_len - vld_win_len
        vld_end_pos = vld_len - 1
        start_ofs = vld_start_pos % blk_size
        start_blk = vld_start_pos // blk_size
        end_blk = vld_end_pos // blk_size
        start_blk_id = ori_block_table[b_idx, start_blk].max(0)
        start_blk = pypto.view(kv_win_2d, [blk_size, dn], [start_blk_id * blk_size, 0])
        end_blk_id = ori_block_table[b_idx, end_blk].max(0)
        end_blk = pypto.view(kv_win_2d, [blk_size, dn], [end_blk_id * blk_size, 0])

        win_blks = pypto.concat([start_blk, end_blk], dim=0)
        vld_win_blk = pypto.view(win_blks, [win, dn], [start_ofs, 0], valid_shape=[vld_win_len, dn])

        kv_assemble = pypto.tensor([combine_s2_tile, dn], kv_2d.dtype, "kj_assemble")
        kv_assemble[0:blk_size, :] = vld_win_blk
        blk_idx = cmp_block_table[b_idx, 0]
        blk_idx_valid = blk_idx.max(0)
        kv_assemble[blk_size:, :] = pypto.view(kv_2d, [blk_size, dn], [blk_idx_valid * blk_size, 0])

        qi = pypto.view(q_2d, [g_tile, dn], oi_ofs)
        pypto.set_pass_options(sg_set_scope=-1)
        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
        mm1 = pypto.matmul(qi, kv_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
        pypto.set_pass_options(sg_set_scope=1)
        mm1 = pypto.view(mm1, [g_tile, combine_s2_tile], [0, 0], valid_shape=[g_tile, blk_size + vld_cmp_seq])
        muls = pypto.mul(mm1, softmax_scale)
        max = pypto.amax(muls, dim=-1, keepdim=True)
        sub = pypto.sub(muls, max)
        exp = pypto.exp(sub)

        sum = pypto.sum(exp, dim=-1, keepdim=True)
        sum_local = pypto.add(sum, pypto.exp(attn_sink_2d - max))
        softmax = pypto.div(exp, sum_local)
        softmax_16 = pypto.cast(softmax, dtype)
        pypto.set_pass_options(sg_set_scope=-1)
        pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
        out_view = pypto.matmul(softmax_16, kv_assemble, dtype)
        atten_out[bs_ofs * g:, :] = out_view


@pypto.jit(
    runtime_options={"stitch_function_num_initial": 128,
                     "stitch_function_outcast_memory": 128,
                     "stitch_function_inner_memory": 128,
                     "device_sched_mode": 1},
    
    # 当子图大小达到上界不允许与其他子图合并
    pass_options={
                 "cube_l1_reuse_setting": {-1: 2},
                  "cube_nbuffer_setting":{-1:2},
                  "vec_nbuffer_mode":2,
                  "vec_nbuffer_setting": {-1:4}}
)
def c128_decode(q, cmp_kv, sinks, cmp_block_table, seqused_kv, ori_kv, ori_block_table, \
              atten_out, cmp_ratio, unroll_list):
    kernel(q, cmp_kv, sinks, cmp_block_table, seqused_kv, ori_kv, ori_block_table, 
              atten_out, cmp_ratio, unroll_list)
