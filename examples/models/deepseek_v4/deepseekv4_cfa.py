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
GLM-4.5 Attention Module

This module implements the Attention mechanism for GLM-4.5 model, which uses
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
import pytest
import numpy as np
import os
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from utils.get_format import get_format


np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


def check_args(
    query,
    key_cache,
    value_cache,
    blk_tbl,
    actual_seqs,
    attn_res
):
    assert query.dim() == 3
    assert get_format(query) == 'ND'
    assert query.dtype == torch.bfloat16
    assert key_cache.dim() == 4
    assert get_format(key_cache) == 'ND'
    assert key_cache.dtype == torch.bfloat16
    assert value_cache.dim() == 4
    assert get_format(value_cache) == 'ND'
    assert value_cache.dtype == torch.bfloat16
    assert blk_tbl.dim() == 2
    assert get_format(blk_tbl) == 'ND'
    assert blk_tbl.dtype == torch.int32
    assert actual_seqs.dim() == 1
    assert get_format(actual_seqs) == 'ND'
    assert actual_seqs.dtype == torch.int32
    assert attn_res.dim() == 3
    assert get_format(attn_res) == 'ND'
    assert attn_res.dtype == torch.bfloat16

@dataclass
class AttentionConfig:
    b: int
    s1: int
    s2: int
    n1: int
    n2: int
    q_d: int
    kv_d: int
    block_size: int = 128
    cmp_r: int = 128
    max_blocks: int = 0
    actual_seq: torch.Tensor = None  # 改为 torch.Tensor 类型
    block_table_batch: int = 0
    kv_num_blocks: int = 0


def get_case_info(device="cpu"):
    b = 4
    s1 = 4
    s2 = 128 * 1024
    q_d = 512
    nq = 64
    nkv = 1
    block_table_batch = b
    block_size = 128
    cmp_r = 128
    kv_num_blocks = b * ((s2 + block_size - 1) // block_size)
    actual_seq_values = [s2] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)
    attn_cfg = AttentionConfig(b=b, s1=s1, s2=s2, n1=nq, n2=nkv,
                                q_d=q_d, kv_d=q_d, block_size=block_size, block_table_batch=block_table_batch,
                                kv_num_blocks=kv_num_blocks, actual_seq=actual_seq_tensor, cmp_r=cmp_r)
    attn_cfg.max_blocks = (s2 + block_size - 1) // block_size
    return attn_cfg


@pypto.jit(
    runtime_options={"stitch_function_num_initial": 128, 
    "stitch_function_outcast_memory": 1024,
    "stitch_function_inner_memory": 1024},
    host_options={"only_codegen": True},
    debug_options={"runtime_debug_mode": 1},
    # codegen_options={"codegen_expression_fusion": True},
    # 当子图大小达到上界不允许与其他子图合并
    pass_options={"pg_upper_bound": 1536,
    # Q常驻，0代表第一组mmad，4代表4次matmul合并
    "cube_l1_reuse_setting": {0: 4}}
)
def ifa_flash(q, k, v, attn_sink, block_table, start_pos, atten_out, cmp_r=1, unroll_list=[]):
    pypto.experimental.set_operation_config(combine_axis=True)
    # pypto.set_debug_options(runtime_debug_mode=2) #开启AICPU抢跑
    shape_q = q.shape
    shape_k = k.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    softmax_scale = dn ** -0.5
    b_scalar = start_pos.shape[0]

    dtype = q.dtype
    n2_sym = nkv

    m_tile = 128
    cube_tile = 128
    s2_tile = 512

    g_tile = min(32, nq)
    c1_tile = [[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]]
    v1_tile = [m_tile, s2_tile]
    c2_tile = [[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]]
    v2_tile = [m_tile, cube_tile]

    # 3. 得到动态tensor的shape
    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)
    attn_sink_2d_shape = (nq, 1)

    k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    attn_sink_2d = pypto.reshape(attn_sink, attn_sink_2d_shape, inplace=True)
    # 4. 实现kernel逻辑，循环展开B动态轴
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = (start_pos[b_idx] + 1 + s1_idx) // cmp_r
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=unroll_list):
                    block_num = s2_tile // block_size
                    idx = s2_idx * block_num
                    bs_ofs = b_idx * s1_scalar + s1_idx
                    n1g_ofs = g_idx * g_tile
                    actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                    oi_ofs = [bs_ofs, n1g_ofs, 0]
                    # 5. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                    qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                    

                    kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                    for i in range(block_num):
                        block_idx = block_table[b_idx, idx + i]
                        block_idx_valid = block_idx.max(0)
                        kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                            pypto.view(k_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                    kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[s2_tile, dn])

                    # c1
                    # 6. 下面是flash attention的计算逻辑
                    pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                    sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False,
                                        b_trans=True)
                    sij = pypto.view(sij, [g_tile, s2_tile], [0, 0],
                                        valid_shape=[g_tile, actual_s2_tile])
                    # v1
                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                    if pypto.cond(pypto.is_loop_begin(s2_idx)):
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

                        tsub = pypto.sub(sij_scale, tilda_mij)
                        tilda_pij = pypto.exp(tsub)
                        tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                        sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                        max_update[:] = tilda_mij

                        # c2
                        vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                        vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                    [0, 0], valid_shape=[actual_s2_tile, dn])
                        pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                        oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        oi_update[:] = oi_tmp
                    else:
                        pypto.set_pass_options(sg_set_scope=1)
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        max_new = pypto.maximum(max_update, tilda_mij)
                        tsub = pypto.sub(sij_scale, max_new)
                        tilda_pij = pypto.exp(tsub)
                        tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                        sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                        pypto.set_pass_options(sg_set_scope=-1)

                        pypto.set_pass_options(sg_set_scope=2)
                        tsub2 = pypto.sub(max_update, max_new)
                        max_update[:] = max_new
                        update_mul = pypto.exp(tsub2)
                        sum_update[:] = sum_update * update_mul + sum_local
                        pypto.set_pass_options(sg_set_scope=-1)

                        #c2
                        vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                        vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                    [0, 0], valid_shape=[actual_s2_tile, dn])
                        pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                        oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

                        # v2
                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        oi_update[:] = oi_update * update_mul + oi_tmp
                    if pypto.cond(pypto.is_loop_end(s2_idx)):
                        attn_sink_tile = pypto.view(attn_sink_2d, [g_tile, 1], [0, 0])
                        sum_local = pypto.add(sum_update, attn_sink_tile)
                        oi_final = pypto.div(oi_update, sum_local)
                        
                        pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                        oi_final_3d = pypto.cast(
                            pypto.reshape(oi_final, [1, g_tile, dn]),
                            dtype)
                        # 7. 将结果搬运到输出tensor上
                        pypto.assemble(oi_final_3d, oi_ofs, atten_out)
                        
                        
class MM(torch.nn.Module):
    def forward(
        self,  
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        blk_tbl: torch.Tensor,
        start_pos: torch.Tensor,
        attn_res: torch.Tensor,
        cmp_r: int = 1,
        unroll_list: list | None = None
    ):
        attention(query, key_cache, value_cache, attn_sink, blk_tbl, start_pos, attn_res, cmp_r, unroll_list)
        return attn_res
                            
@pytest.mark.skip(reason="large test case")
def test_ifa(enable_flash: bool, enable_high_perf: bool, enable_graph: bool, device_id: int):
    device_id = max(device_id, int(os.environ.get('DEVICE_ID', 0)))
    device = f'npu:{device_id}'
    attn_cfg = get_case_info(device=device)    
    torch_dtype = torch.bfloat16
    b = attn_cfg.b
    s1 = attn_cfg.s1
    d = attn_cfg.q_d
    nq = attn_cfg.n1
    nkv = attn_cfg.n2
    cmp_r = attn_cfg.cmp_r

    block_size = attn_cfg.block_size
    max_blocks = attn_cfg.max_blocks
    orig_act_seq = attn_cfg.actual_seq

    q_shape = [b * s1, nq, d]
    kv_shape = [attn_cfg.kv_num_blocks, block_size, nkv, d]
    blk_tbl_shape = [attn_cfg.block_table_batch, max_blocks]
    
    empty_kwargs = {"dtype": torch_dtype, "device": device}
    q = torch.empty(q_shape, **empty_kwargs).uniform_(-1, 1)
    k = torch.empty(kv_shape, **empty_kwargs).uniform_(-1, 1)
    v = torch.empty(kv_shape, **empty_kwargs).uniform_(-1, 1)
    attn_sink = torch.empty(nq, dtype=torch.float32, device=device).uniform_(-1, 1)
    
    import utils.golden.attn_golden as attn_golden
    output = torch.zeros(q_shape, **empty_kwargs)
    output_flash = torch.zeros(q_shape, **empty_kwargs)

    blk_tbl = attn_golden.gen_block_table(orig_act_seq, block_size, blk_tbl_shape, cmp_r=cmp_r)
    start_pos = orig_act_seq - s1
    out_npu = torch.zeros(q_shape, **empty_kwargs)

    debug_str = os.environ.get("HIGH_PERFORMANCE", "False").lower()
    unroll_list = []
    if debug_str in ["true", "1"] or enable_high_perf:
        unroll_list = [8, 4, 2, 1]    
    attention(q,k,v,attn_sink,blk_tbl,start_pos,out_npu,cmp_r,unroll_list)
    from utils.np_compare import detailed_allclose_manual as compare
    
    attn_golden.ifa_golden(q, k, v, attn_sink, blk_tbl, start_pos, output, enable_flash=False,cmp_r=cmp_r)
    attn_golden.ifa_golden(q, k, v, attn_sink, blk_tbl, start_pos, output_flash, enable_flash=True,cmp_r=cmp_r)
    threhold = 5e-4
    compare(output, output_flash, "no flash golden vs flash golden", rtol=threhold, atol=threhold)
    compare(output_flash, out_npu, "golden vs npu", rtol=threhold, atol=threhold)
    
    # acl graph
    model = torch.compile(MM(), backend="eager", dynamic=True)
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        y = model(q,k,v,attn_sink,blk_tbl,start_pos,out_npu,cmp_r,unroll_list)
    g.replay()
    pypto.runtime._device_synchronize()
    compare(output_flash, y, "golden vs graph npu", rtol=threhold, atol=threhold)   
     


@allow_in_graph
def attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_sink: torch.Tensor,
    blk_tbl: torch.Tensor,
    start_pos: torch.Tensor,
    attn_res: torch.Tensor,
    cmp_r: int = 1,
    unroll_list: list | None = None
) -> None:
    """
    Main attention function with Attention support.

    This function implements scaled dot-product attention using Attention
    mechanism, which efficiently handles variable-length sequences and dynamic
    batch sizes by managing KV cache in non-contiguous blocks.

    Args:
        query: Query tensor with shape [num_tokens, num_head, head_size]
        key_cache: Key cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        value_cache: Value cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        blk_tbl: Block mapping table with shape [batch_size, max_blocks]
        start_pos: Actual sequence lengths with shape [batch_size]
        attn_res: Output attention tensor with shape [num_tokens, num_head, head_size]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(query, FakeTensor):
        return
    check_args(
        query,
        key_cache,
        value_cache,
        blk_tbl,
        start_pos,
        attn_res
    )

    inputs = {
        query: [0],
        key_cache: [0],
        value_cache: [0],
        attn_sink: [],
        blk_tbl: [],
        start_pos: [0]
    }
    outputs = {
        attn_res: [],
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    if unroll_list is None:
        unroll_list = []
    ifa_flash(*pto_inputs, *pto_outputs, cmp_r, unroll_list)
    pypto.runtime._device_synchronize()#内部接口，不推荐使用

if __name__ == "__main__":
    import argparse as ap
    p = ap.ArgumentParser(description="参数配置")
    p.add_argument("-f", "--enable-flash", action="store_true", help="开启flash模式")
    p.add_argument("-p", "--high-perf", action="store_true", help="启用高性能模式")
    p.add_argument("-g", "--enable-graph", action="store_true", help="启用高性能模式")
    p.add_argument("-c", "--device-id", type=int, default=0, help="显卡序号，默认0")
    args = p.parse_args()
    test_ifa(enable_flash=args.enable_flash, enable_high_perf=args.high_perf, enable_graph=args.enable_graph,device_id=args.device_id)