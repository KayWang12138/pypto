#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
GLM-4.5 Attention Module with IFA (Incremental Flash Attention) and PFA (Prompt Flash Attention)

Key Differences:
    1. Query Length:
       - IFA: s1 = 1 (decode phase, one token at a time)
       - PFA: s1 >= 1 (prefill phase, full prompt processing)
    
    2. KV Cache Handling:
       - IFA: Reads existing KV cache from block_table (decode stage)
       - PFA: K and V are current input, organized by block_table (prefill stage)
    
    3. Causal Mask:
       - IFA: Each query sees all previous KV (already computed)
       - PFA: Each query position sees current and previous positions only (causal attention)
"""
import os
import sys

# 设置必需的环境变量（如果未设置）
if 'PTO_TILE_LIB_CODE_PATH' not in os.environ:
    pto_isa_path = '/mnt/workspace/gitCode/cann/mce/pto-isa'
    os.environ['PTO_TILE_LIB_CODE_PATH'] = pto_isa_path
    print(f"[INFO] Auto-setting PTO_TILE_LIB_CODE_PATH={pto_isa_path}")

if 'TILE_FWK_DEVICE_ID' not in os.environ:
    os.environ['TILE_FWK_DEVICE_ID'] = '0'
    print(f"[INFO] Auto-setting TILE_FWK_DEVICE_ID=0")

import math
from dataclasses import dataclass
import torch
import torch_npu
import pytest
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
from utils.get_format import get_format

DEBUG_PRINT = True

def debug_print(tag, msg):
    if DEBUG_PRINT:
        print(f"[DEBUG][{tag}] {msg}")

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


# ============================================================================
# 参数校验
# ============================================================================

def check_args(query, key_cache, value_cache, block_tables, actual_seqs, attn_res):
    debug_print("CHECK", f"Query shape: {query.shape}, KV shape: {key_cache.shape}")
    assert query.dim() == 3
    assert get_format(query) == 'ND'
    assert query.dtype == torch.bfloat16
    assert key_cache.dim() == 4
    assert get_format(key_cache) == 'ND'
    assert key_cache.dtype == torch.bfloat16
    assert value_cache.dim() == 4
    assert get_format(value_cache) == 'ND'
    assert value_cache.dtype == torch.bfloat16
    assert block_tables.dim() == 2
    assert get_format(block_tables) == 'ND'
    assert block_tables.dtype == torch.int32
    assert actual_seqs.dim() == 1
    assert get_format(actual_seqs) == 'ND'
    assert actual_seqs.dtype == torch.int32
    assert attn_res.dim() == 3
    assert get_format(attn_res) == 'ND'
    assert attn_res.dtype == torch.bfloat16


# ============================================================================
# 配置数据结构
# ============================================================================

@dataclass
class AttentionTileConfig:
    g_tile: int
    s2_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


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
    max_num_blocks_per_query: int = 0
    softmax_scale: float = 1.0
    kv_layout: str = "PA_BSND"
    actual_seq: torch.Tensor = None
    block_table_batch: int = 0
    kv_num_blocks: int = 0


# ============================================================================
# 配置生成函数
# ============================================================================

def get_pfa_config(device="cpu"):
    """PFA (Decode) 配置"""
    b = 4
    s1 = 32 # 16 # 4 
    s2 = 16384 # 4096 #4096 #128 #4096 # 4096边界 16374
    q_d = 128
    nq = 12
    nkv = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_table_batch = b
    block_size = 128
    kv_num_blocks = b * ((s2 + block_size - 1) // block_size)

    actual_seq_values = [s2] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    atten_cfg = AttentionConfig(
        b=b, s1=s1, s2=s2, n1=nq, n2=nkv, softmax_scale=softmax_scale, kv_layout=kv_layout,
        q_d=q_d, kv_d=q_d, block_size=block_size, block_table_batch=block_table_batch,
        kv_num_blocks=kv_num_blocks, actual_seq=actual_seq_tensor
    )
    atten_cfg.max_num_blocks_per_query = (s2 + block_size - 1) // block_size
    
    cube_tile = 128
    m_tile = 128
    s2_tile = 1024
    tile_cfg = AttentionTileConfig(
        nq,
        s2_tile,
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [m_tile, s2_tile],
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [m_tile, cube_tile])
    return atten_cfg, tile_cfg


# ============================================================================
# 辅助函数
# ============================================================================
def gen_block_table(actual_seq_len, block_size, block_table_shape):
    debug_print("GEN_BLOCK_TABLE", f"actual_seq_len: {actual_seq_len}, block_size: {block_size}, shape: {block_table_shape}")
    
    block_num_per_batch = []
    block_num = 0

    if isinstance(actual_seq_len, torch.Tensor):
        if actual_seq_len.device.type != 'cpu':
            actual_seq_len_cpu = actual_seq_len.cpu()
        else:
            actual_seq_len_cpu = actual_seq_len
        for actual_seq in actual_seq_len_cpu:
            block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
            block_num += math.ceil(actual_seq.item() / block_size)
    else:
        for actual_seq in actual_seq_len:
            block_num_per_batch.append(math.ceil(actual_seq / block_size))
            block_num += math.ceil(actual_seq / block_size)

    # 原始代码（已注释）：
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]  # 随机打乱
    
    # # 新代码：按顺序生成 block 索引（因果注意力需要顺序访问）
    # block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
    # block_idx = 0
    # block_table_batch_idx = 0
    
    # for idx in block_num_per_batch:
    #     for j in range(idx):
    #         # 新方式：直接按顺序分配 block 索引
    #         block_table[block_table_batch_idx][j] = block_idx
    #         block_idx += 1
    #     block_table_batch_idx += 1
    # # ==========

    block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    
    debug_print("GEN_BLOCK_TABLE", f"Generated block_table with {block_num} blocks")
    return block_table


def kv_cache_concat_bsnd(kr_cache_out, kv_cache_out, block_table, atten_config):
    b = atten_config.b
    n2 = atten_config.n2
    kv_lora_rank = atten_config.q_d
    rope_dim = atten_config.kv_d
    block_size = atten_config.block_size
    kv_cache_actual_seq = atten_config.actual_seq
    dtype = kv_cache_out.dtype

    if isinstance(kv_cache_actual_seq, torch.Tensor):
        if kv_cache_actual_seq.device.type != 'cpu':
            kv_cache_actual_seq_cpu = kv_cache_actual_seq.cpu()
        else:
            kv_cache_actual_seq_cpu = kv_cache_actual_seq
        kv_max = (torch.max(kv_cache_actual_seq_cpu).item() + block_size - 1) // block_size * block_size
    else:
        kv_max = (max(kv_cache_actual_seq) + block_size - 1) // block_size * block_size

    device = kr_cache_out.device
    k_cache = torch.zeros([b, kv_max, n2, kv_lora_rank], dtype=dtype, device=device)
    v_cache = torch.zeros([b, kv_max, n2, rope_dim], dtype=dtype, device=device)

    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, kv_max, n2, kv_lora_rank], dtype=dtype, device=device)
        kv_rope_temp_tensor = torch.zeros([1, kv_max, n2, rope_dim], dtype=dtype, device=device)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size
            kv_nope_temp_tensor[:, start_idx:end_idx, :, :] = kv_cache_out[block_idx:block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, start_idx:end_idx, :, :] = kr_cache_out[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        v_cache[b_idx:b_idx + 1, :, :, :] = kv_nope_temp_tensor
        k_cache[b_idx:b_idx + 1, :, :, :] = kv_rope_temp_tensor

    return k_cache, v_cache


def softmax(x, is_fp16=False):
    if is_fp16:
        original_dtype = x.dtype
        x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    if is_fp16:
        ans = ans.to(original_dtype)
    return ans, x_max, x_sum


def pfa_func(q_shape, kv_shape, block_table_shape):
    debug_print("PFA_FUNC", f"Creating PFA kernel with q_shape={q_shape}, kv_shape={kv_shape}")
    
    out_shape = q_shape
    q_shape = (pypto.frontend.dynamic("qshape"), q_shape[1], q_shape[2])
    kv_shape = (pypto.frontend.dynamic("kvshape"), kv_shape[1], kv_shape[2], kv_shape[3])
    bs = pypto.frontend.dynamic("bs")

    @pypto.frontend.jit(
        runtime_options={
            "stitch_function_num_initial": 128,
            "stitch_function_outcast_memory": 1024,
            "stitch_function_inner_memory": 1024
        },
        # pass_options={
        #     "pg_upper_bound": 1536,
        #     "cube_l1_reuse_setting": {0: 4}
        # },
        pass_options={
            "cube_l1_reuse_setting": {-1:16}, 
            "cube_nbuffer_setting":{-1:16}, 
            "vec_nbuffer_mode":2, 
            "vec_nbuffer_setting":{-1:8}
        },
        # verify_options = {
        #     "enable_pass_verify": True,
        #     "pass_verify_save_tensor": True
        # },
        debug_options={"runtime_debug_mode":1}
    )
    def pfa_func_kernel(
        q: pypto.Tensor(q_shape, pypto.DT_BF16),
        k: pypto.Tensor(kv_shape, pypto.DT_BF16),
        v: pypto.Tensor(kv_shape, pypto.DT_BF16),
        block_table: pypto.Tensor(block_table_shape, pypto.DT_INT32),
        kv_act_seqs: pypto.Tensor((bs,), pypto.DT_INT32),
        atten_out: pypto.Tensor(out_shape, pypto.DT_BF16)
    ):
        pypto.experimental.set_operation_options(combine_axis=True)
        
        atten_cfg, tile_cfg = get_pfa_config()
        softmax_scale = atten_cfg.softmax_scale
        
        shape_q = q.shape
        shape_k = k.shape
        bs_scalar = shape_q[0]
        nq = shape_q[1]
        block_num_scalar = shape_k[0]
        block_size = shape_k[1]
        nkv = shape_k[2]
        dn = shape_k[3]
        b_scalar = kv_act_seqs.shape[0]
        
        dtype = q.dtype
        group = nq // nkv
        n2_sym = nkv
        
        g_tile = tile_cfg.g_tile
        s2_tile = tile_cfg.s2_tile
        c1_tile = tile_cfg.c1_tile_shape
        v1_tile = tile_cfg.v1_tile_shape
        c2_tile = tile_cfg.c2_tile_shape
        v2_tile = tile_cfg.v2_tile_shape
        
        s1_scalar = bs_scalar // b_scalar
        g = nq // nkv
        g_loop = g // g_tile
        g_loop_merged = g_loop * n2_sym
        block_num = s2_tile // block_size
        
        k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
        q_2d_shape = (b_scalar * s1_scalar * nq, dn)
        
        k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
        v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
        q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
        
        for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
            for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
                cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
                s2_loop = (cur_seq + s2_tile - 1) // s2_tile
                bs_ofs = b_idx * s1_scalar + s1_idx
                  
                for g_idx_merged in pypto.loop(g_loop_merged, name="LOOP_g_merged", idx_name="g_idx_merged"): # , unroll_list=[12]
                    n2_idx = g_idx_merged // g_loop
                    g_idx = g_idx_merged % g_loop
                    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")

                    n1g_ofs = n2_idx * group + g_idx * g_tile
                    oi_ofs = [bs_ofs, n1g_ofs, 0]
                    
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8,4,2,1]):
                        idx = s2_idx * block_num
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                        
                        kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                        vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(k_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                            vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                        kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[s2_tile, dn])
                        vj_assemble = pypto.view(vj_assemble, [s2_tile, dn], [0, 0], valid_shape=[actual_s2_tile, dn])
                        
                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
                        sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
                        
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        if pypto.is_loop_begin(s2_idx):
                            pypto.set_pass_options(sg_set_scope=1)
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                            tsub = pypto.sub(sij_scale, tilda_mij)
                            tilda_pij = pypto.exp(tsub)
                            tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                            
                            sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            max_update[:] = tilda_mij
                            pypto.set_pass_options(sg_set_scope=-1)
                            
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)
                            
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update[:] = oi_tmp
                        else:
                            pypto.set_pass_options(sg_set_scope=2)
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                            max_new = pypto.maximum(max_update, tilda_mij)
                            tsub = pypto.sub(sij_scale, max_new)
                            tilda_pij = pypto.exp(tsub)
                            tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                            sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            tsub2 = pypto.sub(max_update, max_new)
                            max_update[:] = max_new
                            update_mul = pypto.exp(tsub2)
                            sum_update[:] = sum_update * update_mul + sum_local
                            pypto.set_pass_options(sg_set_scope=-1)
                            
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)
                            
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update[:] = oi_update * update_mul + oi_tmp
                        
                        if pypto.is_loop_end(s2_idx):
                            oi_final = pypto.div(oi_update, sum_update)
                            
                            pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                            oi_final_3d = pypto.cast(pypto.reshape(oi_final, [1, g_tile, dn]), dtype)
                            pypto.assemble(oi_final_3d, oi_ofs, atten_out)
    
    return pfa_func_kernel


# ============================================================================
# 对外接口
# ============================================================================

@allow_in_graph
def attention_pfa(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seqs: torch.Tensor,
    attn_res: torch.Tensor
) -> None:
    debug_print("ATTENTION_PFA", f"Input query shape: {query.shape}")
    
    if isinstance(query, FakeTensor):
        return
    check_args(query, key_cache, value_cache, block_tables, actual_seqs, attn_res)
    
    q_shape = query.shape
    kv_shape = key_cache.shape
    block_table_shape = block_tables.shape
    shapes = [q_shape, kv_shape, block_table_shape]
    inputs = [query, key_cache, value_cache, block_tables, actual_seqs, attn_res]
    pfa_func(*shapes)(*inputs)
    
    debug_print("ATTENTION_PFA", "PFA kernel execution completed")


# ============================================================================
# 测试函数
# ============================================================================
def run_pfa_test(atten_cfg):
    debug_print("PFA_TEST", "Starting PFA test...")
    
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch_dtype = torch.bfloat16
    torch.npu.set_device(int(device_id))
    
    b = atten_cfg.b
    s1 = atten_cfg.s1
    d = atten_cfg.q_d
    nq = atten_cfg.n1
    nkv = atten_cfg.n2
    block_size = atten_cfg.block_size
    max_num_blocks_per_query = atten_cfg.max_num_blocks_per_query
    kv_cache_actual_seq = atten_cfg.actual_seq
    
    debug_print("PFA_TEST", f"Config: b={b}, s1={s1}, nq={nq}, nkv={nkv}, d={d}")
    
    q_shape = [b * s1, nq, d]
    kv_shape = [atten_cfg.kv_num_blocks, block_size, nkv, d]
    block_table_shape = [atten_cfg.block_table_batch, max_num_blocks_per_query]
    
    device = f'npu:{device_id}'
    q = torch.empty(q_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    k = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    v = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    attention_output = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)
    
    debug_print("PFA_TEST", f"Created tensors: q={q.shape}, k={k.shape}, v={v.shape}")
    
    block_table = gen_block_table(kv_cache_actual_seq, block_size, block_table_shape)
    debug_print("PFA_TEST", f"Generated block_table: shape={block_table.shape}")
    k_cache_bsnd, v_cache_bsnd = kv_cache_concat_bsnd(k, v, block_table, atten_cfg)
    debug_print("PFA_TEST", f"Created tensors: k_cache_bsnd={k_cache_bsnd.shape}, v_cache_bsnd={v_cache_bsnd.shape}")
    
    debug_print("PFA_TEST", "Running PyTorch reference implementation...")
    for i in range(b):
        for j in range(s1):
            for n2_idx in range(nkv):
                # seq_len = j + 1 # 修改4 PFA 因果注意力: 位置 j 只能看到位置 0 到 j 的 KV - 启用新的计算方式

                kv_seq_len = kv_cache_actual_seq[i].item()
                seq_len = kv_seq_len - s1 + 1 + j

                q_bs = q[i * s1 + j]
                k_bs = k_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                v_bs = v_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                
                qk_bmm_res = torch.matmul(q_bs, k_bs.transpose(1, 0))
                
                # # 检查点1：保存 QK matmul 结果 (对应 kernel 的 sij)
                # if i == 0 and j == 0:
                #     qk_bmm_res.cpu().float().numpy().tofile(f"golden_sij_b{i}_s1{j}_s20.bin")
                
                qk_ele_res = qk_bmm_res * atten_cfg.softmax_scale
                softmax_res, _, _ = softmax(qk_ele_res, True)
                
                # # 检查点2：保存 softmax 结果 (对应 kernel 的 tilda_pij)
                # if i == 0 and j == 0:
                #     softmax_res.cpu().float().numpy().tofile(f"golden_softmax_b{i}_s1{j}_s20.bin")
                
                bmm2_res = torch.matmul(softmax_res, v_bs)
                
                # # 检查点3：保存最终输出 (对应 kernel 的 oi_final)
                # if i == 0 and j == 0:
                #     bmm2_res.cpu().float().numpy().tofile(f"golden_oi_final_b{i}_s1{j}_s20.bin")
                
                attention_output[i * s1 + j] = bmm2_res    
    
    block_table_torch = block_table.to(dtype=torch.int32, device=device)
    act_seq_torch = kv_cache_actual_seq.to(dtype=torch.int32, device=device)
    out_torch = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)
    
    inputs = [q, k, v, block_table_torch, act_seq_torch, out_torch]
    
    debug_print("PFA_TEST", "Running PFA kernel...")
    attention_pfa(*inputs)
    
    debug_print("PFA_TEST", "Comparing results...")
    assert_allclose(
        np.array(attention_output.cpu().flatten().tolist()),
        np.array(out_torch.cpu().flatten().tolist()),
        rtol=0.0078125, atol=0.0001
    )
    
    debug_print("PFA_TEST", "PFA test PASSED!")


@pytest.mark.skip(reason="large test case")
def test_pfa():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    device = f'npu:{device_id}'
    atten_cfg, _ = get_pfa_config(device=device)
    
    if atten_cfg.actual_seq.device.type != 'cpu':
        actual_seq_cpu = atten_cfg.actual_seq.cpu()
    else:
        actual_seq_cpu = atten_cfg.actual_seq
    
    assert atten_cfg.b == len(atten_cfg.actual_seq), \
        f'B={atten_cfg.b} must equal actual_seq length={len(atten_cfg.actual_seq)}'
    assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "All values must be <= s2"
    
    run_pfa_test(atten_cfg)


if __name__ == "__main__":
    print("=" * 60)
    print("GLM-4.5 Attention Module - IFA & PFA")
    print("=" * 60)
    print()
    print("PFA (Prompt Flash Attention):")
    print("  - For prefill phase, s1>=1")
    print("  - Causal attention (Q[i] sees K[0:i+1])")
    print()
    print("=" * 60)
    
    test_pfa()
