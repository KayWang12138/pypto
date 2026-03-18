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

def check_args(query, key_cache, value_cache, block_tables, causal_table, actual_seqs, attn_res):
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
    assert get_format(causal_table) == 'ND'
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
    """PFA (Prefill) 配置 - s1 >= 1"""
    b = 8
    s1 = 128       # PFA: query 长度可以大于 1
    s2 = s1        # PFA: KV 长度等于 query 长度
    q_d = 128
    nq = 12
    nkv = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_table_batch = b
    block_size = 128
    kv_num_blocks = b * ((s1 + block_size - 1) // block_size)

    actual_seq_values = [s1] * b  # 配置实例为长度为b的tensor，每个b代表一次任务中的token数即s1,实际任务b可能小于s1，代表传入矩阵中的有效序列长度，其余内容会padding，s1决定内存分配
    # s1负责内存对齐和批处理需求，actual_seq_values实现计算效率优化
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    atten_cfg = AttentionConfig(
        b=b, s1=s1, s2=s2, n1=nq, n2=nkv, softmax_scale=softmax_scale, kv_layout=kv_layout,
        q_d=q_d, kv_d=q_d, block_size=block_size, block_table_batch=block_table_batch,
        kv_num_blocks=kv_num_blocks, actual_seq=actual_seq_tensor
    )
    atten_cfg.max_num_blocks_per_query = (max(s1,s2) + block_size - 1) // block_size
    
    cube_tile = 128
    m_tile = 128
    s2_tile = 128
    tile_cfg = AttentionTileConfig(
        nq,
        s2_tile,
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],  #[[128,128], [128,512], [128,128]], 劣化30 # [[128,128], [64,256], [256,256]], # 劣化60 # [[256,256], [64,256], [128,128]], # 劣化50us 936
        [m_tile, s2_tile],
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]], 
        [m_tile, s2_tile],)
    return atten_cfg, tile_cfg


def get_pfa_config_v2(device="cpu"):
    """PFA (Prefill) 配置 - s1 >= 1"""
    b = 16
    s1 = 512       # PFA: query 长度可以大于 1
    s2 = 1024      # PFA: KV 长度等于 query 长度
    q_d = 128
    nq = 12
    nkv = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_table_batch = b
    block_size = 128
    kv_num_blocks = b * ((s1 + block_size - 1) // block_size)

    actual_seq_values = [s1] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    atten_cfg = AttentionConfig(
        b=b, s1=s1, s2=s2, n1=nq, n2=nkv, softmax_scale=softmax_scale, kv_layout=kv_layout,
        q_d=q_d, kv_d=q_d, block_size=block_size, block_table_batch=block_table_batch,
        kv_num_blocks=kv_num_blocks, actual_seq=actual_seq_tensor
    )
    atten_cfg.max_num_blocks_per_query = (s1 + block_size - 1) // block_size
    
    cube_tile = 128
    m_tile = 128
    s2_tile = 128
    tile_cfg = AttentionTileConfig(
        nq,
        s2_tile,
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]], # 建议128，中间64 256
        [m_tile, s2_tile],
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]], 
        [m_tile, s2_tile],)
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

    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]

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


def create_causal_mask_bak(s1_len, s2_len):
    """
    生成形状为 (s1_len, s2_len) 的因果掩码
    - s1_len: Query 序列长度
    - s2_len: Key/Value 序列长度
    """
    mask = torch.zeros((s1_len, s2_len))
    
    # 2. 生成因果布尔矩阵：j > i 的位置为 True（需要遮挡） i 是 Query 位置，j 是 Key 位置
    causal_bool = torch.arange(s2_len) > torch.arange(s1_len)[:, None]
    
    # 3. 将需要遮挡的位置设为 -inf
    mask.masked_fill_(causal_bool, float('-inf'))
    print(mask)
    return mask


def create_causal_mask(s1_len, s2_len, step):
    """
    生成形状为 (s1_len, s2_len) 的阶梯状因果掩码
    - s1_len: Query 序列长度
    - s2_len: Key/Value 序列长度
    - step: 阶梯块的单位长宽（每个块的大小）
    逻辑：
        1. 将 Query/Key 按 step 分块（如 step=4 → 0-3为第0块，4-7为第1块...）
        2. 第i个Query块能看到 0~i 个Key块的全部内容，i+1及以后的Key块遮挡
        3. 遮挡位置设为 -inf，可见位置设为 0
    """
    # 1. 初始化全0掩码矩阵
    mask = torch.zeros((s1_len, s2_len), dtype=torch.float32)
    
    # 2. 生成Query/Key的块索引（每个位置属于哪个step块）
    #    例：step=4 → 位置0-3→0，4-7→1，8-11→2...
    q_block_idx = torch.arange(s1_len) // step  # shape: (s1_len,)
    k_block_idx = torch.arange(s2_len) // step  # shape: (s2_len,)
    
    # 3. 生成阶梯状因果布尔矩阵：Key块索引 > Query块索引 时遮挡
    #    广播为 (s1_len, s2_len) 的布尔矩阵
    causal_bool = k_block_idx > q_block_idx[:, None]
    
    # 4. 将需要遮挡的位置设为 -inf
    mask.masked_fill_(causal_bool, float('-inf'))
    
    print(f"阶梯因果掩码 (s1_len={s1_len}, s2_len={s2_len}, step={step}):")
    print(mask)
    return mask

# ============================================================================
# PFA (Prompt Flash Attention) - Prefill 阶段
# ============================================================================

def pfa_func(q_shape, kv_shape, block_table_shape, causal_table_shape):
    """
    PFA Kernel - Prefill 阶段
    
    关键区别（与 IFA）:
    1. s1 >= 1（处理完整 prompt）
    2. 因果注意力：Q[i] 只能看 K[0:i+1] 和 V[0:i+1]
    3. KV 来自当前输入，按 block_table 组织
    
    因果注意力实现：
    - 对于位置 s1_idx 的 query，只能看到位置 0 到 s1_idx 的 KV
    - cur_seq = s1_idx + 1（因果掩码）
    """
    debug_print("PFA_FUNC", f"Creating PFA kernel with q_shape={q_shape}, kv_shape={kv_shape}")
    
    out_shape = q_shape
    q_shape = (pypto.frontend.dynamic("qshape"), q_shape[1], q_shape[2])
    kv_shape = (pypto.frontend.dynamic("kvshape"), kv_shape[1], kv_shape[2], kv_shape[3])
    bs = pypto.frontend.dynamic("bs")

    @pypto.frontend.jit(
        runtime_options={
            "stitch_function_max_num": 128
        }, 
        pass_options={
            "pg_upper_bound": 20000, #15360,
            "cube_l1_reuse_setting": {}, # {-1: 16},
            "cube_nbuffer_setting":{}, # {-1: 16},
            "vec_nbuffer_setting":{}
        },
        debug_options={"runtime_debug_mode":1,
                        "compile_debug_mode":0}
    )
    def pfa_func_kernel(
        q: pypto.Tensor(q_shape, pypto.DT_BF16),
        k: pypto.Tensor(kv_shape, pypto.DT_BF16),
        v: pypto.Tensor(kv_shape, pypto.DT_BF16),
        block_table: pypto.Tensor(block_table_shape, pypto.DT_INT32),
        causal_table: pypto.Tensor(causal_table_shape,  pypto.DT_FP32),
        query_act_seqs: pypto.Tensor((bs,), pypto.DT_INT32),
        atten_out: pypto.Tensor(out_shape, pypto.DT_BF16)
    ):
        pypto.experimental.set_operation_options(combine_axis=True)
        
        atten_cfg, tile_cfg = get_pfa_config()
        softmax_scale = atten_cfg.softmax_scale
        
        shape_q = q.shape  # 原始的qkvshape(b)-->(8*128,12,128)(b*s1,n1,d)
        shape_k = k.shape  # -->(8, 128,1,128)(b,s2,n2,d)
        bs_scalar = shape_q[0] #dynamic  b*s1
        nq = shape_q[1]        #n1
        block_num_scalar = shape_k[0] #dynamic b
        block_size = shape_k[1]  # s2动态轴，blocksize也会变，需要调整
        s2_scalar = shape_k[1]
        nkv = shape_k[2]         # n2
        dn = shape_k[3]          # d
        b_scalar = query_act_seqs.shape[0] #dymnamic  序列代表每个b实际有效s1，第一维度即b
        
        g_tile = tile_cfg.g_tile
        s2_tile = tile_cfg.s2_tile
        # 为什么没有s1_tile
        c1_tile = tile_cfg.c1_tile_shape
        v1_tile = tile_cfg.v1_tile_shape
        c2_tile = tile_cfg.c2_tile_shape
        v2_tile = tile_cfg.v2_tile_shape
        
        dtype = q.dtype
        s1_scalar = bs_scalar // b_scalar  # 动态轴拆分
        group = nq // nkv
        n2_sym = nkv
        g = nq // nkv
        g_loop = g // g_tile  # 12个group，分块策略全量12
        g_loop_merged = g_loop * n2_sym  # 静态轴合并后的总循环次数
        block_num = s2_tile // block_size
        s2_loop = (s2_scalar + s2_tile - 1) // s2_tile # 显式因果掩码矩阵替代迭代中仅使用当前sq寻找对应kv的因果逻辑，转而遍历所有kv分块
        
        step = 16  # 阶梯因果掩码步长
        s1_block_num = (s1_scalar + step - 1) // step  # s1方向的块数
        
        k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
        q_2d_shape = (b_scalar * s1_scalar * nq, dn)
        
        k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
        v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
        q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
        
        for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"): # b=8 , unroll_list=[8,4]
            for s1_block_idx in pypto.loop(s1_block_num, name="LOOP_s1_block", idx_name="s1_block_idx"):
                s1_start = s1_block_idx * step
                s1_end = (s1_block_idx + 1) * step
                actual_s1_in_block = (s1_scalar - s1_start).min(step)
                
                s2_max_for_block = (s1_block_idx + 1) * step
                s2_loop_for_block = (s2_max_for_block + s2_tile - 1) // s2_tile
                
                for s1_offset in pypto.loop(actual_s1_in_block, name="LOOP_s1_offset", idx_name="s1_offset"):
                    s1_idx = s1_start + s1_offset
                    bs_ofs = b_idx * s1_scalar + s1_idx

                    for g_idx_merged in pypto.loop(g_loop_merged, name="LOOP_g_merged", idx_name="g_idx_merged"):
                        # 从合并索引解耦出原始 n2_idx 和 g_idx
                        n2_idx = g_idx_merged // g_loop
                        g_idx = g_idx_merged % g_loop
                        
                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                    
                        oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")  # (12,128)
                        sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                        max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")

                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                    
                        for s2_idx in pypto.loop(s2_loop_for_block, name="LOOP_s2", idx_name="s2_idx"):  # 优化：只遍历可见的KV块
                            idx = s2_idx * block_num
                            actual_s2_tile = (s2_scalar - s2_idx * s2_tile).min(s2_tile)

                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                            
                            # 从 block_table 组装 K、V
                            pypto.set_pass_options(sg_set_scope=4)
                            kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                            vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]  # idx没法提前
                                block_idx_valid = block_idx.max(0)
                                kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                    pypto.view(k_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                    pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                            kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[actual_s2_tile, dn])
                            vj_assemble = pypto.view(vj_assemble, [s2_tile, dn], [0, 0], valid_shape=[actual_s2_tile, dn])
                            pypto.set_pass_options(sg_set_scope=-1)

                            # QK^T 计算
                            pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2], enable_multi_data_load=True)
                            sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
                            sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
                            # Softmax 计算
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            if pypto.is_loop_begin(s2_idx):
                                pypto.set_pass_options(sg_set_scope=2)
                                # sij_scale = pypto.mul(sij, softmax_scale)
                                print(f'Sij_shape is {sij.shape}') # （12,128）--if+else共执行4次-->(g_tile, actual_s2_tile)
                                # pypto.set_vec_tile_shapes(g_tile, s2_tile) # 因果掩码处理
                                causal_mask_row = pypto.view(causal_table, [1, s2_tile], [s1_idx, s2_idx * s2_tile], valid_shape=[1, actual_s2_tile])
                                causal_mask_broadcast = pypto.expand_clone(causal_mask_row, [g_tile, s2_tile], valid_shape=[g_tile, actual_s2_tile])
                                causal_mask_fp32 = pypto.cast(causal_mask_broadcast, pypto.DT_FP32)

                                sij_scale = pypto.add(pypto.mul(sij, softmax_scale), causal_mask_fp32)
                                tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                                tsub = pypto.sub(sij_scale, tilda_mij)
                                tilda_pij = pypto.exp(tsub)
                                tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                                sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                                max_update[:] = tilda_mij
                                pypto.set_pass_options(sg_set_scope=-1)
                                
                                pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2], enable_multi_data_load=True)
                                oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)
                                
                                pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                oi_update[:] = oi_tmp
                            else:
                                pypto.set_pass_options(sg_set_scope=1)
                                # sij_scale = pypto.mul(sij, softmax_scale)
                                print(f'Sij_shape is {sij.shape}')
                                # pypto.set_vec_tile_shapes(g_tile, s2_tile) # 因果掩码处理
                                causal_mask_row = pypto.view(causal_table, [1, s2_tile], [s1_idx, s2_idx * s2_tile], valid_shape=[1, actual_s2_tile])
                                causal_mask_broadcast = pypto.expand_clone(causal_mask_row, [g_tile, s2_tile], valid_shape=[g_tile, actual_s2_tile])
                                causal_mask_fp32 = pypto.cast(causal_mask_broadcast, pypto.DT_FP32)

                                sij_scale = pypto.add(pypto.mul(sij, softmax_scale), causal_mask_fp32)
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
                                
                                pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2], enable_multi_data_load=True)
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
    causal_table: torch.Tensor,
    query_seqs: torch.Tensor,
    attn_res: torch.Tensor
) -> None:
    """PFA - Prefill 阶段接口"""
    debug_print("ATTENTION_PFA", f"Input query shape: {query.shape}")
    
    if isinstance(query, FakeTensor):
        return
    check_args(query, key_cache, value_cache, block_tables, causal_table, query_seqs, attn_res)
    
    q_shape = query.shape
    kv_shape = key_cache.shape
    block_table_shape = block_tables.shape
    causal_table_shape = causal_table.shape
    
    shapes = [q_shape, kv_shape, block_table_shape, causal_table_shape]
    inputs = [query, key_cache, value_cache, block_tables, causal_table, query_seqs, attn_res]
    pfa_func(*shapes)(*inputs)
    
    debug_print("ATTENTION_PFA", "PFA kernel execution completed")


# ============================================================================
# 测试函数
# ============================================================================

def run_pfa_test(atten_cfg):
    """运行 PFA 测试"""
    debug_print("PFA_TEST", "Starting PFA test...")
    
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch_dtype = torch.bfloat16
    torch.npu.set_device(int(device_id))
    
    b = atten_cfg.b
    s1 = atten_cfg.s1
    s2 = atten_cfg.s2
    d = atten_cfg.q_d
    nq = atten_cfg.n1
    nkv = atten_cfg.n2
    block_size = atten_cfg.block_size
    max_num_blocks_per_query = atten_cfg.max_num_blocks_per_query
    query_seq_len = atten_cfg.actual_seq
    
    debug_print("PFA_TEST", f"Config: b={b}, s1={s1}, nq={nq}, nkv={nkv}, d={d}")
    debug_print("PFA_TEST", f"block_size={block_size}, max_blocks={max_num_blocks_per_query}")
    
    q_shape = [b * s1, nq, d]
    kv_shape = [atten_cfg.kv_num_blocks, block_size, nkv, d]  # block_size命名需要调整
    block_table_shape = [atten_cfg.block_table_batch, max_num_blocks_per_query]
    
    # PFA: 生成因果注意力掩码
    debug_print("PFA_TEST", "create_causal_mask...")
    s2_tile = 128
    step = 16
    causal_table = create_causal_mask(s1, ((s2 + s2_tile -1)//s2_tile)* s2_tile, step) # 对齐到s2_tile防止访问越界

    device = f'npu:{device_id}'
    q = torch.empty(q_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    k = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    v = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    attention_output = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)
    
    debug_print("PFA_TEST", f"Created tensors: q={q.shape}, k={k.shape}, v={v.shape}")
    
    # PFA: 生成 block_table
    block_table = gen_block_table(query_seq_len, block_size, block_table_shape)
    debug_print("PFA_TEST", f"Generated block_table: shape={block_table.shape}")
    
    # 转换 KV cache 格式
    k_cache_bsnd, v_cache_bsnd = kv_cache_concat_bsnd(k, v, block_table, atten_cfg)
    debug_print("PFA_TEST", f"Converted to BSND: k_cache_bsnd={k_cache_bsnd.shape}, v_cache_bsnd={v_cache_bsnd.shape}")
    
    # ========== PFA 关键: 阶梯因果注意力的 PyTorch 参考实现 ==========
    debug_print("PFA_TEST", "Running PyTorch reference (step=16 causal attention)...")
    step = 16  # 阶梯掩码步长，与kernel一致
    for i in range(b):
        seq_len = query_seq_len[i].item()
        for j in range(s1):
            # 阶梯因果掩码: Query位置j可以看到所在块及之前的所有KV
            cur_kv_len = ((j // step) + 1) * step
            cur_kv_len = min(cur_kv_len, s2)  # 防止越界
            for n2_idx in range(nkv):
                q_bs = q[i * s1 + j]  # [nq, d]
                k_bs = k_cache_bsnd[i, :cur_kv_len, n2_idx:n2_idx + 1].reshape(cur_kv_len, d)
                v_bs = v_cache_bsnd[i, :cur_kv_len, n2_idx:n2_idx + 1].reshape(cur_kv_len, d)
                
                qk_bmm_res = torch.matmul(q_bs, k_bs.transpose(1, 0))  # [nq, cur_kv_len]
                qk_ele_res = qk_bmm_res * atten_cfg.softmax_scale
                softmax_res, _, _ = softmax(qk_ele_res, True)  # [nq, cur_kv_len]
                bmm2_res = torch.matmul(softmax_res, v_bs)  # [nq, d]
                
                attention_output[i * s1 + j] = bmm2_res
    
    block_table_torch = block_table.to(dtype=torch.int32, device=device)
    causal_table_torch = causal_table.to(dtype=torch.float32, device=device)
    query_seq_torch = query_seq_len.to(dtype=torch.int32, device=device)
    out_torch = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)
    
    inputs = [q, k, v, block_table_torch, causal_table_torch, query_seq_torch, out_torch]

    debug_print("PFA_TEST", "Running PFA kernel...")
    attention_pfa(*inputs)
    
    debug_print("PFA_TEST", "Comparing results...")
    print('over')

    # Print some intermediate results for debugging
    ref_flat = np.array(attention_output.float().cpu().flatten().tolist())
    out_flat = np.array(out_torch.float().cpu().flatten().tolist())
    
    diff = np.abs(ref_flat - out_flat)
    debug_print("PFA_TEST", f"Max diff: {diff.max():.6f}")
    debug_print("PFA_TEST", f"Mean diff: {diff.mean():.6f}")
    debug_print("PFA_TEST", f"Mismatched count: {np.sum(diff > 0.001)} / {len(diff)}")
    
    # Use tolerance appropriate for BF16 precision with causal attention
    # The valid_shape mechanism has small precision issues for partial tiles
    assert_allclose(
        ref_flat,
        out_flat,
        rtol=0.05, atol=0.005
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
    assert all(x <= atten_cfg.s1 for x in actual_seq_cpu), "All values must be <= s1"
    
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
