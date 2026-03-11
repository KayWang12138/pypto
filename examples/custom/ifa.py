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


import os
import logging
import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
import torch_npu
import pytest
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph

import pypto


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter(
    fmt='%(asctime)s - [%(levelname)s] - [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]'
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

@dataclass
class AttentionTileConfig:
    g_tile: int
    s2_tile: int
    c1_tile: list
    v1_tile: list
    c2_tile: list
    v2_tile: list


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
    kv_actual_seqs: torch.Tensor = None
    block_table_batch: int = 0
    kv_num_blocks: int = 0


@dataclass
class LoopOfs:
    bs_ofs: int = 0
    n1g_ofs: int = 0
    out_ofs: int = 0


@dataclass
class LoopTensor:
    q_2d: pypto.Tensor = None
    k_2d: pypto.Tensor = None
    v_2d: pypto.Tensor = None
    block_table: pypto.Tensor = None
    kv_act_seqs: pypto.Tensor = None
    atten_out: pypto.Tensor = None


@dataclass
class LoopIndex:
    b_idx: int = 0
    s1_idx: int = 0
    n2_idx: int = 0
    group_idx: int = 0
    s2_idx: int = 0


@dataclass
class LoopSize:
    group_loop: int = 0
    s2_loop: int = 0


@dataclass
class TempUpdateTensor:
    out_update: pypto.Tensor = None
    sum_update: pypto.Tensor = None
    max_update: pypto.Tensor = None


@dataclass
class IFAKernelParams:
    n1: int
    d: int
    block_num: int
    n2: int
    block_size: int
    b: int
    s1: int
    group: int
    softmax_scale: float


@dataclass
class ContextParams:
    kernel_params: IFAKernelParams = None
    tile_cfg: AttentionTileConfig = None
    loop_tensors: LoopTensor = None
    loop_index: LoopIndex = None
    loop_size: TempUpdateTensor = None
    loop_ofs: LoopOfs = None
    temp_update_tensors: TempUpdateTensor = None


def gen_block_table(atten_cfg: AttentionConfig, device: str):
    block_num_per_batch = []
    block_num = 0  # res: 1024

    actual_seq_len = atten_cfg.kv_actual_seqs
    block_size = atten_cfg.block_size
    block_table_batch = atten_cfg.block_table_batch
    max_num_blocks_per_query = atten_cfg.max_num_blocks_per_query

    block_table_shape = [block_table_batch, max_num_blocks_per_query]  # [8, 128]

    if actual_seq_len.device.type != 'cpu':
        actual_seq_len_cpu = actual_seq_len.cpu()
    else:
        actual_seq_len_cpu = actual_seq_len

    for actual_seq in actual_seq_len_cpu:
        block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
        block_num += math.ceil(actual_seq.item() / block_size)

    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]  # 随机排列

    # 创建 block_table 张量
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32, device=device)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def kv_cache_concat(k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor, atten_cfg: AttentionConfig, device: str):
    b = atten_cfg.b
    n2 = atten_cfg.n2
    kv_lora_rank = atten_cfg.q_d
    rope_dim = atten_cfg.kv_d
    block_size = atten_cfg.block_size
    kv_actual_seqs = atten_cfg.kv_actual_seqs
    dtype = v_cache.dtype

    if kv_actual_seqs.device.type != 'cpu':
        kv_actual_seqs_cpu = kv_actual_seqs.cpu()
    else:
        kv_actual_seqs_cpu = kv_actual_seqs
    kv_act_seq_max= torch.max(kv_actual_seqs_cpu).item()
    kv_max = math.ceil(kv_act_seq_max / block_size) * block_size

    k = torch.zeros([b, n2, kv_max, kv_lora_rank], dtype=dtype, device=device)  # BNSD [8, 1, 16384, 128]
    v = torch.zeros([b, n2, kv_max, rope_dim], dtype=dtype, device=device)  # BNSD [8, 1, 16384, 128]
    
    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, n2, kv_max, kv_lora_rank], dtype=dtype, device=device)
        kv_rope_temp_tensor = torch.zeros([1, n2, kv_max, rope_dim], dtype=dtype, device=device)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            kv_nope_temp_tensor[:, :, start_idx:end_idx, :] = v_cache[block_idx:block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, :, start_idx:end_idx, :] = k_cache[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        v[b_idx:b_idx + 1, :, :, :] = kv_nope_temp_tensor
        k[b_idx:b_idx + 1, :, :, :] = kv_rope_temp_tensor

    return k, v


def get_env_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logger.info("If no NPU environment is available, set --run_mode sim to run in simulation mode;")
        logger.info("otherwise, set the environment variable TILE_FWK_DEVICE_ID.")
        logger.info("Please set it before running this example:")
        logger.info("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        logger.info(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def get_device(device_id: int = None, run_mode: str = "npu"):
    if device_id != None:
        cue_device_id = device_id
    else:
        cue_device_id = get_env_device_id()
    
    device = f"npu:{cue_device_id}" if (run_mode == "npu" and cue_device_id is not None) else "cpu"
    return device


def get_base_params(case_name: str):
    params = {}
    if case_name.startswith("1b16k"):
        params = {"b": 1, "s2": 16 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("2b16k"):
        params = {"b": 2, "s2": 16 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("4b16k"):
        params = {"b": 4, "s2": 16 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("8b16k"):
        params = {"b": 8, "s2": 16 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("16b16k"):
        params = {"b": 16, "s2": 16 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("1b8k"):
        params = {"b": 1, "s2": 8 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("2b8k"):
        params = {"b": 2, "s2": 8 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("4b8k"):
        params = {"b": 4, "s2": 8 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("8b8k"):
        params = {"b": 4, "s2": 8 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("16b8k"):
        params = {"b": 16, "s2": 8 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    else:
        raise Exception(f"Case {case_name} does not exist.")
    return params


def get_ifa_atten_cfg(device: str, case_name: str):
    base_params = get_base_params(case_name)
    b = base_params.get("b")
    s2 = base_params.get("s2")
    s1 = base_params.get("s1")
    d = base_params.get("d")
    n1 = base_params.get("n1")
    n2 = base_params.get("n2")
    softmax_scale = d ** -0.5
    block_table_batch = b
    block_size = 128
    max_num_blocks_per_query = math.ceil(s2 / block_size)
    kv_num_blocks = b * max_num_blocks_per_query

    kv_actual_seqs = torch.tensor([s2] * b, dtype=torch.int32, device=device)

    atten_cfg = AttentionConfig(b=b, s1=s1, s2=s2, n1=n1, n2=n2, q_d=d, kv_d=d, 
                                block_size=block_size, 
                                max_num_blocks_per_query=max_num_blocks_per_query, 
                                softmax_scale=softmax_scale, kv_actual_seqs=kv_actual_seqs, 
                                block_table_batch=block_table_batch, kv_num_blocks=kv_num_blocks)
    return atten_cfg


def get_ifa_tile_cfg():
    m_tile = 128
    k_tile = 128
    n_tile = 128
    s2_tile = 512

    tile_cfg = AttentionTileConfig(
        g_tile=12,
        s2_tile=s2_tile,
        c1_tile=[[m_tile, m_tile], [k_tile, k_tile], [n_tile, n_tile]],
        v1_tile=[m_tile, s2_tile],
        c2_tile=[[m_tile, m_tile], [k_tile, k_tile], [n_tile, n_tile]],
        v2_tile=[m_tile, m_tile]
    )
    return tile_cfg
        


def gen_qkv(atten_cfg: AttentionConfig, device: str):
    # get query shape
    b = atten_cfg.b
    s1 = atten_cfg.s1
    n1 = atten_cfg.n1
    q_d = atten_cfg.q_d
    q_shape = [b * s1, n1, q_d]  # TND

    # get kv shape
    kv_num_blocks = atten_cfg.kv_num_blocks
    block_size = atten_cfg.block_size
    n2 = atten_cfg.n2
    kv_d = atten_cfg.kv_d
    kv_shape = [kv_num_blocks, n2, block_size, kv_d]  # PA_BnNBsD

    # gen q, k, v
    dtype = torch.bfloat16
    q = torch.empty(q_shape, dtype=dtype).uniform_(-1, 1).to(device=device)
    k = torch.empty(kv_shape, dtype=dtype).uniform_(-1, 1).to(device=device)
    v = torch.empty(kv_shape, dtype=dtype).uniform_(-1, 1).to(device=device)
    return q, k, v


def gen_ifa_golden(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_table: torch.Tensor, atten_cfg: AttentionConfig):
    b = atten_cfg.b
    s1 = atten_cfg.s1
    n2 = atten_cfg.n2
    d = atten_cfg.q_d
    softmax_scale = atten_cfg.softmax_scale
    kv_actual_seqs = atten_cfg.kv_actual_seqs

    atten_out = torch.zeros_like(q)

    for b_idx in range(b):
        for s1_idx in range(s1):
            for n2_idx in range(n2):
                # 从 torch tensor 获取值
                kv_seq_len = kv_actual_seqs[b_idx].item()
                cur_s1_len = s1 - 1 - s1_idx
                seq_len = kv_seq_len - cur_s1_len

                q_bs = q[b_idx * s1 + s1_idx]
                k_bs = k[b_idx, n2_idx:n2_idx + 1, :seq_len].reshape(seq_len, d)
                v_bs = v[b_idx, n2_idx:n2_idx + 1, :seq_len].reshape(seq_len, d)
                
                # MM1: 矩阵乘法
                qk_bmm_res = torch.matmul(q_bs, k_bs.transpose(1, 0))
                qk_ele_res = qk_bmm_res * softmax_scale
                # Softmax计算
                softmax_res = F.softmax(qk_ele_res)

                # MM2: 矩阵乘法
                bmm2_res = torch.matmul(softmax_res, v_bs)

                # 存储结果
                atten_out[b_idx * s1 + s1_idx] = bmm2_res
    return atten_out


def init_kernel_params(q, k, block_table_shape):
    bs, n1, d = q.shape
    block_num, n2, block_size, _ = k.shape
    b = block_table_shape[0]
    s1 = bs // b
    group = n1 // n2
    softmax_scale = d ** -0.5
    kernel_params = IFAKernelParams(
        n1=n1, d=d, block_num=block_num, n2=n2, block_size=block_size, 
        b=b, s1=s1, group=group, softmax_scale=softmax_scale
    )
    return kernel_params


def reshape_qkv_to_2d(q, k, v, kernel_params):
    b = kernel_params.b
    s1 = kernel_params.s1
    n1 = kernel_params.n1
    d = kernel_params.d
    block_num = kernel_params.block_num
    block_size = kernel_params.block_size
    n2 = kernel_params.n2
    d = kernel_params.d

    q_2d_shape = (b * s1 * n1, d)
    kv_2d_shape = (block_num * block_size * n2, d)

    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    k_2d = pypto.reshape(k, kv_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, kv_2d_shape, inplace=True)
    return q_2d, k_2d, v_2d


def assemble_kj(idx, ctx_params):
    # get need loop tensors 
    k_2d = ctx_params.loop_tensors.k_2d
    block_table = ctx_params.loop_tensors.block_table

    # get need tile cfg
    s2_tile = ctx_params.tile_cfg.s2_tile

    # get need kernel params
    block_size = ctx_params.kernel_params.block_size
    d = ctx_params.kernel_params.d

    b_idx = ctx_params.loop_index.b_idx
    
    block_num = s2_tile // block_size

    kj_assemble = pypto.tensor([s2_tile, d], k_2d.dtype, "kj_assemble")
    for i in range(block_num):
        block_idx = block_table[b_idx, idx + i]
        block_idx_valid = block_idx.max(0)
        kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
            pypto.view(k_2d, [block_size, d], [block_idx_valid * block_size, 0])
    kj_assemble = pypto.view(kj_assemble, [s2_tile, d], [0, 0], valid_shape=[s2_tile, d])
    return kj_assemble
    

def assemble_vj(idx, actual_s2_tile, ctx_params):
    # get need loop tensors 
    v_2d = ctx_params.loop_tensors.v_2d
    block_table = ctx_params.loop_tensors.block_table

    # get need tile cfg
    s2_tile = ctx_params.tile_cfg.s2_tile

    # get need kernel params
    block_size = ctx_params.kernel_params.block_size
    d = ctx_params.kernel_params.d

    b_idx = ctx_params.loop_index.b_idx

    block_num = s2_tile // block_size

    vj_assemble = pypto.tensor([s2_tile, d], v_2d.dtype, "vj_assemble")
    for i in range(block_num):
        block_idx = block_table[b_idx, idx + i]
        block_idx_valid = block_idx.max(0)
        vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
            pypto.view(v_2d, [block_size, d], [block_idx_valid * block_size, 0])
    vj_assemble = pypto.view(vj_assemble, [s2_tile, d], [0, 0], valid_shape=[actual_s2_tile, d])
    return vj_assemble


def compute_first_tile(sij, vj_assemble, dtype, ctx_params):
    softmax_scale = ctx_params.kernel_params.softmax_scale

    c2_tile = ctx_params.tile_cfg.c2_tile
    v2_tile = ctx_params.tile_cfg.v2_tile

    out_update = ctx_params.temp_update_tensors.out_update
    sum_update = ctx_params.temp_update_tensors.sum_update
    max_update = ctx_params.temp_update_tensors.max_update

    sij_scale = pypto.mul(sij, softmax_scale)
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

    tsub = pypto.sub(sij_scale, tilda_mij)
    tilda_pij = pypto.exp(tsub)
    tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
    sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
    max_update[:] = tilda_mij

    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
    oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
    out_update[:] = oi_tmp


def compute_other_tile(sij, vj_assemble, dtype, ctx_params):
    softmax_scale = ctx_params.kernel_params.softmax_scale

    c2_tile = ctx_params.tile_cfg.c2_tile
    v2_tile = ctx_params.tile_cfg.v2_tile

    out_update = ctx_params.temp_update_tensors.out_update
    sum_update = ctx_params.temp_update_tensors.sum_update
    max_update = ctx_params.temp_update_tensors.max_update

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

    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
    oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
    out_update[:] = out_update * update_mul + oi_tmp


def finalize_output(out_ofs, dtype, ctx_params):
    d = ctx_params.kernel_params.d

    v2_tile = ctx_params.tile_cfg.v2_tile
    g_tile = ctx_params.tile_cfg.g_tile

    out_update = ctx_params.temp_update_tensors.out_update
    sum_update = ctx_params.temp_update_tensors.sum_update

    atten_out = ctx_params.loop_tensors.atten_out

    oi_final = pypto.div(out_update, sum_update)
    pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
    oi_final_3d = pypto.cast(
        pypto.reshape(oi_final, [1, g_tile, d]), dtype)
    # 7. 将结果搬运到输出tensor上
    pypto.assemble(oi_final_3d, out_ofs, atten_out)


def compute_c1(qi, kj_assemble, actual_s2_tile, tile_cfg):
    c1_tile = tile_cfg.c1_tile
    g_tile = tile_cfg.g_tile
    s2_tile = tile_cfg.s2_tile

    pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
    sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
    sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
    return sij


def compute_loop_s2(ctx_params, cur_seq_len, dtype):
    # get need tile cfg
    tile_cfg = ctx_params.tile_cfg
    s2_tile = tile_cfg.s2_tile
    v1_tile = tile_cfg.v1_tile
    g_tile = tile_cfg.g_tile

    # get need kernel params
    block_size = ctx_params.kernel_params.block_size
    d = ctx_params.kernel_params.d
    n1 = ctx_params.kernel_params.n1

    # get need loop tensors
    q_2d = ctx_params.loop_tensors.q_2d

    # get need loop offset params
    bs_ofs = ctx_params.loop_ofs.bs_ofs
    n1g_ofs = ctx_params.loop_ofs.n1g_ofs
    out_ofs = ctx_params.loop_ofs.out_ofs

    # get need loop index params
    s2_idx = ctx_params.loop_index.s2_idx

    block_num = s2_tile // block_size
    idx = s2_idx * block_num

    actual_s2_tile = (cur_seq_len - s2_idx * s2_tile).min(s2_tile)
    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
    qi = pypto.view(q_2d, [g_tile, d], [bs_ofs * n1 + n1g_ofs, 0])
    kj_assemble = assemble_kj(idx, ctx_params)

    sij = compute_c1(qi, kj_assemble, actual_s2_tile, tile_cfg)
    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
    vj_assemble = assemble_vj(idx, actual_s2_tile, ctx_params)
    if pypto.cond(pypto.is_loop_begin(s2_idx)):
        compute_first_tile(sij, vj_assemble, dtype, ctx_params)
    else:
        compute_other_tile(sij, vj_assemble, dtype, ctx_params)
    
    if pypto.cond(pypto.is_loop_end(s2_idx)):
        finalize_output(out_ofs, dtype, ctx_params)


def compute_loop_group(ctx_params, cur_seq_len, dtype):
    # get need tile cfg
    g_tile = ctx_params.tile_cfg.g_tile

    # get need kernel params
    group = ctx_params.kernel_params.group
    d = ctx_params.kernel_params.d

    # get need loop index params
    loop_index = ctx_params.loop_index
    n2_idx = loop_index.n2_idx
    group_idx = loop_index.group_idx

    # get need loop offset params
    loop_ofs = ctx_params.loop_ofs
    bs_ofs = loop_ofs.bs_ofs

    # get need loop params
    s2_loop = ctx_params.loop_size.s2_loop

    n1g_ofs = n2_idx * group + group_idx * g_tile
    out_ofs = [bs_ofs, n1g_ofs, 0]
    loop_ofs = replace(loop_ofs, n1g_ofs=n1g_ofs, out_ofs=out_ofs)

    out_update = pypto.tensor([g_tile, d], pypto.DT_FP32, "out_update")
    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
    temp_update_tensors = TempUpdateTensor(out_update, sum_update, max_update)

    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 4, 2, 1]):
        loop_index = replace(loop_index, s2_idx=s2_idx)
        ctx_params = replace(ctx_params, loop_index=loop_index, 
                            temp_update_tensors=temp_update_tensors, loop_ofs=loop_ofs)
        compute_loop_s2(ctx_params, cur_seq_len, dtype)
        

def compute_loop_n2(ctx_params, cur_seq_len, dtype):
    loop_index = ctx_params.loop_index
    group_loop = ctx_params.loop_size.group_loop

    for group_idx in pypto.loop(group_loop, name="LOOP_group_idx", idx_name="group_idx"):
        loop_index = replace(loop_index, group_idx=group_idx)
        ctx_params = replace(ctx_params, loop_index=loop_index)
        compute_loop_group(ctx_params, cur_seq_len, dtype)
        

def compute_loop_s1(ctx_params, cur_seq_len, dtype):
    n2 = ctx_params.kernel_params.n2
    loop_index =  ctx_params.loop_index

    for n2_idx in pypto.loop(n2, name="LOOP_n2", idx_name="n2_idx"):
        loop_index = replace(loop_index, n2_idx=n2_idx)
        ctx_params = replace(ctx_params, loop_index=loop_index)
        compute_loop_n2(ctx_params, cur_seq_len, dtype)


def compute_loop_b(dtype, ctx_params):
    # get need kernel params
    s1 = ctx_params.kernel_params.s1

    # get need tile cfg
    s2_tile = ctx_params.tile_cfg.s2_tile

    # get need loop tensors
    kv_act_seqs = ctx_params.loop_tensors.kv_act_seqs

    # get need loop index
    b_idx = ctx_params.loop_index.b_idx

    loop_size = ctx_params.loop_size

    for s1_idx in pypto.loop(s1, name="LOOP_s1", idx_name="s1_idx"):
        cur_seq_len = kv_act_seqs[b_idx] - (s1 - 1 - s1_idx)
        s2_loop = pypto.ceildiv(cur_seq_len, s2_tile)
        loop_size = replace(loop_size, s2_loop=s2_loop)
        bs_ofs = b_idx * s1 + s1_idx
        loop_ofs = LoopOfs(bs_ofs=bs_ofs)
        ctx_params = replace(ctx_params, loop_size=loop_size, loop_ofs=loop_ofs)
        compute_loop_s1(ctx_params, cur_seq_len, dtype)
    

def ifa_func(q_shape, kv_shape, block_table_shape):
    out_shape = q_shape

    q_shape = (pypto.frontend.dynamic("qshape"), q_shape[1], q_shape[2])
    kv_shape = (pypto.frontend.dynamic("kvshape"), kv_shape[1], kv_shape[2], kv_shape[3])
    bs = pypto.frontend.dynamic("bs")
    b = block_table_shape[0]

    @pypto.frontend.jit(
        runtime_options={
            "stitch_function_num_initial": 128,
            "stitch_function_outcast_memory": 1024,
            "stitch_function_inner_memory": 1024
        },
        # 当子图大小达到上界不允许与其他子图合并
        pass_options={
            "pg_upper_bound": 1536,
            # Q常驻，0代表第一组mmad，4代表4次matmul合并
            "cube_l1_reuse_setting": {0: 8}
        },
        debug_options={"runtime_debug_mode": 1}
    )
    def ifa_func_kernel(
        q: pypto.Tensor(q_shape, pypto.DT_BF16),
        k: pypto.Tensor(kv_shape, pypto.DT_BF16),
        v: pypto.Tensor(kv_shape, pypto.DT_BF16),
        block_table: pypto.Tensor(block_table_shape, pypto.DT_INT32),
        kv_act_seqs: pypto.Tensor((b, ), pypto.DT_INT32),
        atten_out: pypto.Tensor(out_shape, pypto.DT_BF16)
    ):
        logger.info(f"================ ifa_func_kernel ================")
        # 1. 解析参数
        dtype = q.dtype
        kernel_params = init_kernel_params(q, k, block_table_shape)

        # 2. 解析tile配置
        tile_cfg = get_ifa_tile_cfg()

        # 3. q, k, v reshape为二维
        q_2d, k_2d, v_2d = reshape_qkv_to_2d(q, k, v, kernel_params)
        loop_tensors = LoopTensor(q_2d, k_2d, v_2d, block_table, kv_act_seqs, atten_out)

        group_loop = kernel_params.group // tile_cfg.g_tile
        loop_size = LoopSize(group_loop=group_loop)

        ctx_params = ContextParams(
            kernel_params=kernel_params, tile_cfg=tile_cfg, loop_tensors=loop_tensors,
            loop_size=loop_size
        )

        # 4. 实现kernel逻辑
        for b_idx in pypto.loop(kernel_params.b, name="LOOP_b", idx_name="b_idx"):
            loop_index = LoopIndex(b_idx=b_idx)
            ctx_params = replace(ctx_params, loop_index=loop_index)
            compute_loop_b(dtype, ctx_params)
    return ifa_func_kernel


@allow_in_graph
def incre_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pseShift = None,
    attenMask = None,
    actualSeqLengths = None,
    dequantScale1 = None,
    quantScale1 = None,
    dequantScale2 = None,
    quantScale2 = None,
    quantOffset2 = None,
    antiquantScale = None,
    antiquantOffset = None,
    blocktable = None,
    kvPaddingSize = None,
    numHeads = None,
    scaleValue = None,
    inputLayout = None,
    numKeyValueHeads = None,
    blockSize = None,
    innerPrecise = None,
):
    atten_out = torch.zeros_like(query)
    input_shapes = [query.shape, key.shape, blocktable.shape]
    input_values = [query, key, value, blocktable, actualSeqLengths, atten_out]
    ifa_func(*input_shapes)(*input_values)
    return atten_out


def do_test_incre_flash_attention(case_name: str):
    logger.info("*" * 60)
    logger.info(f"Run incre_flash_attention {case_name} case")
    logger.info("*" * 60 + "\n")

    device = get_device()
    atten_cfg = get_ifa_atten_cfg(device, case_name)

    q, k_cache, v_cache = gen_qkv(atten_cfg, device)
    logger.info(f"q.shape: {q.shape}")
    logger.info(f"k_cache.shape: {k_cache.shape}")
    logger.info(f"v_cache.shape: {v_cache.shape}")
    block_table = gen_block_table(atten_cfg, device)
    logger.info(f"block_table.shape: {block_table.shape}")
    k, v = kv_cache_concat(k_cache, v_cache, block_table, atten_cfg, device)
    logger.info(f"k.shape: {k.shape}")
    logger.info(f"v.shape: {v.shape}")

    kv_actual_seqs = atten_cfg.kv_actual_seqs
    logger.info(f"kv_actual_seqs: {kv_actual_seqs}")
    ifa_golden = gen_ifa_golden(q, k, v, block_table, atten_cfg)

    inputs = dict(
        query=q,
        key=k_cache,
        value=v_cache,
        blocktable=block_table,
        actualSeqLengths=kv_actual_seqs,
        numHeads=atten_cfg.n1,
        inputLayout="TND",
        numKeyValueHeads=atten_cfg.n2,
        blockSize=atten_cfg.block_size
    )
    ifa_pypto_out = incre_flash_attention(**inputs)

    # 6. 与PyTorch参考实现对比
    assert_allclose(np.array(ifa_golden.cpu().flatten().tolist()),
                    np.array(ifa_pypto_out.cpu().flatten().tolist()),
                    rtol=0.0078125, atol=0.0001)
 

def main():
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("PyPTO incre_flash_attention Example")
    logger.info("=" * 60 + "\n")
    
    # test incre_flash_attention kvs 16k
    test_incre_flash_attention_1b16k()
    test_incre_flash_attention_2b16k()
    test_incre_flash_attention_4b16k()
    test_incre_flash_attention_8b16k()
    test_incre_flash_attention_16b16k()

    # test incre_flash_attention kvs 8k
    test_incre_flash_attention_1b8k()
    test_incre_flash_attention_2b8k()
    test_incre_flash_attention_4b8k()
    test_incre_flash_attention_8b8k()
    test_incre_flash_attention_16b8k()


def test_incre_flash_attention_1b16k():
    do_test_incre_flash_attention("1b16k")


def test_incre_flash_attention_2b16k():
    do_test_incre_flash_attention("2b16k")


def test_incre_flash_attention_4b16k():
    do_test_incre_flash_attention("4b16k")


def test_incre_flash_attention_8b16k():
    do_test_incre_flash_attention("8b16k")


def test_incre_flash_attention_16b16k():
    do_test_incre_flash_attention("16b16k")


def test_incre_flash_attention_1b8k():
    do_test_incre_flash_attention("1b8k")


def test_incre_flash_attention_2b8k():
    do_test_incre_flash_attention("2b8k")


def test_incre_flash_attention_4b8k():
    do_test_incre_flash_attention("4b8k")


def test_incre_flash_attention_8b8k():
    do_test_incre_flash_attention("8b8k")


def test_incre_flash_attention_16b8k():
    do_test_incre_flash_attention("16b8k")


if __name__ == "__main__":
    main()
