#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Flash Attention with Paged Attention (PA) Example for Qwen3 Model

This example demonstrates Flash Attention implementation with Paged Attention format using PyPTO:
- Flash Attention algorithm for efficient attention computation
- Paged Attention (PA) format for KV cache management
- Block-based processing for variable sequence lengths
- Dynamic batch and sequence length support
- Online softmax computation for numerical stability

This is a highly optimized attention implementation used in production LLM inference,
supporting variable-length sequences through block-based KV cache management.
"""

import os
import sys
import argparse
import numpy as np
import math
from dataclasses import dataclass
import pypto
import torch
import torch.nn.functional as F
import torch_npu
from numpy.testing import assert_allclose


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.
    
    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("ERROR: Environment variable TILE_FWK_DEVICE_ID is not set.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    
    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


@dataclass
class TileConfig:
    """Configuration for attention tiling parameters."""
    head_num_q_tile: int 
    c1_tile_shape: list  # Cube tile shape for Q@K^T
    v1_tile_shape: list  # Vector tile shape for softmax
    c2_tile_shape: list  # Cube tile shape for attention@V
    v2_tile_shape: list  # Vector tile shape for output


@dataclass
class AttentionConfig:
    """Configuration for attention computation."""
    b: int  # Batch size
    s_q: int  # Number of new tokens
    skv: int  # Maximum KV cache length
    n_q: int  # Number of query heads
    n_kv: int  # Number of key/value heads
    dim: int  # Query/Key/value dimension
    block_size: int = 128  # Block size for PA format
    block_num: int = 64  # Block num initialization
    is_nz_format: bool = False
    dtype: torch.dtype = torch.float32
    max_unroll_times: int = 1
    

def get_qwen_common_config() -> tuple[AttentionConfig, TileConfig]:
    """
    Get common Qwen3 attention configuration for testing.
    
    Parameters
    ----------
    device : str
        Device to create tensors on
        
    Returns
    -------
    tuple[AttentionConfig, TileConfig]
        Tuple of attention config and tile config
    """
    # Attention configuration
    b = 24
    s_q = 2
    skv = 4096
    n_q = 32
    n_kv = 1
    dim = 512
    block_size = 128
    block_num = 64
    is_nz_format = False
    dtype = torch.float32
    max_unroll_times = 1
    
    atten_cfg = AttentionConfig(
        b = b, s_q = s_q, skv = skv, n_q = n_q, n_kv = n_kv,
        dim = dim, block_size = block_size, block_num = block_num,
        is_nz_format = is_nz_format, dtype = dtype,
        max_unroll_times = max_unroll_times
    )
    
    # Tile configuration
    head_num_q_tile = 32
    c1_tile_shape = (32, 32, 64, 64, 128, 128)
    v1_tile_shape = (32, 64)
    c2_tile_shape = (32, 32, 64, 64, 128, 128)
    v2_tile_shape = (32, 64)
    tile_cfg = TileConfig(
        head_num_q_tile,
        c1_tile_shape,
        v1_tile_shape,
        c2_tile_shape,
        v2_tile_shape
    )
    
    return atten_cfg, tile_cfg


def detailed_allclose_manual(cpu: np.ndarray, npu: np.ndarray, name: str,
                             rtol: float = 1e-3, atol: float = 1e-3,
                             max_prints: int = 50, force_print_first_n: int = 5) -> bool:
    """
    Detailed manual implementation of np.allclose with verbose output.
    
    Prints values that exceed tolerance and NaN values for debugging.
    
    Parameters
    ----------
    cpu : np.ndarray
        CPU reference array
    npu : np.ndarray
        NPU output array
    name : str
        Name for logging
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    max_prints : int
        Maximum number of mismatches to print
    force_print_first_n : int
        Force print first N elements regardless of errors
        
    Returns
    -------
    bool
        True if arrays are close, False otherwise
    """
    # Check shape consistency
    if cpu.shape != npu.shape:
        print(f"Error: Shape mismatch - cpu {cpu.shape} vs npu {npu.shape}")
        return False
    
    total_elements = cpu.size
    abnormal_count = 0
    nan_count = 0
    exceed_tolerance_count = 0
    
    print(f"Starting array comparison, shape: {cpu.shape}, total elements: {total_elements}")
    print(f"Tolerance conditions: rtol={rtol}, atol={atol}")
    print("=" * 80)
    
    # Color codes
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    # Flatten arrays for iteration
    cpu_flat = cpu.reshape(-1)
    npu_flat = npu.reshape(-1)
    
    # Helper function to get multi-dimensional index
    def get_multi_index(flat_index: int, shape: tuple) -> tuple:
        indices = []
        remaining = flat_index
        for dim in reversed(shape):
            indices.append(remaining % dim)
            remaining = remaining // dim
        return tuple(reversed(indices))
    
    # Force print first N elements
    if force_print_first_n > 0:
        print(f"{YELLOW}Force printing first {force_print_first_n} elements:{RESET}")
        for flat_idx in range(min(force_print_first_n, total_elements)):
            cpu_val = cpu_flat[flat_idx]
            npu_val = npu_flat[flat_idx]
            multi_idx = get_multi_index(flat_idx, cpu.shape)
            
            if np.isnan(cpu_val) or np.isnan(npu_val):
                diff_str = "NaN"
            else:
                diff_val = np.abs(cpu_val - npu_val)
                diff_str = f"{diff_val:.6e}"
            
            cpu_str = "NaN" if np.isnan(cpu_val) else f"{cpu_val:.6e}"
            npu_str = "NaN" if np.isnan(npu_val) else f"{npu_val:.6e}"
            
            print(f"{YELLOW}Index {multi_idx}: cpu={cpu_str}, npu={npu_str}, diff={diff_str}{RESET}")
        
        print("-" * 80)
    
    # Check all elements for anomalies
    for flat_idx in range(total_elements):
        cpu_val = cpu_flat[flat_idx]
        npu_val = npu_flat[flat_idx]
        multi_idx = get_multi_index(flat_idx, cpu.shape)
        
        # Check for NaN in NPU output
        if np.isnan(npu_val):
            abnormal_count += 1
            nan_count += 1
            
            if abnormal_count <= max_prints:
                cpu_str = "NaN" if np.isnan(cpu_val) else f"{cpu_val:.6e}"
                print(f"Index {multi_idx}: cpu={cpu_str}, npu=NaN, diff=NaN (NPU contains NaN)")
        
        # Check for NaN in CPU reference
        elif np.isnan(cpu_val):
            abnormal_count += 1
            exceed_tolerance_count += 1
            
            if abnormal_count <= max_prints:
                print(f"Index {multi_idx}: cpu=NaN, npu={npu_val:.6e}, diff=NaN (CPU contains NaN)")
        
        else:
            # Check if difference exceeds tolerance
            abs_diff = np.abs(cpu_val - npu_val)
            allowed_diff = atol + rtol * np.abs(npu_val)
            
            if abs_diff > allowed_diff:
                abnormal_count += 1
                exceed_tolerance_count += 1
                
                if abnormal_count <= max_prints:
                    print(f"Index {multi_idx}: cpu={cpu_val:.6e}, npu={npu_val:.6e}, "
                          f"diff={abs_diff:.6e} (exceeds tolerance {allowed_diff:.6e})")
    
    # Print statistics
    print("=" * 80)
    print(f"\033[1m\033[95m{name} Comparison Statistics:\033[0m")
    print(f"Total elements: {total_elements}")
    print(f"Abnormal elements: {abnormal_count}")
    print(f"  - NaN count: {nan_count}")
    print(f"  - Exceed tolerance count: {exceed_tolerance_count}")
    print(f"Abnormal ratio: {abnormal_count / total_elements * 100:.4f}%")
    
    is_allclose = (abnormal_count == 0)
    print(f"\nnp.allclose equivalent result: {is_allclose}")
    
    if abnormal_count > max_prints:
        print(f"\nNote: Only showing first {max_prints} anomalies, total {abnormal_count} abnormal elements")
    
    assert_allclose(cpu, npu, rtol, atol)
    return is_allclose


def flash_attn_pa_golden(inputs: list, outputs: list, atten_cfg: AttentionConfig, tile_cfg: TileConfig):
    b = atten_cfg.b
    n_q = atten_cfg.n_q
    s_q = atten_cfg.s_q
    n_kv = atten_cfg.n_kv
    kv_lora_rank = atten_cfg.dim
    d_q = kv_lora_rank
    d_k = kv_lora_rank
    d_v = kv_lora_rank
    n_tile = tile_cfg.head_num_q_tile
    block_size = atten_cfg.block_size
    block_num = atten_cfg.block_num
    query, k_cache, v_cache, block_table, actual_seq_len = inputs
    
    q_bnsd = query.reshape(b, n_q, s_q, kv_lora_rank)
    k_cache = k_cache.reshape(block_num, block_size, n_kv * kv_lora_rank)
    v_cache = v_cache.reshape(block_num, block_size, n_kv * d_v)
    scalar = d_q ** -0.5
    tiled_out = []
    block_num_per_batch = []
    for actual_seq in actual_seq_len:
        block_num_per_batch.append(int(math.ceil(actual_seq / block_size)))
    n_loop = int(math.ceil(n_q / n_tile))
    for b_index in range(b):
        matmul_dtype = torch.float32
        cur_seq = actual_seq_len[b_index]
        bn_per_batch = int(math.ceil(cur_seq / block_size))
        for n_idx in range(n_loop):
            oi_update = []
            li_update = []
            mi_update = []
            qi = q_bnsd[b_index, n_idx * n_tile: (n_idx + 1) * n_tile, :, :]
            qi = qi.reshape(-1, qi.shape[-1])
            for bn in range(block_num_per_batch[b_index]):
                cur_block_idx = block_table[b_index][bn]
                s2_tile_cur = min(block_size, cur_seq - bn * block_size)
                kj = k_cache[cur_block_idx, 0:s2_tile_cur, :]
                vj = v_cache[cur_block_idx, 0:s2_tile_cur, :]
                kj = kj.reshape(s2_tile_cur, d_k)
                vj = vj.reshape(s2_tile_cur, d_v)
                
                sij = torch.matmul(
                    qi.to(matmul_dtype),
                    kj.to(matmul_dtype).mT
                )
                sij_scale = sij * scalar
                tilda_mij = sij_scale.max(dim=-1, keepdim=True).values
                t_sub = sij_scale - tilda_mij
                tilda_pij = torch.exp(t_sub)
                tilda_lij = tilda_pij.sum(dim=-1, keepdim=True)

                if bn == 0:
                    oi_tmp = torch.matmul(
                        tilda_pij.to(matmul_dtype),
                        vj.to(matmul_dtype)
                    )
                    if bn_per_batch == 1:
                        oi_update = oi_tmp / tilda_lij
                    else:
                        oi_update = oi_tmp
                    li_update = tilda_lij
                    mi_update = tilda_mij
                    continue
                oi = oi_update
                li = li_update
                mi = mi_update
                
                mi_new = torch.maximum(mi, tilda_mij)
                t1 = mi - mi_new
                t2 = torch.exp(t1)
                t3 = tilda_mij - mi_new
                t4 = torch.exp(t3)
                t5 = t4 * tilda_lij
                t6 = t2 * li
                li_new = t6 + t5
                q3 = oi * t2
                q1 = torch.matmul(tilda_pij.to(matmul_dtype), vj.to(matmul_dtype))
                q2 = q1 * t4
                oi_tmp = q3 + q2
                if bn == block_num_per_batch[b_index] - 1:
                    oi_update = oi_tmp / li_new
                else:
                    oi_update = oi_tmp
                li_update = li_new
                mi_update = mi_new
            tiled_out.append(oi_update)
    attent_out = torch.cat(tiled_out, dim=0)
    return attent_out


@pypto.jit
def flash_attention_pa(inputs: list, outputs: list, atten_cfg: AttentionConfig, tile_cfg: TileConfig):
    """
    PyPTO implementation of Flash Attention with Paged Attention format.
    
    This function implements the Flash Attention algorithm with online softmax
    computation for numerical stability, supporting variable-length sequences
    through block-based KV cache management.
    
    Parameters
    ----------
    inputs : list
        List containing [query, k_cache, v_cache, block_table, actual_seq_len]
    outputs : list
        List containing [atten_out]
    """
    # Enable dynamic unaligned support
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    pypto.set_host_options(only_codegen=True)
    
    block_size = atten_cfg.block_size
    tile_config = tile_cfg
    max_unroll_times = atten_cfg.max_unroll_times
    is_nz_format = atten_cfg.is_nz_format
    d_n = atten_cfg.dim
    softmax_scale = d_n ** -0.5
    n_tile = tile_config.head_num_q_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape
    query, k_cache, v_cache, block_table, actual_seq_len = inputs
    dtype = k_cache.dtype
    attention_out = outputs[0]
    with pypto.function("MAIN", [query, k_cache, v_cache, block_table, actual_seq_len], [attention_out]):
        def inside_main_function():
            batch_size = block_table.shape[0]
            n_q = query.shape[0] // batch_size
            n_loop = n_q // n_tile
            for b_idx in pypto.loop(0, batch_size, 1, name="LOOP_L0_bIdx", idx_name="b_idx"):
                def inside_b_idx_loop(b_idx):
                    cur_seq = actual_seq_len[b_idx]
                    bn_per_batch = (cur_seq + block_size - 1) // block_size
                    bn_per_batch.as_variable()
                    for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                        def inside_n_idx_loop(b_idx, n_idx, bn_per_batch):
                            nonlocal n_tile
                            cur_n_tile = n_tile
                            oi_update = pypto.tensor([n_tile, d_n], pypto.DT_FP32, "oi_update")
                            li_update = pypto.tensor([n_tile, 1], pypto.DT_FP32, "li_update")
                            mi_update = pypto.tensor([n_tile, 1], pypto.DT_FP32, "mi_update")
                            cur_offset = b_idx * n_q + n_idx * n_tile
                            oi_offset = [cur_offset, 0]  
                            
                            for bn in pypto.loop(0, bn_per_batch, 1, name="LOOP_L2_bn", 
                                            idx_name="bn", unroll_List={max_unroll_times}):
                                def inside_bn_loop(**kwargs):
                                    b_idx = kwargs.get("b_idx")
                                    block_table = kwargs.get("block_table")
                                    cur_seq = kwargs.get("cur_seq")
                                    bn = kwargs.get("bn")
                                    block_size = kwargs.get("block_size")
                                    nonlocal oi_update, li_update, mi_update
                                    cur_s2_tile = block_size
                                    qi = pypto.view(query, [cur_n_tile, d_n], [cur_offset, 0])
                                    cur_block_idx = block_table[b_idx, bn]
                                    cur_block_idx.as_variable()
                                    kj_format = pypto.TileOpFormat.TILEOP_NZ if is_nz_format else (
                                        pypto.TileOpFormat.TILEOP_ND
                                    )
                                    kj = pypto.tensor([cur_s2_tile, d_n], dtype, "kj", kj_format)
                                    kj = pypto.view(k_cache, [cur_s2_tile, d_n], 
                                                    [cur_block_idx * block_size, 0],
                                                valid_shape=[(cur_seq - bn * block_size).min(block_size), d_n])
                                    vj = pypto.view(v_cache, 
                                                    [cur_s2_tile, d_n], 
                                                    [cur_block_idx * block_size, 0], 
                                                    valid_shape=[(cur_seq - bn * block_size).min(block_size), d_n])

                                    pypto.set_semantic_label("MatMul")
                                    pypto.set_cube_tile_shapes(
                                        [c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]],
                                        [c1_tile[4], c1_tile[5]])
                                    pypto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
                                    sij = pypto.matmul(qi, kj, pypto.DT_FP32, b_trans=True)
                                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                    pypto.set_semantic_label("SoftMax")
                                    sij_scale = pypto.mul(sij, float(softmax_scale))
                                    pypto.set_semantic_label("SoftMax")
                                    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                                    tsub = pypto.sub(sij_scale, tilda_mij)
                                    tilda_pij = pypto.exp(tsub)
                                    tilda_pij_f16 = pypto.cast(tilda_pij, dtype)
                                    tilda_lij = pypto.sum(tilda_pij, dim=-1, keepdim=True)

                                    if pypto.cond(pypto.is_loop_begin(bn)):
                                        def inside_if_loop_begin():
                                            nonlocal oi_update, li_update, mi_update
                                            pypto.set_cube_tile_shapes(
                                                [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]],
                                                [c2_tile[4], c2_tile[5]])
                                            pypto.set_semantic_label("b1-matmul2")
                                            pypto.set_matrix_size(
                                                [tilda_pij_f16.shape[0], tilda_pij_f16.shape[1],
                                                    vj.shape[1]])
                                            oi_tmp = pypto.matmul(tilda_pij_f16,
                                                                vj, pypto.DT_FP32)
                                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                            pypto.set_semantic_label("b1-after-matmul2")
                                            if pypto.cond(pypto.is_loop_end(bn)):
                                                pypto.set_semantic_label("b1-after-matmul2")
                                                oi_update[:] = (pypto.div(oi_tmp, tilda_lij))
                                                pypto.assemble(oi_update, oi_offset, attention_out)
                                            else:
                                                oi_update[:] = (oi_tmp)
                                            li_update[:] = (tilda_lij)
                                            mi_update[:] = (tilda_mij)
                                        inside_if_loop_begin()
                                    else:
                                        def inside_else_loop_begin():
                                            nonlocal oi_update, li_update, mi_update
                                            oi = oi_update
                                            li = li_update
                                            mi = mi_update
                                            pypto.set_semantic_label("Softmax-acc")
                                            mi_new = pypto.maximum(mi, tilda_mij)
                                            t1 = pypto.sub(mi, mi_new)
                                            t2 = pypto.exp(t1)
                                            t3 = pypto.sub(tilda_mij, mi_new)
                                            t4 = pypto.exp(t3)
                                            t5 = pypto.mul(t4, tilda_lij)
                                            t6 = pypto.mul(t2, li)
                                            li_new = pypto.add(t6, t5)
                                            q3 = pypto.mul(oi, t2)
                                            pypto.set_semantic_label("bn-matmul2")
                                            pypto.set_cube_tile_shapes(
                                            [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4],
                                            c2_tile[5]])
                                            pypto.set_matrix_size(
                                                [tilda_pij_f16.shape[0],
                                                    tilda_pij_f16.shape[1], vj.shape[1]])
                                            q1 = pypto.matmul(tilda_pij_f16, vj, pypto.DT_FP32)
                                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                            pypto.set_semantic_label("bn-after-matmul2")
                                            q2 = pypto.mul(q1, t4)
                                            oi_tmp = pypto.add(q3, q2)
                                            if pypto.cond(pypto.is_loop_end(bn)):
                                                oi_update[:] = (pypto.div(oi_tmp, li_new))
                                                pypto.assemble(oi_update, oi_offset, attention_out)
                                            else:
                                                oi_update[:] = (oi_tmp)
                                            li_update[:] = (li_new)
                                            mi_update[:] = (mi_new)
                                        inside_else_loop_begin()
                                inside_bn_loop(
                                    b_idx=b_idx,
                                    block_table=block_table,
                                    cur_seq=cur_seq,
                                    bn=bn,
                                    block_size=block_size,
                                    bn_per_batch=bn_per_batch)
                        inside_n_idx_loop(b_idx, n_idx, bn_per_batch)
                inside_b_idx_loop(b_idx)
        inside_main_function()


def get_input_from_param(atten_cfg: AttentionConfig):
    def gen_uniform_data(data_shape, min_value, max_value, dtype):
        if min_value == 0 and max_value == 0:
            return torch.zeros(data_shape, dtype=dtype)
        if dtype == torch.bool:
            return torch.rand(data_shape) < 0.5
        return (torch.rand(data_shape) * (max_value - min_value) + min_value).to(dtype)
    
    def convert_tensors_contiguous(tensor_list):
        for idx, t in enumerate(tensor_list):
            if isinstance(t, torch.Tensor):
                tensor_list[idx] = t if t.is_contiguous() else t.contiguous()
        return tensor_list
    # Initialization of hyperparameters related to data generation
    b = atten_cfg.b
    n_q = atten_cfg.n_q
    skv = atten_cfg.skv
    block_size = atten_cfg.block_size
    dtype = atten_cfg.dtype
    s_q = atten_cfg.s_q
    n_kv = atten_cfg.n_kv
    kv_lora_rank = atten_cfg.dim
    
    d_q = kv_lora_rank
    d_k = kv_lora_rank
    d_v = kv_lora_rank
    actual_seq_len = torch.full((b,), skv, dtype=torch.int32) 
    s_max = max(actual_seq_len)
    shape_q = [b * n_q * s_q, d_q]
    shape_k = [b, s_max, n_kv * d_k]
    block_num_per_batch = []
    block_num_min = 0
    # Generate q, k, v data
    q_bnsd = gen_uniform_data(shape_q, -1, 1, dtype)
    k_tensor_bsh_raw = gen_uniform_data(shape_k, -1, 1, dtype)
    v_tensor_bsh_raw = k_tensor_bsh_raw[:, :, :n_kv * d_v]
    for actual_seq in actual_seq_len:
        block_num_per_batch.append(int(math.ceil(actual_seq / block_size)))
        block_num_min += int(math.ceil(actual_seq / block_size))
    block_table_shape = [b, int(math.ceil(s_max / block_size))]
    block_num = block_num_min
    atten_cfg.block_num = block_num
    block_idx_list = torch.arange(0, block_num, 1)
    block_idx_list = torch.randperm(len(block_idx_list), dtype=torch.int32)
    block_idx = 0
    block_table = torch.full((block_table_shape[0], block_table_shape[1]), -1, dtype=torch.int32)
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1
    k_cache = torch.zeros(block_num, block_size, n_kv * d_k, dtype=dtype)
    v_cache = torch.zeros(block_num, block_size, n_kv * d_v, dtype=dtype)
    k_tensor_bsh = torch.zeros(b, block_table_shape[1] * block_size, n_kv * d_k, dtype=dtype)
    v_tensor_bsh = torch.zeros(b, block_table_shape[1] * block_size, n_kv * d_v, dtype=dtype)
    k_tensor_bsh[:, :k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]
    v_tensor_bsh[:, :v_tensor_bsh_raw.shape[1], :] = v_tensor_bsh_raw[:, :, :]
    for b_idx in range(b):
        for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
            block_offset = block_i * block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                k_cache[kv_cache_blk_id, 0:block_size, :] = k_tensor_bsh[
                                                            b_idx, block_offset:(block_offset + block_size), :]
                v_cache[kv_cache_blk_id, 0:block_size, :] = v_tensor_bsh[
                                                            b_idx, block_offset:(block_offset + block_size), :]
    query = q_bnsd[:, :kv_lora_rank]
    k_cache_h = kv_lora_rank * n_kv
    k_cache = k_cache[:, :, : k_cache_h]
    k_cache = k_cache.reshape(k_cache.shape[0] * k_cache.shape[1], k_cache.shape[-1])
    v_cache = v_cache.reshape(v_cache.shape[0] * v_cache.shape[1], v_cache.shape[-1])
    kernel_inputs = [query, k_cache, v_cache, block_table, actual_seq_len]
    kernel_inputs = convert_tensors_contiguous(kernel_inputs)
    return kernel_inputs


def run_flash_attention_pa():
    """
    Test Flash Attention with PA format implementation.
    
    Parameters
    ----------
    atten_cfg : AttentionConfig
        Attention configuration
    """
    device_id = torch.npu.current_device()
    device = f'npu:{device_id}'
    torch_dtype = torch.float32
    atten_cfg, tile_cfg = get_qwen_common_config()
    # Prepare inputs for PyPTO kernel
    kernel_inputs = get_input_from_param(atten_cfg)
    kernel_inputs = [x.to(device=device) for x in kernel_inputs]
    shape_q = kernel_inputs[0].shape
    out_torch = torch.full(shape_q, 9, dtype=torch_dtype, device=device)
    outputs = [out_torch]
    # Execute PyPTO kernel
    flash_attention_pa(kernel_inputs, outputs, atten_cfg, tile_cfg)
    pypto.runtime._device_synchronize()
    # Verify results
    attention_output = flash_attn_pa_golden(kernel_inputs, outputs, atten_cfg, tile_cfg)
    y_data = out_torch.cpu()
    detailed_allclose_manual(
        np.array(attention_output.cpu()).flatten(),
        np.array(y_data).flatten(),
        "flash_attention_pa"
    )


def test_ifa():
    """Test Flash Attention with PA format."""
    print("=" * 60)
    print("Test: Flash Attention with Paged Attention (Qwen3)")
    print("=" * 60)
    run_flash_attention_pa()
    print("\n✓ Flash Attention with PA test passed")


def main():
    """Run Flash Attention example.
    
    Usage:
        python qwen3_attention.py          # Run example
        python qwen3_attention.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Flash Attention with Paged Attention Example (Qwen3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run the example
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=int,
        nargs='?',
        help='Example ID to run (1). If not specified, the example will run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available examples and exit'
    )
    
    args = parser.parse_args()
    
    # Define available examples
    examples = {
        1: {
            'name': 'Flash Attention with PA',
            'description': 'Flash Attention with Paged Attention format',
            'function': test_ifa,
            'requires_npu': True
        }
    }
    
    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            npu_req = " (Requires NPU)" if ex_info['requires_npu'] else " (No NPU required)"
            print(f"  {ex_id}. {ex_info['name']}{npu_req}")
            print(f"     {ex_info['description']}\n")
        return
    
    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("PyPTO Flash Attention with Paged Attention Example (Qwen3)")
    print("=" * 60 + "\n")
    
    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []
    
    if args.example_id is not None:
        # Run single example
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        # Run all examples
        examples_to_run = list(examples.items())
    
    # Check if any example requires NPU
    requires_npu = any(ex_info['requires_npu'] for _, ex_info in examples_to_run)
    
    if requires_npu:
        device_id = get_device_id()
        if device_id is None:
            return
        # Set the device once for all examples
        torch.npu.set_device(device_id)
    
    try:
        for ex_id, ex_info in examples_to_run:
            if ex_info['requires_npu'] and device_id is None:
                print(f"Skipping example {ex_id} ({ex_info['name']}): NPU device not configured")
                continue
            
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function']()
        
        if len(examples_to_run) > 1:
            print("\n" + "=" * 60)
            print("All tests completed successfully!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
