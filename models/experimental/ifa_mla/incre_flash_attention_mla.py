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
from dataclasses import dataclass
import math

import torch

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
# Set up logging format with timestamp, level, filename, and line number
formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]'
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)


@dataclass
class MlaConfig:
    """
    Configuration parameters for mla computation.

    Attributes:
        layout: Input layout
        b: Batch size
        n1: Number of query heads
        s1: Query sequence length
        q_d: Query head dimension
        q_rope_d: Query rope dimension
        n2: Number of key/value heads (for grouped query attention)
        s2: Key/Value sequence length (maximum)
        kv_d: Key/Value head dimension
        k_rope_d: Key rope dimension
        block_size: Size of each block in paged KV cache (default: 128)
        max_num_blocks_per_query: Maximum number of blocks per query sequence
        kv_num_blocks: Total number of KV blocks
    """
    layout: str = "BNSD"
    b: int = 32
    n1: int = 128
    s1: int = 1
    q_d: int = 512
    q_rope_d: int = 64
    n2: int = 1
    s2: int = 4096
    kv_d: int = 512
    k_rope_d: int = 64
    block_size: int = 128


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
        raise ValueError(f"Please set TILE_FWK_DEVICE_ID.")

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        logger.info(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def get_device(device_id: int = None, run_mode: str = "npu"):
    """
    Get the appropriate device string for computation.

    Args:
        device_id: Explicit device ID (optional)
        run_mode: Execution mode - "npu" for hardware, "sim" for simulation

    Returns:
        str: Device string (e.g., "npu:0" or "cpu")
    """
    if device_id is not None:
        cue_device_id = device_id
    else:
        cue_device_id = get_env_device_id()

    device = f"npu:{cue_device_id}" if (run_mode == "npu" and cue_device_id is not None) else "cpu"
    return device


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, data_shape, dtype=dtype)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    else:
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def gen_inputs(mla_config: MlaConfig, device: str):
    query_shape = [mla_config.b, mla_config.n1, mla_config.s1, mla_config.q_d]  # BNSD
    query = gen_uniform_data(query_shape, -1, 1, torch.bfloat16).to(device=device)
    query_rope_shape = [mla_config.b, mla_config.n1, mla_config.s1, mla_config.q_rope_d]  # BNSD
    query_rope = gen_uniform_data(query_rope_shape, -1, 1, torch.bfloat16).to(device=device)

    max_num_blocks_per_query = math.ceil(mla_config.s2 / mla_config.block_size)
    kv_num_blocks = mla_config.b * max_num_blocks_per_query
    key_cache_shape = [kv_num_blocks, mla_config.n2, mla_config.block_size, mla_config.kv_d]  # PA_BnNBsD format
    key_cache = gen_uniform_data(key_cache_shape, -1, 1, torch.bfloat16).to(device=device)
    value_cache = key_cache

    key_rope_cache_shape = [kv_num_blocks, mla_config.n2, mla_config.block_size, mla_config.k_rope_d]  # PA_BnNBsD format
    key_rope_cache = gen_uniform_data(key_rope_cache_shape, -1, 1, torch.bfloat16).to(device=device)

    kv_actual_seqs = torch.tensor([mla_config.s2] * mla_config.b, dtype=torch.int32, device=device)

    block_table = gen_block_table(mla_config, kv_actual_seqs, device)

    key = kv_cache_concat(key_cache, kv_actual_seqs, block_table, mla_config, device)
    value = key

    key_rope = kv_cache_concat(key_rope_cache, kv_actual_seqs, block_table, mla_config, device)

    logger.info(f"query_shape: {query.shape}")
    logger.info(f"key_shape: {key.shape}")
    logger.info(f"value_shape: {value.shape}")
    logger.info(f"key_cache_shape: {key_cache.shape}")
    logger.info(f"value_cache_shape: {value_cache.shape}")
    logger.info(f"query_rope_shape: {query_rope.shape}")
    logger.info(f"key_rope_shape: {key_rope.shape}")
    logger.info(f"key_rope_cache_shape: {key_rope_cache.shape}")
    logger.info(f"block_table_shape: {block_table.shape}")
    logger.info(f"layout: {mla_config.layout}")
    logger.info(f"kv_actual_seqs_shape: {kv_actual_seqs.shape}")
    logger.info(f"kv_actual_seqs: {kv_actual_seqs}")
    logger.info(f"num_heads: {mla_config.n1}")
    logger.info(f"key_num_heads: {mla_config.n2}")
    logger.info(f"block_size: {mla_config.block_size}")

    mla_inputs = dict(
        query=query, key=key, value=value, key_cache=key_cache, 
        value_cache=value_cache, query_rope=query_rope, key_rope=key_rope,
        key_rope_cache=key_rope_cache, block_table=block_table, 
        kv_actual_seqs=kv_actual_seqs,
    )
    
    return mla_inputs


def gen_block_table(mla_config, kv_actual_seqs, device: str):
    """
    Generate a block table for paged KV cache.

    The block table maps logical block indices to physical block indices,
    enabling non-contiguous memory access patterns. This is essential for
    efficient memory management in autoregressive generation.

    Args:
        mla_config: Mla configuration
        kv_actual_seqs: Tensor containing kv actual sequence lengths for each batch
        device: Device to create tensors on (e.g., 'npu:0', 'cpu')

    Returns:
        torch: Block table tensor of shape [batch_size, max_blocks_per_query]
               Contains physical block indices, or -1 for invalid blocks
    """
    block_num_per_batch = []
    block_num = 0  # Total number of blocks needed
    
    block_size = mla_config.block_size
    block_table_batch = mla_config.b
    max_num_blocks_per_query = math.ceil(mla_config.s2 / block_size)
    block_table_shape = [block_table_batch, max_num_blocks_per_query]

    # Calculate number of blocks needed for each batch element
    for actual_seq in kv_actual_seqs:
        block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
        block_num += math.ceil(actual_seq.item() / block_size)

    # Create all block indices and randomly permute them
    # This simulates non-contiguous physical memory allocation
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]

    # Create block table
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32, device=device)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def kv_cache_concat(cache_tensor, kv_actual_seqs, block_table, mla_config, device: str):
    batch_size = mla_config.b
    kv_num_heads = mla_config.n2 
    kv_seqs_len = mla_config.s2
    d = cache_tensor.shape[3]  # BNSD

    block_size = mla_config.block_size
    dtype = cache_tensor.dtype

    # Initializes output tensors
    result = torch.zeros([batch_size, kv_num_heads, kv_seqs_len, d], dtype=dtype, device=device)
    
    # Reconstruct tensors by following block table
    for b_idx in range(batch_size):
        block_list = block_table[b_idx]
        temp_tensor = torch.zeros([1, kv_num_heads, kv_seqs_len, d], dtype=dtype, device=device)
        s_idx = 0

        # Copy blocks according to block table
        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            # Copy block from cache to temp tensor
            temp_tensor[:, :, start_idx:end_idx, :] = cache_tensor[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        result[b_idx:b_idx + 1, :, :, :] = temp_tensor
    return result


def do_test_incre_flash_attention_mla(case_name):
    logger.info("*" * 60)
    logger.info(f"Run incre_flash_attention_mla {case_name} case")
    logger.info("*" * 60 + "\n")

    device = get_device()

    mla_config = MlaConfig()
    gen_inputs(mla_config, device)


def test_incre_flash_attention_mla_32b4k():
    do_test_incre_flash_attention_mla("32b4k")


def main():
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("PyPTO incre_flash_attention_mla example")
    logger.info("=" * 60 + "\n")

    test_incre_flash_attention_mla_32b4k()


if __name__ == "__main__":
    main()