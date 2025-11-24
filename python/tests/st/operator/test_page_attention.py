#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import sys
import os
from pathlib import Path
import torch
import math
from op_page_attention import op_page_attention, op_page_attention_golden
sys.path.append(str(Path(os.path.abspath(__file__)).parents[4].joinpath("framework/tests/cmake/scripts/helper")))
from pypto_test import TestBuilder


class PATest(TestBuilder):
    def __init__(self, params: tuple, kernel, kernel_golden, tiling: int):
        super().__init__(params, kernel, kernel_golden, tiling)
        
    def get_input_from_param(self):
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
        # 数据生成相关超参数初始化
        data_params = self.params[0]
        b = data_params["b"]
        n_q = data_params["n_q"]
        skv = data_params["skv"]
        block_size = data_params["block_size"]
        dtype = data_params["dtype"]
        s_q = data_params["s_q"]
        n_kv = data_params["n_kv"]
        kv_lora_rank = data_params["kv_lora_rank"]
        qk_rope_dim = data_params["qk_rope_dim"]
        
        d_q = kv_lora_rank + qk_rope_dim
        d_k = kv_lora_rank + qk_rope_dim
        d_v = kv_lora_rank
        actual_seq_len = torch.full((b,), skv, dtype=torch.int32) 
        s_max = max(actual_seq_len)
        shape_q = [b * n_q * s_q, d_q]
        shape_k = [b, s_max, n_kv * d_k]
        shape_v = [b, s_max, n_kv * d_v]
        block_num_per_batch = []
        block_num_min = 0
        # 生成 q k v 数据
        q_bnsd = gen_uniform_data(shape_q, -1, 1, dtype)
        k_tensor_bsh_raw = gen_uniform_data(shape_k, -1, 1, dtype)
        v_tensor_bsh_raw = k_tensor_bsh_raw[:, :, :kv_lora_rank]
        for actual_seq in actual_seq_len:
            block_num_per_batch.append(math.ceil(actual_seq / block_size))
            block_num_min += math.ceil(actual_seq / block_size)
        block_table_shape = [b, math.ceil(s_max / block_size)]
        block_num = block_num_min
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
        q_nope = q_bnsd[:, :kv_lora_rank]
        q_rope = q_bnsd[:, kv_lora_rank:]
        k_cache_nope_h = kv_lora_rank * n_kv
        k_cache_nope = k_cache[:, :, : k_cache_nope_h]
        k_cache_rope = k_cache[:, :, k_cache_nope_h:]
        k_cache_nope = k_cache_nope.reshape(k_cache_nope.shape[0] * k_cache_nope.shape[1], k_cache_nope.shape[-1])
        k_cache_rope = k_cache_rope.reshape(k_cache_rope.shape[0] * k_cache_rope.shape[1], k_cache_rope.shape[-1])
        v_cache = v_cache.reshape(v_cache.shape[0] * v_cache.shape[1], v_cache.shape[-1])
        kernel_inputs = [q_nope, k_cache_nope, v_cache, q_rope, k_cache_rope,
            block_table, actual_seq_len]
        kernel_inputs = convert_tensors_contiguous(kernel_inputs)
        self.setup_inputs(*kernel_inputs)
        self.set_tol(rtol=5e-4, atol=5e-4)
        golden_hyper_params = self.params[0]
        golden_hyper_params["block_num"] = block_num
        golden_inputs = [q_nope, k_cache_nope, v_cache, q_rope, k_cache_rope,
            block_table, actual_seq_len, golden_hyper_params]
        return golden_inputs


class TileConfig:
    def __init__(self, head_num_q_tile, c1_tile_shape, v1_tile_shape,
    c2_tile_shape, v2_tile_shape):
        self.head_num_q_tile = head_num_q_tile  
        self.c1_tile_shape = c1_tile_shape  
        self.v1_tile_shape = v1_tile_shape 
        self.c2_tile_shape = c2_tile_shape 
        self.v2_tile_shape = v2_tile_shape  


def test():
    # 数据生成及验证函数相关超参数配置
    data_golden_params = {
        "b": 4,
        "n_q": 32,
        "skv": 256,
        "block_size": 128,
        "dtype": torch.float32,
        "s_q": 1,
        "n_kv": 1,
        "kv_lora_rank": 512,
        "qk_rope_dim": 64,
        "n_tile": 32
    }
    # 目标上板函数相关超参数配置
    model_hyper_params = {
        "block_size": 128,
        "tile_config": TileConfig(head_num_q_tile=32, 
            c1_tile_shape=(32, 32, 64, 64, 128, 128), 
            v1_tile_shape=(32, 64),
            c2_tile_shape=(32, 32, 64, 64, 128, 128),
            v2_tile_shape=(32, 64)),
        "max_unroll_times": 1,
        "is_nz_format": False
    }
    st = PATest((data_golden_params, model_hyper_params), op_page_attention, op_page_attention_golden, tiling=32)
    st()


if __name__ == "__main__":
    test()