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
from dataclasses import dataclass
from typing import List
import logging
import math
import pto

TILE_VEC_DIMS = 2

NUM_16 = 16


@dataclass
class KvSlcTileShapeConfig:
    v0_tile_shape = [0] * TILE_VEC_DIMS


def kv_slc_compute(**kwargs):
    top_k_indcies: pto.tensor = kwargs.get("top_k_indcies")
    top_k_tensor_shape: pto.tensor = kwargs.get("top_k_tensor_shape")
    kv_nope_cache: pto.tensor = kwargs.get("kv_nope_cache")
    k_rope_cache: pto.tensor = kwargs.get("k_rope_cache")
    kv_act_seqs: pto.tensor = kwargs.get("kv_act_seqs")
    front: int = kwargs.get("front")
    near: int = kwargs.get("near")
    topk: int = kwargs.get("topk")
    l_prime: int = kwargs.get("l_prime")
    n2: int = kwargs.get("n2")
    block_table: pto.tensor = kwargs.get("block_table")
    block_size: int = kwargs.get("block_size")
    k_slc_out: pto.tensor = kwargs.get("k_slc_out")
    v_slc_out: pto.tensor = kwargs.get("v_slc_out")
    kv_slc_act_seqs: pto.tensor = kwargs.get("kv_slc_act_seqs")
    tile_config: KvSlcTileShapeConfig = kwargs.get("tile_config")
    debug: bool = kwargs.get("debug")

    v0_tile = tile_config.v0_tile_shape
    b = top_k_indcies.shape[0]
    s = top_k_indcies.shape[1]
    kv_lora_rank = kv_nope_cache.shape[1] // n2
    rope_dim = k_rope_cache.shape[1] // n2

    with pto.loop_function("LOOP_L0_batchIdx", "batch_idx", pto.loop_range(0, b, 1)) as batch_idx_loop:
        for batch_idx in batch_idx_loop:
            def inside_batch_idx_loop(batch_idx):
                cur_act_seq = pto.get_tensor_data(kv_act_seqs, [batch_idx])
                with pto.loop_function("LOOP_L1_slcIdx", "slc_idx", pto.loop_range(0, s, 1)) as slc_idx_loop:
                    for slc_idx in slc_idx_loop:
                        def inside_slc_idx_loop(slc_idx):
                            with pto.loop_function("LOOP_L2_kvSlcIdx", "nkv_idx",
                            pto.loop_range(0, n2, 1)) as nkv_idx_loop:
                                for nkv_idx in nkv_idx_loop:
                                    def inside_nkv_idx_loop(nkv_idx):
                                        pto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
                                        s_slc = pto.get_tensor_data(top_k_tensor_shape, [batch_idx, slc_idx])
                                        positions = 0
                                        prime_value = l_prime
                                        slc_seq_len = 0
                                        for top_k_idx in range(topk):
                                            if top_k_idx < front:
                                                positions = top_k_idx * l_prime
                                            else:
                                                topk_index = None
                                                if debug:
                                                    pto.set_vec_tile_shapes(1, 1, NUM_16)
                                                    topk_index = pto.get_tensor_data(top_k_indcies, [batch_idx, slc_idx,
                                                    pto.symbolic_scalar(top_k_idx - front)])
                                                else:
                                                    topk_index = pto.get_tensor_data(top_k_indcies, [batch_idx, slc_idx, 
                                                    pto.symbolic_scalar(top_k_idx - front)])
                                                positions = topk_index * prime_value
                                            slc_seq_len = slc_seq_len + prime_value
                                            block_idx_in_batch = positions // pto.symbolic_scalar(block_size)
                                            tail = positions % block_size
                                            slc_block_idx = pto.get_tensor_data(block_table, 
                                            [batch_idx, block_idx_in_batch])
                                            pto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
                                            kv_slc_block = pto.view(kv_nope_cache, [l_prime, kv_lora_rank],
                                            [slc_block_idx * block_size + tail, nkv_idx * kv_lora_rank])
                                            k_rope_slc_block = pto.view(k_rope_cache, [l_prime, rope_dim],
                                            [slc_block_idx * block_size + tail, nkv_idx * rope_dim])
                                            pto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
                                            kv_slc_block_fp32 = pto.cast(kv_slc_block, pto.DT_FP32)
                                            k_rope_slc_block_fp32 = pto.cast(k_rope_slc_block, pto.DT_FP32)
                                            pto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
                                            kv_slc_block_tiled = pto.mul_s(kv_slc_block_fp32,
                                            pto.element(kv_slc_block_fp32.get_dtype(), float(1)))
                                            k_rope_slc_block_tiled = pto.mul_s(k_rope_slc_block_fp32,
                                            pto.element(k_rope_slc_block_fp32.get_dtype(), float(1)))
                                            pto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
                                            kv_slc_block_fp16 = pto.cast(kv_slc_block_tiled, k_slc_out.get_dtype())
                                            k_rope_slc_block_fp16 = pto.cast(k_rope_slc_block_tiled,
                                            v_slc_out.get_dtype())
                                            pto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
                                            output_axis1_value = batch_idx * s * n2 * topk * l_prime
                                            + slc_idx * n2 * topk * l_prime
                                            + nkv_idx * topk * l_prime + top_k_idx * l_prime
                                            pto.assemble(kv_slc_block_fp16, [output_axis1_value, 0], k_slc_out)
                                            pto.assemble(k_rope_slc_block_fp16,
                                            [output_axis1_value, kv_lora_rank], k_slc_out)
                                            pto.assemble(kv_slc_block_fp16, [output_axis1_value, 0], v_slc_out)
                                        pto.set_tensor_data(pto.symbolic_scalar(slc_seq_len),
                                        [batch_idx, slc_idx], kv_slc_act_seqs)
                                    inside_nkv_idx_loop(slc_idx)
                        inside_slc_idx_loop(slc_idx)
            inside_batch_idx_loop(batch_idx)


if __name__ == '__main__':
    tile_config = KvSlcTileShapeConfig()
    n_tile = 32
    tile_config.v0_tile_shape = [n_tile, 32]

    input_param: List[int] = [1, 1, 1, 64, 64, 1, 2, 4, 16, 32]
    b = input_param[0]
    s = input_param[1]
    n2 = input_param[2]
    kv_lora_rank = input_param[3]
    rope_dim = input_param[4]
    front = input_param[5]
    near = input_param[6]
    topk = input_param[7]
    l_prime = input_param[8]
    block_size = input_param[9]

    block_num = 0
    seq = [1] * 1024
    for s_item in seq:
        block_num += (s_item + block_size - 1) // block_size
    max_seq_all_batch = max(seq)
    max_block_num_per_batch = math.ceil(max_seq_all_batch / block_size)

    top_k_indcies = pto.tensor([b, s, topk - front - near], pto.DT_INT32, "top_k_indcies")
    top_k_tensor_shape = pto.tensor([b, s], pto.DT_INT32, "top_k_tensor_shape")
    kv_nope_cache = pto.tensor([int(block_num * block_size), n2 * kv_lora_rank], pto.DT_FP16, "kv_nope_cache")
    k_rope_cache = pto.tensor([int(block_num * block_size), n2 * rope_dim], pto.DT_FP16, "k_rope_cache")
    kv_act_seqs = pto.tensor([b], pto.DT_INT32, "kv_act_seqs")
    block_table = pto.tensor([b, max_block_num_per_batch], pto.DT_INT32, "block_table")
    k_slc_out = pto.tensor([b * s * n2 * topk * l_prime, rope_dim + kv_lora_rank], pto.DT_FP16, "k_slc_out")
    v_slc_out = pto.tensor([b * s * n2 * topk * l_prime, kv_lora_rank], pto.DT_FP16, "v_slc_out")
    kv_slc_act_seqs = pto.tensor([b, s], pto.DT_INT32, "kv_slc_act_seqs")
    input_tensors = [
        top_k_indcies, top_k_tensor_shape, kv_nope_cache,
        k_rope_cache, kv_act_seqs, block_table
        ]
    output_tensors = [k_slc_out, v_slc_out, kv_slc_act_seqs]
    with pto.function("main_slc", input_tensors, output_tensors):
        kv_slc_compute(
            top_k_indcies=top_k_indcies,
            top_k_tensor_shape=top_k_tensor_shape,
            kv_nope_cache=kv_nope_cache,
            k_rope_cache=k_rope_cache,
            kv_act_seqs=kv_act_seqs,
            front=front,
            near=near,
            topk=topk,
            l_prime=l_prime,
            n2=n2,
            block_table=block_table,
            block_size=block_size,
            k_slc_out=k_slc_out,
            v_slc_out=v_slc_out,
            kv_slc_act_seqs=kv_slc_act_seqs,
            tile_config=tile_config
        )
