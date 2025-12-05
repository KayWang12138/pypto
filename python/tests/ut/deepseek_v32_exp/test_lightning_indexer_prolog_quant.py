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
"""

import pypto
import torch
import os
from examples.models.deepseek_v32_exp.lightning_indexer_prolog_quant import IndexerPrologQuantInput, \
    IndexerPrologQuantOutput, IndexerPrologQuantAttr, IndexerPrologQuantConfigs, lightning_indexer_prolog_quant


def build_indexer_args(
        # ==== match C++ test config ====
        t=4,  # params.t
        dim=7168,  # paras.h
        n_q=64,  # params.head_num
        n_kv=1,  # params.n_kv
        qk_nope_head_dim=64,  # headDim - ropeHeadDim
        qk_rope_head_dim=64,  # ropeHeadDim
        q_lora_rank=1536,  # params.q_lora_rank
        # cache//grid
        block_size=128,  # params.block_size
        block_num=448,  # params.block_num
):
    """
    Build args for lightning_indexer_prolog with required dataclasses.
    Returns (kwargs, meta).
    """
    pypto.set_host_options(only_codegen=True)

    # ---- DTypes ----
    base_dtype = torch.bfloat16  # C++ uses BF16
    NZ_B8_C0 = 32
    NZ_FIRST_DIM = 16
    NZ_B16_C0 = 16

    # ---- Derived sizes ----
    head_num = n_q
    rope_head_dim = qk_rope_head_dim
    head_dim = qk_nope_head_dim + qk_rope_head_dim  # 64 + 64 = 128

    # ---- Inputs tensors match C++ ----
    x = torch.empty([t, dim], dtype=base_dtype)
    q_norm = torch.empty([t, q_lora_rank], dtype=torch.int8)  # [b,seq,qLoraRank]
    q_norm_scale = torch.empty([t, 1], dtype=torch.float32)
    w_qb = torch.empty([head_num * head_dim // NZ_B8_C0, q_lora_rank // NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B8_C0],
                       dtype=torch.int8)
    w_qb_scale = torch.empty([head_num * head_dim, 1], dtype=torch.float32)
    wk = torch.empty([head_dim // NZ_B16_C0, dim // NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B16_C0],
                     dtype=base_dtype)  # [dim, headDim]
    w_proj = torch.empty([head_num // NZ_B16_C0, dim // NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B16_C0], dtype=base_dtype)
    ln_gamma_k = torch.empty([head_dim], dtype=base_dtype)  # [headDim]
    ln_beta_k = torch.empty([head_dim], dtype=base_dtype)  # [headDim]
    cos_idx_rope = torch.empty([t, rope_head_dim], dtype=base_dtype)
    sin_idx_rope = torch.empty([t, rope_head_dim], dtype=base_dtype)
    hadamard_q = torch.empty([head_dim, head_dim], dtype=base_dtype)
    hadamard_k = torch.empty([head_dim, head_dim], dtype=base_dtype)
    k_cache = torch.empty([block_num, block_size, n_kv, head_dim], dtype=torch.int8)
    k_cache_scale = torch.empty([block_num, block_size, n_kv, 1], dtype=torch.float16)
    # C++: kCacheIndex {b, seq}
    k_cache_index = torch.empty([t], dtype=torch.int64)

    inputs = IndexerPrologQuantInput(
        x=x,
        q_norm=q_norm,
        q_norm_scale=q_norm_scale,
        w_qb=w_qb,
        w_qb_scale=w_qb_scale,
        wk=wk,
        w_proj=w_proj,
        ln_gamma_k=ln_gamma_k,
        ln_beta_k=ln_beta_k,
        cos_idx_rope=cos_idx_rope,
        sin_idx_rope=sin_idx_rope,
        hadamard_q=hadamard_q,
        hadamard_k=hadamard_k,
        k_cache=k_cache,
        k_cache_scale=k_cache_scale,
        k_cache_index=k_cache_index,
    )

    # ---- Outputs tensors match C++ ----
    q_int8 = torch.empty([t, head_num, head_dim], dtype=torch.int8)
    q_scale = torch.empty([t, head_num, 1], dtype=torch.float16)
    k_int8 = torch.empty([block_num, block_size, n_kv, head_dim], dtype=torch.int8)
    k_scale = torch.empty([block_num, block_size, n_kv, 1], dtype=torch.float16)
    weights = torch.empty([t, head_num], dtype=torch.float16)

    outputs = IndexerPrologQuantOutput(q_int8=q_int8, q_scale=q_scale, k_int8=k_int8, k_scale=k_scale, weights=weights)

    # ---- Attrs ----
    attrs = IndexerPrologQuantAttr(
        eps=1e-6,
        layerout_query="TND",
        layerout_key="PA_BSND",
    )

    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        l1_reuse_param={1: 4},
        copy_in_threshold=2 * 1024 * 1024,
        cycle_upper_bound=8192,
        block_size=128
    )

    kwargs = {"inputs": inputs, "outputs": outputs, "attrs": attrs, "configs": configs}

    return kwargs


def lighting_indexer_prolog_quant_dyn(inputs: IndexerPrologQuantInput, outputs: IndexerPrologQuantOutput,
                                      attrs: IndexerPrologQuantAttr, configs: IndexerPrologQuantConfigs):
    input_tensors = {
        inputs.x: [0],
        inputs.q_norm: [0],
        inputs.q_norm_scale: [0],
        inputs.w_qb: [],
        inputs.w_qb_scale: [],
        inputs.wk: [],
        inputs.w_proj: [],
        inputs.ln_gamma_k: [],
        inputs.ln_beta_k: [],
        inputs.cos_idx_rope: [0],
        inputs.sin_idx_rope: [0],
        inputs.hadamard_q: [],
        inputs.hadamard_k: [],
        inputs.k_cache: [0],
        inputs.k_cache_scale: [0],
        inputs.k_cache_index: [0],
    }
    output_tensors = {
        outputs.q_int8: [],
        outputs.q_scale: [],
        outputs.k_int8: [],
        outputs.k_scale: [],
        outputs.weights: []
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in input_tensors.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in output_tensors.items()]
    lightning_indexer_prolog_quant(pto_inputs, pto_outputs, attrs, configs)


def do_test_lighting_indexer_prolog_quant(case_name):

    print(f"=== run test case: {case_name} ===")

    t = 8
    dim = 7168
    q_lora_rank = 1536
    head_dim = 128
    head_num = 64
    rope_head_dim = 64
    block_size = 128
    block_num = 10  # a rand num
    n_kv = 1

    kwargs = build_indexer_args(t, dim, head_num, n_kv, head_dim - rope_head_dim, rope_head_dim, q_lora_rank,
                                block_size, block_num)
    inputs = kwargs["inputs"]
    outputs = kwargs["outputs"]
    attrs = kwargs["attrs"]
    configs = kwargs["configs"]

    lighting_indexer_prolog_quant_dyn(inputs, outputs, attrs, configs)

    print(f"=== {case_name}: PASS ===")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    case_name = "QuantLightningIndexerPrologUTest.b4_s1_2_s2_block_10"
    do_test_lighting_indexer_prolog_quant(case_name)


def test_b4_s1_2_s2_64k():
    do_test_lighting_indexer_prolog_quant("QuantLightningIndexerPrologUTest.b4_s1_2_s2_block_10")
