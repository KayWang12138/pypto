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
from dataclasses import dataclass
import pypto
import math
import torch
import torch_npu
from typing import List
import examples.models.deepseek_v32_exp.lightning_indexer_prolog_quant as ip
import examples.models.deepseek_v32_exp.mla_prolog_quant as mla

PRINT_DEBUG = False

@pypto.jit
def _mla_prolog_indexer_prolog_debug(inputs, outputs, mla_epsilon_cq, mla_epsilon_ckv, mla_cache_mode, mla_tile_config,
                                     ip_attrs, ip_configs):
    (token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
     mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
     mla_k_scale_cache, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in, ip_w_proj_in,
     ip_ln_gamma_k_in, ip_ln_beta_k_in, ip_hadamard_q_in, ip_hadamard_k_in, ip_k_cache, ip_k_cache_scale,
     ) = inputs
    (mla_query_nope_out, mla_query_rope_out, mla_kv_cache_out, mla_kr_cache_out,
     mla_k_scale_cache_out, ip_q_int8_out, ip_q_scale_out, ip_k_int8_out, ip_k_scale_out, ip_weights_out, mla_q_norm_out, mla_q_norm_scale_out) = outputs

    mla_input_tensors = (token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
                         mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
                         mla_k_scale_cache)
    mla_output_tensors = (mla_q_norm_out, mla_q_norm_scale_out, mla_query_nope_out, mla_query_rope_out,
                          mla_kv_cache_out, mla_kr_cache_out, mla_k_scale_cache_out)
    mla.mla_prolog_compute_p(mla_input_tensors, mla_output_tensors, mla_epsilon_cq, mla_epsilon_ckv, mla_cache_mode,
                             mla_tile_config)

    ip_input_tensors = (token_x, mla_q_norm_out, mla_q_norm_scale_out, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in,
                        ip_w_proj_in, ip_ln_gamma_k_in, ip_ln_beta_k_in, cos, sin, ip_hadamard_q_in, ip_hadamard_k_in,
                        ip_k_cache, ip_k_cache_scale, cache_index)
    ip_output_tensors = (ip_q_int8_out, ip_q_scale_out, ip_k_int8_out, ip_k_scale_out, ip_weights_out)
    ip.lightning_indexer_prolog_quant_compute(ip_input_tensors, ip_output_tensors, ip_attrs, ip_configs)

@pypto.jit
def _mla_prolog_indexer_prolog(inputs, outputs, mla_epsilon_cq, mla_epsilon_ckv, mla_cache_mode, mla_tile_config,
                               ip_attrs, ip_configs):
    (token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
     mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
     mla_k_scale_cache, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in, ip_w_proj_in,
     ip_ln_gamma_k_in, ip_ln_beta_k_in, ip_hadamard_q_in, ip_hadamard_k_in, ip_k_cache, ip_k_cache_scale,
     ) = inputs
    (mla_query_nope_out, mla_query_rope_out, mla_kv_cache_out, mla_kr_cache_out,
     mla_k_scale_cache_out, ip_q_int8_out, ip_q_scale_out, ip_k_int8_out, ip_k_scale_out, ip_weights_out) = outputs

    t = token_x.shape[0]
    q_lora_rank = ip_w_qb_in.shape[1]*16
    mla_q_norm_out = pypto.Tensor([t, q_lora_rank], pypto.DT_INT8)
    mla_q_norm_scale_out = pypto.Tensor([t, 1], pypto.DT_FP32)

    ##################### mla #######################
    #TODO mla options

    mla_input_tensors = (token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
                         mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
                         mla_k_scale_cache)
    mla_output_tensors = (mla_q_norm_out, mla_q_norm_scale_out, mla_query_nope_out, mla_query_rope_out,
                          mla_kv_cache_out, mla_kr_cache_out, mla_k_scale_cache_out)
    mla.mla_prolog_compute_p(mla_input_tensors, mla_output_tensors, mla_epsilon_cq, mla_epsilon_ckv, mla_cache_mode,
                             mla_tile_config)

    ##################### ip #######################
    pypto.set_runtime_options(workspace_recycle_period=512, estimated_stitch_task_max_loop_num=512)

    pypto.set_pass_options(nbuffer_merge_mode=0)
    pypto.set_pass_options(l1_reuse_map=ip_configs.l1_reuse_param)
    pypto.set_pass_options(copyin_threshold=ip_configs.copy_in_threshold)
    pypto.set_pass_options(cycle_upper_bound=ip_configs.cycle_upper_bound)

    ip_input_tensors = (token_x, mla_q_norm_out, mla_q_norm_scale_out, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in,
                        ip_w_proj_in, ip_ln_gamma_k_in, ip_ln_beta_k_in, cos, sin, ip_hadamard_q_in, ip_hadamard_k_in,
                        ip_k_cache, ip_k_cache_scale, cache_index)
    ip_output_tensors = (ip_q_int8_out, ip_q_scale_out, ip_k_int8_out, ip_k_scale_out, ip_weights_out)
    ip.lightning_indexer_prolog_quant_compute(ip_input_tensors, ip_output_tensors, ip_attrs, ip_configs)


mla_prolog_indexer_prolog = _mla_prolog_indexer_prolog_debug if PRINT_DEBUG else _mla_prolog_indexer_prolog
