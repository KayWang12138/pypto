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
import math
import torch
import torch_npu
import pypto
from typing import List
import lightning_indexer_prolog_quant as ip
import mla_prolog_quant as mla


@pypto.jit(
    runtime_options={"stitch_function_inner_memory": 512, "stitch_function_outcast_memory": 512,
                     "device_sched_mode": 2}
)
def mla_indexer_prolog_quant_debug(token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale,
                                   mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
                                   mla_gamma_ckv, cos, sin, cache_index,
                                   mla_kv_cache, mla_kr_cache, mla_k_scale_cache,
                                   ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in, ip_w_proj_in,
                                   ip_ln_gamma_k_in, ip_ln_beta_k_in, ip_hadamard_q_in,
                                   ip_hadamard_k_in, ip_k_cache, ip_k_cache_scale,
                                   mla_query_nope_out, mla_query_rope_out,
                                   mla_kv_cache_out, mla_kr_cache_out,
                                   mla_k_scale_cache_out, ip_q_int8_out, ip_q_scale_out,
                                   ip_k_int8_out, ip_k_scale_out, ip_weights_out,
                                   mla_q_norm_out, mla_q_norm_scale_out, mla_epsilon_cq,
                                   mla_epsilon_ckv, mla_cache_mode, mla_tile_config,
                                   ip_attrs, ip_configs):

    pypto.set_codegen_options(support_dynamic_unaligned=True)
    ##################### mla #######################
    pypto.set_pass_options(cube_l1_reuse_setting={0: 2, 1: 1, 2: 1, 3: 4, 4: 4, 5: 1},
                           cube_nbuffer_setting={3: 4},
                           mg_copyin_upper_bound=16 * 1024 * 1024)

    mla_input_tensors = (token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
                         mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
                         mla_k_scale_cache)
    mla_output_tensors = (mla_q_norm_out, mla_q_norm_scale_out, mla_query_nope_out, mla_query_rope_out,
                          mla_kv_cache_out, mla_kr_cache_out, mla_k_scale_cache_out)
    mla.mla_prolog_quant_compute(*mla_input_tensors, *mla_output_tensors, mla_epsilon_cq, mla_epsilon_ckv,
                                 mla_cache_mode,
                                 mla_tile_config, rope_cfg)

    ##################### ip #######################
    pypto.set_pass_options(cube_l1_reuse_setting=ip_configs.l1_reuse_param)
    pypto.set_pass_options(mg_copyin_upper_bound=ip_configs.mg_copy_in_upper_bound)
    pypto.set_pass_options(pg_upper_bound=ip_configs.pg_upper_bound)

    ip_input_tensors = (token_x, mla_q_norm_out, mla_q_norm_scale_out, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in,
                        ip_w_proj_in, ip_ln_gamma_k_in, ip_ln_beta_k_in, cos, sin, ip_hadamard_q_in, ip_hadamard_k_in,
                        ip_k_cache, ip_k_cache_scale, cache_index)
    ip_output_tensors = (ip_q_int8_out, ip_q_scale_out, ip_k_int8_out, ip_k_scale_out, ip_weights_out)
    ip.lightning_indexer_prolog_quant_compute(*ip_input_tensors, *ip_output_tensors, ip_attrs, ip_configs)


@pypto.jit(
    runtime_options={"stitch_function_inner_memory": 512, "stitch_function_outcast_memory": 512,
                     "device_sched_mode": 2}
)
def mla_indexer_prolog_quant(token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
                             mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
                             mla_k_scale_cache, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in, ip_w_proj_in,
                             ip_ln_gamma_k_in, ip_ln_beta_k_in, ip_hadamard_q_in, ip_hadamard_k_in,
                             ip_k_cache, ip_k_cache_scale, mla_query_nope_out, mla_query_rope_out,
                             mla_kv_cache_out, mla_kr_cache_out,
                             mla_k_scale_cache_out, ip_q_int8_out, ip_q_scale_out, ip_k_int8_out,
                             ip_k_scale_out, ip_weights_out, mla_epsilon_cq, mla_epsilon_ckv,
                             mla_cache_mode, mla_tile_config,
                             ip_attrs, ip_configs, rope_cfg):
    t = token_x.shape[0]
    q_lora_rank = ip_w_qb_in.shape[0]
    mla_q_norm_out = pypto.Tensor([t, q_lora_rank], pypto.DT_INT8)
    mla_q_norm_scale_out = pypto.Tensor([t, 1], pypto.DT_FP32)

    pypto.set_codegen_options(support_dynamic_unaligned=mla_tile_config.dynamic_unaligned_enable)
    ##################### mla #######################
    pypto.set_pass_options(vec_nbuffer_mode=mla_tile_config.vec_nbuffer_mode,
                           cube_l1_reuse_mode=mla_tile_config.cube_l1_reuse_mode,
                           cube_l1_reuse_setting=mla_tile_config.cube_l1_reuse_setting,
                           cube_nbuffer_setting=mla_tile_config.cube_nbuffer_setting,
                           mg_copyin_upper_bound=mla_tile_config.mg_copyin_upper_bound)

    mla_input_tensors = (token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
                         mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
                         mla_k_scale_cache)
    mla_output_tensors = (mla_q_norm_out, mla_q_norm_scale_out, mla_query_nope_out, mla_query_rope_out,
                          mla_kv_cache_out, mla_kr_cache_out, mla_k_scale_cache_out)
    mla.mla_prolog_quant_compute(*mla_input_tensors, *mla_output_tensors, mla_epsilon_cq, mla_epsilon_ckv,
                                   mla_cache_mode, mla_tile_config, rope_cfg)

    ##################### ip #######################
    pypto.set_pass_options(vec_nbuffer_mode=ip_configs.vec_nbuffer_mode)
    pypto.set_pass_options(cube_l1_reuse_setting=ip_configs.l1_reuse_param)
    pypto.set_pass_options(mg_copyin_upper_bound=ip_configs.mg_copyin_upper_bound)
    pypto.set_pass_options(pg_upper_bound=ip_configs.pg_upper_bound)

    ip_input_tensors = (token_x, mla_q_norm_out, mla_q_norm_scale_out, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in,
                        ip_w_proj_in, ip_ln_gamma_k_in, ip_ln_beta_k_in, cos, sin, ip_hadamard_q_in, ip_hadamard_k_in,
                        ip_k_cache, ip_k_cache_scale, cache_index)
    ip_output_tensors = (ip_q_int8_out, ip_q_scale_out, ip_k_int8_out, ip_k_scale_out, ip_weights_out)
    ip.lightning_indexer_prolog_quant_compute(*ip_input_tensors, *ip_output_tensors, ip_attrs, ip_configs)
