#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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

INT8 = pto.DT_INT8
INT32 = pto.DT_INT32
INT64 = pto.DT_INT64
FP32 = pto.DT_FP32
FP16 = pto.DT_FP16

NBUFFER_MERGE_MODE = "nbuffer_merge_mode"
L1_REUSE = "l1_reuse"
CUBE_NBUFFER_MAP = "cube_nbuffer_map"
COPYIN_THRESHOLD = "copyin_threshold"

SHAPE_DIM3 = 3
SHAPE_DIM4 = 4

NUM_NEG_2 = -2
NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_8 = 8
NUM_16 = 16
NUM_20 = 20
NUM_32 = 32
NUM_64 = 64
NUM_128 = 128
NUM_256 = 256
NUM_512 = 512
NUM_1024 = 1024

tensor = pto.tensor


@dataclass
class MlaQuantInputs:
    dequant_scale_x: tensor = tensor()
    dequant_scale_w_dq: tensor = tensor()
    dequant_scale_w_uq_qr: tensor = tensor()
    dequant_scale_w_dkv_kr: tensor = tensor()
    quant_scale_ckv: tensor = tensor()
    quant_scale_ckr: tensor = tensor()
    smooth_scales_cq: tensor = tensor()


@dataclass
class TestShapeParams:
    b: int = 0
    s: int = 0
    s2: int = 0
    n: int = 0
    h: int = 0
    q_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    kv_lora_rank: int = 0
    block_size: int = 0


class SimpleParams:
    b: int = 0
    s: int = 0
    s2: int = 0
    d: int = 0
    m: int = 0
    k: int = 0
    n: int = 0
    n2: int = 0
    right: int = 0
    h: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    q_head_dim: int = 0
    cache_mode: str = ""
    block_size: int = 0
    vec_tile: list = list()
    cube_m_tile: list = list()
    cube_k_tile: list = list()
    cube_n_tile: list = list()
    tile_b: int = 0
    
    @staticmethod
    def get_common_params():
        params: SimpleParams = SimpleParams()
        params.n2 = 1
        params.s = 1
        params.h = 7168 # 7168
        params.q_lora_rank = 1536 # 1536
        params.kv_lora_rank = 512 # 512
        params.qk_rope_head_dim = 64 # 64
        params.qk_nope_head_dim = 128 # 128
        params.q_head_dim = params.qk_rope_head_dim + params.qk_nope_head_dim
        params.cache_mode = "BNSD"
        params.block_size = 128 # 128
        return params

    @staticmethod
    def get_low_params():
        params: SimpleParams = SimpleParams.get_common_params()
        params.b = 4 # 4
        params.n = 32 # 32
        params.s2 = 256 # 256
        return params

    @staticmethod
    def get_high_params():
        params: SimpleParams = SimpleParams.get_common_params()
        params.b = 32 # 32
        params.n = 128 # 128
        params.s2 = 4096 # 4096
        return params


# tile config
@dataclass
class MlaTileConfig:
    tile_b: int = 8 # tile_b is 8
    tile_s: int = 1


def mla_pre(**kwargs):
    token_x: tensor = kwargs.get("token_x")
    w_dq: tensor = kwargs.get("w_dq")
    w_uq_qr: tensor = kwargs.get("w_uq_qr")
    w_dkv_kr: tensor = kwargs.get("w_dkv_kr")
    gamma_cq: tensor = kwargs.get("gamma_cq")
    epsilon_cq: float = kwargs.get("epsilon_cq")
    quant_inputs: MlaQuantInputs = kwargs.get("quant_inputs")
    split_k: bool = kwargs.get("split_k")
    is_smooth: bool = kwargs.get("is_smooth")
    # quant
    dequant_scale_w_uq_qr = quant_inputs.dequant_scale_w_uq_qr
    is_quant = (dequant_scale_w_uq_qr.has_storage() if dequant_scale_w_uq_qr is not None else False)
    smooth_scales_cq = quant_inputs.smooth_scales_cq

    b = token_x.shape[0]
    s = token_x.shape[1]
    h = token_x.shape[NUM_2]
    bs = b * s
    q_lora_rank = w_dq.shape[1]

    d_type = token_x.get_dtype()
    d_type_quant_out = pto.DT_INT32 if is_quant else d_type
    qkv_pre_res = []

    input_tensor = pto.reshape(token_x, [bs, h])  # [b,s,h] -> [b*s,h]

    ###### q ########
    c0 = 16  # 16
    m = (min(32, bs) + c0 - 1) // c0 * c0  # 32
    tie_m = min(32, m)  # 32
    pto.set_cube_tile_shapes([tie_m, tie_m], [256, 256], [64, 64])  # 256, 64
    
    q_mm_res = tensor()
    if split_k:
        def inside_if_split_k():
            nonlocal q_mm_res
            tmp_c = tensor(dtype=FP32, shape=[bs, q_lora_rank], name="tmp_q")
            pto.set_vec_tile_shapes(min(32, bs), 128)  # 32, 128
            tmp_c[:] = pto.mul_s(tmp_c, pto.element(pto.DT_FP32, 0.0))
            matmul_result = []
            k_split = 7
            k_split_size = h // k_split
            for ki in range(k_split):
                input_mk = pto.view(input_tensor, [bs, k_split_size], [0, ki * k_split_size])
                input_kn = pto.view(w_dq, [k_split_size, q_lora_rank], [ki * k_split_size, 0])
                tmp = pto.matmul(pto.DT_FP32, input_mk, input_kn)  # [b*s,h/2] * [h/2,q_lora_rank]
                matmul_result.append(tmp)
            q_mm_res_f32 = pto.reduce(matmul_result, pto.reduce_mode.ATOMIC_ADD)
            pto.set_vec_tile_shapes(min(32, bs), 128)  # 32, 128
            q_mm_res[:] = (pto.cast(q_mm_res_f32, d_type))
        inside_if_split_k()
    else:
        q_mm_res[:] = pto.matmul(d_type, input_tensor, w_dq)  # bf16

    pto.set_vec_tile_shapes(min(8, bs), q_lora_rank)  # 8
    norm_res = pto.rms_norm(q_mm_res, gamma_cq, epsilon_cq)

    norm_dequant_scale = tensor()
    norm_quant_res = None
    if is_quant:
        if is_smooth:
            norm_quant_res = pto.quant(norm_res, True, True, smooth_scales_cq)
        else:
            norm_quant_res = pto.quant(norm_res)  # int8
        norm_res[:] = norm_quant_res[0]
        norm_dequant_scale[:] = norm_quant_res[1]
        pto.set_cube_tile_shapes(
            [tie_m, tie_m], [256, 256], [256, 256])  # 256
    else:
        # use tileM will core dump
        pto.set_cube_tile_shapes([tie_m, tie_m], [256, 256], [64, 64])  # 256, 64
    
    q = pto.matmul(d_type_quant_out, norm_res, w_uq_qr)  # bf16  // quant: A8W8O32 -> bf16
    qkv_pre_res.append(q)

    ###### kv ########
    pto.set_cube_tile_shapes([m, m], [256, 256], [64, 64])  # 256, 64
    compressed_kv = tensor()
    if split_k:
        def inside_if_split():
            nonlocal compressed_kv
            pto.set_vec_tile_shapes(min(32, bs), 64)  # 32, 64
            kv_n = w_dkv_kr.shape[1]
            tmp_c_kv = tensor(dtype=FP32, shape=[bs, kv_n], name="tmpKv")
            tmp_c_kv[:] = pto.mul_s(tmp_c_kv, pto.element(pto.DT_FP32, 0.0))
            matmul_result_kv = []
            k_split_kv = 7
            k_split_size_kv = h // k_split_kv
            for ki in range(k_split_kv):
                input_mk = pto.view(input_tensor, [bs, k_split_size_kv], [0, ki * k_split_size_kv])
                input_kn = pto.view(w_dkv_kr, [k_split_size_kv, kv_n], [ki * k_split_size_kv, 0])
                tmp = pto.matmul(pto.DT_FP32, input_mk, input_kn)  # [b*s,h/2] * [h/2,kv_n] = [b*s,kv_n]
                matmul_result_kv.append(tmp)
            kv_mm_res_f32 = pto.reduce(matmul_result_kv, pto.reduce_mode.ATOMIC_ADD)
            pto.set_vec_tile_shapes(min(32, bs), 64)  # 32, 64
            compressed_kv[:] = pto.cast(kv_mm_res_f32, d_type)
        inside_if_split()
    else:
        compressed_kv[:] = pto.matmul(d_type, input_tensor, w_dkv_kr)  # bf16
    
    compressed_kv_res = pto.reshape(compressed_kv, [b, s, w_dkv_kr.shape[1]])
    qkv_pre_res.append(compressed_kv_res)

    if is_quant:
        qkv_pre_res.append(norm_dequant_scale)

    return qkv_pre_res


def mla_prolog(**kwargs):
    token_x: tensor = kwargs.get("token_x")
    w_dq: tensor = kwargs.get("w_dq")
    w_uq_qr: tensor = kwargs.get("w_uq_qr")
    w_uk: tensor = kwargs.get("w_uk")
    w_dkv_kr: tensor = kwargs.get("w_dkv_kr")
    gamma_cq: tensor = kwargs.get("gamma_cq")
    gamma_ckv: tensor = kwargs.get("gamma_ckv")
    sin: tensor = kwargs.get("sin")
    cos: tensor = kwargs.get("cos")
    cache_index: tensor = kwargs.get("cache_index")
    kv_cache: tensor = kwargs.get("kv_cache")
    kr_cache: tensor = kwargs.get("kr_cache")
    quant_inputs: MlaQuantInputs = kwargs.get("quant_inputs")
    rope_config: pto.rope_tile_shape_config_new = kwargs.get("rope_config")
    query_out: tensor = kwargs.get("query_out")
    query_rope_out: tensor = kwargs.get("query_rope_out")
    kv_cache_out: tensor = kwargs.get("kv_cache_out")
    kr_cache_out: tensor = kwargs.get("kr_cache_out")
    epsilon_cq: float = kwargs.get("epsilon_cq")
    epsilon_ckv: float = kwargs.get("epsilon_ckv")
    cache_mode: str = kwargs.get("cache_mode")
    split_k: bool = kwargs.get("split_k")
    is_smooth: bool = kwargs.get("is_smooth")
    # params check
    if len(token_x.shape) != SHAPE_DIM3 or len(w_uk.shape) != SHAPE_DIM3 or len(sin.shape) != SHAPE_DIM3:
        raise ValueError(f"Following tensors should be {SHAPE_DIM3}-dimensional:"
        f"\n\ttoken_x dim: {len(token_x.shape)}"
        f"\n\tw_uk dim: {len(w_uk.shape)}"
        f"\n\tsin dim: {len(sin.shape)}.")
    if cache_mode not in ["BNSD", "PA_BSND", "PA_NZ"]:
        raise ValueError(f"cache_mode buse be one of: 'BNSD', 'PA_BSND' or 'PA_NZ'. "
        f"Received: '{cache_mode}'")
    d_type: pto = token_x.get_dtype()
    b = token_x.shape[0]
    s = token_x.shape[1] # s=1
    h = token_x.shape[NUM_2]
    s2 = kv_cache.shape[NUM_2]
    # -> [n, qk_nope_head_dim, kv_lora_rank]
    n = w_uk.shape[0]
    qk_nope_head_dim = w_uk.shape[1]
    kv_lora_rank = w_uk.shape[NUM_2]
    qk_rope_head_dim = sin.shape[NUM_2] # [b,s,qk_rope_head_dim]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    tile_b = b
    tile_bs = tile_b * s
    b_loop = b // tile_b

    input_tensors = [
        token_x, w_dq, w_uq_qr, w_uk,
        w_dkv_kr, gamma_cq, gamma_ckv,
        sin, cos, cache_index, kv_cache,
        kr_cache, quant_inputs.dequant_scale_w_uq_qr,
        quant_inputs.smooth_scales_cq]
    output_tensors = [query_out, query_rope_out, kv_cache_out, kr_cache_out]

    with pto.function("main", input_tensors, output_tensors):
        def inside_main_function():
            for b_idx in pto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="bIdx"):
                def inside_b_idx_loop_prolog(b_idx, tile_b):
                    b_offset = b_idx * tile_b
                    output_offset = [b_offset, 0, 0, 0]

                    dequant_scale_w_uq_qr: tensor = quant_inputs.dequant_scale_w_uq_qr
                    is_quant = (dequant_scale_w_uq_qr.has_storage() if dequant_scale_w_uq_qr is not None else False)

                    x_view = pto.view(token_x, [tile_b, s, h], [b_offset, 0, 0])
                    q_kv = mla_pre(
                        token_x=x_view,
                        w_dq=w_dq,
                        w_uq_qr=w_uq_qr,
                        w_dkv_kr=w_dkv_kr,
                        gamma_cq=gamma_cq,
                        epsilon_cq=epsilon_cq,
                        quant_inputs=quant_inputs,
                        split_k=split_k,
                        is_smooth=is_smooth
                    )
                    q: tensor = q_kv[0] # [b*s, n*q_head_dim]
                    kv_tmp: tensor = q_kv[1] # [b,s,kv_lora_rank+qk_rope_head_dim]

                    # dequant32 -> fp32 -> *scale -> fp16/bf16
                    if is_quant:
                        def inside_quant():
                            tile_shape = [min(NUM_32, tile_bs), NUM_64]
                            pto.set_vec_tile_shapes(*tile_shape)
                            q_tmp_fp_32 = pto.cast(q, pto.DT_FP32)
                            q_tmp_dequant_scale = q_kv[NUM_2]
                            q_tmp_dequant_per_token = pto.mul(q_tmp_fp_32, q_tmp_dequant_scale)
                            q_tmp_dequant_channel = pto.mul(q_tmp_dequant_per_token, dequant_scale_w_uq_qr)

                            q[:] = pto.cast(q_tmp_dequant_channel, d_type)
                        inside_quant()

                    q_tmp = pto.reshape(q, [tile_b, s, n, q_head_dim])
                    tile_shape = [min(NUM_32, tile_b), 1, 1, NUM_64]
                    pto.set_vec_tile_shapes(*tile_shape)

                    ########## q ##########
                    # -> [b,s,n,qkNopeHeadDim]
                    q_nope: tensor = pto.view(q_tmp, [tile_b, s, n, qk_nope_head_dim], [0, 0, 0, 0])
                    tile_shape = [tile_b, 1, 1, NUM_128] # 128
                    pto.set_vec_tile_shapes(*tile_shape)
                    # -> [bs,n,qk_nope_head_dim]
                    q_nope_res: tensor = pto.reshape(q_nope, [tile_bs, n, qk_nope_head_dim])
                    tile_shape = [min(NUM_32, tile_bs), 1, qk_nope_head_dim] # [NUM_2, NUM_32, qkNopeHeadDim]
                    pto.set_vec_tile_shapes(*tile_shape)
                    q_nope_trans: tensor = pto.transpose(q_nope_res, [0, 1]) # [n,bs,qkNopeHeadDim]

                    c0 = NUM_16 # 16
                    m = (min(NUM_32, tile_bs) + c0 - 1) // c0 * c0
                    pto.set_cube_tile_shapes([m, m], [NUM_128, NUM_128], [NUM_128, NUM_128]) # 128
                    q_nope_new: tensor = pto.batch_matmul(d_type, q_nope_trans, w_uk)

                    tile_shape = [1, min(NUM_32, tile_bs), kv_lora_rank] # 32
                    pto.set_vec_tile_shapes(*tile_shape)
                    # -> [bs,n,kv_lora_rank]
                    q_nope_new_trans: tensor = pto.transpose(q_nope_new, [0, 1])
                    # -> [b,s,n,kv_lora_rank], output1
                    query_out_dview = pto.reshape(q_nope_new_trans, [tile_b, s, n, kv_lora_rank])

                    ########## kv ##########
                    # -> [b,s,kvLoraRank]
                    compressed_kv: tensor = pto.view(kv_tmp, [tile_b, s, kv_lora_rank], [0, 0, 0])
                    tile_shape = [NUM_2, 1, NUM_512] # 512
                    pto.set_vec_tile_shapes(*tile_shape)
                    # -> [b,s,kvLoraRank]
                    compressed_kv_norm: tensor = pto.rms_norm(compressed_kv, gamma_ckv, epsilon_ckv)
                    # -> [b,1,s,kvLoraRank]
                    k_nope: tensor = pto.reshape(compressed_kv_norm, [tile_b, 1, s, kv_lora_rank])

                    ########## RoPE ##########
                    # -> [b,s,qk_rope_head_dim]
                    k_pe_view: tensor = pto.view(kv_tmp, [tile_b, s, qk_rope_head_dim], [0, 0, kv_lora_rank])
                    tile_shape = [min(NUM_32, tile_b), 1, qk_rope_head_dim]
                    pto.set_vec_tile_shapes(*tile_shape)
                    # -> [b,s,1,qk_rope_head_dim]
                    k_pe_res: tensor = pto.reshape(k_pe_view, [tile_b, s, 1, qk_rope_head_dim])
                    q_pe_view: tensor = pto.view(
                        q_tmp, [tile_b, s, n, qk_rope_head_dim], [0, 0, 0, qk_nope_head_dim])
                    cos_view: tensor = pto.view(cos, [tile_b, s, qk_rope_head_dim], [b_offset, 0, 0])
                    sin_view: tensor = pto.view(sin, [tile_b, s, qk_rope_head_dim], [b_offset, 0, 0])
                    # -> [b,1,s,qk_rope_head_dim]
                    k_rope_view: tensor = tensor(
                        dtype=k_pe_res.get_dtype(), shape=[tile_b, s, 1, qk_rope_head_dim], name="kRopeView")
                    q_rope_view: tensor = tensor(
                        dtype=k_pe_res.get_dtype(), shape=[tile_b, s, n, qk_rope_head_dim], name="qRopeView")
                    pto.apply_rotary_pos_emb_v2(
                        q_pe_view, k_pe_res, cos_view, sin_view, q_rope_view, k_rope_view, NUM_2, rope_config) # 2
                    if cache_mode != "BNSD":
                        def inside_if_cache_mode():
                            block_num = kv_cache.shape[0]
                            block_size = kv_cache.shape[1]
                            n2 = kv_cache.shape[NUM_2]
                            kv_cache_res: tensor = pto.reshape(
                                kv_cache, [block_num * block_size * n2, kv_lora_rank])
                            kr_cache_res: tensor = pto.reshape(
                                kr_cache, [block_num * block_size * n2, qk_rope_head_dim])
                            cache_index_dview = pto.view(cache_index, [tile_b, s], [b_offset, 0])
                            k_nope = pto.reshape(k_nope, [tile_b * s, kv_lora_rank]) # [b*s,kv_lora_rank]
                            k_rope_res: tensor = pto.reshape(k_rope_view, [tile_b * s * 1, qk_rope_head_dim])
                            
                            ########## kvCache ##########
                            tile_shape = [1, kv_lora_rank]
                            pto.set_vec_tile_shapes(*tile_shape)
                            # kv_cache: [block_num*block_size*n2,kv_lora_rank], output3
                            kv_cache_out_dview = pto.scatter_update(
                                kv_cache_res, cache_index_dview, k_nope, NUM_NEG_2, cache_mode, block_size)

                            ########## krCache ##########
                            tile_shape = [1, qk_rope_head_dim]
                            pto.set_vec_tile_shapes(*tile_shape)
                            # krCache: [block_num*block_size*n2,qk_rope_head_dim], ouytput4
                            kr_cache_out_dview = pto.scatter_update(
                                kr_cache_res, cache_index_dview, k_rope_res, NUM_NEG_2, cache_mode, block_size)
                            
                            kv_cache_out = pto.reshape(
                                kv_cache_out_dview, [block_num, block_size, n2, kv_lora_rank])
                            kr_cache_out = pto.reshape(
                                kr_cache_out_dview, [block_num, block_size, n2, qk_rope_head_dim])
                        inside_if_cache_mode()
                    else:
                        def inside_else_cache_mode():
                            k_rope_res: tensor = pto.reshape(k_rope_view, [tile_b, 1, s, qk_rope_head_dim])
                            cache_index_dview = pto.view(cache_index, [tile_b, s], [b_offset, 0])
                            tile_shape = [1, 1, 1, kv_lora_rank]
                            pto.set_vec_tile_shapes(*tile_shape)
                            kv_cache_dview = pto.view(kv_cache, [tile_b, 1, s2, kv_lora_rank], [b_offset, 0, 0, 0])
                            kv_cache_out = pto.scatter_update(kv_cache_dview, cache_index_dview, k_nope, NUM_NEG_2)

                            tile_shpe = [1, 1, 1, qk_rope_head_dim]
                            pto.set_vec_tile_shapes(*tile_shape)
                            kr_cache_dview = pto.view(
                                kr_cache, [tile_b, 1, s2, qk_rope_head_dim], [b_offset, 0, 0, 0])
                            kr_cache_out = pto.scatter_update(
                                kr_cache_dview, cache_index_dview, k_rope_res, NUM_NEG_2)
                        inside_else_cache_mode()
                    pto.assemble(query_out_dview, output_offset, query_out)
                    pto.assemble(q_rope_view, output_offset, query_rope_out)
                inside_b_idx_loop_prolog(b_idx, tile_b)
        inside_main_function()


def de_quant(d_type: pto, data: tensor, scale: tensor, w_scale: tensor):
    dequant_res: tensor = pto.cast(data, FP32)
    dequant_res[:] = pto.mul(dequant_res, scale)
    dequant_res[:] = pto.mul(dequant_res, w_scale)
    return pto.cast(dequant_res, d_type)


def pre_compute(**kwargs):
    token_x: tensor = kwargs.get("token_x")
    w_dq: tensor = kwargs.get("w_dq")
    w_uq_qr: tensor = kwargs.get("w_uq_qr")
    w_dkv_kr: tensor = kwargs.get("w_dkv_kr")
    gamma_cq: tensor = kwargs.get("gamma_cq")
    epsilon_cq: float = kwargs.get("epsilon_cq")
    quant_inputs: MlaQuantInputs = kwargs.get("quant_inputs")

    # quant
    dequant_scale_w_dq: tensor = quant_inputs.dequant_scale_w_dq
    dequant_scale_w_dkv_kr: tensor = quant_inputs.dequant_scale_w_dkv_kr
    dequant_scale_w_uq_qr: tensor = quant_inputs.dequant_scale_w_uq_qr
    is_quant_a = ((dequant_scale_w_dq.has_storage() if dequant_scale_w_dq is not None else False) and
        (dequant_scale_w_dkv_kr.has_storage() if dequant_scale_w_dkv_kr is not None else False))
    is_quant_b = (dequant_scale_w_uq_qr.has_storage() if dequant_scale_w_uq_qr is not None else False)
    smooth_scales_cq: tensor = quant_inputs.smooth_scales_cq
    is_smooth = (smooth_scales_cq.has_storage() if smooth_scales_cq is not None else False)

    b = token_x.shape[0]
    s = token_x.shape[1]
    h = token_x.shape[NUM_2]
    bs = b * s
    q_lora_rank = w_dq.shape[1]

    d_type: pto = token_x.get_dtype()
    d_type_quant_a_out: pto = INT32 if is_quant_a else d_type
    d_type_quant_b_out: pto = INT32 if is_quant_b else d_type
    qkv_pre_res = list()

    pto.set_semantic_label("pre_reshape")
    data: tensor = pto.reshape(token_x, [bs, h]) # [b,s,h] -> [b*s,h]
    input_quant: tensor = tensor()
    input_quant_scale: tensor = tensor()

    ###### q ########
    c0 = NUM_16 # 16
    m = (min(NUM_32, bs) + c0 - 1) // c0 * c0 # 32
    mv = min(NUM_8, bs)
    q_a_proj: tensor = tensor()
    if is_quant_a:
        def inside_if_is_quant_a():
            nonlocal q_a_proj
            pto.set_vec_tile_shapes(mv, q_lora_rank)
            pto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_256, NUM_256])
            # no smooth
            pto.set_semantic_label("Quant_x")
            quant_res: List[tensor] = pto.quant(data)
            input_quant[:] = quant_res[0]
            input_quant_scale[:] = quant_res[1]
            pto.set_semantic_label("QuantMatmul_qa")
            q_a_proj = pto.matmul(d_type_quant_a_out, input_quant, w_dq)
            pto.set_semantic_label("Dequant_qa")
            q_a_proj = de_quant(d_type, q_a_proj, input_quant_scale, dequant_scale_w_dq)
        inside_if_is_quant_a()
    else:
        def inside_else_is_quant_a():
            nonlocal q_a_proj
            pto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_64, NUM_64])
            pto.set_semantic_label("Matmul_qa")
            q_a_proj = pto.matmul(d_type, data, w_dq)
        inside_else_is_quant_a()
    
    # rmsnorm
    pto.set_vec_tile_shapes(mv, q_lora_rank)
    pto.set_semantic_label("RmsNorm_qa")
    norm_res: tensor = pto.rms_norm(q_a_proj, gamma_cq, epsilon_cq)

    q_b_proj: tensor = tensor()
    if is_quant_b:
        def inside_if_is_quant_b():
            nonlocal q_b_proj
            norm_quant: tensor = tensor()
            norm_quant_scale: tensor = tensor()
            pto.set_vec_tile_shapes(mv, q_lora_rank)
            pto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_256, NUM_256])
            pto.set_semantic_label("Quant_qMmRes")
            quant_res = None
            if is_smooth:
                quant_res = pto.quant(norm_res, True, True, smooth_scales_cq)
            else:
                quant_res = pto.quant(norm_res, True, False)
            norm_quant[:] = quant_res[0]
            norm_quant_scale[:] = quant_res[1]
            pto.set_semantic_label("QuantMatmul_qb")
            q_b_proj[:] = pto.matmul(d_type_quant_b_out, norm_quant, w_uq_qr)
            pto.set_semantic_label("Dequant_qb")
            q_b_proj[:] = de_quant(d_type, q_b_proj, norm_quant_scale, dequant_scale_w_uq_qr)
        inside_if_is_quant_b()
    else:
        def inside_else_is_quant_b():
            nonlocal q_b_proj
            pto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_64, NUM_64])
            pto.set_semantic_label("Matmul_qb")
            q_b_proj[:] = pto.matmul(d_type, norm_res, w_uq_qr)
        inside_else_is_quant_b()
    qkv_pre_res.append(q_b_proj)
    
    ###### kv ######
    compressed_kv: tensor = None
    if is_quant_a:
        def inside_if_is_quant_a():
            nonlocal compressed_kv
            pto.set_vec_tile_shapes(mv, q_lora_rank)
            pto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_256, NUM_256])
            # no smooth
            pto.set_semantic_label("QuantMatmul_kva")
            compressed_kv = pto.matmul(d_type_quant_a_out, input_quant, w_dkv_kr)
            pto.set_semantic_label("Dequant_kva")
            compressed_kv = de_quant(d_type, compressed_kv, input_quant_scale, dequant_scale_w_dkv_kr)
        inside_if_is_quant_a()
    else:
        def inside_else_is_quant_a():
            nonlocal compressed_kv
            pto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_64, NUM_64])
            pto.set_semantic_label("Matmul_kva")
            compressed_kv = pto.matmul(d_type, data, w_dkv_kr)
        inside_else_is_quant_a()
    qkv_pre_res.append(compressed_kv)

    return qkv_pre_res


# NSA MlaProlog, b and s is dynamic, support:
# b: 16, 32, 64, 24, 48, 96
# s: 1, 2
def mla_prolog_compute(**kwargs):
    token_x: tensor = kwargs.get("token_x")
    w_dq: tensor = kwargs.get("w_dq")
    w_uq_qr: tensor = kwargs.get("w_uq_qr")
    w_uk: tensor = kwargs.get("w_uk")
    w_dkv_kr: tensor = kwargs.get("w_dkv_kr")
    gamma_cq: tensor = kwargs.get("gamma_cq")
    gamma_ckv: tensor = kwargs.get("gamma_ckv")
    sin: tensor = kwargs.get("sin")
    cos: tensor = kwargs.get("cos")
    cache_index: tensor = kwargs.get("cache_index")
    kv_cache: tensor = kwargs.get("kv_cache")
    kr_cache: tensor = kwargs.get("kr_cache")
    quant_inputs: MlaQuantInputs = kwargs.get("quant_inputs")
    tile_config: MlaTileConfig = kwargs.get("tile_config")
    query_out: tensor = kwargs.get("query_out")
    query_rope_out: tensor = kwargs.get("query_rope_out")
    kv_cache_out: tensor = kwargs.get("kv_cache_out")
    kr_cache_out: tensor = kwargs.get("kr_cache_out")
    epsilon_cq: float = kwargs.get("epsilon_cq")
    epsilon_ckv: float = kwargs.get("epsilon_ckv")
    cache_mode: str = kwargs.get("cache_mode")
    # params check
    if len(token_x.shape) != SHAPE_DIM3 or len(w_uk.shape) != SHAPE_DIM3 or len(sin.shape) != SHAPE_DIM3:
        raise ValueError(f"Following tensors should be {SHAPE_DIM3}-dimensional:"
        f"\n\ttoken_x dim: {len(token_x.shape)}"
        f"\n\tw_uk dim: {len(w_uk.shape)}"
        f"\n\tsin dim: {len(sin.shape)}.")
    if len(kv_cache.shape) != SHAPE_DIM4 or len(kr_cache.shape) != SHAPE_DIM4:
        raise ValueError(f"Following tensors should be {SHAPE_DIM4}-dimensional:"
        f"\n\tkv_cache dim: {len(kv_cache.shape)}"
        f"\n\tkr_cache dim: {len(kr_cache.shape)}.")
    if cache_mode not in ["PA_BSND", "PA_NZ"]:
        raise ValueError(f"cache_mode buse be one of: 'PA_BSND' or 'PA_NZ'. "
        f"Received: '{cache_mode}'")

    d_type: pto = token_x.get_dtype()
    h = token_x.shape[NUM_2]
    # -> [n, qk_nope_head_dim, kv_lora_rank]
    n = w_uk.shape[0]
    qk_nope_head_dim = w_uk.shape[1]
    kv_lora_rank = w_uk.shape[NUM_2]
    qk_rope_head_dim = sin.shape[NUM_2] # [b, s, qk_rope_head_dim], 2
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    block_num = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    n2 = kv_cache.shape[NUM_2]
    if qk_nope_head_dim != NUM_128 and qk_rope_head_dim != NUM_64:
        raise ValueError("qk_nope_head_dim must be 128 or qk_rope_head_dim must be 64"
        " (support 128, 64)")
    
    tile_b = tile_config.tile_b
    tile_s = tile_config.tile_s
    tile_bs = tile_b * tile_s

    rope_config: pto.rope_tile_shape_config_new = pto.rope_tile_shape_config_new()
    rope_config.three_dims_tile_shape = [tile_b, tile_s, qk_rope_head_dim]
    rope_config.four_dims_tile_shape_q = [tile_b, tile_s, 1, qk_rope_head_dim]
    rope_config.four_dims_tile_shape_k = [tile_b, tile_s, 1, qk_rope_head_dim]
    rope_config.five_dims_tile_shape = [tile_b, tile_s, 1, qk_rope_head_dim // 2, 2]

    b = pto.get_input_shape(token_x, 0)
    s = pto.get_input_shape(token_x, 1)

    b_loop = b // tile_b
    s_loop = s // tile_s

    for b_idx in pto.loop(0, b_loop, 1, name="MLA_LOOP_L0_bIdx", idx_name="bIdx"):
        def inside_b_idx_loop(b_idx):
            b_offset = b_idx * tile_b
            for s_idx in pto.loop(0, s_loop, 1, name="MLA_LOOP_L1_sIdx", idx_name="sIdx"):
                def inside_s_idx_loop(s_idx):
                    s_offset = s_idx * tile_s
                    output_offset = [b_offset, s_offset, 0, 0]
                    pto.set_vec_tile_shapes(tile_b, tile_s, NUM_128)
                    x_view = pto.view(token_x, [tile_b, tile_s, h], [b_offset, s_offset, 0])
                    q_kv = pre_compute(
                        token_x=x_view,
                        w_dq=w_dq,
                        w_uq_qr=w_uq_qr,
                        w_dkv_kr=w_dkv_kr,
                        gamma_cq=gamma_cq,
                        epsilon_cq=epsilon_cq,
                        quant_inputs=quant_inputs
                    )
                    q: tensor = q_kv[0] # [b * s, n * q_head_dim]
                    kv_tmp: tensor = q_kv[1] # [b * s, kv_lora_rank + qk_rope_head_dim]
                    q_tmp = pto.reshape(q, [tile_b, tile_s, n, q_head_dim])

                    ##### q #####
                    pto.set_semantic_label("Prepare_qNope")
                    q_nope: tensor = pto.view(q_tmp, [tile_b, tile_s, n, qk_nope_head_dim], [0, 0, 0, 0])
                    tile_shape = [tile_b, tile_s, 1, NUM_128]
                    pto.set_vec_tile_shapes(*tile_shape)
                    q_nope_res: tensor = pto.reshape(q_nope, [tile_bs, n, qk_nope_head_dim])
                    tile_shape = [min(NUM_32, tile_bs), 1, qk_nope_head_dim]
                    pto.set_vec_tile_shapes(*tile_shape)
                    q_nope_trans: tensor = pto.transpose(q_nope_res, [0, 1])

                    c0 = NUM_16
                    m = (min(NUM_32, tile_bs) + c0 - 1) // c0 * c0
                    pto.set_semantic_label("Matmul_qNope_wUk")
                    pto.set_cube_tile_shapes([m, m], [NUM_128, NUM_128], [NUM_128, NUM_128])
                    q_nope_new = pto.batch_matmul(d_type, q_nope_trans, w_uk)

                    pto.set_semantic_label("queryOut")
                    tile_shape = [1, min(NUM_32, tile_bs), kv_lora_rank]
                    pto.set_vec_tile_shapes(*tile_shape)
                    q_nope_new_trans: tensor = pto.transpose(q_nope_new, [0, 1]) # [bs, n, kv_lora_rank]
                    query_out_view = pto.reshape(q_nope_new_trans, [tile_b, tile_s, n, kv_lora_rank])

                    ##### kv #####
                    compressed_kv: tensor = pto.view(kv_tmp, [tile_bs, kv_lora_rank], [0, 0])
                    tile_shape = [NUM_2, NUM_512]
                    pto.set_semantic_label("RmsNorm_compressedKv")
                    pto.set_vec_tile_shapes(*tile_shape)
                    # -> [b * s, kv_lora_rank]
                    k_nope: tensor = pto.rms_norm(compressed_kv, gamma_ckv, epsilon_ckv)

                    ##### rope #####
                    pto.set_semantic_label("RotaryPosEmb")
                    k_pe_view: tensor = pto.view(kv_tmp, [tile_bs, qk_rope_head_dim], [0, kv_lora_rank])
                    k_pe_res: tensor = pto.reshape(k_pe_view, [tile_b, tile_s, 1, qk_rope_head_dim])
                    q_pe_view: tensor = pto.view(
                        q_tmp, [tile_b, tile_s, n, qk_rope_head_dim], [0, 0, 0, qk_nope_head_dim])
                    cos_view: tensor = pto.view(cos, [tile_b, tile_s, qk_rope_head_dim], [b_offset, s_offset, 0])
                    sin_view: tensor = pto.view(sin, [tile_b, tile_s, qk_rope_head_dim], [b_offset, s_offset, 0])
                    q_rope_view: tensor = tensor(dtype=k_pe_res.get_dtype(),
                        shape=[tile_b, tile_s, n, qk_rope_head_dim], name="qRopeView")
                    k_rope_view: tensor = tensor(dtype=k_pe_res.get_dtype(),
                        shape=[tile_b, tile_s, 1, qk_rope_head_dim], name="kRopeView")
                    # 2 is unsqueeze dim
                    pto.apply_rotary_pos_emb_v2(
                        q_pe_view, k_pe_res, cos_view, sin_view, q_rope_view, k_rope_view, NUM_2, rope_config)
                    
                    # PA_BSND, PA_NZ
                    kv_cache_res: tensor = pto.reshape(kv_cache, [block_num * block_size * n2, kv_lora_rank])
                    kr_cache_res: tensor = pto.reshape(kr_cache, [block_num * block_size * n2, qk_rope_head_dim])
                    k_rope_res: tensor = pto.reshape(k_rope_view, [tile_bs * 1, qk_rope_head_dim])
                    index_view: tensor = pto.view(cache_index, [tile_b, tile_s], [b_offset, s_offset])

                    ##### kv_cache #####
                    pto.set_semantic_label("ScatterUpdate_kvCache")
                    tile_shape = [1, kv_lora_rank]
                    pto.set_vec_tile_shapes(*tile_shape)
                    # kv_cache: [block_num * block_size * n2, kv_lora_rank] , output3
                    kv_cache_out_view: tensor = pto.scatter_update(
                        kv_cache_res, index_view, k_nope, NUM_NEG_2, cache_mode, block_size)

                    ##### kr_cache #####
                    pto.set_semantic_label("ScatterUpdate_krCache")
                    tile_shape = [1, qk_rope_head_dim]
                    pto.set_vec_tile_shapes(*tile_shape)
                    # kr_cache: [block_num * block_size * n2, qk_rope_head_dim], output4
                    kr_cache_out_view: tensor = pto.scatter_update(
                        kr_cache_res, index_view, k_rope_res, NUM_NEG_2, cache_mode, block_size)

                    # 输入和输出相同shape时，即 n2 = 1, tensor graph上无法看到该reshape
                    kv_cache_out[:] = pto.reshape(kv_cache_out_view, [block_num * block_size, n2 * kv_lora_rank])
                    kr_cache_out[:] = pto.reshape(kr_cache_out_view, [block_num * block_size, n2 * qk_rope_head_dim])

                    pto.set_semantic_label("Assemble_queryOut")
                    pto.set_vec_tile_shapes(1, 1, NUM_32, NUM_128)
                    pto.assemble(query_out_view, output_offset, query_out) # output1
                    pto.set_semantic_label("Assemble_qRope")
                    pto.set_vec_tile_shapes(1, 1, NUM_32, NUM_64)
                    pto.assemble(q_rope_view, output_offset, query_rope_out) # output2
                    pto.set_semantic_label("")
                inside_s_idx_loop(s_idx)
        inside_b_idx_loop(b_idx)


def mla_prolog_main(**kwargs):
    token_x: tensor = kwargs.get("token_x")
    w_dq: tensor = kwargs.get("w_dq")
    w_uq_qr: tensor = kwargs.get("w_uq_qr")
    w_uk: tensor = kwargs.get("w_uk")
    w_dkv_kr: tensor = kwargs.get("w_dkv_kr")
    gamma_cq: tensor = kwargs.get("gamma_cq")
    gamma_ckv: tensor = kwargs.get("gamma_ckv")
    sin: tensor = kwargs.get("sin")
    cos: tensor = kwargs.get("cos")
    cache_index: tensor = kwargs.get("cache_index")
    kv_cache: tensor = kwargs.get("kv_cache")
    kr_cache: tensor = kwargs.get("kr_cache")
    quant_inputs: MlaQuantInputs = kwargs.get("quant_inputs")
    tile_config: MlaTileConfig = kwargs.get("tile_config")
    query_out: tensor = kwargs.get("query_out")
    query_rope_out: tensor = kwargs.get("query_rope_out")
    kv_cache_out: tensor = kwargs.get("kv_cache_out")
    kr_cache_out: tensor = kwargs.get("kr_cache_out")
    epsilon_cq: float = kwargs.get("epsilon_cq")
    epsilon_ckv: float = kwargs.get("epsilon_ckv")
    cache_mode: str = kwargs.get("cache_mode")
    input_tensors = [
        token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr,
        gamma_cq, gamma_ckv, sin, cos, cache_index,
        kv_cache, kr_cache,
        quant_inputs.dequant_scale_w_dq,
        quant_inputs.dequant_scale_w_dkv_kr,
        quant_inputs.dequant_scale_w_uq_qr,
        quant_inputs.smooth_scales_cq
        ]
    output_tensors = [query_out, query_rope_out, kv_cache_out, kr_cache_out]
    with pto.function("main", input_tensors, output_tensors):
        mla_prolog_compute(
            token_x=token_x,
            w_dq=w_dq,
            w_uq_qr=w_uq_qr,
            w_uk=w_uk,
            w_dkv_kr=w_dkv_kr,
            gamma_cq=gamma_cq,
            gamma_ckv=gamma_ckv,
            sin=sin,
            cos=cos,
            cache_index=cache_index,
            kv_cache=kv_cache,
            kr_cache=kr_cache,
            quant_inputs=quant_inputs,
            tile_config=tile_config,
            query_out=query_out,
            query_rope_out=query_rope_out,
            kv_cache_out=kv_cache_out,
            kr_cache_out=kr_cache_out,
            epsilon_cq=epsilon_cq,
            epsilon_ckv=epsilon_ckv,
            cache_mode=cache_mode
        )


def test_dynamic_mla_prolog(t_params: list, shape_params: SimpleParams, tile_config: MlaTileConfig, cache_mode: str):

    b = shape_params.b
    s = shape_params.s
    s2 = shape_params.s2
    n = shape_params.n
    n2 = 1
    h = shape_params.h
    q_lora_rank = shape_params.q_lora_rank
    qk_nope_head_dim = shape_params.qk_nope_head_dim
    qk_rope_head_dim = shape_params.qk_rope_head_dim
    kv_lora_rank = shape_params.kv_lora_rank
    block_size = shape_params.block_size
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    is_quant_a = t_params[0]
    is_quant_b = t_params[1]
    is_smooth = t_params[2]
    nz = t_params[3]
    use_prefetch = t_params[4]

    d_type = FP16
    d_type_quant_a = INT8 if is_quant_a else d_type
    d_type_quant_b = INT8 if is_quant_b else d_type

    x_shape = [b, s, h]
    w_dq_shape = [h, q_lora_rank]
    w_uq_qr_shape = [q_lora_rank, n * q_head_dim]
    w_dkv_kr_shape = [h, kv_lora_rank + qk_rope_head_dim]
    w_uk_shape = [n, qk_nope_head_dim, kv_lora_rank]
    cos_shape = [b, s, qk_rope_head_dim]
    gamma_cq_shape = [q_lora_rank]
    gamma_ckv_shape = [kv_lora_rank]
    kv_len_shape = [b, s]
    block_num = b * (s2 // block_size)
    kv_cache_shape = [block_num, block_size, n2, kv_lora_rank]
    kr_cache_shape = [block_num, block_size, n2, qk_rope_head_dim]
    kv_cache_out_shape = [block_num * block_size, n2 * kv_lora_rank]
    kr_cache_out_shape = [block_num, block_size, n2 * qk_rope_head_dim]
    scale_w_dq_shape = [1, q_lora_rank]
    scale_w_uq_qr_shape = [1, n * q_head_dim]
    scale_w_dkv_kr_shape = [1, kv_lora_rank + qk_rope_head_dim]
    smooth_cq_shape = [1, q_lora_rank]

    q_out_shape = [b, s, n, kv_lora_rank]
    q_rope_out_shape = [b, s, n, qk_rope_head_dim]

    weight_format = pto.tile_op_format.TILEOP_NZ if nz else pto.tile_op_format.TILEOP_ND
    w_dq: tensor = tensor(dtype=d_type_quant_a, shape=w_dq_shape, name="wDq", format=weight_format)
    w_uq_qr: tensor = tensor(dtype=d_type_quant_b, shape=w_uq_qr_shape, name="wUqQr", format=weight_format)
    if use_prefetch:
        w_dq.set_cache_policy(pto.cache_policy.PREFETCH, True)
        w_uq_qr.set_cache_policy(pto.cache_policy.PREFETCH, True)
    w_dkv_kr: tensor = tensor(dtype=d_type_quant_a, shape=w_dkv_kr_shape, name="wDkvKr", format=weight_format)
    w_uk: tensor = tensor(dtype=d_type, shape=w_uk_shape, name="wUk", format=weight_format)

    gamma_cq: tensor = tensor(dtype=d_type, shape=gamma_cq_shape, name="gammaCq")
    gamma_ckv: tensor = tensor(dtype=d_type, shape=gamma_ckv_shape, name="gammaCkv")

    kv_cache: tensor = tensor(dtype=d_type, shape=kv_cache_shape, name="kvCache")
    kr_cache: tensor = tensor(dtype=d_type, shape=kr_cache_shape, name="krCache")

    scale_w_dq: tensor = tensor(dtype=FP32, shape=scale_w_dq_shape, name="scaleWDq")
    scale_w_uq_qr: tensor = tensor(dtype=FP32, shape=scale_w_uq_qr_shape, name="scaleWUqQr")
    scale_w_dkv_kr: tensor = tensor(dtype=FP32, shape=scale_w_dkv_kr_shape, name="scaleWDkvKr")

    smooth_cq: tensor = tensor(dtype=FP32, shape=smooth_cq_shape, name="smoothCq")
    # output
    output_kv_cache: tensor = tensor(dtype=d_type, shape=kv_cache_out_shape, name="outputKvCache")
    output_kr_cache: tensor = tensor(dtype=d_type, shape=kr_cache_out_shape, name="outputKrCache")

    # dynamic shape
    dynamic_x: tensor = tensor(dtype=d_type, shape=[-1, -1, h], name="dynamicX")
    dynamic_cos: tensor = tensor(dtype=d_type, shape=[-1, -1, qk_rope_head_dim], name="dynamicCos")
    dynamic_sin: tensor = tensor(dtype=d_type, shape=[-1, -1, qk_rope_head_dim], name="dynamicSin")
    dynamic_cache_index: tensor = tensor(dtype=INT64, shape=[-1, -1], name="dynamicCacheIndex")
    dynamic_output_q: tensor = tensor(
        dtype=d_type, shape=[-1, pto.get_input_shape(dynamic_x, 1), n, kv_lora_rank], name="dynamicOutputQ")
    dynamic_output_q_rope: tensor = tensor(
        dtype=d_type, shape=[-1, pto.get_input_shape(dynamic_x, 1), n, qk_rope_head_dim], name="dynamicOutputQRope")

    quant_inputs: MlaQuantInputs = MlaQuantInputs()
    if is_quant_a:
        quant_inputs.dequant_scale_w_dq = scale_w_dq
        quant_inputs.dequant_scale_w_dkv_kr = scale_w_dkv_kr
    if is_quant_b:
        quant_inputs.dequant_scale_w_uq_qr = scale_w_uq_qr
        if is_smooth:
            quant_inputs.smooth_scales_cq = smooth_cq
    
    mla_prolog_main(
        token_x=dynamic_x,
        w_dq=w_dq,
        w_uq_qr=w_uq_qr,
        w_uk=w_uk,
        w_dkv_kr=w_dkv_kr,
        gamma_cq=gamma_cq,
        gamma_ckv=gamma_ckv,
        sin=dynamic_sin,
        cos=dynamic_cos,
        cache_index=dynamic_cache_index,
        kv_cache=kv_cache,
        kr_cache=kr_cache,
        quant_inputs=quant_inputs,
        tile_config=tile_config,
        query_out=dynamic_output_q,
        query_rope_out=dynamic_output_q_rope,
        kv_cache_out=output_kv_cache,
        kr_cache_out=output_kr_cache,
        epsilon_cq=1e-5,
        epsilon_ckv=1e-5,
        cache_mode=cache_mode,
    )


def main():
    t_params = [False, True, True, False, True]
    shape_params: TestShapeParams = TestShapeParams(1, 2, 128, 6, 512, 64, 64, 64, 32, 32)
    cache_mode = "PA_BSND"
    tile_config: MlaTileConfig = MlaTileConfig(32, 1)

    pto.set_config(NBUFFER_MERGE_MODE, 1)
    pto.set_config(L1_REUSE, NUM_4)
    pto.set_config(CUBE_NBUFFER_MAP, {NUM_3: NUM_4})
    pto.set_config(COPYIN_THRESHOLD, NUM_2 * NUM_1024 * NUM_1024)

    test_dynamic_mla_prolog(t_params, shape_params, tile_config, cache_mode)

    graph_dump = pto.dump()
    with open("dump_dynamic_mla_py.txt", "w", encoding="utf-8") as f:
        f.write(graph_dump)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    main()
