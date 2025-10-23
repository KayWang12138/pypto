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
SHAPE_DIM3 = 3
SHAPE_DIM4 = 4


@dataclass
class MlaQuantInputs:
    dequant_scale_x: pto.tensor = pto.tensor()
    dequant_scale_w_dq: pto.tensor = pto.tensor()
    dequant_scale_w_uq_qr: pto.tensor = pto.tensor()
    dequant_scale_w_dkv_kr: pto.tensor = pto.tensor()
    quant_scale_ckv: pto.tensor = pto.tensor()
    quant_scale_ckr: pto.tensor = pto.tensor()
    smooth_scales_cq: pto.tensor = pto.tensor()


def qkv_pre(**kwargs) -> List[pto.tensor]:
    token_x = kwargs.get("token_x")
    w_dq = kwargs.get("w_dq")
    w_uq_qr = kwargs.get("w_uq_qr")
    w_dkv_kr = kwargs.get("w_dkv_kr")
    gamma_cq = kwargs.get("gamma_cq")
    epsilon_cq = kwargs.get("epsilon_cq")
    quant_inputs = kwargs.get("quant_inputs")
    split_reduce_last_dim = kwargs.get("split_reduce_last_dim")
    split_k = kwargs.get("split_k")

    # quant
    dequant_scale_w_uq_qr = quant_inputs.dequant_scale_w_uq_qr
    is_quant = (dequant_scale_w_uq_qr.has_storage() if dequant_scale_w_uq_qr is not None else False)
    smooth_scales_cq = quant_inputs.smooth_scales_cq
    has_smooth = (dequant_scale_w_uq_qr.has_storage() if smooth_scales_cq is not None else False)
    b = token_x.shape[0]
    s = token_x.shape[1]
    h = token_x.shape[2]
    bs = b * s
    q_lora_rank = w_dq.shape[1]

    d_type = token_x.dtype
    d_type_quant_out = pto.DT_INT32 if is_quant else d_type
    qkv_pre_res = []

    input_tensor = pto.reshape(token_x, [bs, h])  # [b,s,h] -> [b*s,h]

    ###### q ########
    c0 = NUM_16  # 16
    tie_m = (bs + c0 - 1) // c0 * c0  # 32
    pto.set_cube_tile_shapes([tie_m, tie_m], [NUM_256, NUM_256], [NUM_64, NUM_64])  # 256, 64

    q_mm_res = pto.tensor()
    if split_k:
        tmp_c = pto.tensor([bs, q_lora_rank], pto.DT_FP32, "tmp_q")
        pto.set_vec_tile_shapes(min(NUM_32, bs), NUM_128)  # 32, 128
        tmp_c[:] = pto.mul_s(tmp_c, pto.element(pto.DT_FP32, 0.0))
        matmul_result = []
        k_split = 7
        k_split_size = h // k_split
        for ki in range(k_split):
            input_mk = pto.view(input_tensor, [bs, k_split_size], [0, ki * k_split_size])
            input_kn = pto.view(w_dq, [k_split_size, q_lora_rank], [ki * k_split_size, 0])
            tmp = pto.matmul(pto.DT_FP32, input_mk, input_kn, tmp_c)  # [b*s,h/2] * [h/2,q_lora_rank]
            matmul_result.append(tmp)
        q_mm_res_f32 = pto.reduce(matmul_result, pto.ReduceMode.ATOMIC_ADD)
        pto.set_vec_tile_shapes(min(NUM_32, bs), NUM_128)  # 32, 128
        q_mm_res[:] = pto.cast(q_mm_res_f32, d_type)
    else:
        q_mm_res[:] = pto.matmul(d_type, input_tensor, w_dq)  # bf16

    if split_reduce_last_dim:
        pto.set_vec_tile_shapes(min(NUM_16, bs), NUM_128)
    else:
        pto.set_vec_tile_shapes(min(NUM_8, bs), q_lora_rank)

    norm_res = pto.rms_norm(q_mm_res, gamma_cq, epsilon_cq)
    norm_dequant_scale = pto.tensor()
    norm_quant_res = None
    if is_quant:
        if has_smooth:
            norm_quant_res = pto.quant(norm_res, True, True, smooth_scales_cq)
        else:
            norm_quant_res = pto.quant(norm_res)
        norm_res[:] = norm_quant_res[0]
        norm_dequant_scale[:] = norm_quant_res[1]
        pto.set_cube_tile_shapes(
            [tie_m, tie_m], [NUM_256, NUM_256], [NUM_256, NUM_256])  # 256
    else:
        pto.set_cube_tile_shapes([tie_m, tie_m], [NUM_256, NUM_256], [NUM_64, NUM_64])  # 256, 64

    q = pto.matmul(d_type_quant_out, norm_res, w_uq_qr, False, False)  # bf16  // quant: A8W8O32 -> bf16
    qkv_pre_res.append(q)

    ####### kv ########
    pto.set_cube_tile_shapes([tie_m, tie_m], [NUM_256, NUM_256], [NUM_64, NUM_64])  # 256, 64
    compressed_kv = pto.tensor()
    if split_k:
        pto.set_vec_tile_shapes(min(NUM_32, bs), NUM_64)  # 32, 64
        kv_n = w_dkv_kr.shape[1]
        tmp_c_kv = pto.tensor([bs, kv_n], pto.DT_FP32, "tmp_kv")
        tmp_c_kv[:] = pto.mul_s(tmp_c_kv, pto.element(pto.DT_FP32, 0.0))
        matmul_result_kv = []
        k_split_kv = 7
        k_split_size_kv = h // k_split_kv
        for ki in range(k_split_kv):
            input_mk = pto.view(input_tensor, [bs, k_split_size_kv], [0, ki * k_split_size_kv])
            input_kn = pto.view(w_dkv_kr, [k_split_size_kv, kv_n], [ki * k_split_size_kv, 0])
            tmp = pto.matmul(pto.DT_FP32, input_mk, input_kn, tmp_c_kv)  # [b*s,h/2] * [h/2,kv_n] = [b*s,kv_n]
            matmul_result_kv.append(tmp)
        kv_mm_res_f32 = pto.reduce(matmul_result_kv, pto.ReduceMode.ATOMIC_ADD)
        pto.set_vec_tile_shapes(min(NUM_32, bs), NUM_64)  # 32, 64
        compressed_kv[:] = pto.cast(kv_mm_res_f32, d_type)
    else:
        compressed_kv[:] = pto.matmul(d_type, input_tensor, w_dkv_kr)  # bf16

    compressed_kv_res = pto.reshape(compressed_kv, [b, s, w_dkv_kr.shape[1]])
    qkv_pre_res.append(compressed_kv_res)

    if is_quant:
        qkv_pre_res.append(norm_dequant_scale)
    return qkv_pre_res


def mla_prolog(**kwargs):
    token_x = kwargs.get("token_x")
    w_dq = kwargs.get("w_dq")
    w_uq_qr = kwargs.get("w_uq_qr")
    w_uk = kwargs.get("w_uk")
    w_dkv_kr = kwargs.get("w_dkv_kr")
    gamma_cq = kwargs.get("gamma_cq")
    gamma_ckv = kwargs.get("gamma_ckv")
    sin = kwargs.get("sin")
    cos = kwargs.get("cos")
    cache_index = kwargs.get("cache_index")
    kv_cache = kwargs.get("kv_cache")
    kr_cache = kwargs.get("kr_cache")
    quant_inputs = kwargs.get("quant_inputs")
    rope_config = kwargs.get("rope_config")
    query_out = kwargs.get("query_out")
    query_rope_out = kwargs.get("query_rope_out")
    kv_cache_out = kwargs.get("kv_cache_out")
    kr_cache_out = kwargs.get("kr_cache_out")
    epsilon_cq = kwargs.get("epsilon_cq")
    epsilon_ckv = kwargs.get("epsilon_ckv")
    cache_mode = kwargs.get("cache_mode")
    split_reduce_last_dim = kwargs.get("split_reduce_last_dim")
    split_k = kwargs.get("split_k")

    try:
        if len(token_x.shape) != SHAPE_DIM3:
            raise ValueError(f"token_x.shape must have {SHAPE_DIM3} dimensions")
        if len(w_uk.shape) != SHAPE_DIM3:
            raise ValueError(f"w_uk.shape must have {SHAPE_DIM3} dimensions")
        if len(sin.shape) != SHAPE_DIM3:
            raise ValueError(f"sin.shape must have {SHAPE_DIM3} dimensions")
        if len(kv_cache.shape) != SHAPE_DIM4:
            raise ValueError(f"kv_cache.shape must have {SHAPE_DIM4} dimensions")
        if len(kr_cache.shape) != SHAPE_DIM4:
            raise ValueError(f"kr_cache.shape must have {SHAPE_DIM4} dimensions")
        if cache_mode not in {"BNSD", "PA_BSND", "PA_NZ"}:
            raise ValueError('cache_mode must be one of BNSD, PA_BSND, PA_NZ')
    except ValueError as e:
        print(f"ValueError:{e}")

    dequant_scale_w_uq_qr = quant_inputs.dequant_scale_w_uq_qr
    is_quant = (dequant_scale_w_uq_qr.has_storage() if dequant_scale_w_uq_qr is not None else False)
    print("is_quant +++ ", is_quant)

    d_type = token_x.dtype
    b = token_x.shape[0]
    s = token_x.shape[1]  # s=1
    b_s = b * s

    n = w_uk.shape[0]
    qk_nope_head_dim = w_uk.shape[1]
    kv_lora_rank = w_uk.shape[2]
    qk_rope_head_dim = sin.shape[2]  # [b,s,qkRopeHeadDim]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    q_kv = qkv_pre(
        token_x=token_x,
        w_dq=w_dq,
        w_uq_qr=w_uq_qr,
        w_dkv_kr=w_dkv_kr,
        gamma_cq=gamma_cq,
        epsilon_cq=epsilon_cq,
        quant_inputs=quant_inputs,
        split_reduce_last_dim=split_reduce_last_dim,
        split_k=split_k
    )
    q = q_kv[0]
    kv_tmp = q_kv[1]

    if is_quant:
        tile_shape = [b_s, NUM_64]
        pto.set_vec_tile_shapes(*tile_shape)
        q_tmp_fp32 = pto.cast(q, pto.DT_FP32)
        q_tmp_dequant_scale = q_kv[2]
        q_tmp_dequant_per_token = pto.mul(q_tmp_fp32, q_tmp_dequant_scale)
        q_tmp_dequant_channel = pto.mul(q_tmp_dequant_per_token, dequant_scale_w_uq_qr)
        q[:] = pto.cast(q_tmp_dequant_channel, d_type)

    q_tmp = pto.reshape(q, [b, s, n, q_head_dim])
    tile_shape = [b, 1, 1, NUM_64]
    pto.set_vec_tile_shapes(*tile_shape)

    ########## q ##########
    q_nope = pto.view(q_tmp, [b, s, n, qk_nope_head_dim], [0, 0, 0, 0])

    tile_shape = [b, 1, 1, NUM_128]
    pto.set_vec_tile_shapes(*tile_shape)
    q_nope_res = pto.reshape(q_nope, [b_s, n, qk_nope_head_dim]); # [bs,n,qkNopeHeadDim]
    tile_shape = [b_s, 1, qk_nope_head_dim];     # {NUM_2, NUM_32, qkNopeHeadDim}
    pto.set_vec_tile_shapes(*tile_shape)
    q_nope_trans = pto.transpose(q_nope_res, [0, 1]); # [n,bs,qkNopeHeadDim]

    c0 = NUM_16
    m = (b_s + c0 - 1) // c0 * c0
    pto.set_cube_tile_shapes([m, m], [NUM_128, NUM_128], [NUM_128, NUM_128])
    q_nope_new = pto.batch_matmul(d_type, q_nope_trans, w_uk)

    tile_shape = [1, b_s, kv_lora_rank]  # {NUM_16, NUM_2, kvLoraRank}
    pto.set_vec_tile_shapes(*tile_shape)
    q_nope_new_trans = pto.transpose(q_nope_new, [0, 1])  # [bs,n,kvLoraRank]
    query_out[:] = pto.reshape(q_nope_new_trans, [b, s, n, kv_lora_rank])

    ########## kv ##########
    compressed_kv = pto.view(kv_tmp, [b, s, kv_lora_rank], [0, 0, 0])
    tile_shape = [NUM_2, 1, NUM_512]
    pto.set_vec_tile_shapes(*tile_shape)
    compressed_kv_norm = pto.rms_norm(compressed_kv, gamma_ckv, epsilon_ckv)

    ########## RoPE ##########
    q_pe = pto.view(q_tmp, [b, s, n, qk_rope_head_dim], [0, 0, 0, qk_nope_head_dim])
    k_pe = pto.view(kv_tmp, [b, s, qk_rope_head_dim], [0, 0, kv_lora_rank])
    tile_shape = [b_s, 1, NUM_64]
    pto.set_vec_tile_shapes(*tile_shape)
    k_pe_res = pto.reshape(k_pe, [b, s, 1, qk_rope_head_dim])

    k_rope = pto.tensor([b, s, 1, qk_rope_head_dim], k_pe_res.dtype, "kRope")
    pto.apply_rotary_pos_emb_v2(q_pe, k_pe_res, cos, sin, query_rope_out, k_rope, 2, rope_config)

    if cache_mode == "PA_BSND":
        block_num = kv_cache.shape[0]
        block_size = kv_cache.shape[1]
        n2 = kv_cache.shape[2]
        kv_cache_res = pto.reshape(kv_cache, [block_num * block_size * n2, kv_lora_rank])
        kr_cache_res = pto.reshape(kr_cache, [block_num * block_size * n2, qk_rope_head_dim])
        k_nope = pto.reshape(compressed_kv_norm, [b * s, kv_lora_rank])
        k_rope_res = pto.reshape(k_rope, [b * s * 1, qk_rope_head_dim])

        ########## kvCache ##########
        tile_shape = [1, kv_lora_rank]
        pto.set_vec_tile_shapes(*tile_shape)
        kv_cache_updata = pto.scatter_update(kv_cache_res, cache_index, k_nope, -2, cache_mode)
        kv_cache_out[:] = pto.reshape(kv_cache_updata, [block_num, block_size, n2, kv_lora_rank])

        ########## krCache ##########
        tile_shape = [1, qk_rope_head_dim]
        pto.set_vec_tile_shapes(*tile_shape)
        kr_cache_update = pto.scatter_update(kr_cache_res, cache_index, k_rope_res, -2, cache_mode)
        kr_cache_out[:] = pto.reshape(kr_cache_update, [block_num, block_size, n2, qk_rope_head_dim])
    else:
        k_nope = pto.reshape(compressed_kv_norm, [b, 1, s, kv_lora_rank])
        k_rope_res = pto.reshape(k_rope, [b, 1, s, qk_rope_head_dim])

        ########## kvCache ##########
        tile_shape = [1, 1, 1, kv_lora_rank]
        pto.set_vec_tile_shapes(*tile_shape)
        kv_cache_out[:] = pto.scatter_update(kr_cache, cache_index, k_rope_res, -2)


def test_mla_prolog_v2(**kwargs):
    params = kwargs.get("params")
    is_quant = kwargs.get("is_quant", False)
    has_smooth = kwargs.get("has_smooth", False)
    block_size = kwargs.get("block_size", 128)
    cache_mode = kwargs.get("cache_mode", "BNSD")
    nz = kwargs.get("nz", False)
    use_pre_fetch = kwargs.get("use_pre_fetch", False)
    split_reduce_last_dim = kwargs.get("split_reduce_last_dim", True)
    split_k = kwargs.get("split_k", False)

    b = params[0]
    s = params[1]
    s2 = params[2]
    n = params[3]
    h = params[4]
    q_lora_rank = params[5]
    qk_nope_head_dim = params[6]
    qk_rope_head_dim = params[7]
    kv_lora_rank = params[8]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    d_type = pto.DT_BF16
    d_type_quant_in = pto.DT_INT8 if is_quant else d_type

    x_shape = [b, s, h]
    w_qa_shape = [h, q_lora_rank]
    w_qb_shape = [q_lora_rank, n * q_head_dim]
    w_kv_a_shape = [h, kv_lora_rank + qk_rope_head_dim]
    w_kv_b_k_shape = [n, qk_nope_head_dim, kv_lora_rank]
    cos_shape = [b, s, qk_rope_head_dim]
    gamma_cq_shape = [q_lora_rank]
    gamma_ckv_shape = [kv_lora_rank]
    kv_len_shape = [b, s]
    kv_cache_shape = [b, 1, s2, kv_lora_rank]
    kr_cache_shape = [b, 1, s2, qk_rope_head_dim]
    # output
    q_out_shape = [b, s, n, kv_lora_rank]
    q_rope_out_shape = [b, s, n, qk_rope_head_dim]
    if cache_mode == "PA_BSND":
        block_num = b * (s2 // block_size)
        kv_cache_shape = [block_num, block_size, 1, kv_lora_rank]
        kr_cache_shape = [block_num, block_size, 1, qk_rope_head_dim]


    x = pto.tensor(x_shape, d_type, "x")
    weight_format = pto.TileOpFormat.TILEOP_NZ if nz else pto.TileOpFormat.TILEOP_ND
    w_dq = pto.tensor(w_qa_shape, d_type, "w_dq", weight_format)
    w_uq_qr = pto.tensor(w_qb_shape, d_type_quant_in, "w_uq_qr", weight_format)
    if use_pre_fetch:
        w_dq.set_cache_policy(pto.CachePolicy.PREFETCH, True)
        w_uq_qr.set_cache_policy(pto.CachePolicy.PREFETCH, True)


    w_dkv_kr = pto.tensor(w_kv_a_shape, d_type, "w_dkv_kr", weight_format)
    w_uk = pto.tensor(w_kv_b_k_shape, d_type, "w_uk", weight_format)
    gamma_cq = pto.tensor(gamma_cq_shape, d_type, "gamma_cq")
    gamma_ckv = pto.tensor(gamma_ckv_shape, d_type, "gamma_ckv")
    cos = pto.tensor(cos_shape, d_type, "cos")
    sin = pto.tensor(cos_shape, d_type, "sin")
    kv_len = pto.tensor(kv_len_shape, pto.DT_INT64, "kv_len")
    kv_cache = pto.tensor(kv_cache_shape, d_type, "kv_cache")
    kr_cache = pto.tensor(kr_cache_shape, d_type, "kr_cache")

    # output
    output_q = pto.tensor(q_out_shape, d_type, "output_q")
    output_q_rope = pto.tensor(q_rope_out_shape, d_type, "output_q_rope")

    rope_config = pto.rope_tile_shape_config_new()
    rope_config.three_dims_tile_shape = [b, 1, 64]
    rope_config.four_dims_tile_shape_q = [b, 1, 1, 64]
    rope_config.four_dims_tile_shape_k = [b, 1, 1, 64]
    rope_config.five_dims_tile_shape = [b, 1, 1, 32, 2]

    quant_inputs = MlaQuantInputs()
    if is_quant:
        w_qb_scale_shape = [1, n * q_head_dim]
        smooth_cq_shape = [1, q_lora_rank]
        w_qb_scale = pto.tensor(w_qb_scale_shape, pto.DT_FP32, "w_qb_scale")
        quant_inputs.dequant_scale_w_uq_qr = w_qb_scale
        smooth_cq = pto.tensor(smooth_cq_shape, pto.DT_FP32, "smooth_cq_shape")
        if has_smooth:
            quant_inputs.smooth_scales_cq = smooth_cq
            smooth_cq.set_cache_policy(pto.CachePolicy.PREFETCH, True)

        graph_t = pto.GraphType.TENSOR_GRAPH
        func_t = pto.FunctionType.STATIC
        with pto.pto_function("MlaPrologUt", graph_t, func_t,
                x, w_dq, w_uq_qr, w_qb_scale, smooth_cq, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos,
                 kv_len, kv_cache, kr_cache, output_q, output_q_rope):
            mla_prolog(
                token_x=x,
                w_dq=w_dq,
                w_uq_qr=w_uq_qr,
                w_uk=w_uk,
                w_dkv_kr=w_dkv_kr,
                gamma_cq=gamma_cq,
                gamma_ckv=gamma_ckv,
                sin=sin,
                cos=cos,
                cache_index=kv_len,
                kv_cache=kv_cache,
                kr_cache=kr_cache,
                quant_inputs=quant_inputs,
                rope_config=rope_config,
                query_out=output_q,
                query_rope_out=output_q_rope,
                kv_cache_out=kv_cache,
                kr_cache_out=kr_cache,
                epsilon_cq=1e-5,
                epsilon_ckv=1e-5,
                cache_mode=cache_mode,
                split_reduce_last_dim=split_reduce_last_dim,
                split_k=split_k
                )
    else:
        graph_t = pto.GraphType.TENSOR_GRAPH
        func_t = pto.FunctionType.STATIC
        with pto.pto_function("MlaPrologUt", graph_t, func_t,
                x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos,
                 kv_len, kv_cache, kr_cache, output_q, output_q_rope):
            mla_prolog(
                token_x=x,
                w_dq=w_dq,
                w_uq_qr=w_uq_qr,
                w_uk=w_uk,
                w_dkv_kr=w_dkv_kr,
                gamma_cq=gamma_cq,
                gamma_ckv=gamma_ckv,
                sin=sin,
                cos=cos,
                cache_index=kv_len,
                kv_cache=kv_cache,
                kr_cache=kr_cache,
                quant_inputs=quant_inputs,
                rope_config=rope_config,
                query_out=output_q,
                query_rope_out=output_q_rope,
                kv_cache_out=kv_cache,
                kr_cache_out=kr_cache,
                epsilon_cq=1e-5,
                epsilon_ckv=1e-5,
                cache_mode=cache_mode,
                split_reduce_last_dim=split_reduce_last_dim,
                split_k=split_k
                )



if __name__ == "__main__":
    b = 32
    s = 1
    s2 = 4096
    h = 7168
    n = 128
    q_lora_rank = 1536
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    block_size = 128
    cache_mode = "PA_BSND"
    split_reduce_last_dim = False
    split_k = False
    nz = True
    use_pre_fetch = False
    params = [b, s, s2, n, h, q_lora_rank, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank]
    test_mla_prolog_v2(params=params, is_quant=True, has_smooth=True, block_size=block_size, cache_mode=cache_mode,
                    nz=nz, use_pre_fetch=use_pre_fetch, split_reduce_last_dim=split_reduce_last_dim, split_k=split_k)