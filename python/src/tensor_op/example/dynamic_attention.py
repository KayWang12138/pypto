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

# RuntimeConfig KEYS
SG_PARALLEL_NUM = "parallel_threshold"
SG_CYCLE_UPPER_BOUND = "cycle_upper_bound"
SG_CYCLE_LOWER_BOUND = "cycles_threshold"
L1_REUSE = "l1_reuse"
L1_REUSE_MAP = "l1_reuse_map"
CUBE_NBUFFER = "cube_nbuffer"
CUBE_NBUFFER_MAP = "cube_nbuffer_map"
LOAD_BALANCE = "load_balance"
COPYIN_THRESHOLD = "copyin_threshold"
MACHINE_CONFIG = "machine_config"
OOO_PRESCHEDULE_METHOD = "ooo_preschedule_method"
NBUFFER_MERGE_MODE = "nbuffer_merge_mode"
VEC_NBUFFER_MAP = "vec_nbuffer_map"
SG_CUBE_PARALLEL_NUM = "sg_cube_parallel_num"
SG_VEC_PARALLEL_NUM = "sg_vec_parallel_num"
KEY_ONLY_CODEGEN = "ONLY_CODEGEN"

SCATTER_UPDATE_DIM = -2
NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_16 = 16
NUM_20 = 20
NUM_32 = 32
NUM_64 = 64
NUM_128 = 128
NUM_256 = 256
NUM_512 = 512
NUM_1024 = 1024


def ceil_div(a, b):
    return (a + b - 1) // b


@dataclass
class MlaQuantInputs:
    dequant_scale_x: pto.tensor = pto.tensor()
    dequant_scale_w_dq: pto.tensor = pto.tensor()
    dequant_scale_w_uq_qr: pto.tensor = pto.tensor()
    dequant_scale_w_dkv_kr: pto.tensor = pto.tensor()
    quant_scale_ckv: pto.tensor = pto.tensor()
    quant_scale_ckr: pto.tensor = pto.tensor()
    smooth_scales_cq: pto.tensor = pto.tensor()


def mla_pre(**kwargs) -> List[pto.tensor]:
    token_x = kwargs.get("token_x")
    w_dq = kwargs.get("w_dq")
    w_uq_qr = kwargs.get("w_uq_qr")
    w_dkv_kr = kwargs.get("w_dkv_kr")
    gamma_cq = kwargs.get("gamma_cq")
    epsilon_cq = kwargs.get("epsilon_cq")
    quant_inputs = kwargs.get("quant_inputs")
    split_k = kwargs.get("split_k")
    is_smooth = kwargs.get("is_smooth")

    # quant
    dequant_scale_w_uq_qr = quant_inputs.dequant_scale_w_uq_qr
    is_quant = (dequant_scale_w_uq_qr.has_storage() if dequant_scale_w_uq_qr is not None else False)
    smooth_scales_cq = quant_inputs.smooth_scales_cq

    b = token_x.shape[0]
    s = token_x.shape[1]
    h = token_x.shape[2]
    bs = b * s
    q_lora_rank = w_dq.shape[1]

    d_type = token_x.get_dtype()
    d_type_quant_out = pto.data_type.DT_INT32 if is_quant else d_type
    qkv_pre_res = []

    input_tensor = pto.reshape(token_x, [bs, h])  # [b,s,h] -> [b*s,h]

    ###### q ########
    c0 = 16  # 16
    m = (min(32, bs) + c0 - 1) // c0 * c0  # 32
    tie_m = min(32, m)  # 32
    pto.set_cube_tile_shapes([tie_m, tie_m], [256, 256], [64, 64])  # 256, 64

    q_mm_res = pto.tensor()
    if split_k:
        def inside_if_split_k():
            nonlocal q_mm_res
            tmp_c = pto.tensor(pto.data_type.DT_FP32, [bs, q_lora_rank], "tmp_q")
            pto.set_vec_tile_shapes(min(32, bs), 128)  # 32, 128
            tmp_c.move(pto.mul_s(tmp_c, pto.element(pto.data_type.DT_FP32, 0.0)))
            matmul_result = []
            k_split = 7
            k_split_size = h // k_split
            for ki in range(k_split):
                input_mk = pto.view(input_tensor, [bs, k_split_size], [0, ki * k_split_size])
                input_kn = pto.view(w_dq, [k_split_size, q_lora_rank], [ki * k_split_size, 0])
                tmp = pto.matmul(pto.data_type.DT_FP32, input_mk, input_kn)  # [b*s,h/2] * [h/2,q_lora_rank]
                matmul_result.append(tmp)
            q_mm_res_f32 = pto.reduce(matmul_result, pto.reduce_mode.ATOMIC_ADD)
            pto.set_vec_tile_shapes(min(32, bs), 128)  # 32, 128
            q_mm_res.move(pto.cast(q_mm_res_f32, d_type))
        inside_if_split_k()
    else:
        q_mm_res.move(pto.matmul(d_type, input_tensor, w_dq))  # bf16

    pto.set_vec_tile_shapes(min(8, bs), q_lora_rank)  # 8
    norm_res = pto.rms_norm(q_mm_res, gamma_cq, epsilon_cq)

    norm_dequant_scale = pto.tensor()
    norm_quant_res = None
    if is_quant:
        if is_smooth:
            norm_quant_res = pto.quant(norm_res, True, True, smooth_scales_cq)
        else:
            norm_quant_res = pto.quant(norm_res)  # int8
        norm_res.move(norm_quant_res[0])
        norm_dequant_scale.move(norm_quant_res[1])
        pto.set_cube_tile_shapes(
            [tie_m, tie_m], [256, 256], [256, 256])  # 256
    else:
        # use tileM will core dump
        pto.set_cube_tile_shapes([tie_m, tie_m], [256, 256], [64, 64])  # 256, 64

    q = pto.matmul(d_type_quant_out, norm_res, w_uq_qr)  # bf16  // quant: A8W8O32 -> bf16
    qkv_pre_res.append(q)

    ###### kv ########
    pto.set_cube_tile_shapes([m, m], [256, 256], [64, 64])  # 256, 64
    compressed_kv = pto.tensor()
    if split_k:
        def inside_if_split():
            nonlocal compressed_kv
            pto.set_vec_tile_shapes(min(32, bs), 64)  # 32, 64
            kv_n = w_dkv_kr.shape[1]
            tmp_c_kv = pto.tensor([bs, kv_n], pto.data_type.DT_FP32, "tmp_kv")
            tmp_c_kv.move(pto.mul_s(tmp_c_kv, pto.element(pto.data_type.DT_FP32, 0.0)))
            matmul_result_kv = []
            k_split_kv = 7
            k_split_size_kv = h // k_split_kv
            for ki in range(k_split_kv):
                input_mk = pto.view(input_tensor, [bs, k_split_size_kv], [0, ki * k_split_size_kv])
                input_kn = pto.view(w_dkv_kr, [k_split_size_kv, kv_n], [ki * k_split_size_kv, 0])
                tmp = pto.matmul(pto.data_type.DT_FP32, input_mk, input_kn)  # [b*s,h/2] * [h/2,kv_n] = [b*s,kv_n]
                matmul_result_kv.append(tmp)
            kv_mm_res_f32 = pto.reduce(matmul_result_kv, pto.reduce_mode.ATOMIC_ADD)
            pto.set_vec_tile_shapes(min(32, bs), 64)  # 32, 64
            compressed_kv.move(pto.cast(kv_mm_res_f32, d_type))
        inside_if_split()
    else:
        compressed_kv.move(pto.matmul(d_type, input_tensor, w_dkv_kr))  # bf16

    compressed_kv_res = pto.reshape(compressed_kv, [b, s, w_dkv_kr.shape[1]])
    qkv_pre_res.append(compressed_kv_res)

    if is_quant:
        qkv_pre_res.append(norm_dequant_scale)

    return qkv_pre_res

NUM_100000 = 100000
NUM_500000 = 500000


def attention(**kwargs):
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
    q_nope_out = kwargs.get("q_nope_out")
    q_rope_out = kwargs.get("q_rope_out")
    kv_cache_out = kwargs.get("kv_cache_out")
    kr_cache_out = kwargs.get("kr_cache_out")
    quant_inputs = kwargs.get("quant_inputs")
    rope_config = kwargs.get("rope_config")
    block_table = kwargs.get("block_table")
    act_seqs = kwargs.get("act_seqs")
    pa_out = kwargs.get("pa_out")
    block_size = kwargs.get("block_size")
    softmax_scale = kwargs.get("softmax_scale")
    pa_tile_config = kwargs.get("pa_tile_config")
    weight_uv = kwargs.get("weight_uv")
    weight_o = kwargs.get("weight_o")
    weight_o_scale_w = kwargs.get("weight_o_scale_w")
    post_out = kwargs.get("post_out")
    epsilon_cq = kwargs.get("epsilon_cq")
    epsilon_ckv = kwargs.get("epsilon_ckv")
    cache_mode = kwargs.get("cache_mode")

    pa_format = pto.tile_op_format.TILEOP_NZ if cache_mode == "PA_NZ" else pto.tile_op_format.TILEOP_ND
    dtype = token_x.get_dtype()
    b = token_x.shape[0]
    s = token_x.shape[1]  # s=1
    h = token_x.shape[2]
    s2 = kv_cache.shape[2]

    n = w_uk.shape[0]
    qk_nope_head_dim = w_uk.shape[1]
    kv_lora_rank = w_uk.shape[2]
    qk_rope_head_dim = sin.shape[2]  # [b,s,qkRopeHeadDim]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    tile_b = b
    tile_bs = tile_b * s

    # 入参B*S*N合轴
    d_n = kv_lora_rank
    d_r = qk_rope_head_dim

    n_tile = pa_tile_config.head_num_q_tile
    c1_tile = pa_tile_config.c1_tile_shape
    v1_tile = pa_tile_config.v1_tile_shape
    c2_tile = pa_tile_config.c2_tile_shape
    v2_tile = pa_tile_config.v2_tile_shape

    v_head_dim = weight_uv.shape[2]

    pa_out_shape = [b * s * n, kv_lora_rank]


    input_tensors = [token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, cache_index,
                           kv_cache, kr_cache,
                           quant_inputs.dequant_scale_w_uq_qr, quant_inputs.smooth_scales_cq,
                           block_table, act_seqs, weight_uv, weight_o, weight_o_scale_w]

    output_tensors = [post_out]
    inplace_tensors = [[kv_cache_out, kv_cache], [kr_cache_out, kr_cache]]
    pto.set_pass_config("PVC2_OOO", "InferMemoryConflict", "DISABLE_PASS", True)
    with pto.dyn_function("main", input_tensors, output_tensors, inplace_tensors):
        def inside_main_function():
            nonlocal n_tile, pa_out, kv_cache_out, kr_cache_out
            ########## mla_prolog ##########
            b_loop = b // tile_b
            pto.set_config(NBUFFER_MERGE_MODE, 1)
            pto.set_config(L1_REUSE, NUM_4)  # L1reuse合并的左矩阵或者右矩阵数量
            # 从NUM_3个mm开始设置CubeNBuffer数量为NUM_4；CubeNBuffer：设置同构的mm计算合并入一个图
            pto.set_config(CUBE_NBUFFER_MAP, {NUM_3: NUM_4})
            pto.set_config(COPYIN_THRESHOLD, NUM_2 * NUM_1024 * NUM_1024)   # CubeNBuffer、L1reuse合并时copyin的cycle上限
            pto.set_config(SG_CYCLE_UPPER_BOUND, NUM_100000)    # 设置切图与合图后子图的Latency的上限
            # 设置子图合并的并行度下限（子图数量大于等于parallelThreshold才可合并）
            pto.set_config(SG_PARALLEL_NUM, NUM_2)
            # LOOP("LOOP_L0_bIdx_mla_prolog", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1)) {
            with pto.loop_function("LOOP_L0_bIdx_mla_prolog", "b_idx", pto.loop_range(0, b_loop, 1)) as b_idx_loop:
                for b_idx in b_idx_loop:
                    def inside_b_idx_loop_prolog(b_idx):
                        nonlocal n_tile, pa_out, kv_cache_out, kr_cache_out
                        b_offset = b_idx * tile_b
                        output_offset = [b_offset, 0, 0, 0]

                        dequant_scale_w_uq_qr = quant_inputs.dequant_scale_w_uq_qr
                        is_quant = (dequant_scale_w_uq_qr.has_storage() if dequant_scale_w_uq_qr is not None else False)
                        smooth_scales_cq = quant_inputs.smooth_scales_cq
                        is_smooth = (smooth_scales_cq.has_storage() if smooth_scales_cq is not None else False)
                        logging.info(f"is_quant +++ {is_quant}")
                        x_view = pto.view(token_x, [tile_b, s, h], [b_offset, 0, 0])
                        pto.set_semantic_label("mlaPre")

                        q_kv = mla_pre(
                            token_x=x_view,
                            w_dq=w_dq,
                            w_uq_qr=w_uq_qr,
                            w_dkv_kr=w_dkv_kr,
                            gamma_cq=gamma_cq,
                            epsilon_cq=epsilon_cq,
                            quant_inputs=quant_inputs,
                            split_k=False,
                            is_smooth=is_smooth)
                        q = q_kv[0]      # [b*s, n*qHeadDim]
                        kv_tmp = q_kv[1]  # [b,s,kvLoraRank+qkRopeHeadDim]

                        # dequant: int32 -> fp32 -> *scale -> fp16/bf16
                        if is_quant:
                            def inside_quant():
                                nonlocal q
                                pto.set_semantic_label("Quant")
                                tile_shape = [min(NUM_32, tile_bs), NUM_64]
                                pto.set_vec_tile_shapes(*tile_shape)
                                q_tmp_fp32 = pto.cast(q, pto.data_type.DT_FP32)
                                q_tmp_dequant_scale = q_kv[2]
                                q_tmp_dequant_per_token = pto.mul(q_tmp_fp32, q_tmp_dequant_scale)
                                q_tmp_dequant_channel = pto.mul(q_tmp_dequant_per_token, dequant_scale_w_uq_qr)

                                q.move(pto.cast(q_tmp_dequant_channel, dtype))
                            inside_quant()

                        pto.set_semantic_label("Reshape0")
                        q_tmp = pto.reshape(q, [tile_b, s, n, q_head_dim])
                        tile_shape = [min(NUM_32, tile_b), 1, 1, NUM_64]
                        pto.set_vec_tile_shapes(*tile_shape)

                        ########## q ##########
                        pto.set_semantic_label("q")
                        q_nope = pto.view(q_tmp, [tile_b, s, n, qk_nope_head_dim], [0, 0, 0, 0]) # [b,s,n,qkNopeHeadDim]
                        tile_shape = [tile_b, 1, 1, NUM_128]
                        pto.set_vec_tile_shapes(*tile_shape)
                        q_nope_res = pto.reshape(q_nope, [tile_bs, n, qk_nope_head_dim])  # [bs,n,qkNopeHeadDim]
                        tile_shape = [min(NUM_32, tile_bs), 1, qk_nope_head_dim]     # {NUM_2, NUM_32, qkNopeHeadDim}
                        pto.set_vec_tile_shapes(*tile_shape)
                        pto.set_semantic_label("Transpose0")
                        q_nope_trans = pto.transpose(q_nope_res, [0, 1])  # [n,bs,qkNopeHeadDim]

                        c0 = NUM_16
                        m = (min(NUM_32, tile_bs) + c0 - 1) // c0 * c0
                        pto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_128, NUM_128], True)
                        pto.set_semantic_label("BatchMatmul")
                        q_nope_new = pto.batch_matmul(dtype, q_nope_trans, w_uk)

                        tile_shape = [1, min(NUM_32, tile_bs), kv_lora_rank]  # {NUM_16, NUM_2, kvLoraRank}
                        pto.set_vec_tile_shapes(*tile_shape)
                        pto.set_semantic_label("Transpose1")
                        q_nope_new_trans = pto.transpose(q_nope_new, [0, 1])  # [bs,n,kvLoraRank]

                        ########## kv ##########
                        pto.set_semantic_label("kv")
                        compressed_kv = pto.view(kv_tmp, [tile_b, s, kv_lora_rank], [0, 0, 0])  # [b,s,kvLoraRank]
                        tile_shape = [NUM_2, 1, NUM_512]
                        pto.set_vec_tile_shapes(*tile_shape)
                        pto.set_semantic_label("RmsNorm")
                        compressed_kv_norm = pto.rms_norm(compressed_kv, gamma_ckv, epsilon_ckv)  # [b,s,kvLoraRank]
                        k_nope = pto.reshape(compressed_kv_norm, [tile_b, 1, s, kv_lora_rank])    # [b,1,s,kvLoraRank]

                        ########## RoPE ##########
                        pto.set_semantic_label("RoPE")
                        # -> [b,s,qkRopeHeadDim]
                        k_pe_view = pto.view(kv_tmp, [tile_b, s, qk_rope_head_dim], [0, 0, kv_lora_rank])
                        tile_shape = [min(NUM_32, tile_b), 1, qk_rope_head_dim]
                        pto.set_vec_tile_shapes(*tile_shape)
                        # -> [b,s,1,qkRopeHeadDim]
                        k_pe_res = pto.reshape(k_pe_view, [tile_b, s, 1, qk_rope_head_dim])
                        q_pe_view = pto.view(q_tmp, [tile_b, s, n, qk_rope_head_dim], [0, 0, 0, qk_nope_head_dim])
                        cos_view = pto.view(cos, [tile_b, s, qk_rope_head_dim], [b_offset, 0, 0])
                        sin_view = pto.view(sin, [tile_b, s, qk_rope_head_dim], [b_offset, 0, 0])
                        ## -> [b,1,s,qkRopeHeadDim]
                        k_rope_view = pto.tensor([tile_b, s, 1, qk_rope_head_dim], k_pe_res.get_dtype(), "kRopeView")
                        q_rope_view = pto.tensor([tile_b, s, n, qk_rope_head_dim], k_pe_res.get_dtype(), "qRopeView")
                        pto.set_semantic_label("ApplyRotaryPosEmbV2")
                        pto.apply_rotary_pos_emb_v2(q_pe_view, k_pe_res, cos_view, sin_view, q_rope_view,
                                                     k_rope_view, NUM_2, rope_config)

                        if cache_mode != "BNSD":
                            def inside_if_cache_mode():
                                nonlocal kv_cache_out, kr_cache_out
                                block_num = kv_cache.shape[0]
                                n2 = kv_cache.shape[2]
                                kv_cache_res = pto.reshape(kv_cache, [block_num * block_size * n2, kv_lora_rank])
                                kr_cache_res = pto.reshape(kr_cache, [block_num * block_size * n2, qk_rope_head_dim])
                                cache_index_dview = pto.view(cache_index, [tile_b, s], [b_offset, 0])
                                k_nope.move(pto.reshape(k_nope, [tile_b * s, kv_lora_rank]))  # [b*s,kvLoraRank]
                                k_rope_res = pto.reshape(k_rope_view, [tile_b * s * 1, qk_rope_head_dim])

                                ########## kvCache ##########
                                tile_shape = [1, kv_lora_rank]
                                pto.set_vec_tile_shapes(*tile_shape)
                                kv_cache_out_dview = pto.scatter_update(kv_cache_res, cache_index_dview, k_nope,
                                                                        SCATTER_UPDATE_DIM, cache_mode, block_size)

                                ########## krCache ##########
                                tile_shape = [1, qk_rope_head_dim]
                                pto.set_vec_tile_shapes(*tile_shape)
                                kr_cache_out_dview = pto.scatter_update(kr_cache_res, cache_index_dview, k_rope_res,
                                                                        SCATTER_UPDATE_DIM, cache_mode, block_size)

                                kv_cache_out.move(pto.reshape(kv_cache_out_dview,
                                                              [block_num, block_size, n2, kv_lora_rank]))
                                kr_cache_out.move(pto.reshape(kr_cache_out_dview,
                                                              [block_num, block_size, n2, qk_rope_head_dim]))
                            inside_if_cache_mode()
                        else:
                            def inside_else_cache_mode():
                                nonlocal kv_cache_out, kr_cache_out
                                pto.set_semantic_label("Reshape1")
                                k_rope_res = pto.reshape(k_rope_view, [tile_b, 1, s, qk_rope_head_dim])
                                pto.set_semantic_label("kvCache")
                                cache_index_dview = pto.view(cache_index, [tile_b, s], [b_offset, 0])
                                ########## kvCache ##########
                                tile_shape = [1, 1, 1, kv_lora_rank]
                                pto.set_vec_tile_shapes(*tile_shape)
                                # kvCache: [b,1,s2,kvLoraRank], output3
                                kv_cache_dview = pto.view(kv_cache, [tile_b, 1, s2, kv_lora_rank], [b_offset, 0, 0, 0])
                                pto.set_semantic_label("ScatterUpdate0")
                                kv_cache_out_dview = pto.scatter_update(kv_cache_dview, cache_index_dview, k_nope, -2)

                                ########## krCache ##########
                                pto.set_semantic_label("krCache")
                                tile_shape = [1, 1, 1, qk_rope_head_dim]
                                pto.set_vec_tile_shapes(*tile_shape)
                                # krCache: [b,1,s2,qkRopeHeadDim], output4
                                kr_cache_dview = pto.view(kr_cache, [tile_b, 1, s2, qk_rope_head_dim],
                                                          [b_offset, 0, 0, 0])
                                pto.set_semantic_label("ScatterUpdate1")
                                kr_cache_out_dview = pto.scatter_update(kr_cache_dview, cache_index_dview,
                                                                        k_rope_res, -2)

                                kv_cache_out_dview_new = pto.reshape(kv_cache_out_dview,
                                                                     [tile_b * 1 * s2, kv_lora_rank])
                                kr_cache_out_dview_new = pto.reshape(kr_cache_out_dview,
                                                                     [tile_b * 1 * s2, qk_rope_head_dim])
                                pto.assemble(kv_cache_out_dview_new, [b_offset * s2, 0], kv_cache_out)
                                pto.assemble(kr_cache_out_dview_new, [b_offset * s2, 0], kr_cache_out)
                            inside_else_cache_mode()

                        query_out_dview_new = pto.reshape(q_nope_new_trans, [tile_b * s * n, kv_lora_rank])
                        q_rope_view_new = pto.reshape(q_rope_view, [tile_b * s * n, qk_rope_head_dim])
                        pto.assemble(query_out_dview_new, [b_offset * s * n, 0], q_nope_out)
                        pto.assemble(q_rope_view_new, [b_offset * s, 0], q_rope_out)
                    inside_b_idx_loop_prolog(b_idx)
            # } # LOOP("LOOP_L0_bIdx_mla_prolog") ends
            ########## pa ##########
            batch_size_scalar = block_table.shape[0]
            n_q = q_nope_out.shape[0] // batch_size_scalar
            n_loop = n_q // n_tile

            pto.set_config(CUBE_NBUFFER_MAP,  std::map<int64_t, int64_t>{})
            pto.set_config(L1_REUSE, 0)
            pto.set_config(COPYIN_THRESHOLD, 1 * NUM_1024 * NUM_1024)
            pto.set_config(SG_CYCLE_UPPER_BOUND, NUM_100000)
            pto.set_config(SG_PARALLEL_NUM, NUM_2)
            pto.set_config(CUBE_NBUFFER, NUM_2)
            pto.set_operation_config("FORCE_COMBINE_AXIS", True)

            # LOOP("LOOP_L0_bIdx_pa", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSizeScalar, 1), {}, true) {
            with pto.loop_function("LOOP_L0_bIdx_pa", "b_idx",
                                   pto.loop_range(0, batch_size_scalar, 1), set(), True) as b_idx_loop:
                for b_idx in b_idx_loop:
                    def inside_b_idx_loop_pa(b_idx):
                        nonlocal pa_out
                        cur_seq = pto.get_input_data(act_seqs, [b_idx])
                        bn_per_batch = (cur_seq + block_size - 1) // block_size
                        bn_per_batch.as_intermediate_variable()
                        # LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nLoop, 1)) {
                        with pto.loop_function("LOOP_L1_nIdx", "n_idx", pto.loop_range(0, n_loop, 1)) as n_idx_loop:
                            for n_idx in n_idx_loop:
                                def inside_n_idx_loop(b_idx, n_idx, bn_per_batch):
                                    nonlocal pa_out, n_tile
                                    cur_n_tile = n_tile
                                    oi_update = pto.tensor([n_tile, d_n], pto.data_type.DT_FP32, "oiUpdate")
                                    li_update = pto.tensor([n_tile, 1], pto.data_type.DT_FP32, "liUpdate")
                                    mi_update = pto.tensor([n_tile, 1], pto.data_type.DT_FP32, "miUpdate")
                                    # 当前curOffset没放到更内层循环，避免重复bnPerBatch次的DAssemble操作
                                    cur_offset = b_idx * n_q + n_idx * n_tile
                                    oi_offset = [cur_offset, 0]  # (B*N*S, d)

                                    # LoopRange(0, bnPerBatch, 1), PowersOf2(1)) {
                                    with pto.loop_function("LOOP_L2_bn", "bn", pto.loop_range(0, bn_per_batch, 1),
                                                            pto.powers_of_2(1)) as bn_loop:
                                        for bn in bn_loop:
                                            def inside_bn_loop(**kwargs):
                                                b_idx = kwargs.get("b_idx")
                                                block_table = kwargs.get("block_table")
                                                cur_seq = kwargs.get("cur_seq")
                                                bn = kwargs.get("bn")
                                                block_size = kwargs.get("block_size")
                                                bn_per_batch = kwargs.get("bn_per_batch")
                                                nonlocal oi_update, li_update, mi_update
                                                pto.set_semantic_label("pa")
                                                # 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                                                cur_s2_tile = block_size
                                                qn = pto.view(q_nope_out, [cur_n_tile, d_n], [cur_offset, 0])
                                                qr = pto.view(q_rope_out, [cur_n_tile, d_r], [cur_offset, 0])
                                                qi = pto.tensor([cur_n_tile, d_n + d_r], dtype, "qi")
                                                pto.assemble(qn, [0, 0], qi)
                                                pto.assemble(qr, [0, d_n], qi)

                                                cur_block_idx = pto.get_input_data(block_table, [b_idx, bn])
                                                cur_block_idx.as_intermediate_variable()
                                                kn = pto.view(kv_cache_out, [cur_s2_tile, d_n],
                                                              [min(cur_seq - bn * block_size, block_size), d_n],
                                                              [cur_block_idx * block_size, 0])
                                                kr = pto.view(kr_cache_out, [cur_s2_tile, d_r],
                                                              [min(cur_seq - bn * block_size, block_size), d_r],
                                                              [cur_block_idx * block_size, 0])
                                                kj = pto.tensor([cur_s2_tile, d_n + d_r], dtype, "kj", pa_format)
                                                pto.assemble(kn, [0, 0], kj)
                                                pto.assemble(kr, [0, d_n], kj)
                                                vj = pto.view(kv_cache_out, [cur_s2_tile, d_n],
                                                              [min(cur_seq - bn * block_size, block_size), d_n],
                                                              [cur_block_idx * block_size, 0])

                                                pto.set_cube_tile_shapes(
                                                    [c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]],
                                                    [c1_tile[4], c1_tile[5]], True)
                                                pto.set_semantic_label("paQkMM")
                                                pto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
                                                # (curNTile, dN+dR), (curS2Tile, dN+dR) -> (curNTile, curS2Tile)
                                                sij = pto.matmul(pto.data_type.DT_FP32, qi, kj, False, True)
                                                pto.set_semantic_label("paQkvec1")
                                                pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                                ## -> (curNTile, curS2Tile)
                                                sij_scale = pto.mul_s(
                                                    sij, pto.element(pto.data_type.DT_FP32, float(softmax_scale)))
                                                # (curNTile, curS2Tile) -> (curNTile, 1)
                                                tilda_mij = pto.row_max_single(sij_scale)
                                                tsub = pto.sub(sij_scale, tilda_mij)
                                                tilda_pij = pto.exp(tsub)
                                                tilda_pij_f16 = pto.cast(tilda_pij, dtype)
                                                # (nTileCur, s2TileCur) -> (nTileCur, 1)
                                                tilda_lij = pto.row_sum_single(tilda_pij)

                                                if pto.cond(pto.is_loop_begin(bn, 0)):
                                                    def inside_if_loop_begin():
                                                        nonlocal oi_update, li_update, mi_update
                                                        pto.set_cube_tile_shapes(
                                                            [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]],
                                                            [c2_tile[4], c2_tile[5]], True)
                                                        pto.set_semantic_label("paKvMm")
                                                        pto.set_matrix_size(
                                                            [tilda_pij_f16.shape[0], tilda_pij_f16.shape[1],
                                                             vj.shape[1]])
                                                        oi_tmp = pto.matmul(pto.data_type.DT_FP32, tilda_pij_f16,
                                                                            vj, False, False)
                                                        pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                        if pto.cond(pto.is_loop_end(bn, bn_per_batch)):
                                                            pto.set_semantic_label("paKvVec2")
                                                            # (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                                                            oi_update.move(pto.div(oi_tmp, tilda_lij))
                                                            pto.assemble(oi_update, oi_offset, pa_out)
                                                        else:
                                                            oi_update.move(oi_tmp)
                                                        li_update.move(tilda_lij)
                                                        mi_update.move(tilda_mij)
                                                    inside_if_loop_begin()
                                                else:
                                                    def inside_else_loop_begin():
                                                        nonlocal oi_update, li_update, mi_update
                                                        pto.set_semantic_label("paUpdateVec2")
                                                        oi = oi_update
                                                        li = li_update
                                                        mi = mi_update
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        mi_new = pto.maximum(mi, tilda_mij)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        t1 = pto.sub(mi, mi_new)
                                                        t2 = pto.exp(t1)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        t3 = pto.sub(tilda_mij, mi_new)
                                                        t4 = pto.exp(t3)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        t5 = pto.mul(t4, tilda_lij)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        t6 = pto.mul(t2, li)
                                                        # (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                                                        li_new = pto.add(t6, t5)
                                                        # (curNTile, dN), (curNTile, 1) -> (curNTile, dN)
                                                        q3 = pto.mul(oi, t2)
                                                        pto.set_cube_tile_shapes(
                                                            [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]],
                                                            [c2_tile[4], c2_tile[5]], True)
                                                        pto.set_semantic_label("paUpdateMM2")
                                                        pto.set_matrix_size(
                                                            [tilda_pij_f16.shape[0],
                                                             tilda_pij_f16.shape[1], vj.shape[1]])
                                                        q1 = pto.matmul(pto.data_type.DT_FP32, tilda_pij_f16, vj,
                                                                        False, False)
                                                        pto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                        # (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                                                        q2 = pto.mul(q1, t4)
                                                        # (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                                                        oi_tmp = pto.add(q3, q2)
                                                        if pto.cond(pto.is_loop_end(bn, bn_per_batch)):
                                                            # (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                                                            oi_update.move(pto.div(oi_tmp, li_new))
                                                            pto.assemble(oi_update, oi_offset, pa_out)
                                                        else:
                                                            oi_update.move(oi_tmp)
                                                        li_update.move(li_new)
                                                        mi_update.move(mi_new)
                                                    inside_else_loop_begin()
                                            inside_bn_loop(
                                                b_idx=b_idx,
                                                block_table=block_table,
                                                cur_seq=cur_seq,
                                                bn=bn,
                                                block_size=block_size,
                                                bn_per_batch=bn_per_batch)
                                    # } # LOOP("LOOP_L2_bn") ends
                                inside_n_idx_loop(b_idx, n_idx, bn_per_batch)
                        # } # LOOP("LOOP_L1_nIdx") ends
                    inside_b_idx_loop_pa(b_idx)
            # } # LOOP("LOOP_L0_bIdx_pa") ends

            ########## post ##########
            pto.set_config(COPYIN_THRESHOLD, 1 * NUM_1024 * NUM_1024)
            pto.set_config(SG_CYCLE_UPPER_BOUND, NUM_500000)
            pto.set_config(SG_PARALLEL_NUM, NUM_20)
            pto.set_config(CUBE_NBUFFER, 1)
            pto.set_operation_config("FORCE_COMBINE_AXIS", False)
            pto.set_config(CUBE_NBUFFER_MAP, {0: 4})
            pto.set_matrix_size([])
            # LOOP("PaPost", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(bLoop), {}, true) {
            with pto.loop_function("PaPost", "b_idx", pto.loop_range(b_loop), set(), True) as b_idx_loop:
                for b_idx in b_idx_loop:
                    def inside_b_idx_loop_post(**kwargs):
                        pa_out = kwargs.get("pa_out")
                        tile_b = kwargs.get("tile_b")
                        s = kwargs.get("s")
                        n = kwargs.get("n")
                        kv_lora_rank = kwargs.get("kv_lora_rank")
                        b_idx = kwargs.get("b_idx")
                        pto.set_semantic_label("Post")
                        post_in_unit = pto.view(pa_out, [tile_b * s * n, kv_lora_rank], [b_idx * tile_b * s * n, 0])

                        r1_res = pto.reshape(post_in_unit, [tile_b * s, n, kv_lora_rank])  # 128个
                        pto.set_vec_tile_shapes(min(NUM_32, tile_b * s), NUM_2, kv_lora_rank)
                        cast1 = pto.cast(r1_res, pto.data_type.DT_FP16)
                        t1_res = pto.transpose(cast1, [0, 1])  # (n, tileB * s, kvLoraRank)    # 128个

                        pto.set_cube_tile_shapes([min(NUM_32, tile_b * s), min(NUM_32, tile_b * s)],
                            [min(256, kv_lora_rank), min(512, kv_lora_rank)],
                            [v_head_dim, v_head_dim], True)  # raw tileB*1  512   128   # 128/4个
                        # (n, tileB, kvLoraRank) * (n, kvLoraRank, vHeadDim) -> (n, tileB, vHeadDim)
                        bmm_res = pto.batch_matmul(dtype, t1_res, weight_uv)

                        pto.set_vec_tile_shapes(NUM_4, min(NUM_32, tile_b * s), v_head_dim)  # raw (128, tileB*1, 128)
                        t3_res = pto.transpose(bmm_res, [0, 1])  # (n, tileB, vHeadDim) -> (tileB, n, vHeadDim) # 128个
                        # (tileB * s, n, vHeadDim) -> (tileB * s, n*vHeadDim)
                        r2_res = pto.reshape(t3_res, [tile_b * s, n * v_head_dim])

                        pto.set_vec_tile_shapes(1, n * v_head_dim)  # raw (tileB*1, 128*128)
                        quant_a = pto.quant(r2_res)
                        quantized_a = quant_a[0]  # (tileB * s, n*vHeadDim)
                        dequant_scale_a = quant_a[1]  # (tileB * s, 1)

                        pto.set_cube_tile_shapes(
                            [min(32, tile_b * s), min(32, tile_b * s)],
                            [min(512, n * v_head_dim), min(512, n * v_head_dim)],
                            [min(64, h), min(64, h)], True)  # raw  tileB*1  16k  7168
                        res = pto.matmul(pto.data_type.DT_INT32, quantized_a, weight_o)

                        pto.set_vec_tile_shapes(min(NUM_32, tile_b * s), min(NUM_32, h))  # raw (tileB*1, 7168)
                        res.move(pto.cast(res, pto.data_type.DT_FP32))
                        res.move(pto.mul(res, dequant_scale_a))   # (B*s, 1)
                        weight_o_scale_w_2dim = pto.reshape(weight_o_scale_w, [1, h])
                        res.move(pto.mul(res, weight_o_scale_w_2dim))   # (1, h)  # 224个
                        bmm5_res = pto.cast(res, pto.data_type.DT_FP16, pto.cast_mode.CAST_RINT)
                        post_out_tmp = pto.reshape(bmm5_res, [tile_b, s, h])

                        dyn_offset = [b_idx * tile_b, 0, 0]
                        pto.assemble(post_out_tmp, dyn_offset, post_out)
                    inside_b_idx_loop_post(
                        pa_out=pa_out,
                        tile_b=tile_b,
                        s=s,
                        n=n,
                        kv_lora_rank=kv_lora_rank,
                        b_idx=b_idx)
            # } # LOOP("PaPost") ends
        inside_main_function()


def test_dynamic_attention(params, pa_tile_config, is_quant=False, cache_mode="BNSD", use_pre_fetch=False):
    # b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim
    pto.set_host_config(KEY_ONLY_CODEGEN, True)

    b = params[0]
    s = params[1]
    s2 = params[2]
    n = params[3]
    h = params[4]
    q_lora_rank = params[5]
    qk_nope_head_dim = params[6]
    qk_rope_head_dim = params[7]
    kv_lora_rank = params[8]

    v_head_dim = params[9]
    block_size = params[10]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    atc_seqs = [s2] * b

    block_num = 0
    for seq in atc_seqs:
        block_num += ceil_div(seq, block_size)

    softmax_scale = 1.0 / math.sqrt(kv_lora_rank + qk_rope_head_dim)

    max_seq_all_batch = max(atc_seqs)
    max_block_num_per_batch = ceil_div(max_seq_all_batch, block_size)

    d_type = pto.data_type.DT_FP16
    d_type_quant_in = pto.data_type.DT_INT8 if is_quant else d_type

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
    if cache_mode != "BNSD":
        kv_cache_shape = [block_num, block_size, 1, kv_lora_rank]
        kr_cache_shape = [block_num, block_size, 1, qk_rope_head_dim]
    # pa
    block_table_shape = [b, 1, s2, qk_rope_head_dim]
    # output
    q_out_shape = [b, s, n, kv_lora_rank]
    q_rope_out_shape = [b, s, n, qk_rope_head_dim]
    kv_cache_out_shape = [b, 1, s2, kv_lora_rank]
    kr_cache_out_shape = [b, 1, s2, qk_rope_head_dim]
    fake_out_shape = [b, s, kv_lora_rank + qk_rope_head_dim]
    fake_out_shape1 = [n, b * s, qk_nope_head_dim]

    w_qb_scale_shape = []
    if is_quant:
        w_qb_scale_shape = [1, n * q_head_dim]

    weight_format = pto.tile_op_format.TILEOP_NZ if False else pto.tile_op_format.TILEOP_ND  # nz = false
    pa_format = pto.tile_op_format.TILEOP_NZ if cache_mode == "PA_NZ" else pto.tile_op_format.TILEOP_ND
    #mla_prolog
    x = pto.tensor(x_shape, d_type, "x")
    w_dq = pto.tensor(w_qa_shape, d_type, "wDq", weight_format)
    w_uq_qr = pto.tensor(w_qb_shape, d_type_quant_in, "wUqQr", weight_format)
    if use_pre_fetch:
        w_dq.set_cache_policy(pto.cache_policy.PREFETCH, True)
        w_uq_qr.set_cache_policy(pto.cache_policy.PREFETCH, True)

    w_dkv_kr = pto.tensor(w_kv_a_shape, d_type, "wDkvKr", weight_format)
    w_uk = pto.tensor(w_kv_b_k_shape, d_type, "wUk", weight_format)
    gamma_cq = pto.tensor(gamma_cq_shape, d_type, "gamma_cq")
    gamma_ckv = pto.tensor(gamma_ckv_shape, d_type, "gamma_ckv")
    cos = pto.tensor(cos_shape, d_type, "cos")
    sin = pto.tensor(cos_shape, d_type, "sin")
    kv_len = pto.tensor(kv_len_shape, pto.data_type.DT_INT64, "kv_len")
    kv_cache = pto.tensor(kv_cache_shape, d_type, "kv_cache", pa_format)
    kr_cache = pto.tensor(kr_cache_shape, d_type, "kr_cache", pa_format)

    output_q = pto.tensor([b * s * n, kv_lora_rank], d_type, "output_q")
    output_q_rope = pto.tensor([b * s * n, qk_rope_head_dim], d_type, "output_q_rope")
    output_kv_cache = pto.tensor([b * 1 * s2, kv_lora_rank], d_type, "output_kv_cache", pa_format)
    output_kr_cache = pto.tensor([b * 1 * s2, qk_rope_head_dim], d_type, "output_kr_cache", pa_format)

    fake_out = pto.tensor([b * s, n, qk_nope_head_dim], d_type, "fakeOut")
    fake_out1 = pto.tensor([n, b * s, qk_nope_head_dim], d_type, "fakeOut1")
    # pa
    block_table = pto.tensor([b, max_block_num_per_batch], pto.data_type.DT_INT32, "blockTable")
    act_seqs = pto.tensor([b], pto.data_type.DT_INT32, "actSeqs")
    #out mla
    pa_out = pto.tensor([b * n * s, kv_lora_rank], pto.data_type.DT_FP32, "paOut")
    #post
    weight_uv = pto.tensor([n, kv_lora_rank, v_head_dim], d_type, "weightUV")
    weight_o = pto.tensor([n * v_head_dim, h], pto.data_type.DT_INT8, "weightO")
    weight_o_scale_w = pto.tensor([1, h], pto.data_type.DT_FP32, "weightOScaleW")
    # output
    post_out = pto.tensor([b, s, h], d_type, "postOut")

    tile_b = b
    rope_config = pto.rope_tile_shape_config_new()
    rope_config.three_dims_tile_shape = [tile_b, 1, 64]
    rope_config.four_dims_tile_shape_q = [tile_b, 1, 1, 64]
    rope_config.four_dims_tile_shape_k = [tile_b, 1, 1, 64]
    rope_config.five_dims_tile_shape = [tile_b, 1, 1, 32, 2]

    quant_inputs = MlaQuantInputs()
    # The Attention is implemented in another cpp and dont need to translate in this file
    attention(
        token_x=x,
        w_dq=w_dq,
        w_uq_qr=w_uq_qr,
        w_uk=w_uk,
        w_dkv_kr=w_dkv_kr,
        gamma_cq=gamma_cq,
        gamma_ckv=gamma_ckv,
        sin=sin,
        cos=cos,
        cache_index=kv_len, # ?
        kv_cache=kv_cache,
        kr_cache=kr_cache,
        q_nope_out=output_q,
        q_rope_out=output_q_rope,
        kv_cache_out=output_kv_cache,
        kr_cache_out=output_kr_cache,
        quant_inputs=quant_inputs,
        rope_config=rope_config,
        block_table=block_table,
        act_seqs=act_seqs,
        pa_out=pa_out,
        block_size=block_size,
        softmax_scale=softmax_scale,
        pa_tile_config=pa_tile_config,
        weight_uv=weight_uv,
        weight_o=weight_o,
        weight_o_scale_w=weight_o_scale_w,
        post_out=post_out,
        epsilon_cq=1e-5,
        epsilon_ckv=1e-5,
        cache_mode=cache_mode)


def main():
    b = 4
    s = 1
    s2 = 256
    h = 7168
    n = 32
    q_lora_rank = 1536
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    block_size = 256
    params = [b, s, s2, n, h, q_lora_rank, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, block_size]

    split_reduce_last_dim = False
    split_k = False
    nz = False

    tile_config = pto.pa_tile_shape_config()
    n_tile = 32
    tile_config.head_num_q_tile = n_tile
    tile_config.v0_tile_shape = [n_tile, 64]
    tile_config.c1_tile_shape = [n_tile, n_tile, 64, 64, 128, 128]
    tile_config.v1_tile_shape = [n_tile, 64]
    tile_config.c2_tile_shape = [n_tile, n_tile, 64, 64, 128, 128]
    tile_config.v2_tile_shape = [n_tile, 64]

    test_dynamic_attention(params, tile_config, False)
    logging.info("finished")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    main()
