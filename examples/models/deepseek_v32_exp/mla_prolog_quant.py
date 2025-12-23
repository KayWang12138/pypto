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
from typing import List
import pypto

from pypto import pypto_impl
from pypto.operation import op_wrapper


@op_wrapper
def scalar_div(input, other, is_reserve=False):
    return pypto_impl.ScalarDivS(input, pypto_impl.Element(input.dtype, other), is_reserve)


@dataclass
class IndexerPrologQuantConfigs:
    q_linear: List[int]
    q_hd: List[int]
    k_linear: List[int]
    w_linear: List[int]
    unroll_list: List[int]

    cube_l1_reuse_setting: dict[int, int]
    mg_copyin_upper_bound: int
    pg_upper_bound: int
    block_size: int
    t_sub_tile: int
    chunk_size: int
    vec_nbuffer_mode: int


@dataclass
class MlaTileConfig:
    def __init__(self):
        self.tile_b = 8
        self.tile_s = 1
        self.tile_bs = 8
        self.m_tile = 16
        self.mv_tile = 16
        self.pre_quant_cube_tile = [16, 16, 256, 256, 128, 128]
        self.unroll_list = [32, 16, 8, 4, 2, 1]
        self.q_vec_tile0 = 16
        self.q_vec_tile1 = 16
        self.k_vec_tile0 = 16
        self.k_vec_tile1 = 16
        self.cube_l1_reuse_mode = 4
        self.cube_l1_reuse_setting = {}
        self.mg_copyin_upper_bound = 2 * 1024 * 1024
        self.pg_upper_bound = 8192
        self.vec_nbuffer_mode = 1
        self.cube_nbuffer_setting = {3: 4}
        self.dynamic_unaligned_enable = False

@dataclass
class MlaQuantInputs:
    dequant_scale_x: pypto.tensor = None
    dequant_scale_w_dq: pypto.tensor = None
    dequant_scale_w_uq_qr: pypto.tensor = None
    dequant_scale_w_dkv_kr: pypto.tensor = None
    quant_scale_ckv: pypto.tensor = None
    quant_scale_ckr: pypto.tensor = None
    smooth_scales_cq: pypto.tensor = None


@dataclass
class RopeTileShapeConfig:
    two_dim: List[int]
    three_dim: List[int]
    four_dim: List[int]


def k_nope_quant(x):
    x_fp32 = pypto.cast(x, pypto.DataType.DT_FP32)
    abs_res = pypto.abs(x_fp32)
    max_value = pypto.amax(abs_res, -1, keepdim=True)
    scale_quant = pypto.div(pypto.full(max_value.shape, 127.0, pypto.DataType.DT_FP32), max_value)
    out_fp32 = pypto.mul(x_fp32, scale_quant)
    out_int32 = pypto.cast(out_fp32, pypto.DataType.DT_INT32, pypto.CastMode.CAST_RINT)
    out_half = pypto.cast(out_int32, pypto.DataType.DT_FP16, pypto.CastMode.CAST_ROUND)
    out_int8 = pypto.cast(out_half, pypto.DataType.DT_INT8, pypto.CastMode.CAST_TRUNC)

    scale_de_quant = pypto.div(pypto.full(scale_quant.shape, 1.0, pypto.DataType.DT_FP32), scale_quant)
    return out_int8, scale_de_quant


def rms_norm(input_tensor, gamma, epsilon):
    input_fp32 = pypto.cast(input_tensor, pypto.DataType.DT_FP32)
    dim = len(input_tensor.shape)
    shape = [1] * dim
    shape[dim - 1] = gamma.shape[0]
    gamma_cast = pypto.reshape(gamma, shape)
    gamma_fp32 = pypto.cast(gamma_cast, pypto.DataType.DT_FP32)
    y = pypto.mul(input_fp32, input_fp32)
    y = pypto.mul(y, 1.0 / input_tensor.shape[dim - 1])
    y = pypto.sum(y, -1, keepdim=True)
    y = pypto.add(y, epsilon)
    y = pypto.sqrt(y)
    ones_vector = pypto.full(y.shape, 1.0, pypto.DataType.DT_FP32)
    y = pypto.div(ones_vector, y)
    y = pypto.mul(input_fp32, y)
    y = pypto.mul(gamma_fp32, y)
    y = pypto.cast(y, input_tensor.dtype)
    return y


def quant(input_tensor, is_symmetry=False, has_smooth_factor=False, smooth_factor=None):
    input_fp32 = pypto.cast(input_tensor, pypto.DataType.DT_FP32)
    if has_smooth_factor:
        input_fp32 = pypto.mul(input_fp32, smooth_factor)
    if is_symmetry:
        abs_res = pypto.abs(input_fp32)
        max_value = pypto.amax(abs_res, -1, keepdim=True)
        scale_quant = scalar_div(max_value, 127.0, True)
        out_fp32 = pypto.mul(input_fp32, scale_quant)
        out_int32 = pypto.cast(out_fp32, pypto.DataType.DT_INT32, pypto.CastMode.CAST_RINT)
        out_half = pypto.cast(out_int32, pypto.DataType.DT_FP16, pypto.CastMode.CAST_ROUND)
        out_int8 = pypto.cast(out_half, pypto.DataType.DT_INT8, pypto.CastMode.CAST_TRUNC)
        scale_de_quant = scalar_div(scale_quant, 1.0, True)
        return out_int8, scale_de_quant
    else:
        max_value = pypto.amax(input_fp32, -1, keepdim=True)
        min_value = pypto.amin(input_fp32, -1, keepdim=True)
        scale_de_quant = pypto.max(pypto.div(pypto.sub(max_value, min_value), 255.0), 1e-12)
        offset = pypto.sub(127.0, pypto.div(max_value, scale_de_quant))
        scale_quant = scalar_div(max_value, 1.0, True)
        out_fp32 = pypto.mul(input_fp32, scale_quant)
        out_int32 = pypto.cast(out_fp32, pypto.DataType.DT_INT32, pypto.CastMode.CAST_RINT)
        out_half = pypto.cast(out_int32, pypto.DataType.DT_FP16, pypto.CastMode.CAST_ROUND)
        out_int8 = pypto.cast(out_half, pypto.DataType.DT_INT8, pypto.CastMode.CAST_TRUNC)
        return out_int8, scale_de_quant


def dequant(dtype, input_tensor, scale, w_scale):
    dequant_res = pypto.cast(input_tensor, pypto.DataType.DT_FP32)
    dequant_res = dequant_res * scale
    dequant_res = dequant_res * w_scale
    return pypto.cast(dequant_res, dtype)


def rotate_half(input_tensor):
    shape = input_tensor.shape
    shape_size = len(shape)
    assert shape_size >= 1, "rope rotate_half input dim less than 1"
    assert shape[shape_size - 1] % 2 == 0, "rope rotate_half last dim shape is even"

    new_shape = list(shape)
    new_shape[shape_size - 1] //= 2

    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = new_shape[shape_size - 1]

    x1 = pypto.view(input_tensor, new_shape, offset1)
    x2 = pypto.view(input_tensor, new_shape, offset2)

    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rope_v2(x, cos, sin, tile_config):
    assert len(x.shape) == 2 and len(cos.shape) == 2 and len(sin.shape) == 2
    seq_size = x.shape[0]
    d_r = x.shape[1]
    x_dtype = x.dtype

    pypto.set_vec_tile_shapes(tile_config.two_dim[0], tile_config.two_dim[1])
    cast_x = pypto.cast(x, pypto.DataType.DT_FP32)
    cast_cos = pypto.cast(cos, pypto.DataType.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DataType.DT_FP32)

    pypto.set_vec_tile_shapes(*tile_config.three_dim)
    x_view = pypto.reshape(cast_x, [seq_size, d_r // 2, 2])
    x_trans = pypto.transpose(x_view, 1, 2)
    x_re_second = pypto.reshape(x_trans, [seq_size, d_r])

    pypto.set_vec_tile_shapes(tile_config.two_dim[0], tile_config.two_dim[1])
    x_embded = x_re_second * cast_cos + rotate_half(x_re_second) * cast_sin

    return pypto.cast(x_embded, x.dtype)


def rope_3d_v2(x, cos, sin):
    assert len(x.shape) == 3 and len(cos.shape) == 2 and len(sin.shape) == 2

    pypto.set_vec_tile_shapes(1, 64)
    cast_cos = pypto.cast(cos, pypto.DataType.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DataType.DT_FP32)

    pypto.set_vec_tile_shapes(1, 64, 64)
    cast_x = pypto.cast(x, pypto.DataType.DT_FP32)
    cast_cos = pypto.reshape(cast_cos, [x.shape[0], 1, x.shape[2]])
    cast_sin = pypto.reshape(cast_sin, [x.shape[0], 1, x.shape[2]])

    pypto.set_vec_tile_shapes(1, 64, 128, 128)
    x_view = pypto.reshape(cast_x, [x.shape[0], x.shape[1], x.shape[2] // 2, 2])
    x_trans = pypto.transpose(x_view, 2, 3)
    x_re_second = pypto.reshape(x_trans, x.shape)
    x_embed = x_re_second * cast_cos + rotate_half(x_re_second) * cast_sin

    return pypto.cast(x_embed, x.dtype)


def pre_compute_2d(token_x, w_dq, w_uq_qr, w_dkv_kr, gamma_cq, epsilon_cq, quant_inputs, tile_config):
    dequant_scale_w_dq = quant_inputs.dequant_scale_w_dq
    dequant_scale_w_dkv_kr = quant_inputs.dequant_scale_w_dkv_kr
    dequant_scale_w_uq_qr = quant_inputs.dequant_scale_w_uq_qr

    is_quant_a = (dequant_scale_w_dq is not None) and (dequant_scale_w_dkv_kr is not None)
    is_quant_b = dequant_scale_w_uq_qr is not None

    smooth_scales_cq = quant_inputs.smooth_scales_cq
    is_smooth = smooth_scales_cq is not None

    bs = token_x.shape[0]
    q_lora_rank = w_dq.shape[1]

    dtype = token_x.dtype
    dtype_quant_a_out = pypto.DataType.DT_INT32 if is_quant_a else dtype
    dtype_quant_b_out = pypto.DataType.DT_INT32 if is_quant_b else dtype
    qkv_pre_res = []

    pypto.set_semantic_label("pre_reshape")

    mv = tile_config.mv_tile

    if is_quant_a:
        pypto.set_vec_tile_shapes(mv, q_lora_rank)
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                                   [256, 256], [256, 256])
        pypto.set_semantic_label("Quant_x")
        quant_res = quant(token_x)
        input_quant = quant_res[0]
        input_quant_scale = quant_res[1]
        pypto.set_semantic_label("QuantMatmul_qa")
        q_a_proj = pypto.matmul(input_quant, w_dq, dtype_quant_a_out)
        pypto.set_semantic_label("Dequant_qa")
        q_a_proj[:] = dequant(dtype, q_a_proj, input_quant_scale, dequant_scale_w_dq)
    else:
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                                   [tile_config.pre_quant_cube_tile[2], tile_config.pre_quant_cube_tile[3]],
                                   [tile_config.pre_quant_cube_tile[4], tile_config.pre_quant_cube_tile[5]])
        pypto.set_semantic_label("Matmul_qa")
        q_a_proj = pypto.matmul(token_x, w_dq, dtype)

    pypto.set_vec_tile_shapes(mv, q_lora_rank)
    pypto.set_semantic_label("RmsNorm_qa")
    norm_res = rms_norm(q_a_proj, gamma_cq, epsilon_cq)

    if is_quant_b:
        pypto.set_vec_tile_shapes(mv, q_lora_rank)
        pypto.set_semantic_label("Quant_qMnRes")
        if is_smooth:
            quant_res = quant(norm_res, True, True, smooth_scales_cq)
        else:
            quant_res = quant(norm_res, True, False)
        norm_quant = quant_res[0]
        norm_quant_scale = quant_res[1]
        pypto.set_semantic_label("QuantMatmul_qb")
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                                   [256, 256], [256, 256])
        q_b_proj_tmp = pypto.matmul(norm_quant, w_uq_qr, dtype_quant_b_out)
        pypto.set_semantic_label("Dequant_qb")
        q_b_proj = dequant(dtype, q_b_proj_tmp, norm_quant_scale, dequant_scale_w_uq_qr)
    else:
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                                   [tile_config.pre_quant_cube_tile[2], tile_config.pre_quant_cube_tile[3]],
                                   [tile_config.pre_quant_cube_tile[4], tile_config.pre_quant_cube_tile[5]])
        pypto.set_semantic_label("Matmul_qb")
        q_b_proj = pypto.matmul(norm_res, w_uq_qr, dtype)

    qkv_pre_res.append(q_b_proj)

    ####### kv ##########
    if is_quant_a:
        pypto.set_vec_tile_shapes(mv, q_lora_rank)
        pypto.set_cube_tile_shapes(tile_config.m_tile, [256, 256], [256, 256])
        pypto.set_semantic_label("QuantMatmul_kva")
        compressed_kv = pypto.matmul(input_quant, w_dkv_kr, dtype_quant_a_out)
        pypto.set_semantic_label("Dequant_kva")
        compressed_kv[:] = dequant(dtype, compressed_kv, input_quant_scale, dequant_scale_w_dkv_kr)
    else:
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                                   [tile_config.pre_quant_cube_tile[2], tile_config.pre_quant_cube_tile[3]],
                                   [tile_config.pre_quant_cube_tile[4], tile_config.pre_quant_cube_tile[5]])
        pypto.set_semantic_label("Matmul_kva")
        compressed_kv = pypto.matmul(token_x, w_dkv_kr, dtype)

    qkv_pre_res.append(compressed_kv)
    if is_quant_b:
        qkv_pre_res.append(norm_quant)
        qkv_pre_res.append(norm_quant_scale)
    else:
        qkv_pre_res.append(norm_res)
    return qkv_pre_res


def mla_prolog_quant_compute(token_x, w_dq, w_uq_qr, dequant_scale, w_uk,
                             w_dkv_kr, gamma_cq, gamma_ckv, cos,
                             sin, cache_index, kv_cache, kr_cache, k_scale_cache,
                             q_norm_out, q_norm_scale_out, query_nope_out,
                             query_rope_out, kv_cache_out,
                             kr_cache_out, k_scale_cache_out, epsilon_cq,
                             epsilon_ckv, cache_mode, tile_config, rope_cfg):

    assert len(token_x.shape) == 2 and len(w_uk.shape) == 3 and len(sin.shape) == 2
    assert len(kv_cache.shape) == 4 and len(kr_cache.shape) == 4
    assert cache_mode in ["PA_BSND", "PA_NZ"]

    dtype = token_x.dtype
    h = token_x.shape[1]
    n1 = w_uk.shape[0]
    q_lora_rank = w_dq.shape[1]
    qk_nope_head_dim = w_uk.shape[1]
    kv_lora_rank = w_uk.shape[2]
    qk_rope_head_dim = sin.shape[1]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    assert qk_nope_head_dim == 128 or qk_rope_head_dim == 64

    tile_bs = tile_config.tile_bs

    t = token_x.shape[0]
    bs_loop = (t + tile_bs - 1) // tile_bs

    quant_inputs = MlaQuantInputs()

    k_cache_index_2d = pypto.reshape(cache_index, [t, 1], inplace=True)
    if dequant_scale is not None:
        dequant_scale_wuqr_reshape = pypto.reshape(dequant_scale, [1, n1 * q_head_dim], inplace=True)
        quant_inputs.dequant_scale_w_uq_qr = dequant_scale_wuqr_reshape

    unroll_list = tile_config.unroll_list
    for bs_offset, unrollLength in pypto.loop_unroll(0, t, 1, name="MLA_BS_LOOP", idx_name="bs_offset",
                                                unroll_list=unroll_list, ):
        tile_bs = unrollLength
        output_offset = [bs_offset, 0, 0]

        pypto.set_vec_tile_shapes(tile_bs, 128)
        x_view = pypto.view(token_x, [tile_bs, h], [bs_offset, 0])
        q_kv = pre_compute_2d(x_view, w_dq, w_uq_qr, w_dkv_kr, gamma_cq, epsilon_cq, quant_inputs, tile_config)
        q = q_kv[0]
        kv_tmp = q_kv[1]

        ############# q_norm #############
        pypto.set_semantic_label("Assemble_qNorm")
        q_norm = q_kv[2]
        pypto.set_vec_tile_shapes(tile_bs, q_lora_rank)
        pypto.assemble(q_norm, [bs_offset, 0], q_norm_out)
        q_norm_scale = q_kv[3]
        pypto.set_vec_tile_shapes(tile_bs, 1)
        pypto.assemble(q_norm_scale, [bs_offset, 0], q_norm_scale_out)

        ########### q ##############
        q_tmp = pypto.reshape(q, [tile_bs, n1, q_head_dim])
        pypto.set_semantic_label("Prepare_qNope")
        q_nope = pypto.view(q_tmp, [tile_bs, n1, qk_nope_head_dim], [0, 0, 0])
        tile_shape = [min(16, tile_bs), 32, qk_nope_head_dim]
        pypto.set_vec_tile_shapes(*tile_shape)
        q_nope_trans = pypto.transpose(q_nope, 0, 1)

        m = tile_config.m_tile
        pypto.set_semantic_label("Matmul_qNope_wUk")
        pypto.set_cube_tile_shapes([m, m], [128, 128], [128, 128])
        q_nope_new = pypto.matmul(q_nope_trans, w_uk, dtype)

        tile_shape = [1, min(32, tile_bs), kv_lora_rank]
        pypto.set_vec_tile_shapes(*tile_shape)
        q_nope_new_trans = pypto.transpose(q_nope_new, 0, 1)

        pypto.set_semantic_label("Assemble_queryOut")
        pypto.set_vec_tile_shapes(tile_config.q_vec_tile0, tile_config.q_vec_tile1, 128)
        pypto.assemble(q_nope_new_trans, output_offset, query_nope_out)
        
        if tile_bs >= 128:
            pypto.set_vec_tile_shapes(tile_config.q_vec_tile0, tile_config.q_vec_tile1, 64)
        q_pe_view = pypto.view(q_tmp, [tile_bs, n1, qk_rope_head_dim], [0, 0, qk_nope_head_dim])
        cos_2d_view = pypto.view(cos, [tile_bs, qk_rope_head_dim], [bs_offset, 0])
        sin_2d_view = pypto.view(sin, [tile_bs, qk_rope_head_dim], [bs_offset, 0])
        pypto.set_semantic_label("Rope_qRope")
        q_rope_view = rope_3d_v2(q_pe_view, cos_2d_view, sin_2d_view)
        pypto.set_semantic_label("Assemble_qRope")
        pypto.set_vec_tile_shapes(tile_config.q_vec_tile0, tile_config.q_vec_tile1, 64)
        pypto.assemble(q_rope_view, output_offset, query_rope_out)

        ########### RoPE #################
        pypto.set_vec_tile_shapes(tile_config.k_vec_tile0, tile_config.k_vec_tile1)
        pypto.set_semantic_label("RotaryPosEmb")
        k_pe_view = pypto.view(kv_tmp, [tile_bs, qk_rope_head_dim], [0, kv_lora_rank])
        k_rope_2d = rope_v2(k_pe_view, cos_2d_view, sin_2d_view, rope_cfg)

        ############### kNope ##############

        compressed_kv = pypto.view(kv_tmp, [tile_bs, kv_lora_rank], [0, 0])
        pypto.set_semantic_label("RmsNorm_compressedkv")
        pypto.set_vec_tile_shapes(tile_config.k_vec_tile0, tile_config.k_vec_tile1)
        k_nope = rms_norm(compressed_kv, gamma_ckv, epsilon_ckv)

        ########### kNope Quant ############
        pypto.set_semantic_label("Quant_knope")
        pypto.set_vec_tile_shapes(32, kv_lora_rank)
        k_nope_split = pypto.reshape(k_nope, [tile_bs, 4, kv_lora_rank // 4])
        pypto.set_vec_tile_shapes(32, 4, kv_lora_rank // 4)
        k_nope_quant_res = k_nope_quant(k_nope_split)
        k_nope_quant_tensor = k_nope_quant_res[0]
        k_nope_scale = k_nope_quant_res[1]

        pypto.set_vec_tile_shapes(32, 4, kv_lora_rank // 4)
        k_nope_2d = pypto.reshape(k_nope_quant_tensor, [tile_bs, kv_lora_rank])
        k_scale_2d = pypto.reshape(k_nope_scale, [tile_bs, 4])

        k_rope_4d = pypto.reshape(k_rope_2d, [tile_bs, 1, 1, qk_rope_head_dim], inplace=True)
        k_nope_4d = pypto.reshape(k_nope_2d, [tile_bs, 1, 1, kv_lora_rank], inplace=True)
        k_scale_4d = pypto.reshape(k_scale_2d, [tile_bs, 1, 1, 4], inplace=True)
        index = pypto.view(k_cache_index_2d, [tile_bs, 1], [bs_offset, 0])
        pypto.set_semantic_label("ScatterUpdate_krCache")
        pypto.set_vec_tile_shapes(32, 1, 1, qk_rope_head_dim)
        kr_cache_out[:] = pypto.scatter_update(kr_cache, -2, index, k_rope_4d)
        pypto.set_semantic_label("ScatterUpdate_kvCache")
        pypto.set_vec_tile_shapes(32, 1, 1, kv_lora_rank)
        kv_cache_out[:] = pypto.scatter_update(kv_cache, -2, index, k_nope_4d)
        pypto.set_semantic_label("ScatterUpdate_kScaleCache")
        pypto.set_vec_tile_shapes(32, 1, 1, 4)
        k_scale_cache_out[:] = pypto.scatter_update(k_scale_cache, -2, index, k_scale_4d)


@pypto.jit(
    pass_options={"vec_nbuffer_mode": 1,
                "cube_l1_reuse_mode": 4,
                "cube_l1_reuse_setting": {3: 4},
                "mg_copyin_upper_bound": 2 * 1024 * 1024},
    host_options={"only_codegen": True}
)
def mla_prolog_quant_p(
                       token_x, w_dq, w_uq_qr, dequant_scale, w_uk,
                       w_dkv_kr, gamma_cq, gamma_ckv, cos,
                       sin, cache_index, kv_cache, kr_cache, k_scale_cache,
                       q_norm_out, q_norm_scale_out, query_nope_out,
                       query_rope_out, kv_cache_out,
                       kr_cache_out, k_scale_cache_out, epsilon_cq,
                       epsilon_ckv, cache_mode, tile_config, rope_cfg):
    '''
    prefill
    '''
    mla_prolog_quant_compute(
                             token_x, w_dq, w_uq_qr, dequant_scale, w_uk,
                             w_dkv_kr, gamma_cq, gamma_ckv, cos,
                             sin, cache_index, kv_cache, kr_cache, k_scale_cache,
                             q_norm_out, q_norm_scale_out, query_nope_out,
                             query_rope_out, kv_cache_out,
                             kr_cache_out, k_scale_cache_out, epsilon_cq,
                             epsilon_ckv, cache_mode, tile_config, rope_cfg
    )


@pypto.jit(
    pass_options={"vec_nbuffer_mode": 1,
                "cube_l1_reuse_mode": 4,
                "cube_l1_reuse_setting": {3: 4},
                "mg_copyin_upper_bound": 2 * 1024 * 1024},
    host_options={"only_codegen": True}
)
def mla_prolog_quant_d(
                       token_x, w_dq, w_uq_qr, dequant_scale, w_uk,
                       w_dkv_kr, gamma_cq, gamma_ckv, cos,
                       sin, cache_index, kv_cache, kr_cache, k_scale_cache,
                       q_norm_out, q_norm_scale_out, query_nope_out,
                       query_rope_out, kv_cache_out,
                       kr_cache_out, k_scale_cache_out, epsilon_cq,
                       epsilon_ckv, cache_mode, tile_config, rope_cfg):
    '''
    decode
    '''
    pypto.set_host_options(only_codegen=True)
    mla_prolog_quant_compute(
                             token_x, w_dq, w_uq_qr, dequant_scale, w_uk,
                             w_dkv_kr, gamma_cq, gamma_ckv, cos,
                             sin, cache_index, kv_cache, kr_cache, k_scale_cache,
                             q_norm_out, q_norm_scale_out, query_nope_out,
                             query_rope_out, kv_cache_out,
                             kr_cache_out, k_scale_cache_out, epsilon_cq,
                             epsilon_ckv, cache_mode, tile_config, rope_cfg
    )
