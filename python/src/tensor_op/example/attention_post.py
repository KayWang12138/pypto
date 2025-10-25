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
import pto


SHAPE_DIM4 = 4
SHAPE_DIM3 = 3
SHAPE_DIM2 = 2


@dataclass
class PostTileConfig:
    tile_b: int = 8
    tile_s: int = 1


@dataclass
class PostTensors:
    weight_u_v: pto.tensor = pto.tensor()
    weight_o: pto.tensor = pto.tensor()
    weight_uv_scale: pto.tensor = pto.tensor()
    smooth_scales_w_uv: pto.tensor = pto.tensor()
    weight_o_scale: pto.tensor = pto.tensor()
    smooth_scales_wo: pto.tensor = pto.tensor()


def post_compute(
        input_tensor: pto.tensor,
        post_tensors: PostTensors,
        tile_config: PostTileConfig,
        post_out: pto.tensor,
):

    if not (len(input_tensor.shape) == SHAPE_DIM4 and
        len(post_tensors.weight_u_v.shape) == SHAPE_DIM3 and
        len(post_tensors.weight_o.shape) == SHAPE_DIM2):
        raise ValueError("Tensor shapes do not match the expected dimensions")
    dtype = input_tensor.dtype
    n = post_tensors.weight_u_v.shape[0]
    kv_lora_rank = post_tensors.weight_u_v.shape[1]
    v_head_dim = post_tensors.weight_u_v.shape[2]
    h = post_tensors.weight_o.shape[1]

    tile_b = tile_config.tile_b
    tile_s = tile_config.tile_s
    tile_b_s = tile_b * tile_s

    is_quant_w_uv = (post_tensors.weight_uv_scale.has_storage()
                        if post_tensors.weight_uv_scale is not None else False)
    is_smooth_w_uv = (post_tensors.smooth_scales_w_uv.has_storage()
                        if post_tensors.smooth_scales_w_uv is not None else False)
    is_quant_wo = (post_tensors.weight_o_scale.has_storage()
                        if post_tensors.weight_o_scale is not None else False)
    is_smooth_wo = (post_tensors.smooth_scales_wo.has_storage()
                        if post_tensors.smooth_scales_wo is not None else False)

    b = input_tensor.shape[0];  # SymbolicScalar b = GetInputShapeDim(input_tensor, 0);
    s = input_tensor.shape[1];  # SymbolicScalar s = GetInputShapeDim(input_tensor, 1);
    b_loop = pto.symbolic_scalar(b // tile_b)
    s_loop = pto.symbolic_scalar(s // tile_s)

    for b_idx in pto.loop(0, b_loop, 1, name="POST_LOOP_L0_bIdx", idx_name="b_idx", submit_before_loop=True):
        def inside_b_idx_loop_attention_post(b_idx):
            nonlocal tile_s, tile_b, kv_lora_rank, tile_b_s, v_head_dim, n
            b_offset = b_idx * tile_b
            for s_idx in pto.loop(0, s_loop, 1, name="POST_LOOP_L1_sIdx", idx_name="s_idx"):
                def inside_s_idx_loop_attention_post(s_idx):
                    nonlocal tile_s, tile_b, kv_lora_rank, tile_b_s, v_head_dim, b_offset, n
                    s_offset = s_idx * tile_s
                    out_offset = [b_offset, s_offset, pto.symbolic_scalar(0)]

                    pto.set_vec_tile_shapes(*[1, 1, 32, kv_lora_rank])
                    input_view = pto.view(input_tensor, [tile_b, tile_s, n, kv_lora_rank], [b_offset, s_offset, 0, 0])
                    pto.set_semantic_label("postReshape1")
                    input_res = pto.reshape(input_view, [tile_b_s, n, kv_lora_rank])
                    pto.set_vec_tile_shapes(*[min(32, tile_b_s), 2, kv_lora_rank])
                    pto.set_semantic_label("postTranspose1")
                    input_trans = pto.transpose(input_res, [0, 1]);  # [n,tileBS,kvLoraRank]

                    pto.set_semantic_label("postBmm")
                    c0 = 16
                    m = (min(32, tile_b_s) + c0 - 1) // c0 * c0
                    pto.set_cube_tile_shapes([m, m], [min(256, kv_lora_rank),
                                        min(512, kv_lora_rank)], [v_head_dim, v_head_dim], True)

                    bmm = pto.tensor()
                    if is_quant_w_uv:
                        def inside_of_if_is_quant_w_uv():
                            nonlocal bmm, kv_lora_rank
                            pto.set_semantic_label("postQuantWUv")
                            pto.set_vec_tile_shapes(*[1, 1, min(512, kv_lora_rank)])
                            quant_res = None
                            if is_smooth_w_uv:
                                quant_res = pto.quant(input_trans, True, True, post_tensors.smooth_scales_w_uv)
                            else:
                                quant_res = pto.quant(input_trans, True, False)
                            input_trans_quant = quant_res[0]
                            scale_dequant = quant_res[1]

                            mm = pto.matmul(input_trans_quant, post_tensors.weight_u_v, pto.DT_INT32)

                            pto.set_semantic_label("postDequantWUv")
                            pto.set_vec_tile_shapes(*[1, min(16, tile_b_s), min(32, v_head_dim)])
                            res = pto.cast(mm, pto.DT_FP32)
                            res[:] = pto.mul(res, scale_dequant)
                            res[:] = pto.mul(res, post_tensors.weight_uv_scale)
                            bmm[:] = pto.cast(res, dtype, pto.cast_mode.CAST_RINT)
                        inside_of_if_is_quant_w_uv()
                    else:
                        bmm[:] = pto.matmul(input_trans, post_tensors.weight_u_v, dtype)

                    pto.set_semantic_label("postTranspose2")
                    pto.set_vec_tile_shapes(*[4, min(32, tile_b_s), v_head_dim])
                    bmm_trans = pto.transpose(bmm, [0, 1])
                    pto.set_semantic_label("postReshape2")
                    bmm_res = pto.reshape(bmm_trans, [tile_b_s, n * v_head_dim])

                    mm_res: pto.tensor = pto.tensor()
                    pto.set_cube_tile_shapes([m, m], [min(512, n * v_head_dim), min(512, n * v_head_dim)],
                                                [min(64, h), min(64, h)], True)
                    if is_quant_wo:
                        def inside_of_if_is_quant_wo():
                            pto.set_semantic_label("postQuantWo")
                            pto.set_vec_tile_shapes(*[1, n * v_head_dim])
                            quant_res = None
                            if is_smooth_wo:
                                quant_res = pto.quant(bmm_res, True, True, post_tensors.smooth_scales_wo)
                            else:
                                quant_res = pto.quant(bmm_res, True, False)
                            bmm_res_quant = quant_res[0]
                            scale_dequant = quant_res[1]

                            pto.set_semantic_label("postMm")
                            mm = pto.matmul(bmm_res_quant, post_tensors.weight_o, pto.DT_INT32)

                            pto.set_semantic_label("postDequantWo")
                            pto.set_vec_tile_shapes(*[min(32, tile_b_s), min(32, h)])
                            res = pto.cast(mm, pto.DT_FP32)
                            res[:] = pto.mul(res, scale_dequant);   # [tileBS, h] * [tileBS, 1] -> [tileBS, h]
                            res[:] = pto.mul(res, post_tensors.weight_o_scale)   # [tileBS, h] * [1, h] -> [tileBS, h]
                            mm_res[:] = pto.cast(res, dtype, pto.cast_mode.CAST_RINT)
                        inside_of_if_is_quant_wo()
                    else:
                        mm_res[:] = pto.matmul(bmm_res, post_tensors.weight_o, dtype)
                    pto.set_semantic_label("postReshape3")
                    post_out_view = pto.reshape(mm_res, [tile_b, tile_s, h])
                    pto.set_vec_tile_shapes(*[1, 1, h])
                    pto.assemble(post_out_view, out_offset, post_out)
                inside_s_idx_loop_attention_post(s_idx)
        inside_b_idx_loop_attention_post(b_idx)


def attention_post_standalone(
        input_tensor: pto.tensor,
        post_tensors: PostTensors,
        tile_config: PostTileConfig,
        post_out: pto.tensor,
):

    input_tensors = [input_tensor, post_tensors.weight_u_v, post_tensors.weight_o,
                        post_tensors.weight_uv_scale, post_tensors.smooth_scales_w_uv,
                            post_tensors.weight_o_scale, post_tensors.smooth_scales_wo]
    output_tensors = [post_out]
    with pto.function("POST_MAIN", input_tensors, output_tensors):
        post_compute(input_tensor, post_tensors, tile_config, post_out)

## UT


@dataclass
class TestPostParams:
    b: int
    n: int
    s: int
    h: int
    kv_lora_rank: int
    v_head_dim: int


def test_attention_post_ut(**kwargs):
    params = kwargs.get("params")
    tile_config = kwargs.get("tile_config")
    d_type = kwargs.get("d_type", pto.DT_FP16)
    nz = kwargs.get("nz", True)
    w_uv_dtype = kwargs.get("w_uv_dtype", pto.DT_INT8)
    is_smooth_wuv = kwargs.get("is_smooth_wuv", False)
    w_o_dtype = kwargs.get("w_o_dtype", pto.DT_INT8)
    is_smooth_wo = kwargs.get("is_smooth_wo", False)

    b = params.b
    n = params.n
    s = params.s
    h = params.h
    kv_lora_rank = params.kv_lora_rank
    v_head_dim = params.v_head_dim

    d_type = pto.DT_FP16 if d_type == pto.DT_FP16 else pto.DT_BF16
    is_quant_w_uv = w_uv_dtype == pto.DT_INT8
    is_quant_wo = w_o_dtype == pto.DT_INT8

    x_shape = [b, s, n, kv_lora_rank]
    w_uv_shape = [n, kv_lora_rank, v_head_dim]
    w_uv_scale_shape = [n, 1, v_head_dim]
    smooth_w_uv_shape = [1, kv_lora_rank]
    wo_shape = [n * v_head_dim, h]
    wo_scale_shape = [1, h]
    smooth_wo_shape = [1, n * v_head_dim]
    out_shape = [b, s, h]

    weight_format = pto.TileOpFormat.TILEOP_NZ if nz else pto.TileOpFormat.TILEOP_ND
    x = pto.Tensor(x_shape, d_type, "x")
    w_uv = pto.Tensor(w_uv_shape, pto.DT_INT8 if is_quant_w_uv else d_type, "wUv")
    w_uv_scale = pto.Tensor()
    smooth_w_uv = pto.Tensor()
    wo = pto.Tensor(wo_shape, pto.DT_INT8 if is_quant_wo else d_type, "wo", weight_format)
    wo_scale = pto.Tensor()
    smooth_wo = pto.Tensor()
    post_out = pto.Tensor(out_shape, d_type, "postOut")

    if is_quant_w_uv:
        scale = pto.Tensor(w_uv_scale_shape, pto.DT_FP32, "wUvScale")
        w_uv_scale[:] = scale
        if is_smooth_wuv:
            smooth = pto.Tensor(smooth_w_uv_shape, pto.DT_FP32, "smoothWUv")
            smooth_w_uv[:] = smooth

    if is_quant_wo:
        scale = pto.Tensor(wo_scale_shape, pto.DT_FP32, "woScale")
        wo_scale[:] = scale
        if is_smooth_wo:
            smooth = pto.Tensor(smooth_wo_shape, pto.DT_FP32, "smoothWo")
            smooth_wo[:] = smooth

    post_tensors = PostTensors(w_uv, wo, w_uv_scale, smooth_w_uv, wo_scale, smooth_wo)
    attention_post_standalone(x, post_tensors, tile_config, post_out)


if __name__ == "__main__":
    params = TestPostParams(32, 128, 2, 7168, 512, 128)
    tile_config = PostTileConfig(16, 1)
    d_type = pto.DT_BF16
    nz = True
    w_uv_dtype = pto.DT_FP16
    is_smooth_wuv = False
    w_o_dtype = pto.DT_INT8
    is_smooth_wo = True

    test_attention_post_ut(params=params, tile_config=tile_config, nz=True, d_type=d_type, is_smooth_wuv=is_smooth_wuv,
                            w_uv_dtype=w_uv_dtype, w_o_dtype=w_o_dtype, is_smooth_wo=is_smooth_wo)