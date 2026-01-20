#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Quant Attention Post Module

This module implements Quant Attention Post with RoPE for DeepSeek V4.

Main Functions:
    - attention_post_quant_compute: Quant attention post with RoPE computation
    - attention_post_quant_kernel: JIT-compiled entry function

Example:
    See deepseekv4_quant_attention_post.py for usage examples.
"""
from dataclasses import dataclass
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from common import inverse_rope_3d
from common import quant
import pypto
import torch


@dataclass
class AttnPostQuantConfig:
    unroll_list: list
    c1_tile: list
    c2_tile: list


def attention_post_quant_compute(
    attn_res: pypto.Tensor,
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    wo_a: pypto.Tensor,
    wo_b: pypto.Tensor,
    wo_b_scale: pypto.Tensor,
    hidden_states: pypto.Tensor,
    tile_config: AttnPostQuantConfig,
):
    """Attention Post compute.

    Args:
        group       name           dtype     shape                              format
        INPUT 0	    attn_res	   DT_BF16	 (t, n_q, dim)	                    ND
        INPUT 1	    cos	           DT_BF16	 (t, rope_dim)	                    ND
        INPUT 2	    sin	           DT_BF16	 (t, rope_dim)	                    ND
        INPUT 3	    wo_a	       DT_BF16	 (ng, n_q * dim // ng, o_lora_rank)	ND
        INPUT 4	    wo_b	       DT_INT8	 (ng * o_lora_rank, h)              ND
        INPUT 5	    wo_b_scale	   DT_FP32	 (h, 1)                             ND
        OUTPUT 0	hidden_states  DT_BF16	 (t, h)	                            ND
    Note:

    """
    assert (
        len(attn_res.shape) == 3
        and len(cos.shape) == 2
        and len(sin.shape) == 2
        and len(wo_a.shape) == 3
    )
    assert (
        len(wo_b.shape) == 2
        and len(wo_b_scale.shape) == 2
        and len(hidden_states.shape) == 2
    )

    dtype = attn_res.dtype
    t = attn_res.shape[0]
    n_q = attn_res.shape[1]
    d = attn_res.shape[2]

    n_g = wo_a.shape[0]
    o_lora_rank = wo_a.shape[2]

    rope_dim = cos.shape[1]
    nope_dim = d - rope_dim

    h = hidden_states.shape[1]

    unroll_list = tile_config.unroll_list
    c1_tile = tile_config.c1_tile
    c2_tile = tile_config.c2_tile

    wo_b_scale_in = pypto.reshape(wo_b_scale, [1, h], inplace=True)

    for t_idx, unrollLength in pypto.loop_unroll(
        0, t, 1, name="ATTN_POST_T_LOOP", idx_name="t_idx", unroll_list=unroll_list
    ):
        tile_t = unrollLength

        # for nope+rope
        tmp_tensor = pypto.tensor([tile_t, n_q, d], dtype, "tmp_tensor")

        # copy nope to tmp_tensor
        pypto.set_semantic_label("Rope_nope")
        pypto.set_vec_tile_shapes(1, n_q // 2, d)
        atten_res_nope = pypto.view(
            attn_res, [tile_t, n_q, nope_dim], [t_idx, 0, 0]
        )  # nope: (tile_t, n_q, nope_dim)
        pypto.assemble(pypto.clone(atten_res_nope), [0, 0, 0], tmp_tensor)

        # apply rope to tmp_tensor
        pypto.set_semantic_label("Rope_rope")
        atten_res_rope = pypto.view(
            attn_res, [tile_t, n_q, rope_dim], [t_idx, 0, nope_dim]
        )  # rope: (tile_t, n_q, 64)
        cos_in = pypto.view(cos, [tile_t, rope_dim], [t_idx, 0])
        sin_in = pypto.view(sin, [tile_t, rope_dim], [t_idx, 0])
        rope_result = inverse_rope_3d(atten_res_rope, cos_in, sin_in)
        pypto.assemble(rope_result, [0, 0, nope_dim], tmp_tensor)  # (tile_t, n_q, d)

        # bmm1 left transpose: (tile_t, n_q, d) -> (n_g, tile_t, n_q * d / n_g)
        pypto.set_semantic_label("bmm_transpose")
        pypto.set_vec_tile_shapes(1, 32, 512)
        atten_reshape = pypto.reshape(tmp_tensor, [tile_t, n_g, n_q * d // n_g])
        pypto.set_vec_tile_shapes(1, 4, 4096)
        atten_trs = pypto.transpose(atten_reshape, 0, 1)

        # batch_matmul: (n_g, tile_t, n_q * d / n_g) @ (n_g, n_q * d / n_g, o_lora_rank) --> (n_g, tile_t, o_lora_rank)
        pypto.set_semantic_label("bmm_cube")
        pypto.set_cube_tile_shapes(*c1_tile)
        attn_res_l = pypto.view(atten_trs, [n_g, tile_t, n_q * d // n_g], [0, 0, 0])
        bmm1_res = pypto.matmul(attn_res_l, wo_a, dtype, a_trans=False, b_trans=False)

        # bmm1 output transpose: (n_g, tile_t, o_lora_rank) -> (tile_t, n_g * o_lora_rank)
        pypto.set_semantic_label("mm_transpose")
        pypto.set_vec_tile_shapes(1, 16, 1024)
        bmm1_res_tmp = pypto.view(bmm1_res, [n_g, tile_t, o_lora_rank], [0, 0, 0])

        bmm1_res_trs = pypto.transpose(bmm1_res_tmp, 0, 1)
        bmm1_res_reshape = pypto.reshape(bmm1_res_trs, [tile_t, n_g * o_lora_rank])

        pypto.set_semantic_label("bmm_quant")
        pypto.set_vec_tile_shapes(2, n_g * o_lora_rank)
        # (tile_t, n_g * o_lora_rank), (tile_t, 1)
        bmm1_int8, bmm1_scale = quant(bmm1_res_reshape)

        # matmul: (t, n_g * o_lora_rank) @ (n_g * o_lora_rank, h) --> (tile_t, h)
        pypto.set_semantic_label("mm_cube")
        pypto.set_cube_tile_shapes(*c2_tile)
        mm2_res = pypto.matmul(
            bmm1_int8, wo_b, pypto.DT_INT32, a_trans=False, b_trans=False
        )

        pypto.set_vec_tile_shapes(4, h)
        mm2_cast = pypto.cast(mm2_res, pypto.DT_FP32)
        mul_res1 = pypto.mul(
            mm2_cast, wo_b_scale_in
        )  # (tile_t, h) * (1, h) --> (tile_t, h)
        mul_res = pypto.mul(
            mul_res1, bmm1_scale
        )  # (tile_t, h) * (tile_t, 1) --> (tile_t, h)
        mul_cast = pypto.cast(mul_res, dtype)
        pypto.assemble(mul_cast, [t_idx, 0], hidden_states)


@pypto.jit(
    pass_options={
        "mg_copyin_upper_bound": 16 * 1024 * 1024,
        "pg_upper_bound": 80000,
        "pg_lower_bound": 512,
        "pg_parallel_lower_bound": 40,
    },
    runtime_options={
        "stitch_function_inner_memory": 128,
        "stitch_function_outcast_memory": 128,
        "stitch_cfgcache_size": 2500000,
    },
)
def attention_post_quant_kernel(
    attn_res: pypto.Tensor,
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    wo_a: pypto.Tensor,
    wo_b: pypto.Tensor,
    wo_b_scale: pypto.Tensor,
    hidden_states: pypto.Tensor,
    tile_config: AttnPostQuantConfig,
):
    """JIT-compiled attention post for attention post quant phase.
    Attention Post Quant compute.

    Args:
        group       name           dtype     shape                              format
        INPUT 0	    attn_res	   DT_BF16	 (t, n_q, dim)	                    ND
        INPUT 1	    cos	           DT_BF16	 (t, rope_dim)	                    ND
        INPUT 2	    sin	           DT_BF16	 (t, rope_dim)	                    ND
        INPUT 3	    wo_a	       DT_BF16	 (ng, n_q * dim // ng, o_lora_rank)	ND
        INPUT 4	    wo_b	       DT_INT8	 (ng * o_lora_rank, h)              ND
        INPUT 5	    wo_b_scale	   DT_FP32	 (h, 1)                             ND
        OUTPUT 0	hidden_states  DT_BF16	 (t, h)	                            ND
    Note:

    """
    attention_post_quant_compute(
        attn_res, cos, sin, wo_a, wo_b, wo_b_scale, hidden_states, tile_config
    )


def check_input_shape_dtype(
    attn_res: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
):
    # shape check
    n_q = 64
    d = 512
    rope_dim = 64
    n_g = 8
    o_lora_rank = 1024
    h = 4096

    assert (
        len(attn_res.shape) == 3 and attn_res.size(1) == n_q and attn_res.size(2) == d
    ), f"expected attn_res shape: (t, 64, 512), but got {attn_res.shape}"
    assert (
        len(cos.shape) == 2 and cos.size(1) == rope_dim
    ), f"expected cos shape: (t, 64), but got {cos.shape}"
    assert (
        len(sin.shape) == 2 and sin.size(1) == rope_dim
    ), f"expected sin shape: (t, 64), but got {sin.shape}"
    assert (
        len(wo_a.shape) == 3
        and wo_a.size(0) == n_g
        and wo_a.size(1) == n_q * d // n_g
        and wo_a.size(2) == o_lora_rank
    ), f"expected wo_a shape: (8, 4096, 1024), but got {wo_a.shape}"
    assert (
        len(wo_b.shape) == 2 and wo_b.size(0) == n_g * o_lora_rank and wo_b.size(1) == h
    ), f"expected wo_b shape: (8192, 4096), but got {wo_b.shape}"
    assert (
        len(wo_b_scale.shape) == 2
        and wo_b_scale.size(1) == h
        and wo_b_scale.size(2) == 1
    ), f"expected wo_b_scale shape: (h, 1), but got {wo_b_scale.shape}"

    assert (
        attn_res.dtype == cos.dtype == sin.dtype == wo_a.dtype == torch.bfloat16
    ), f"expected attn_res, cos, sin and wo_a dtype is torch.bfloat16, but got {attn_res.dtype}, {cos.dtype}, {sin.dtype}, and {wo_a.dtype}"
    assert (
        wo_b.dtype == torch.int8
    ), f"expected wo_b dtype is torch.int8, but got {wo_b.dtype}"
    assert (
        wo_b_scale.dtype == torch.float32
    ), f"expected wo_b_scale dtype is torch.float32, but got {wo_b_scale.dtype}"


@allow_in_graph
def npu_quant_attention_post(
    attn_res: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
):
    """
    torch npu graph interface

    """

    hidden_states = torch.zeros(
        [attn_res.size(0), wo_b.size(1)],
        dtype=attn_res.dtype,
        device=f"{attn_res.device}",
    )

    check_input_shape_dtype(attn_res, cos, sin, wo_a, wo_b, wo_b_scale)
    # notice mark dynamic axis
    atten_res_pto = pypto.from_torch(attn_res, dynamic_axis=[0], name="attn_res")
    cos_pto = pypto.from_torch(cos, dynamic_axis=[0], name="cos")
    sin_pto = pypto.from_torch(sin, dynamic_axis=[0], name="sin")
    wo_a_pto = pypto.from_torch(wo_a, name="wo_a")
    wo_b_pto = pypto.from_torch(wo_b, name="wo_b")
    wo_b_scale_pto = pypto.from_torch(wo_b_scale, name="wo_b_scale")
    hidden_states_pto = pypto.from_torch(
        hidden_states, dynamic_axis=[0], name="hidden_states"
    )

    # tiling
    tile_config = AttnPostQuantConfig(
        unroll_list=[128, 64, 32, 16, 8, 1],
        c1_tile=[[64, 64], [64, 64], [512, 512]],
        c2_tile=[[128, 128], [128, 128], [256, 256]],
    )

    # kernel
    if not isinstance(attn_res, FakeTensor):
        pto_inputs = [
            atten_res_pto,
            cos_pto,
            sin_pto,
            wo_a_pto,
            wo_b_pto,
            wo_b_scale_pto,
        ]
        pto_outputs = [hidden_states_pto]
        attention_post_quant_kernel(*pto_inputs, *pto_outputs, tile_config)

    return hidden_states
