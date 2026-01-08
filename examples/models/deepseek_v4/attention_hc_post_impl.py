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
Attention Hc Post Module

This module implements Attention Post and Hc Post with RoPE for DeepSeek V4.

Main Functions:
    - attention_hc_post_compute: Attention hc post with RoPE computation
    - attention_hc_post_kernel: JIT-compiled decode version

Example:
    See deepseekv4_attention_hc_post.py for usage examples.
"""
from dataclasses import dataclass
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import torch
import pypto

AHP_DIM_2 = 2
AHP_DIM_3 = 3


@dataclass
class AttnHcPostConfig:
    unroll_list: list
    c1_tile: list
    c2_tile: list


def post_rotate_half(input_tensor: pypto.Tensor) -> pypto.Tensor:
    """Rotate half of the tensor dimensions for RoPE computation.

    Splits the last dimension in half and applies rotation transformation:
    [-x2, x1] where x1 is the first half and x2 is the second half.
    This is a key component of RoPE (Rotary Position Embedding).

    Args:
        input_tensor: Input tensor with last dimension divisible by 2

    Returns:
        Rotated tensor with same shape as input

    Raises:
        AssertionError: If input dimension is less than 1 or last dimension
                       is not divisible by 2
    """
    chunk_size = 2
    shape = input_tensor.shape
    shape_size = len(shape)
    assert shape_size >= 1, "rope rotate_half input dim less than 1"
    assert (
        shape[shape_size - 1] % chunk_size == 0
    ), "rope rotate_half last dim shape is even"

    new_shape = list(shape)
    new_shape[shape_size - 1] //= chunk_size

    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = new_shape[shape_size - 1]

    x1 = pypto.view(input_tensor, new_shape, offset1)
    x2 = pypto.view(input_tensor, new_shape, offset2)

    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def post_rope_3d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor) -> pypto.Tensor:
    """Apply 3D Rotary Position Embedding (RoPE).

    Implements RoPE transformation for 3D tensors with shape (t_tile, n_q, rope_dim).
    The RoPE is applied independently to each head using broadcasted cos/sin values.

    Args:
        x: Input tensor of shape (t_tile, n_q, rope_dim)
        cos: Cosine values for RoPE, shape (t_tile, rope_dim)
        sin: Sine values for RoPE, shape (t_tile, rope_dim)

    Returns:
        Tensor with RoPE applied, same shape as input x

    Note:
        The function broadcasts cos and sin to match the head dimension,
        then applies rotation: x_rotated = x * cos + rotate_half(x) * sin
    """
    t_tile = x.shape[0]
    n_q = x.shape[1]
    rope_dim = x.shape[2]
    pypto.set_vec_tile_shapes(1, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DataType.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DataType.DT_FP32)

    pypto.set_vec_tile_shapes(1, n_q, rope_dim)
    cast_x = pypto.cast(x, pypto.DataType.DT_FP32)
    cast_cos = pypto.reshape(cast_cos, [x.shape[0], 1, x.shape[2]])
    cast_sin = pypto.reshape(cast_sin, [x.shape[0], 1, x.shape[2]])

    pypto.set_vec_tile_shapes(1, n_q, rope_dim // 2, 8)
    x_view = pypto.reshape(cast_x, [t_tile, n_q, rope_dim // 2, 2])
    x_trans = pypto.transpose(x_view, 2, 3)
    x_re_second = pypto.reshape(x_trans, x.shape)  # (t_tile, n_q, rope_dim)
    pypto.set_vec_tile_shapes(1, n_q, rope_dim)
    x_embed = x_re_second * cast_cos + post_rotate_half(x_re_second) * cast_sin

    return pypto.cast(x_embed, x.dtype)


def attention_hc_post_compute(
    x: pypto.Tensor,
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    wo_a: pypto.Tensor,
    wo_b: pypto.Tensor,
    residual: pypto.Tensor,
    post: pypto.Tensor,
    comb: pypto.Tensor,
    y: pypto.Tensor,
    tile_config: AttnHcPostConfig,
):
    """Attention Post compute.

    Args:
        group       name           dtype     shape                              format
        INPUT 0	    x	           DT_BF16	 (t, n_q, dim)	                    ND
        INPUT 1	    cos	           DT_BF16	 (t, rope_dim)	                    ND
        INPUT 2	    sin	           DT_BF16	 (t, rope_dim)	                    ND
        INPUT 3	    wo_a	       DT_BF16	 (n_g, n_q*dim // ng, o_lora_rank)	ND
        INPUT 4	    wo_b	       DT_BF16	 (n_g * o_lora_rank, h)	            NZ
        INPUT 5	    residual	   DT_FP32	 (t, hc, h)	                        ND
        INPUT 6	    post	       DT_FP32	 (t, hc)	                        ND
        INPUT 7	    comb	       DT_FP32	 (t, hc, hc)	                    ND
        OUTPUT 0	y              DT_BF16	 (t, hc, h)	                        ND
    Note:

    """
    # do input/output shape len check
    assert (
        len(x.shape) == AHP_DIM_3
        and len(cos.shape) == AHP_DIM_2
        and len(sin.shape) == AHP_DIM_2
    )
    assert (
        len(wo_a.shape) == AHP_DIM_3
        and len(wo_b.shape) == AHP_DIM_2
        and len(residual.shape) == AHP_DIM_3
    )
    assert (
        len(post.shape) == AHP_DIM_2
        and len(comb.shape) == AHP_DIM_3
        and len(y.shape) == AHP_DIM_3
    )

    attn_dtype = x.dtype
    hc_dtype = residual.dtype
    # do input/output dtype check
    assert (
        cos.dtype == attn_dtype and sin.dtype == attn_dtype and wo_a.dtype == attn_dtype
    )
    assert (
        wo_b.dtype == attn_dtype
        and post.dtype == hc_dtype
        and comb.dtype == hc_dtype
        and y.dtype == attn_dtype
    )

    t = x.shape[0]
    n_q = x.shape[1]
    d = x.shape[2]

    n_g = wo_a.shape[0]
    o_lora_rank = wo_a.shape[2]

    rope_dim = cos.shape[1]
    nope_dim = d - rope_dim

    hc = residual.shape[1]
    h = residual.shape[2]

    unroll_list = tile_config.unroll_list
    c1_tile = tile_config.c1_tile
    c2_tile = tile_config.c2_tile

    post_reshape = pypto.reshape(post, [t, hc, 1], inplace=True)
    comb_reshape = pypto.reshape(comb, [t, hc, hc, 1], inplace=True)
    re_reshape = pypto.reshape(residual, [t, hc, 1, h], inplace=True)

    for t_idx, unrollLength in pypto.loop_unroll(
        0, t, 1, name="ATTN_HC_POST", idx_name="t_idx", unroll_list=unroll_list
    ):
        tile_t = unrollLength
        # for nope+rope
        tmp_tensor = pypto.tensor([tile_t, n_q, d], attn_dtype, "tmp_tensor")

        # copy nope to tmp_tensor
        pypto.set_semantic_label("attn-post-nope")
        pypto.set_vec_tile_shapes(1, 32, 512)
        x_nope = pypto.view(
            x, [tile_t, n_q, nope_dim], [t_idx, 0, 0]
        )  # nope: (tile_t, n_q, d - rope_dim)
        pypto.assemble(pypto.clone(x_nope), [0, 0, 0], tmp_tensor)

        # apply rope to tmp_tensor
        pypto.set_semantic_label("attn-post-nope")
        x_rope = pypto.view(
            x, [tile_t, n_q, rope_dim], [t_idx, 0, nope_dim]
        )  # rope: (tile_t, n_q, d - rope_dim)
        cos_in = pypto.view(cos, [tile_t, rope_dim], [t_idx, 0])
        sin_in = pypto.view(sin, [tile_t, rope_dim], [t_idx, 0])
        rope_result = post_rope_3d(x_rope, cos_in, sin_in)  # (t_tile, n_q, rope_dim)
        pypto.assemble(rope_result, [0, 0, nope_dim], tmp_tensor)

        # bmm1 left transpose: (tile_t, n_q, d) -> (n_g, tile_t, n_q * d / n_g)
        pypto.set_semantic_label("attn-post-bmm-trans")
        pypto.set_vec_tile_shapes(1, 32, 512)
        xhape = pypto.reshape(tmp_tensor, [tile_t, n_g, n_q * d // n_g])
        pypto.set_vec_tile_shapes(1, 4, 4096)
        atten_trs = pypto.transpose(xhape, 0, 1)

        # batch_matmul: (n_g, tile_t, n_q * d / n_g) @ (n_g, n_q * d / n_g, o_lora_rank) --> (n_g, tile_t, o_lora_rank)
        pypto.set_semantic_label("attn-post-bmm-cube")
        pypto.set_cube_tile_shapes(*c1_tile)
        x_l = pypto.view(atten_trs, [n_g, tile_t, n_q * d // n_g], [0, 0, 0])
        bmm1_res = pypto.matmul(x_l, wo_a, attn_dtype, a_trans=False, b_trans=False)

        # bmm1 output transpose: (n_g, tile_t, o_lora_rank) -> (tile_t, n_g * o_lora_rank)
        pypto.set_semantic_label("attn-post-mm-trans")
        pypto.set_vec_tile_shapes(1, 16, 1024)
        bmm1_res_tmp = pypto.view(bmm1_res, [n_g, tile_t, o_lora_rank], [0, 0, 0])
        bmm1_res_trs = pypto.transpose(bmm1_res_tmp, 0, 1)
        bmm1_res_reshape = pypto.reshape(bmm1_res_trs, [tile_t, n_g * o_lora_rank])

        # matmul: (t, n_g * o_lora_rank) @ (n_g * o_lora_rank, h) --> (tile_t, h)
        pypto.set_semantic_label("attn-post-mm-cube")
        pypto.set_cube_tile_shapes(*c2_tile)
        mm2_l = pypto.view(bmm1_res_reshape, [tile_t, n_g * o_lora_rank], [0, 0])
        mm2_res = pypto.matmul(mm2_l, wo_b, attn_dtype, a_trans=False, b_trans=False)

        # hc_post
        pypto.set_semantic_label("hc-post-left")
        hc_in = pypto.reshape(mm2_res, [tile_t, 1, h], inplace=True)
        pypto.set_vec_tile_shapes(1, hc, h // 2)
        hc_cast = pypto.cast(hc_in, hc_dtype)
        hc_expand = pypto.expand_clone(hc_cast, [tile_t, hc, h])

        post_in = pypto.view(post_reshape, [tile_t, hc, 1], [t_idx, 0, 0])
        post_res = post_in * hc_expand  # (tile_t, hc, 1) * (tile_t, hc, h)

        pypto.set_semantic_label("hc-post-right")
        pypto.set_vec_tile_shapes(1, hc, 1, h // 2)
        re_in = pypto.view(re_reshape, [tile_t, hc, 1, h], [t_idx, 0, 0, 0])
        re_expand = pypto.expand_clone(re_in, [tile_t, hc, hc, h])
        comb_in = pypto.view(comb_reshape, [tile_t, hc, hc, 1], [t_idx, 0, 0, 0])
        pypto.set_vec_tile_shapes(1, 1, hc, h // 2)
        re_mul = re_expand * comb_in  # (tile_t, hc, hc, h)
        re_sum = pypto.sum(re_mul, 1)  # (tile_t, hc, h)
        pypto.set_vec_tile_shapes(1, hc, h // 2)
        hc_res = pypto.add(post_res, re_sum)  # (tile_t, hc, h) + (tile_t, hc, h)
        hc_cast = pypto.cast(hc_res, attn_dtype)
        pypto.assemble(hc_cast, [t_idx, 0, 0], y)


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
    },
)
def attention_hc_post_kernel(
    x: pypto.Tensor,
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    wo_a: pypto.Tensor,
    wo_b: pypto.Tensor,
    residual: pypto.Tensor,
    post: pypto.Tensor,
    comb: pypto.Tensor,
    y: pypto.Tensor,
    tile_config: AttnHcPostConfig,
):
    """JIT-compiled attention hc post for decode phase.

    Args:

    Note:

    """
    attention_hc_post_compute(
        x, cos, sin, wo_a, wo_b, residual, post, comb, y, tile_config
    )


def check_input_shape_dtype(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
):
    # shape check
    n_q = 64
    d = 512
    rope_dim = 64
    n_g = 8
    o_lora_rank = 1024
    h = 4096
    hc = 4

    assert (
        len(x.shape) == AHP_DIM_3 and x.size(1) == n_q and x.size(2) == d
    ), f"expected x shape: (t, 64, 512), but got {x.shape}"
    assert (
        len(cos.shape) == AHP_DIM_2 and cos.size(1) == rope_dim
    ), f"expected cos shape: (t, 64), but got {cos.shape}"
    assert (
        len(sin.shape) == AHP_DIM_2 and sin.size(1) == rope_dim
    ), f"expected sin shape: (t, 64), but got {sin.shape}"
    assert (
        len(wo_a.shape) == AHP_DIM_3
        and wo_a.size(0) == n_g
        and wo_a.size(1) == n_q * d // n_g
        and wo_a.size(2) == o_lora_rank
    ), f"expected wo_a shape: (8, 4096, 1024), but got {wo_a.shape}"
    assert (
        len(wo_b.shape) == AHP_DIM_2
        and wo_b.size(0) == n_g * o_lora_rank
        and wo_b.size(1) == h
    ), f"expected wo_b shape: (8192, 4096), but got {wo_b.shape}"
    assert (
        len(residual.shape) == AHP_DIM_3
        and residual.size(1) == hc
        and residual.size(2) == h
    ), f"expected residual shape: (t, 4, 4096), but got {residual.shape}"
    assert (
        len(post.shape) == AHP_DIM_2 and post.size(1) == hc
    ), f"expected post shape: (t, 4), but got {post.shape}"
    assert (
        len(comb.shape) == AHP_DIM_3 and comb.size(1) == hc and comb.size(2) == hc
    ), f"expected comb shape: (t, 4, 4), but got {comb.shape}"

    assert (
        x.dtype == cos.dtype == sin.dtype == wo_a.dtype == wo_b.dtype == torch.bfloat16
    ), f"expected x, cos, sin, wo_a and wo_b dtype is torch.bfloat16, but got {x.dtype}, {cos.dtype}, {sin.dtype}, {wo_a.dtype} and {wo_b.dtype}"
    assert (
        residual.dtype == post.dtype == comb.dtype == torch.float32
    ), f"expected residual, post and comb dtype is torch.float32, but got {residual.dtype}, {post.dtype} and {comb.dtype}"


@allow_in_graph
def npu_attention_hc_post(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
):
    """
    torch npu graph interface

    """
    if isinstance(x, FakeTensor):
        return x

    y = torch.zeros([x.shape[0], post.shape[1], residual.shape[2]]).to(x.dtype).npu()

    check_input_shape_dtype(x, cos, sin, wo_a, wo_b, residual, post, comb)
    # mark dynamic_axis
    x_pto = pypto.from_torch(x, dynamic_axis=[0], name="x")
    cos_pto = pypto.from_torch(cos, dynamic_axis=[0], name="cos")
    sin_pto = pypto.from_torch(sin, dynamic_axis=[0], name="sin")
    wo_a_pto = pypto.from_torch(wo_a, name="wo_a")
    wo_b_pto = pypto.from_torch(wo_b, name="wo_b")
    residual_pto = pypto.from_torch(residual, dynamic_axis=[0], name="residual")
    post_pto = pypto.from_torch(post, dynamic_axis=[0], name="post")
    comb_pto = pypto.from_torch(comb, dynamic_axis=[0], name="comb")
    y_pto = pypto.from_torch(y, dynamic_axis=[0], name="y")

    # tiling
    tile_config = AttnHcPostConfig(
        unroll_list=[128, 64, 32, 16, 8, 1],
        c1_tile=[[64, 64], [64, 64], [512, 512]],
        c2_tile=[[128, 128], [128, 128], [256, 256]],
    )

    # kernel
    pto_inputs = [
        x_pto,
        cos_pto,
        sin_pto,
        wo_a_pto,
        wo_b_pto,
        residual_pto,
        post_pto,
        comb_pto,
    ]
    pto_outputs = [y_pto]
    attention_hc_post_kernel(*pto_inputs, *pto_outputs, tile_config)

    return y
