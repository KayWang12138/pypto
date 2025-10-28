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
import logging
from dataclasses import dataclass
import pto

NUM2 = 2
SHAPE_DIM4 = 4
SHAPE_DIM3 = 3
SHAPE_DIM2 = 2
NUM_VALUE_3 = 3
NUM_VALUE_4 = 4


def rope_input_cast(input_tensor: pto.tensor) -> pto.tensor:
    input_dtype = input_tensor.get_dtype()
    if input_dtype == pto.DT_FP32:
        return input_tensor
    return pto.cast(input_tensor, pto.data_type.DT_FP32)


def rotate_half(input_tensor: pto.tensor) -> pto.tensor:
    shape = input_tensor.shape
    shape_size = len(shape)

    logging.basicConfig(level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s')
            
    def check_shape(shape, shape_size):
        try:
            if shape_size < 1:
                raise ValueError("rope rotate_half input dim less than 1")
            if shape[shape_size - 1] % NUM2 != 0:
                raise ValueError("rope rotate_half last dim shape is even.")
        except ValueError as e:
            logging.error(f"Error:{e}")
    check_shape(shape, shape_size)
    shape[shape_size - 1] //= NUM2
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]

    x1 = pto.view(input_tensor, shape, offset1)
    x2 = pto.view(input_tensor, shape, offset2)

    return pto.concat(
        [pto.mul_s(x2, pto.element(x2.get_dtype(), -1.0)), pto.add_s(x1, pto.element(x1.get_dtype(), 0.0))], -1)


def applyRotaryPosEmbV2(**kwargs):
    q = kwargs.get("q")
    k = kwargs.get("k")
    cos = kwargs.get("cos")
    sin = kwargs.get("sin")
    q_embed = kwargs.get("q_embed")
    k_embed = kwargs.get("k_embed")
    unsqueeze_dim = kwargs.get("unsqueeze_dim")
    rope_tile_shape_config_new = kwargs.get("rope_tile_shape_config_new")

    output_dtype = q_embed.get_dtype()

    
    if not (len(q.shape) == SHAPE_DIM4 and
            len(k.shape) == SHAPE_DIM4 and
            len(cos.shape) == SHAPE_DIM3 and
            len(sin.shape) == SHAPE_DIM3):
        raise ValueError("q, k, cos, sin shape are not valid.")
    
    logging.basicConfig(level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s')

    def check_shape(rope_tile_shape_config_new):
        try:
            if len(rope_tile_shape_config_new.three_dims_tile_shape) == 0:
                raise ValueError("rope ThreeDims Tile need to set!")
            if len(rope_tile_shape_config_new.four_dims_tile_shape_q) == 0:
                raise ValueError("rope FourDimsQ Tile need to set!")
            if len(rope_tile_shape_config_new.four_dims_tile_shape_k) == 0:
                raise ValueError("rope FourDimsK Tile need to set!")
            if len(rope_tile_shape_config_new.five_dims_tile_shape) == 0:
                raise ValueError("rope FiveDims Tile need to set!")
        except ValueError as e:
            logging.error(f"Error:{e}")

    check_shape(rope_tile_shape_config_new)
    pto.set_vec_tile_shapes(*rope_tile_shape_config_new.four_dims_tile_shape_q)
    cast_q = rope_input_cast(q)
    pto.set_vec_tile_shapes(*rope_tile_shape_config_new.four_dims_tile_shape_k)
    cast_k = rope_input_cast(k)

    pto.set_vec_tile_shapes(*rope_tile_shape_config_new.three_dims_tile_shape)
    cast_cos = rope_input_cast(cos)
    cast_sin = rope_input_cast(sin)

    cos_unsqueeze = pto.unsqueeze(cast_cos, unsqueeze_dim)
    sin_unsqueeze = pto.unsqueeze(cast_sin, unsqueeze_dim)

    b = cast_q.shape[0]
    s = cast_q.shape[1]
    h = cast_q.shape[2]
    d = cast_q.shape[NUM_VALUE_3]

    q_view = pto.reshape(cast_q, [b, s, h, d // 2, 2])
    pto.set_vec_tile_shapes(*rope_tile_shape_config_new.five_dims_tile_shape)
    q_trans = pto.transpose(q_view, [NUM_VALUE_3, NUM_VALUE_4])
    q_reshape = pto.reshape(q_trans, [b, s, h, d])

    b = cast_k.shape[0]
    s = cast_k.shape[1]
    h = cast_k.shape[2]
    d = cast_k.shape[3]

    pto.set_vec_tile_shapes(*rope_tile_shape_config_new.four_dims_tile_shape_k)
    k_view = pto.reshape(cast_k, [b, s, h, d // 2, 2])
    pto.set_vec_tile_shapes(*rope_tile_shape_config_new.five_dims_tile_shape)
    k_trans = pto.transpose(k_view, [NUM_VALUE_3, NUM_VALUE_4])
    k_reshape = pto.reshape(k_trans, [b, s, h, d])

    pto.set_vec_tile_shapes(*rope_tile_shape_config_new.four_dims_tile_shape_q)
    q_embed[:] = pto.add(pto.mul(q_reshape, cos_unsqueeze), pto.mul(rotate_half(q_reshape), sin_unsqueeze))
    pto.set_vec_tile_shapes(*rope_tile_shape_config_new.four_dims_tile_shape_k)
    k_embed[:] = pto.add(pto.mul(k_reshape, cos_unsqueeze), pto.mul(rotate_half(k_reshape), sin_unsqueeze))

    if output_dtype != q_embed.get_dtype():
        pto.set_vec_tile_shapes(*rope_tile_shape_config_new.four_dims_tile_shape_q)
        q_embed[:] = pto.cast(q_embed, output_dtype)
        pto.set_vec_tile_shapes(*rope_tile_shape_config_new.four_dims_tile_shape_k)
        k_embed[:] = pto.cast(k_embed, output_dtype)


def applyRotaryPosEmb(**kwargs):
    q = kwargs.get("q")
    k = kwargs.get("k")
    cos = kwargs.get("cos")
    sin = kwargs.get("sin")
    position_ids = kwargs.get("position_ids")
    q_embed = kwargs.get("q_embed")
    k_embed = kwargs.get("k_embed")
    unsqueeze_dim = kwargs.get("unsqueeze_dim")
    rope_tile_shape_config = kwargs.get("rope_tile_shape_config")

    output_dtype = q_embed.get_dtype()
    
    if not (len(q.shape) == SHAPE_DIM4 and
            len(k.shape) == SHAPE_DIM4 and
            len(cos.shape) == SHAPE_DIM2 and
            len(sin.shape) == SHAPE_DIM2):
        raise ValueError("q, k, cos, sin shape are not valid.")

    logging.basicConfig(level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s')

    def check_shape(rope_tile_shape_config):
        try:
            if len(rope_tile_shape_config.two_dims_tile_shape) == 0:
                raise ValueError("rope TwoDims Tile need to set!")
            if len(rope_tile_shape_config.three_dims_tile_shape) == 0:
                raise ValueError("rope ThreeDims Tile need to set!")
            if len(rope_tile_shape_config.four_dims_tile_shape) == 0:
                raise ValueError("rope FourDims Tile need to set!")
            if len(rope_tile_shape_config.five_dims_tile_shape) == 0:
                raise ValueError("rope FiveDims Tile need to set!")
        except ValueError as e:
            logging.error(f"Error:{e}")

    check_shape(rope_tile_shape_config)
    pto.set_vec_tile_shapes(*rope_tile_shape_config.four_dims_tile_shape)
    cast_q = rope_input_cast(q)
    cast_k = rope_input_cast(k)

    pto.set_vec_tile_shapes(*rope_tile_shape_config.two_dims_tile_shape)
    cast_cos = rope_input_cast(cos)
    cast_sin = rope_input_cast(sin)
    
    pto.set_vec_tile_shapes(*rope_tile_shape_config.three_dims_tile_shape)
    cos_tensor_indexes = pto.tensor_index(cast_cos, position_ids)
    sin_tensor_indexes = pto.tensor_index(cast_sin, position_ids)

    cos_unsqueeze = pto.unsqueeze(cos_tensor_indexes, unsqueeze_dim)
    sin_unsqueeze = pto.unsqueeze(sin_tensor_indexes, unsqueeze_dim)

    b = cast_q.shape[0]
    h = cast_q.shape[1]
    s = cast_q.shape[2]
    d = cast_q.shape[NUM_VALUE_3]

    pto.set_vec_tile_shapes(*rope_tile_shape_config.four_dims_tile_shape)
    q_view = pto.reshape(cast_q, [b, h, s, d // 2, 2])
    pto.set_vec_tile_shapes(*rope_tile_shape_config.five_dims_tile_shape)
    q_trans = pto.transpose(q_view, [NUM_VALUE_3, NUM_VALUE_4])
    q_reshape = pto.reshape(q_trans, [b, h, s, d])

    b = cast_k.shape[0]
    h = cast_k.shape[1]
    s = cast_k.shape[2]
    d = cast_k.shape[3]

    pto.set_vec_tile_shapes(*rope_tile_shape_config.four_dims_tile_shape)
    k_view = pto.reshape(cast_k, [b, h, s, d // 2, 2])
    pto.set_vec_tile_shapes(*rope_tile_shape_config.five_dims_tile_shape)
    k_trans = pto.transpose(k_view, [NUM_VALUE_3, NUM_VALUE_4])
    k_reshape = pto.reshape(k_trans, [b, h, s, d])

    pto.set_vec_tile_shapes(*rope_tile_shape_config.four_dims_tile_shape)
    q_embed[:] = pto.add(pto.mul(q_reshape, cos_unsqueeze), pto.mul(rotate_half(q_reshape), sin_unsqueeze))
    k_embed[:] = pto.add(pto.mul(k_reshape, cos_unsqueeze), pto.mul(rotate_half(k_reshape), sin_unsqueeze))

    if output_dtype != q_embed.get_dtype():
        q_embed[:] = pto.cast(q_embed, output_dtype)
        k_embed[:] = pto.cast(k_embed, output_dtype)


def testApplyRotaryPosEmbV2():
    b = 32
    n = 128
    s = 1
    qk_rope_head_dim = 64

    q_pe_shape = [b, s, n, qk_rope_head_dim]
    k_pe_shape = [b, s, 1, qk_rope_head_dim]
    cos_sin_shape = [b, s, qk_rope_head_dim]
    q_embed_shape = [b, s, n, qk_rope_head_dim]
    k_embed_shape = [b, s, 1, qk_rope_head_dim]

    q = pto.tensor(q_pe_shape, pto.data_type.DT_FP32, "q")
    k = pto.tensor(k_pe_shape, pto.data_type.DT_FP32, "k")
    cos = pto.tensor(cos_sin_shape, pto.data_type.DT_FP32, "cos")
    sin = pto.tensor(cos_sin_shape, pto.data_type.DT_FP32, "sin")
    q_embed = pto.tensor(q_embed_shape, pto.data_type.DT_FP32, "qEmbed")
    k_embed = pto.tensor(k_embed_shape, pto.data_type.DT_FP32, "kEmbed")

    rope_tile_shape_config_new = pto.rope_tile_shape_config_new()
    rope_tile_shape_config_new.three_dims_tile_shape = [32, 1, 64]
    rope_tile_shape_config_new.four_dims_tile_shape_q = [1, 1, 32, 64]
    rope_tile_shape_config_new.four_dims_tile_shape_k = [32, 1, 1, 64]
    rope_tile_shape_config_new.five_dims_tile_shape = [32, 1, 1, 32, 2]

    input_tensor = []
    output_tensor = []
    with pto.dyn_function("A", input_tensor, output_tensor):
        applyRotaryPosEmbV2(q=q, k=k, cos=cos, sin=sin, q_embed=q_embed, k_embed=k_embed,
                        unsqueeze_dim=2, rope_tile_shape_config_new=rope_tile_shape_config_new)


def testApplyRotaryPosEmb():
    rope_tile_shape_config = pto.rope_tile_shape_config()
    rope_tile_shape_config.two_dims_tile_shape = [64, 64]
    rope_tile_shape_config.three_dims_tile_shape = [1, 64, 64]
    rope_tile_shape_config.four_dims_tile_shape = [1, 64, 1, 64]
    rope_tile_shape_config.five_dims_tile_shape = [1, 64, 1, 32, 2]

    b = 1
    n = 32
    s = 1
    qk_rope_head_dim = 64

    q_pe_shape = [b, s, n, qk_rope_head_dim]
    k_pe_shape = [b, s, qk_rope_head_dim]
    ids_shape = [b, s]
    cos_shape = [s, qk_rope_head_dim]

    q_embed_shape = [b, n, s, qk_rope_head_dim]
    k_embed_shape = [b, 1, s, qk_rope_head_dim]

    q_pe = pto.tensor(q_pe_shape, pto.data_type.DT_BF16, "qPe")
    k_pe = pto.tensor(k_pe_shape, pto.data_type.DT_BF16, "kPe")
    cos = pto.tensor(cos_shape, pto.data_type.DT_BF16, "cos")
    sin = pto.tensor(cos_shape, pto.data_type.DT_BF16, "sin")
    position_ids = pto.tensor(ids_shape, pto.data_type.DT_INT32, "positionIds")
    q_embed = pto.tensor(q_embed_shape, pto.data_type.DT_BF16, "qEmbed")
    k_embed = pto.tensor(k_embed_shape, pto.data_type.DT_BF16, "kEmbed")

    with pto.dyn_function("RoPE", [], []):
        def inside():
            pto.set_vec_tile_shapes(*[1, 1, 64, 64])
            q_pe_trans = pto.transpose(q_pe, [1, 2])

            b = k_pe.shape[0]
            s = k_pe.shape[1]
            d = k_pe.shape[2]

            k_pe_reshape = pto.reshape(k_pe, [b, 1, s, d])
            applyRotaryPosEmb(q=q_pe_trans, k=k_pe_reshape, cos=cos, sin=sin, position_ids=position_ids,
                                    q_embed=q_embed, k_embed=k_embed, unsqueeze_dim=1,
                                        rope_tile_shape_config=rope_tile_shape_config)
        inside()


if __name__ == "__main__":
    
    testApplyRotaryPosEmbV2()