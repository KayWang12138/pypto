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
import pto

# RuntimeConfig KEYS
KEY_ONLY_CODEGEN = "only_codegen"

F_1 = 1.0
F_NEGA_1 = -1.0


def ceil_div(a, b):
    return (a + (b - 1)) // b


def dynamic_ffn(**kwargs):
    hidden_states = kwargs.get("hidden_states")
    ffn_weight1 = kwargs.get("ffn_weight1")
    ffn_weight2 = kwargs.get("ffn_weight2")
    ffn_weight3 = kwargs.get("ffn_weight3")
    out = kwargs.get("out")
    basic_batch = kwargs.get("basic_batch")

    h = hidden_states.shape[1]

    input_tensors = [hidden_states, ffn_weight1, ffn_weight2, ffn_weight3]
    output_tensors = [out]
    if basic_batch == 0:
        raise ValueError("basic_batch is zero")
    with pto.function("main", input_tensors, output_tensors):
        for idx in pto.loop(ceil_div(pto.get_input_shape(hidden_states, 0), basic_batch), name="L0", idx_name="idx"):
            def inside_idx_loop(idx):
                batch_idx = basic_batch * idx
                hidden_states_temp = pto.view(hidden_states, [basic_batch, h], [batch_idx, 0])
                cast_res = pto.cast(hidden_states_temp, pto.DT_FP16)
                gate = pto.matmul(cast_res, ffn_weight1, pto.DT_FP32)
                swish = pto.mul_s(gate, pto.element(pto.DT_FP32, F_NEGA_1))
                swish = pto.exp(swish)
                swish = pto.add_s(swish, pto.element(pto.DT_FP32, F_1))
                swish = pto.div(gate, swish)

                up = pto.matmul(cast_res, ffn_weight2, pto.DT_FP32)
                swish = pto.mul(swish, up)
                swish_fp16 = pto.cast(swish, pto.DT_FP16)

                # down_proj
                mlp_res = pto.matmul(swish_fp16, ffn_weight3, pto.DT_FP32, a_trans=False, b_trans=True)
                pto.assemble(mlp_res, [batch_idx, 0], out)
            inside_idx_loop(idx)


def dynamic_ffn_quant(**kwargs):
    hidden_states_quant = kwargs.get("hidden_states_quant")
    hidden_states_scale = kwargs.get("hidden_states_scale")
    ffn_weight1 = kwargs.get("ffn_weight1")
    ffn_weight2 = kwargs.get("ffn_weight2")
    ffn_weight3 = kwargs.get("ffn_weight3")
    ffn_scale1 = kwargs.get("ffn_scale1")
    ffn_scale2 = kwargs.get("ffn_scale2")
    ffn_scale3 = kwargs.get("ffn_scale3")
    out = kwargs.get("out")
    basic_batch = kwargs.get("basic_batch")

    h = hidden_states_quant.shape[1]

    input_tensors = [hidden_states_quant, hidden_states_scale, ffn_weight1, ffn_weight2,
                    ffn_weight3, ffn_scale1, ffn_scale2, ffn_scale3]
    output_tensors = [out]
    if basic_batch == 0:
        raise ValueError("basic_batch is zero")
    with pto.function("main", input_tensors, output_tensors):
        for idx in pto.loop(ceil_div(pto.get_input_shape(hidden_states_quant, 0), basic_batch),
        name="L0", idx_name="idx"):
            def inside_idx_loop(idx):
                batch_idx = basic_batch * idx

                cast_res = pto.view(hidden_states_quant, [basic_batch, h], [batch_idx, 0])
                cast_res_scale = pto.view(hidden_states_scale, [basic_batch, 1], [batch_idx, 0])
                gate_int32 = pto.matmul(cast_res, ffn_weight1, pto.DT_INT32)

                # dequant: int32 -> fp32 -> *scale -> fp16/bf16
                gate_tmp_fp32 = pto.cast(gate_int32, pto.DT_FP32)
                gate_tmp_dequant_pertoken = pto.mul(gate_tmp_fp32, cast_res_scale)
                gate = pto.mul(gate_tmp_dequant_pertoken, ffn_scale1)

                swish = pto.mul_s(gate, pto.element(pto.DT_FP32, F_NEGA_1))
                swish = pto.exp(swish)
                swish = pto.add_s(swish, pto.element(pto.DT_FP32, F_1))
                swish = pto.div(gate, swish)

                up_int32 = pto.matmul(cast_res, ffn_weight2, pto.DT_INT32)
                # upProj
                up_tmp_fp32 = pto.cast(up_int32, pto.DT_FP32)
                up_tmp_dequant_pertoken = pto.mul(up_tmp_fp32, cast_res_scale)
                up = pto.mul(up_tmp_dequant_pertoken, ffn_scale2)

                swish = pto.mul(swish, up)

                # downProj
                swish_quant_res = pto.quant(swish) # int8
                swish_res = swish_quant_res[0]
                swish_scale = swish_quant_res[1]

                res_int32 = pto.matmul(swish_res, ffn_weight3, pto.DT_INT32, a_trans=False, b_trans=True)
                res_tmp_fp32 = pto.cast(res_int32, pto.DT_FP32)
                res_tmp_dequant_pertoken = pto.mul(res_tmp_fp32, swish_scale)
                res = pto.mul(res_tmp_dequant_pertoken, ffn_scale3)
                pto.assemble(res, [batch_idx, 0], out)
            inside_idx_loop(idx)


def test_ffn():
    pto.set_host_option(KEY_ONLY_CODEGEN, True)

    pto.set_vec_tile_shapes(32, 128)
    pto.set_cube_tile_shapes([32, 32], [128, 128], [128, 128])

    batch_size = 64
    sequence = 1
    h = 7168
    expert_dim = 2048
    bs = batch_size * sequence
    basic_batch = 32

    hidden_states_shape = [bs, h]
    weight_shape = [h, expert_dim]
    out_shape = [bs, h]

    hidden_states = pto.tensor(hidden_states_shape, pto.DT_FP32, "hidden_states")
    ffn_weight1 = pto.tensor(weight_shape, pto.DT_FP16, "weight_shape1")
    ffn_weight2 = pto.tensor(weight_shape, pto.DT_FP16, "weight_shape2")
    ffn_weight3 = pto.tensor(weight_shape, pto.DT_FP16, "weight_shape3")
    ffn_out = pto.tensor(out_shape, pto.DT_FP32, "ffn_out")

    dynamic_ffn(hidden_states=hidden_states,
                ffn_weight1=ffn_weight1,
                ffn_weight2=ffn_weight2,
                ffn_weight3=ffn_weight3,
                out=ffn_out,
                basic_batch=basic_batch)


def test_ffn_quant():
    pto.set_host_option(KEY_ONLY_CODEGEN, True)

    pto.set_vec_tile_shapes(32, 128)
    pto.set_cube_tile_shapes([32, 32], [128, 128], [128, 128])

    batch_size = 32
    sequence = 1
    h = 7168
    expert_dim = 2048
    bs = batch_size * sequence
    basic_batch = 32

    hidden_states_shape = [bs, h]
    weight_shape = [h, expert_dim]
    out_shape = [bs, h]

    hidden_states_quant = pto.tensor(hidden_states_shape, pto.DT_INT8, "hidden_states_quant")
    hidden_states_scale = pto.tensor([bs, 1], pto.DT_FP32, "hidden_states_scale")
    ffn_weight1 = pto.tensor(weight_shape, pto.DT_INT8, "weight_shape1", pto.TILEOP_NZ)
    ffn_weight2 = pto.tensor(weight_shape, pto.DT_INT8, "weight_shape2", pto.TILEOP_NZ)
    ffn_weight3 = pto.tensor(weight_shape, pto.DT_INT8, "weight_shape3", pto.TILEOP_NZ)
    ffn_scale1 = pto.tensor([1, expert_dim], pto.DT_FP32, "ffn_scale1")
    ffn_scale2 = pto.tensor([1, expert_dim], pto.DT_FP32, "ffn_scale2")
    ffn_scale3 = pto.tensor([1, h], pto.DT_FP32, "ffn_scale3")
    ffn_out = pto.tensor(out_shape, pto.DT_FP32, "ffn_out")

    dynamic_ffn_quant(hidden_states_quant=hidden_states_quant,
                    hidden_states_scale=hidden_states_scale,
                    ffn_weight1=ffn_weight1,
                    ffn_weight2=ffn_weight2,
                    ffn_weight3=ffn_weight3,
                    ffn_scale1=ffn_scale1,
                    ffn_scale2=ffn_scale2,
                    ffn_scale3=ffn_scale3,
                    out=ffn_out,
                    basic_batch=basic_batch)