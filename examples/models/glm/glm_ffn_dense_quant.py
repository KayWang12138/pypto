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
import os
import pypto
import numpy as np
from numpy.testing import assert_allclose
from glm_ffn_quant_common import symmetric_quantization_per_token, dequant_dynamic
import torch
import torch_npu
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph


def main():
    test_glm_mlp()


def ffn_golden_quan_per_token(x):
    # y_int8 : int8  scale_dequant : x.dtype
    x_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def ffn_golden_quan_per_channel(x):
    # y_int8 : int8  scale_dequant : x.dtype
    x_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=0, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def ffn_golden_torch(expand_x_tensor, w13_int8, w13_scale, w2):
    bs = expand_x_tensor.shape[0]
    h = expand_x_tensor.shape[1]
    d = w2.shape[0]
    out = torch.zeros_like(expand_x_tensor)
    x_dtype = expand_x_tensor.dtype

    # 获取当前专家的权重
    # 矩阵乘法: [token_count, h] @ [h, d*2] = [token_count, d*2]
    x_int8, x_scale = ffn_golden_quan_per_token(expand_x_tensor)
    gate_output_fp32 = torch.matmul(x_int8, w13_int8).to(torch.int32).to(torch.float32)

    # 分割为left和right
    w13_scale_fp32 = w13_scale.to(torch.float32)
    gate_output = gate_output_fp32 * (w13_scale_fp32 * x_scale)
    split_dim = gate_output.shape[-1] // 2
    left, right = torch.split(gate_output, split_dim, dim=-1)

    # 计算Swish-GLU
    swiglu = left * 1 / (1 + torch.exp(-left))

    # Swish-GLU与right相乘
    swiglu = swiglu * right

    # 下层投影: [token_count, d] @ [d, h] = [token_count, h]
    swiglu_half = swiglu.to(x_dtype)
    out = torch.matmul(swiglu_half, w2).to(x_dtype)
    return out


def moe_torch_npu(expand_x_tensor, w13_int8, w13_scale, w2):
    x_dtype = expand_x_tensor.dtype
    quantized_x, dynamic_scale = torch_npu.npu_dynamic_quant(expand_x_tensor)
    w2 = w2.transpose(0, 1)
    output_w13 = torch_npu.npu_quant_matmul(
            quantized_x,
            w13_int8,
            w13_scale,
            pertoken_scale=dynamic_scale,
            bias=None,
            output_dtype=x_dtype,
        )
    output_w13 = output_w13.to(torch.float32)
    swiglu = torch_npu.npu_swiglu(output_w13)
    swiglu = swiglu.to(x_dtype)
    output = torch.matmul(swiglu.to(torch.float32), w2.to(torch.float32)).to(x_dtype)
    return output


def get_token_acc_table(expert_tokens):
    assert len(expert_tokens.shape) == 1
    token_acc_table = torch.zeros_like(expert_tokens)
    for i in range(1, expert_tokens.shape[0]):
        token_acc_table[i] = torch.sum(expert_tokens[0:i])
    return token_acc_table


def gen_input(b, s, hidden_size, intermediate_size, dtypes, device_id):
    torch.manual_seed(42)
    expand_x_tensor = torch.randn((b * s, hidden_size), dtype = dtypes, device = f'npu:{device_id}') * 0.01 * 2 - 0.01

    weight_gate_upper_tensor = torch.randn((hidden_size, intermediate_size * 2), dtype = dtypes, device = f'npu:{device_id}')  * 0.01 * 2 - 0.01
    w13_int8, w13_scale = ffn_golden_quan_per_channel(weight_gate_upper_tensor)
    w13_scale = w13_scale.reshape(-1).to(dtypes)

    w2 =  torch.randn((hidden_size, intermediate_size), dtype = dtypes, device = f'npu:{device_id}')  * 0.01 * 2 - 0.01

    out_tensor = torch.zeros((b * s, hidden_size), dtype = dtypes, device = f'npu:{device_id}')
    return expand_x_tensor, w13_int8, w13_scale, w2, out_tensor


def expert_infer_base(**kwargs):
    # 入参信息获取
    dense_loop_idx = kwargs.get("dense_loop_idx")
    expand_x = kwargs.get("expand_x")
    w13_int8 = kwargs.get("w13_int8")
    w13_scale = kwargs.get("w13_scale")
    w2 = kwargs.get("w2")
    vec_tile_shape = kwargs.get("vec_tile_shape")
    mm1_cube_tile_shape = kwargs.get("mm1_cube_tile_shape")
    mm2_cube_tile_shape = kwargs.get("mm2_cube_tile_shape")
    ffn_out = kwargs.get("ffn_out")
    loop_base = kwargs.get("loop_base")

    token_size, hidden_size = expand_x.shape[:2]
    intermediate_size = w2.shape[1]
    x_Dtype = expand_x.dtype

    # offset
    expand_x_offset = [dense_loop_idx * loop_base, 0]
    cur_valid_size = pypto.min(token_size - dense_loop_idx * loop_base, loop_base)
    expand_x_actual = pypto.view(expand_x, [loop_base, hidden_size], expand_x_offset, valid_shape=[cur_valid_size, hidden_size])


    pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
    # dynamic per_token_quant
    expand_int8, expand_scale = symmetric_quantization_per_token(expand_x_actual)

    # up_proj的matmul计算
    pypto.set_cube_tile_shapes([mm1_cube_tile_shape[0], mm1_cube_tile_shape[0]], [mm1_cube_tile_shape[1], mm1_cube_tile_shape[1]], [mm1_cube_tile_shape[2], mm1_cube_tile_shape[2]])
    gate_int32 = pypto.matmul(expand_int8, w13_int8, pypto.DT_INT32)

    # dequant
    w13_scale_2d = pypto.unsqueeze(w13_scale, 0)
    pypto.set_vec_tile_shapes(1, intermediate_size * 2)
    gate = dequant_dynamic(gate_int32, w13_scale_2d, expand_scale)

    # SwiGlu and mul :[x / (1 + e^(-x))]
    gate_left = pypto.view(gate, [loop_base, intermediate_size], [0, 0])
    gate_right = pypto.view(gate, [loop_base, intermediate_size], [0, intermediate_size])
    swiglu_a = pypto.mul(gate_left, -1.0)
    swiglu_b = pypto.exp(swiglu_a)
    swiglu_c = pypto.add(swiglu_b, 1.0)
    swiglu_out = pypto.div(gate_left, swiglu_c)
    swiglu = pypto.mul(swiglu_out, gate_right)
    swiglu_half = pypto.cast(swiglu, x_Dtype)

    # down_proj
    pypto.set_cube_tile_shapes([mm2_cube_tile_shape[0], mm2_cube_tile_shape[0]], [mm2_cube_tile_shape[1], mm2_cube_tile_shape[1]], [mm2_cube_tile_shape[2], mm2_cube_tile_shape[2]])
    out = pypto.matmul(swiglu_half, w2, x_Dtype, b_trans=True)
    pypto.assemble(out, expand_x_offset, ffn_out)


# tiling config
vec_tile_shape = (1, 5120)
mm1_cube_tile_shape = (16, 256, 128)
mm2_cube_tile_shape = (64, 64, 256)
loop_base = 16


@pypto.jit
def dense_moe_main(inputs, outputs):
    pypto.set_host_options(only_codegen=True)
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    pypto.set_runtime_options(cfgcache_device_task_num=100)
    pypto.set_runtime_options(cfgcache_root_task_num=1000)
    pypto.set_runtime_options(cfgcache_leaf_task_num=10000)

    expand_x = inputs[0]
    w13_int8 = inputs[1]
    w13_scale = inputs[2]
    w2 = inputs[3]
    ffn_out = outputs[0]

    token_nums = expand_x.shape[0]
    token_loop_times = (token_nums + loop_base - 1) / loop_base
    for dense_loop_idx in pypto.loop(0, token_loop_times, 1, name="dense_loop_idx"):
        def loop_token(dense_loop_idx):
            expert_infer_base(
                dense_loop_idx=dense_loop_idx,
                expand_x=expand_x,
                w13_int8=w13_int8,
                w13_scale=w13_scale,
                w2=w2,
                ffn_out=ffn_out,
                vec_tile_shape=vec_tile_shape,
                mm1_cube_tile_shape=mm1_cube_tile_shape,
                mm2_cube_tile_shape=mm2_cube_tile_shape,
                loop_base=loop_base,
                )
        loop_token(dense_loop_idx)


@allow_in_graph
def glm_dense_quant(hidden_states, w13_int8, w13_scale, w2):
    if isinstance(hidden_states, FakeTensor):
        return
    out_tensor = torch.zeros_like(hidden_states, device=hidden_states.device)
    inputs = {
        hidden_states: [0],
        w13_int8: [],
        w13_scale: [],
        w2: []
    }
    outputs = {
        out_tensor: [0]
    }
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    dense_moe_main(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()
    return out_tensor


def test_glm_mlp():
    x_dtype = torch.bfloat16
    # parameter config
    b = 1
    s = 1
    intermediate_size = 1536
    hidden_size = 5120
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    for i in range(0, 2):
        if (i == 0):
            b = 1
        if (i == 1):
            b = 2
        # expand_x_tensor, w13_int8, w13_scale, w2, out_tensor
        expand_x_tensor, w13_int8, w13_scale, w2, out_tensor = gen_input(b, s, hidden_size, intermediate_size, x_dtype, device_id)
        inputs = {
            expand_x_tensor: [0],
            w13_int8: [],
            w13_scale: [],
            w2: []
        }
        outputs = {
            out_tensor: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        dense_moe_main(pto_inputs, pto_outputs)
        pypto.runtime._device_synchronize()

        # golden
        golden = moe_torch_npu(expand_x_tensor, w13_int8, w13_scale, w2)
        assert_allclose(np.array(out_tensor.cpu().flatten().tolist()), np.array(golden.cpu().flatten().tolist()), rtol=0.0078125, atol=0.0001)


if __name__ == "__main__":
    main()