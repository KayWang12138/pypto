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
import pytest
from numpy.testing import assert_allclose
from glm_ffn_quant_common import symmetric_quantization_per_token, dequant_dynamic
import torch
import torch_npu
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph


def main():
    test_glm4_ffn_router()


def ffn_router_torch_npu(expand_x_int8, expand_x_scale, group_list, w13_int8, w13_scale, w2_int8, w2_scale):
    group_list = group_list.to(torch.int64)
    group_list_cumsum = group_list.cumsum(dim=0)
    output_dtype = w2_scale.dtype
    w13_int8_nz = torch_npu.npu_format_cast(w13_int8, 29)
    hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=expand_x_int8,
                weight=w13_int8_nz,
                bias=None,
                group_list=group_list_cumsum,
                weight_scale=w13_scale,
                x_scale=expand_x_scale)
    hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2_int8],
            scale=[w2_scale],
            bias=None,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype)[0]
    return hidden_states


def get_token_acc_table(group_list):
    assert len(group_list.shape) == 1
    group_list_cumsum = torch.zeros_like(group_list)
    for i in range(1, group_list.shape[0]):
        group_list_cumsum[i] = torch.sum(group_list[0:i])
    return group_list_cumsum


def ffn_golden_quan_per_token(x):
    # y_int8 : int8  scale_dequant : x.dtype
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def ffn_golden_quan_per_channel_3d(x):
    # y_int8 : int8  scale_dequant : x.dtype
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def gen_input(b, s, topk, per_expert_num, hidden_size, intermediate_size, dtypes, device_id):
    torch.manual_seed(42)
    expand_x_tensor = torch.randn((b * s * topk, hidden_size), dtype = dtypes, device = f'npu:{device_id}') * 0.01 * 2 - 0.01
    expand_x_int8, expand_x_scale = ffn_golden_quan_per_token(expand_x_tensor)
    expand_x_scale = expand_x_scale.reshape(-1).to(torch.float32)

    group_list = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype = torch.int32, device = f'npu:{device_id}')
    group_list_cumsum = get_token_acc_table(group_list).to(torch.int32)
    w13 = torch.randn((per_expert_num, hidden_size, intermediate_size * 2), dtype = dtypes, device = f'npu:{device_id}')  * 0.01 * 2 - 0.01
    w13_int8, w13_scale = ffn_golden_quan_per_channel_3d(w13)
    w13_scale = w13_scale.squeeze(1).to(torch.float32)

    w2 = torch.randn((per_expert_num, intermediate_size, hidden_size), dtype = dtypes, device = f'npu:{device_id}')  * 0.01 * 2 - 0.01
    w2_int8, w2_scale = ffn_golden_quan_per_channel_3d(w2)
    w2_scale = w2_scale.squeeze(1).to(dtypes)

    out_tensor = torch.zeros_like(expand_x_tensor, device = f'npu:{device_id}')
    return expand_x_int8, expand_x_scale, group_list, group_list_cumsum, w13_int8, w13_scale, w2_int8, w2_scale, out_tensor


def expert_infer_base(**kwargs):
    # 入参信息获取
    exp_idx = kwargs.get("exp_idx")
    token_loop_idx = kwargs.get("token_loop_idx")
    loop_base = kwargs.get("loop_base")
    expand_x_int8 = kwargs.get("expand_x_int8")
    expand_x_scale = kwargs.get("expand_x_scale")
    group_list = kwargs.get("group_list")
    group_list_cumsum = kwargs.get("group_list_cumsum")
    w13_int8 = kwargs.get("w13_int8")
    w13_scale = kwargs.get("w13_scale")
    w2_int8 = kwargs.get("w2_int8")
    w2_scale = kwargs.get("w2_scale")
    ffn_out = kwargs.get("ffn_out")
    mm1_cube_tile_shape = kwargs.get("mm1_cube_tile_shape")
    mm2_cube_tile_shape = kwargs.get("mm2_cube_tile_shape")

    hidden_size = expand_x_int8.shape[1]
    intermediate_size = w13_int8.shape[1] // 2
    x_Dtype = w2_scale.dtype

    # 计算对应激活专家的偏移地址
    pypto.set_vec_tile_shapes(32)
    # 获取该激活专家的当前loop参与计算的token
    token_num = group_list[exp_idx,]

    # 获取该激活专家在当前loop参与计算部分，有效token的偏移地址和scale偏移地址
    expand_x_offset_start = group_list_cumsum[exp_idx, ]
    expand_x_offset = [expand_x_offset_start + token_loop_idx * loop_base, 0]
    x_scale_offset = [expand_x_offset_start + token_loop_idx * loop_base, 0]

    # 获取该激活专家权重和scale的偏移地址
    weight_13_offset = [exp_idx * hidden_size, 0]
    w13_scale_offset = [exp_idx, 0]
    weight_2_offset = [exp_idx * intermediate_size, 0]
    w2_scale_offset = [exp_idx, 0]

    # 获取当前专家的实际token数和scale
    cur_valid_size = pypto.min(token_num - token_loop_idx * loop_base, loop_base)
    x = pypto.view(expand_x_int8, [loop_base, hidden_size], expand_x_offset, valid_shape=[cur_valid_size, hidden_size])
    x_scale = pypto.view(expand_x_scale, [loop_base, 1], x_scale_offset, valid_shape=[cur_valid_size, 1])

    # 获取当前专家的weght_13和scale
    w13_int8_2d = pypto.view(w13_int8, [hidden_size, intermediate_size * 2], weight_13_offset)
    w13_exp_scale = pypto.view(w13_scale, [1, intermediate_size * 2], w13_scale_offset)

    # # 获取当前专家的weght_2和scale
    w2_int8_2d = pypto.view(w2_int8, [intermediate_size, hidden_size], weight_2_offset)
    w2_exp_scale = pypto.view(w2_scale, [1, hidden_size], w2_scale_offset)

    # up_proj的matmul计算
    pypto.set_cube_tile_shapes([mm1_cube_tile_shape[0], mm1_cube_tile_shape[0]], [mm1_cube_tile_shape[1], mm1_cube_tile_shape[1]], [mm1_cube_tile_shape[2], mm1_cube_tile_shape[2]])
    pypto.set_matrix_size({loop_base, w13_int8_2d.shape[0], w13_int8_2d.shape[1]})
    gate_int32 = pypto.matmul(x, w13_int8_2d, pypto.DT_INT32)

    # dequant
    pypto.set_vec_tile_shapes(1, intermediate_size * 2)
    gate = dequant_dynamic(gate_int32, w13_exp_scale, x_scale)

    gate_left = pypto.view(gate, [loop_base, intermediate_size], [0, 0])
    gate_right = pypto.view(gate, [loop_base, intermediate_size], [0, intermediate_size])

    # SwiGlu & mul : [x / (1 + e^(-x)) * right]
    pypto.set_vec_tile_shapes(1, intermediate_size)
    swiglu_a = pypto.mul(gate_left, -1.0)
    swiglu_b = pypto.exp(swiglu_a)
    swiglu_c = pypto.add(swiglu_b, 1.0)
    swiglu_out = pypto.div(gate_left, swiglu_c)
    swiglu = pypto.mul(swiglu_out, gate_right)

    # down_proj
    # quant
    swiglu_int8, swiglu_int8_scale = symmetric_quantization_per_token(swiglu)

    pypto.set_cube_tile_shapes([mm2_cube_tile_shape[0], mm2_cube_tile_shape[0]], [mm2_cube_tile_shape[1], mm2_cube_tile_shape[1]], [mm2_cube_tile_shape[2], mm2_cube_tile_shape[2]])
    pypto.set_matrix_size({loop_base, w2_int8_2d.shape[0], w2_int8_2d.shape[1]})
    out_int32 = pypto.matmul(swiglu_int8, w2_int8_2d, pypto.DT_INT32)

    # dequant
    pypto.set_vec_tile_shapes(1, hidden_size)
    gate = dequant_dynamic(out_int32, w2_exp_scale, swiglu_int8_scale)
    out = pypto.cast(gate, x_Dtype)
    pypto.assemble(out, expand_x_offset, ffn_out)


# tiling config
mm1_cube_tile_shape = (8, 256, 256)
mm2_cube_tile_shape = (8, 256, 256)
loop_base = 8


@pypto.jit
def moe_router_expert_main(inputs, outputs):
    pypto.set_host_options(only_codegen=True)
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    pypto.set_runtime_options(machine_sched_mode=1)

    # expand_x_int8, expand_x_scale, group_list, group_list_cumsum, w13_int8, w13_scale, w2_int8, w2_scale
    expand_x_int8 = inputs[0]
    expand_x_scale = inputs[1]
    group_list = inputs[2]
    group_list_cumsum = inputs[3]
    w13_int8 = inputs[4]
    w13_scale = inputs[5]
    w2_int8 = inputs[6]
    w2_scale = inputs[7]
    ffn_out = outputs[0]

    # 获取当前device上专家总数
    expert_num = group_list.shape[0]

    w13_2d_shape = (w13_int8.shape[0] * w13_int8.shape[1], w13_int8.shape[2])
    w2_2d_shape = (w2_int8.shape[0] * w2_int8.shape[1], w2_int8.shape[2])
    expand_x_scale_shape = (expand_x_scale.shape[0], 1)

    for _ in pypto.loop(0, 1, 1, name="LOOP_FFN_ROUTER_MLP_RESHAPE", idx_name="reshape_idx"):
        w13_2d = pypto.reshape(w13_int8, w13_2d_shape, inplace=True)
        w2_int8_2d = pypto.reshape(w2_int8, w2_2d_shape, inplace=True)
        expand_x_scale_2d = pypto.reshape(expand_x_scale, expand_x_scale_shape, inplace=True)

    for exp_idx in pypto.loop(0, expert_num, 1, name="LOOP_FFN_ROUTER_MLP_L0", idx_name="exp_idx"):
        def loop_expert(exp_idx):
            # 获取激活专家的token数
            token_num = group_list[exp_idx, ]
            # 每个专家单次计算16token，不足部分会进行pad
            exp_loop_times = (token_num + loop_base - 1) // loop_base

            for token_loop_idx in pypto.loop(0, exp_loop_times, 1, name="LOOP_FFN_ROUTER_MLP_L1", idx_name="token_loop_idx"):
                def loop_token(exp_idx, token_loop_idx):
                    expert_infer_base(
                        exp_idx=exp_idx,
                        token_loop_idx=token_loop_idx,
                        loop_base=loop_base,
                        expand_x_int8=expand_x_int8,
                        expand_x_scale=expand_x_scale_2d,
                        group_list=group_list,
                        group_list_cumsum=group_list_cumsum,
                        w13_int8=w13_2d,
                        w13_scale=w13_scale,
                        w2_int8=w2_int8_2d,
                        w2_scale=w2_scale,
                        ffn_out=ffn_out,
                        mm1_cube_tile_shape=mm1_cube_tile_shape,
                        mm2_cube_tile_shape=mm2_cube_tile_shape,
                        )
                loop_token(exp_idx, token_loop_idx)
        loop_expert(exp_idx)


@allow_in_graph
def glm_router_expert_quant(hidden_states, pertoken_scale, group_list, w13, w13_scale, w2, w2_scale):
    if isinstance(hidden_states, FakeTensor):
        return
    x_dtype = w2_scale.dtype
    b_s_topk, hidden_size = hidden_states.shape[0:2]
    group_list_int32 = group_list.to(torch.int32)
    # cumsum torch_npu
    # group_list_cumsum = (torch.cumsum(group_list_int32, dim=0) - group_list_int32).to(torch.int32)

    from glm_ffn_group_list_cumsum import glm_router_expert_cumsum
    group_list_cumsum = glm_router_expert_cumsum(group_list)

    pypto_out = torch.zeros((b_s_topk, hidden_size), dtype=x_dtype, device=hidden_states.device)
    inputs = {
        hidden_states: [0],
        pertoken_scale: [0],
        group_list_int32: [0],
        group_list_cumsum: [0],
        w13: [],
        w13_scale: [],
        w2: [],
        w2_scale: []
    }
    outputs = {
        pypto_out: [0]
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    moe_router_expert_main(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()
    return pypto_out


@pytest.mark.skip(reason="failure")
def test_glm4_ffn_router():
    dtype = torch.bfloat16
    # parameter config
    b = 1
    s = 1
    intermediate_size = 1536
    hidden_size = 5120
    per_expert_num = 20
    topk = 8
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    for i in range(0, 2):
        if (i == 0):
            b = 1
        if (i == 1):
            b = 2
        # expand_x_int8, expand_x_scale, group_list, group_list_cumsum, w13_int8, w13_scale, w2_int8, w2_scale, out_tensor
        expand_x_int8, expand_x_scale, group_list, group_list_cumsum, w13_int8, w13_scale, w2_int8, w2_scale, out_tensor = \
            gen_input(b, s, topk, per_expert_num, hidden_size, intermediate_size, dtype, device_id)

        inputs = {
            expand_x_int8: [0],
            expand_x_scale: [0],
            group_list: [0],
            group_list_cumsum: [0],
            w13_int8: [],
            w13_scale: [],
            w2_int8: [],
            w2_scale: []
        }
        outputs = {
            out_tensor: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        moe_router_expert_main(pto_inputs, pto_outputs)
        pypto.runtime._device_synchronize()

        # golden
        golden = ffn_router_torch_npu(expand_x_int8, expand_x_scale, group_list, w13_int8, w13_scale, w2_int8, w2_scale)

        # calc valid token num for compare
        vaild_token_cumsum = group_list.cumsum(dim=0)
        valid_size = vaild_token_cumsum[vaild_token_cumsum.shape[0] - 1] * hidden_size
        assert_allclose(np.array(out_tensor.cpu().flatten().tolist()[0 : valid_size]), np.array(golden.cpu().flatten().tolist()[0 : valid_size]), rtol=0.0078125, atol=0.0001)


if __name__ == "__main__":
    main()