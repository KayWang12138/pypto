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
import numpy as np
import torch
import torch_npu
import pypto
from numpy.testing import assert_allclose
from glm_ffn_common_interface import symmetric_quantization_per_token, dequant_dynamic, swiglu
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from utils.get_format import get_format


def check_args(
    hidden_states,
    w13,
    w13_scale,
    w2,
    w2_scale
):
    assert hidden_states.dim() == 2
    assert hidden_states.shape[1] == 5120
    assert get_format(hidden_states) == 'ND'
    assert hidden_states.dtype == torch.bfloat16

    assert w13.dim() == 2
    assert w13.shape[0] == 5120
    assert w13.shape[1] == 384
    assert get_format(w13) == 'NZ'
    assert w13.dtype == torch.int8

    assert w13_scale.dim() == 1
    assert w13_scale.shape[0] == 384
    assert get_format(w13_scale) == 'ND'
    assert w13_scale.dtype == torch.bfloat16

    assert w2.dim() == 2
    assert w2.shape[0] == 192
    assert w2.shape[1] == 5120
    assert get_format(w2) == 'NZ'
    assert w2.dtype == torch.int8

    assert w2_scale.dim() == 1
    assert w2_scale.shape[0] == 5120
    assert get_format(w2_scale) == 'ND'
    assert w2_scale.dtype == torch.bfloat16


def main():
    test_ffn_share()


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


def moe_torch_npu(hidden_states, w13, w13_scale, w2, w2_scale):
    x_dtype = hidden_states.dtype
    quantized_x, dynamic_scale = torch_npu.npu_dynamic_quant(hidden_states)
    output_w13 = torch_npu.npu_quant_matmul(
            quantized_x,
            w13,
            w13_scale,
            pertoken_scale=dynamic_scale,
            bias=None,
            output_dtype=x_dtype,
        )
    swiglu_out = torch_npu.npu_swiglu(output_w13)
    quantized_x, x_scale = torch_npu.npu_dynamic_quant(swiglu_out)
    output = torch_npu.npu_quant_matmul(
            quantized_x,
            w2,
            w2_scale,
            pertoken_scale=x_scale,
            bias=None,
            output_dtype=x_dtype,
        )
    return output


def gen_input(b, s, hidden_size, intermediate_size, dtypes, device_id):
    torch.manual_seed(42)
    hidden_states = torch.randn((b * s, hidden_size), dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01

    weight_gate_upper_tensor = torch.randn((hidden_size, intermediate_size * 2),
                                           dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01
    w13, w13_scale = ffn_golden_quan_per_channel(weight_gate_upper_tensor)
    w13_scale = w13_scale.reshape(-1).to(dtypes)

    weight_down_proj_tensor = torch.randn((intermediate_size, hidden_size),
                                          dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01
    w2, w2_scale = ffn_golden_quan_per_channel(weight_down_proj_tensor)
    w2_scale = w2_scale.reshape(-1).to(dtypes)

    ffn_res = torch.empty((b * s, hidden_size), dtype=dtypes, device=f'npu:{device_id}')
    return hidden_states, w13, w13_scale, w2, w2_scale, ffn_res


def expert_infer_base(hidden_states, w13_params, w2_params, ffn_res, tiling_params, offset_params):
    # 入参信息获取
    w13, w13_scale = w13_params
    w2, w2_scale = w2_params
    share_loop_idx, loop_base = offset_params
    vec_tile_shape, mm1_cube_tile_shape, mm2_cube_tile_shape = tiling_params

    token_size, hidden_size = hidden_states.shape[:2]
    intermediate_size = w2.shape[0]
    x_dtype = hidden_states.dtype

    # offset
    hidden_states_offset = [share_loop_idx * loop_base, 0]
    cur_valid_size = pypto.min(token_size - share_loop_idx * loop_base, loop_base)
    hidden_states_actual = pypto.view(hidden_states, [loop_base, hidden_size], hidden_states_offset,
                                      valid_shape=[cur_valid_size, hidden_size])

    pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
    # dynamic per_token_quant
    hidden_states_quant, hidden_states_scale = symmetric_quantization_per_token(hidden_states_actual)

    # up_proj的matmul计算
    pypto.set_cube_tile_shapes([mm1_cube_tile_shape[0], mm1_cube_tile_shape[0]],
                               [mm1_cube_tile_shape[1], mm1_cube_tile_shape[1] * 2],
                               [mm1_cube_tile_shape[2], mm1_cube_tile_shape[2]], True, True)
    up_proj = pypto.matmul(hidden_states_quant, w13, pypto.DT_INT32)

    # dequant
    w13_scale_2d = pypto.unsqueeze(w13_scale, 0)
    pypto.set_vec_tile_shapes(2, intermediate_size * 2)
    up_proj_dequant = dequant_dynamic(up_proj, w13_scale_2d, hidden_states_scale)
    swiglu_out = swiglu(up_proj_dequant)

    # dynamic per_token_quant
    down_proj_quant, down_proj_scale = symmetric_quantization_per_token(swiglu_out)

    # down_proj
    pypto.set_cube_tile_shapes([mm2_cube_tile_shape[0], mm2_cube_tile_shape[0]],
                               [mm2_cube_tile_shape[1], mm2_cube_tile_shape[1] * 2],
                               [mm2_cube_tile_shape[2], mm2_cube_tile_shape[2]], True, False)
    down_proj = pypto.matmul(down_proj_quant, w2, pypto.DT_INT32)

    # dequant
    w2_scale_2d = pypto.unsqueeze(w2_scale, 0)
    pypto.set_vec_tile_shapes(2, hidden_size)
    down_proj_dequant = dequant_dynamic(down_proj, w2_scale_2d, down_proj_scale)
    out = pypto.cast(down_proj_dequant, x_dtype)
    pypto.assemble(out, hidden_states_offset, ffn_res)


@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_unaligned": True,
                     "codegen_expression_fusion": True},
    runtime_options={"device_sched_mode": 1,
                     "cfgcache_device_task_num": 100,
                     "cfgcache_root_task_num": 1000,
                     "cfgcache_leaf_task_num": 10000}
)
def share_expert_moe_main(hidden_states, w13, w13_scale, w2, w2_scale, ffn_res):
    pypto.set_pass_options(l1_reuse=2)

    # tiling config
    vec_tile_shape = (2, 5120)
    mm1_cube_tile_shape = (8, 128, 384)
    mm2_cube_tile_shape = (8, 192, 256)
    loop_base = 8

    token_nums = hidden_states.shape[0]
    token_loop_times = (token_nums + loop_base - 1) // loop_base

    for share_loop_idx in pypto.loop(token_loop_times, name="share_loop_idx"):
        expert_infer_base(
            hidden_states=hidden_states,
            w13_params=[w13, w13_scale],
            w2_params=[w2, w2_scale],
            ffn_res=ffn_res,
            tiling_params=[vec_tile_shape, mm1_cube_tile_shape, mm2_cube_tile_shape],
            offset_params=[share_loop_idx, loop_base]
        )


@allow_in_graph
def ffn_shared_expert_quant(hidden_states: torch.Tensor,
                           w13: torch.Tensor,
                           w13_scale: torch.Tensor,
                           w2: torch.Tensor,
                           w2_scale: torch.Tensor,
                           ffn_res: torch.Tensor
) -> None:
    inputs = {
        hidden_states: [0],
        w13: [],
        w13_scale: [],
        w2: [],
        w2_scale: []
    }
    outputs = {
        ffn_res: [0]
    }
    if not isinstance(hidden_states, FakeTensor):
        check_args(hidden_states, w13, w13_scale, w2, w2_scale)

        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        share_expert_moe_main(*pto_inputs, *pto_outputs)
        pypto.runtime._device_synchronize()


def test_ffn_share():
    x_dtype = torch.bfloat16
    # parameter config
    b = 1
    s = 1
    intermediate_size = 192
    hidden_size = 5120
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    for i in range(0, 2):
        if (i == 0):
            b = 1
        if (i == 1):
            b = 2
        # hidden_states, w13, w13_scale, w2, w2_scale, ffn_res
        hidden_states, w13, w13_scale, w2, w2_scale, ffn_res = \
            gen_input(b, s, hidden_size, intermediate_size, x_dtype, device_id)
        inputs = {
            hidden_states: [0],
            w13: [],
            w13_scale: [],
            w2: [],
            w2_scale: []
        }
        outputs = {
            ffn_res: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        share_expert_moe_main(*pto_inputs, *pto_outputs)
        pypto.runtime._device_synchronize()

        # golden
        golden = moe_torch_npu(hidden_states, w13, w13_scale, w2, w2_scale)
        assert_allclose(np.array(ffn_res.cpu().flatten().tolist()), np.array(golden.cpu().flatten().tolist()),
                        rtol=0.0078125, atol=0.0001)


if __name__ == "__main__":
    main()
