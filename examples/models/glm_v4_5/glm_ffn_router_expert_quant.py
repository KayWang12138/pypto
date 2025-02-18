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
import torch
import torch_npu
import pypto
import numpy as np
from numpy.testing import assert_allclose
from glm_ffn_common_interface import symmetric_quantization_per_token, dequant_dynamic, swiglu
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from utils.get_format import get_format


def check_args(
        hidden_states: torch.Tensor,
        pertoken_scale: torch.Tensor,
        group_list: torch.Tensor,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor
) -> None:
    assert hidden_states.dim() == 2
    assert hidden_states.shape[1] == 5120
    assert get_format(hidden_states) == 'ND'
    assert hidden_states.dtype == torch.int8

    assert pertoken_scale.dim() == 1
    assert get_format(pertoken_scale) == 'ND'
    assert pertoken_scale.dtype == torch.float32

    assert group_list.dim() == 1
    assert get_format(group_list) == 'ND'
    assert group_list.dtype == torch.int32

    assert w13.dim() == 3
    assert w13.shape[1] == 5120
    assert w13.shape[2] == 3072
    assert get_format(w13) == 'NZ'
    assert w13.dtype == torch.int8

    assert w13_scale.dim() == 2
    assert w13_scale.shape[1] == 3072
    assert get_format(w13_scale) == 'ND'
    assert w13_scale.dtype == torch.float32

    assert w2.dim() == 3
    assert w2.shape[1] == 1536
    assert w2.shape[2] == 5120
    assert get_format(w2) == 'NZ'
    assert w2.dtype == torch.int8

    assert w2_scale.dim() == 2
    assert w2_scale.shape[1] == 5120
    assert get_format(w2_scale) == 'ND'
    assert w2_scale.dtype == torch.bfloat16


def main():
    test_ffn_router()


def ffn_router_torch_npu(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    group_list: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor
) -> torch.Tensor:
    group_list = group_list.to(torch.int64)
    group_list_cumsum = group_list.cumsum(dim=0)
    output_dtype = w2_scale.dtype
    w13_int8_nz = torch_npu.npu_format_cast(w13, 29)
    hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w13_int8_nz,
                bias=None,
                group_list=group_list_cumsum,
                weight_scale=w13_scale,
                x_scale=hidden_states_scale)
    hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
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
    return (torch.cumsum(group_list, dim=0) - group_list).to(group_list.dtype)


def ffn_golden_quan_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor per token (per row).

    Args:
        x: Input tensor to quantize

    Returns:
        Tuple of (quantized_int8_tensor, dequantization_scale)
    """
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def ffn_golden_quan_per_channel_3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor per channel (per column) for 3D tensors.

    Note: This function currently uses per-token quantization (dim=1)
    but is kept for backward compatibility.

    Args:
        x: Input tensor to quantize

    Returns:
        Tuple of (quantized_int8_tensor, dequantization_scale)
    """
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def gen_input(
    b: int,
    s: int,
    topk: int,
    per_expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    dtypes: torch.dtype,
    device_id: int
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(42)
    hidden_states = torch.randn((b * s * topk, hidden_size), dtype = dtypes, device = f'npu:{device_id}') * 0.01 * 2 - 0.01
    hidden_states, hidden_states_scale = ffn_golden_quan_per_token(hidden_states)
    hidden_states_scale = hidden_states_scale.reshape(-1).to(torch.float32)

    group_list = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype = torch.int32, device = f'npu:{device_id}')
    group_list_cumsum = get_token_acc_table(group_list).to(torch.int32)
    w13 = torch.randn((per_expert_num, hidden_size, intermediate_size * 2), dtype = dtypes, device = f'npu:{device_id}')  * 0.01 * 2 - 0.01
    w13, w13_scale = ffn_golden_quan_per_channel_3d(w13)
    w13_scale = w13_scale.squeeze(1).to(torch.float32)

    w2 = torch.randn((per_expert_num, intermediate_size, hidden_size), dtype = dtypes, device = f'npu:{device_id}')  * 0.01 * 2 - 0.01
    w2, w2_scale = ffn_golden_quan_per_channel_3d(w2)
    w2_scale = w2_scale.squeeze(1).to(dtypes)

    ffn_res = torch.empty(hidden_states.shape, dtype =w2_scale.dtype, device = f'npu:{device_id}')
    return hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res


def expert_infer_base(
        hidden_states_params,
        group_list_params,
        w13_params,
        w2_params,
        offset_params,
        tiling_params,
        ffn_res):

    # 入参信息获取
    hidden_states, hidden_states_scale = hidden_states_params
    group_list, group_list_cumsum = group_list_params
    w13, w13_scale = w13_params
    w2, w2_scale = w2_params
    exp_idx, token_loop_idx, loop_base = offset_params
    mm1_cube_tile_shape, mm2_cube_tile_shape = tiling_params

    hidden_size = hidden_states.shape[1]
    intermediate_size = w13.shape[1] // 2
    x_dtype = w2_scale.dtype

    # 计算对应激活专家的偏移地址
    pypto.set_vec_tile_shapes(32)
    # 获取该激活专家的当前loop参与计算的token
    token_num = group_list[exp_idx,]

    # 获取该激活专家在当前loop参与计算部分，有效token的偏移地址和scale偏移地址
    hidden_states_offset_start = group_list_cumsum[exp_idx, ]
    hidden_states_offset = [hidden_states_offset_start + token_loop_idx * loop_base, 0]
    x_scale_offset = [hidden_states_offset_start + token_loop_idx * loop_base, 0]

    # 获取该激活专家权重和scale的偏移地址
    weight_13_offset = [exp_idx * hidden_size, 0]
    w13_scale_offset = [exp_idx, 0]
    weight_2_offset = [exp_idx * intermediate_size, 0]
    w2_scale_offset = [exp_idx, 0]

    # 获取当前专家的实际token数和scale
    cur_valid_size = pypto.min(token_num - token_loop_idx * loop_base, loop_base)
    x = pypto.view(hidden_states, [loop_base, hidden_size], hidden_states_offset, valid_shape=[cur_valid_size, hidden_size])
    x_scale = pypto.view(hidden_states_scale, [loop_base, 1], x_scale_offset, valid_shape=[cur_valid_size, 1])

    # 获取当前专家的weght_13和scale
    w13_weight_2d = pypto.view(w13, [hidden_size, intermediate_size * 2], weight_13_offset)
    w13_scale_valid = pypto.view(w13_scale, [1, intermediate_size * 2], w13_scale_offset)

    # # 获取当前专家的weght_2和scale
    w2_weight_2d = pypto.view(w2, [intermediate_size, hidden_size], weight_2_offset)
    w2_scale_valid = pypto.view(w2_scale, [1, hidden_size], w2_scale_offset)

    # up_proj的matmul计算
    pypto.set_cube_tile_shapes([mm1_cube_tile_shape[0], mm1_cube_tile_shape[0]], [mm1_cube_tile_shape[1], mm1_cube_tile_shape[1] * 2], \
                               [mm1_cube_tile_shape[2], mm1_cube_tile_shape[2]], True, True)
    pypto.set_matrix_size([loop_base, w13_weight_2d.shape[0], w13_weight_2d.shape[1]])
    up_proj = pypto.matmul(x, w13_weight_2d, pypto.DT_INT32)

    # dequant
    pypto.set_vec_tile_shapes(1, intermediate_size * 2)
    up_proj_out = dequant_dynamic(up_proj, w13_scale_valid, x_scale)
    swiglu_out = swiglu(up_proj_out)

    # down_proj
    # quant
    down_proj_quant, down_proj_scale = symmetric_quantization_per_token(swiglu_out)

    pypto.set_cube_tile_shapes([mm2_cube_tile_shape[0], mm2_cube_tile_shape[0]], [mm2_cube_tile_shape[1], mm2_cube_tile_shape[1] * 2], \
                               [mm2_cube_tile_shape[2], mm2_cube_tile_shape[2]], True, True)
    pypto.set_matrix_size([loop_base, w2_weight_2d.shape[0], w2_weight_2d.shape[1]])
    down_proj = pypto.matmul(down_proj_quant, w2_weight_2d, pypto.DT_INT32)

    # dequant
    pypto.set_vec_tile_shapes(1, hidden_size)
    down_proj_dequant = dequant_dynamic(down_proj, w2_scale_valid, down_proj_scale)
    out = pypto.cast(down_proj_dequant, x_dtype)
    pypto.assemble(out, hidden_states_offset, ffn_res)


@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"codegen_expression_fusion": True},
    runtime_options={"device_sched_mode": 1},
    pass_options={"cube_l1_reuse_mode": 2}
)
def moe_router_expert_main(hidden_states, hidden_states_scale,
                           group_list, group_list_cumsum, w13,
                           w13_scale, w2, w2_scale, ffn_res):
    # tiling config
    mm1_cube_tile_shape = (8, 256, 256)
    mm2_cube_tile_shape = (8, 256, 256)
    loop_base = 8

    # 获取当前device上专家总数
    expert_num = group_list.shape[0]

    # 输入Tensor shape转换为2维
    w13_2d_shape = (w13.shape[0] * w13.shape[1], w13.shape[2])
    w2_2d_shape = (w2.shape[0] * w2.shape[1], w2.shape[2])
    hidden_states_scale_shape = (hidden_states_scale.shape[0], 1)

    w13_2d = pypto.reshape(w13, w13_2d_shape, inplace=True)
    w2_2d = pypto.reshape(w2, w2_2d_shape, inplace=True)
    hidden_states_scale_2d = pypto.reshape(hidden_states_scale, hidden_states_scale_shape, inplace=True)

    for exp_idx in pypto.loop(expert_num, name="LOOP_FFN_ROUTER_MLP_L0", idx_name="exp_idx"):
        # 获取激活专家的token数
        token_num = group_list[exp_idx, ]
        # 每个专家单次计算16token，不足部分会进行pad
        exp_loop_times = (token_num + loop_base - 1) // loop_base
        for token_loop_idx in pypto.loop(exp_loop_times, name="LOOP_FFN_ROUTER_MLP_L1", idx_name="token_loop_idx"):
            expert_infer_base(
                hidden_states_params=[hidden_states, hidden_states_scale_2d],
                group_list_params=[group_list, group_list_cumsum],
                w13_params=[w13_2d, w13_scale],
                w2_params=[w2_2d, w2_scale],
                offset_params=[exp_idx, token_loop_idx, loop_base],
                tiling_params=[mm1_cube_tile_shape, mm2_cube_tile_shape],
                ffn_res=ffn_res
            )


@allow_in_graph
def ffn_router_expert_quant(hidden_states: torch.Tensor,
                            pertoken_scale: torch.Tensor,
                            group_list: torch.Tensor,
                            w13: torch.Tensor,
                            w13_scale: torch.Tensor,
                            w2: torch.Tensor,
                            w2_scale: torch.Tensor,
                            ffn_res: torch.Tensor
) -> None:
    group_list_int32 = group_list.to(torch.int32)

    group_list_cumsum = (torch.cumsum(group_list_int32, dim=0) - group_list_int32).to(torch.int32)

    inputs = {
        hidden_states: [0],
        pertoken_scale: [0],
        group_list_int32: [],
        group_list_cumsum: [],
        w13: [],
        w13_scale: [],
        w2: [],
        w2_scale: []
    }
    outputs = {
        ffn_res: [0]
    }
    if not isinstance(hidden_states, FakeTensor):
        check_args(hidden_states, pertoken_scale, group_list_int32, w13, w13_scale, w2, w2_scale)
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        moe_router_expert_main(*pto_inputs, *pto_outputs)
        pypto.runtime._device_synchronize()#内部接口，不推荐使用


def test_ffn_router() -> None:
    dtype = torch.bfloat16
    # parameter config
    s = 1
    intermediate_size = 1536
    hidden_size = 5120
    per_expert_num = 20
    topk = 8
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    # Test with different batch sizes
    for b in [1, 2]:
        # hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res
        hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res = \
            gen_input(b, s, topk, per_expert_num, hidden_size, intermediate_size, dtype, device_id)

        inputs = {
            hidden_states: [0],
            hidden_states_scale: [0],
            group_list: [],
            group_list_cumsum: [],
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
        moe_router_expert_main(*pto_inputs, *pto_outputs)
        pypto.runtime._device_synchronize()#内部接口，不推荐使用

        # golden
        golden = ffn_router_torch_npu(hidden_states, hidden_states_scale, group_list, w13, w13_scale, w2, w2_scale)

        # calc valid token num for compare
        vaild_token_cumsum = group_list.cumsum(dim=0)
        valid_size = vaild_token_cumsum[vaild_token_cumsum.shape[0] - 1] * hidden_size
        assert_allclose(np.array(ffn_res.cpu().flatten().tolist()[0 : valid_size]),\
                        np.array(golden.cpu().flatten().tolist()[0 : valid_size]), rtol=0.0078125, atol=0.0001)


if __name__ == "__main__":
    main()