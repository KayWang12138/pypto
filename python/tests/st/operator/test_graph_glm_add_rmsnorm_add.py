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
import pytest
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph


def main():
    test_rms_norm_main()


@allow_in_graph
def graph_add_rms_norm_custom(inputs, outputs, eps):
    if isinstance(inputs[0], FakeTensor):
        return
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    add_rms_norm_custom(pto_inputs, pto_outputs, eps)
    pypto.runtime._device_synchronize()


def add_rms_norm_golden(hidden_states, residual, gamma, bias_input, eps):
    x_dtype = residual.dtype
    res_add = residual.to(torch.float32) + hidden_states.to(torch.float32)
    mean_coff = 1.0 / res_add.shape[-1]
    x_f32 = res_add
    square = x_f32 * x_f32
    square = square.sum(dim=-1, keepdim=True)
    mean_res = square * mean_coff
    reduce_sum = mean_res + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt
    res = res_div * gamma.to(torch.float32)
    res = res + bias_input.to(res.dtype)
    if x_dtype != torch.float32:
        res = res.to(x_dtype)
        x_out = x_f32.to(x_dtype)
    return res, x_out


def post_attention_layernorm_pto(layer_input_layernorm,
                                 hidden_states: torch.Tensor,
                                 residual: torch.Tensor
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    input_norm_bias = layer_input_layernorm.bias
    input_norm_eps = layer_input_layernorm.variance_epsilon
    input_norm_weight = layer_input_layernorm.weight
    bs, h_num = hidden_states.shape
    device_info = hidden_states.device
    output_hidden_states = torch.zeros((bs, h_num), dtype=hidden_states.dtype, device=device_info)
    output_residual = torch.zeros((bs, h_num), dtype=hidden_states.dtype, device=device_info)
    inputs = [hidden_states, residual, input_norm_weight, input_norm_bias]
    outputs = [output_hidden_states, output_residual]

    graph_add_rms_norm_custom(inputs, outputs, input_norm_eps)
    return output_hidden_states, output_residual


@pypto.jit
def add_rms_norm_custom(in_tensor, out_tensor, eps):
    # 添加支持动态的config
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    pypto.set_host_options(only_codegen=True)

    # 从入参拿到输入和输出tensor
    hidden_states = in_tensor[0]
    residual = in_tensor[1]
    weight = in_tensor[2]
    bias_input = in_tensor[3]

    hidden_states_out = out_tensor[0]
    residual_out = out_tensor[1]

    # 设置axis = 0为动态shape
    pypto.mark_dynamic(residual, 0)
    pypto.mark_dynamic(hidden_states, 0)
    pypto.mark_dynamic(hidden_states_out, 0)
    pypto.mark_dynamic(residual_out, 0)

    calc_dtype = pypto.DT_FP32
    input_dtype = hidden_states.dtype

    bs = hidden_states.shape[0]
    h_num = hidden_states.shape[1]
    view_shape = (128, h_num)
    tile_shape_rmsnorm = [16, 1024]

    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]

    # 实现kernel逻辑， 包在函数中实现变量自动回收
    # 循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_RMS_NORM_L0", idx_name="bs_idx"):
        def bs_loop_func(bs_idx):
            # 通过view得到输入
            tile_residual = pypto.view(residual, view_shape,
                                        [bs_idx * view_shape[0], 0],
                                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), h_num])
            tile_hidden_states = pypto.view(hidden_states, view_shape,
                                            [bs_idx * view_shape[0], 0],
                                            valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                            h_num])

            # 设置set_vec_tile_shape时 尽可能用满UB，但不要超过UB的大小
            pypto.set_vec_tile_shapes(tile_shape_rmsnorm[0], tile_shape_rmsnorm[1])

            mean_coff = 1.0 / tile_hidden_states.shape[-1]

            # 按照计算图实现逻辑
            # cast to calc_dtype
            tile_residual_fp32 = pypto.cast(tile_residual, calc_dtype)
            tile_hidden_states_fp32 = pypto.cast(tile_hidden_states, calc_dtype)

            weight_shape = [1] * len(tile_hidden_states_fp32.shape)
            weight_shape[-1] = weight.shape[0]
            weight_2d = pypto.reshape(weight, weight_shape)
            tile_weight_fp32 = pypto.cast(weight_2d, calc_dtype)

            x_f32 = pypto.add(tile_residual_fp32, tile_hidden_states_fp32)

            square = pypto.mul(x_f32, x_f32)
            mean_res = pypto.mul(square, mean_coff)
            reduce_asum = pypto.sum(mean_res, -1, True)
            reduce_sum = pypto.add(reduce_asum, eps)
            reduce_sqrt = pypto.sqrt(reduce_sum)
            res_div = pypto.div(x_f32, reduce_sqrt)
            res = pypto.mul(res_div, tile_weight_fp32)

            bias_fp32 = pypto.cast(bias_input, res.dtype)
            hidden_states_add = pypto.add(res, bias_fp32)
            hidden_states_out_tmp = pypto.cast(hidden_states_add, input_dtype)

            hidden_states_out[bs_idx * view_shape[0]:, 0:] = hidden_states_out_tmp
            residual_out_16 = pypto.cast(x_f32, input_dtype)
            residual_out[bs_idx * view_shape[0]:, 0:] = residual_out_16

        bs_loop_func(bs_idx)

    assert isinstance(hidden_states_out, pypto.tensor)
    assert isinstance(residual_out, pypto.tensor)


def test_rms_norm_main():
    bs = 8
    h_num = 5120

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    eps = 1e-5

    for i in range(0, 2):
        if (i == 1):
            bs = 2

        # 准备测试数据
        residual_tensor = torch.rand((bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        hidden_states_tensor = torch.rand((bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        weight_tensor = torch.rand((h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        bias_input = torch.rand((h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')

        output_hidden_states = torch.zeros((bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        output_residual = torch.zeros((bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')

        inputs = [hidden_states_tensor, residual_tensor, weight_tensor, bias_input]
        outputs = [output_hidden_states, output_residual]

        pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
        pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
        add_rms_norm_custom(pto_inputs, pto_outputs, eps)
        pypto.runtime._device_synchronize()

        golden_hidden_states, golden_residual = add_rms_norm_golden(hidden_states_tensor,
                                                                    residual_tensor, weight_tensor, bias_input, eps)

        assert_allclose(np.array(output_hidden_states.cpu().flatten().tolist()),
                        np.array(golden_hidden_states.flatten().tolist()), rtol=8e-3, atol=8e-3)
        assert_allclose(np.array(output_residual.cpu().flatten().tolist()),
                        np.array(golden_residual.flatten().tolist()), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    main()
