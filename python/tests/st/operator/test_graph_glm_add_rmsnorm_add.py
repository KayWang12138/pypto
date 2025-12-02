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
import torch
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph


def powers_of_2(n: int) -> set[int]:
    assert n > 0, "n must be positive"
    result = set()
    power = 0
    while True:
        current = 1 << power  # 计算2的power次方
        if current > n:
            break
        result.add(current)
        power += 1
    return result


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
    output_hidden_states = torch.zeros(
        (bs, h_num), dtype=hidden_states.dtype, device=device_info)
    output_residual = torch.zeros(
        (bs, h_num), dtype=hidden_states.dtype, device=device_info)
    inputs = [hidden_states, residual, input_norm_weight, input_norm_bias]
    outputs = [output_hidden_states, output_residual]

    graph_add_rms_norm_custom(inputs, outputs, input_norm_eps)
    return output_hidden_states, output_residual


@pypto.jit
def add_rms_norm_custom(in_tensor, out_tensor, eps):
    # 添加支持动态的config
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    pypto.set_host_options(only_codegen=True)
    pypto.set_runtime_options(cfgcache_device_task_num=100)
    pypto.set_runtime_options(cfgcache_root_task_num=100)
    pypto.set_runtime_options(cfgcache_leaf_task_num=10000)
    # 泳道图使能  pypto.set_option('profile_enable', True)

    # 从入参拿到输入和输出tensor
    x = in_tensor[0]
    residual_input = in_tensor[1]
    x_gamma = in_tensor[2]
    x_bias = in_tensor[3]

    hidden_states_out = out_tensor[0]
    residual_out = out_tensor[1]

    # 设置axis = 0为动态shape
    pypto.mark_dynamic(residual_input, 0)
    pypto.mark_dynamic(x, 0)
    pypto.mark_dynamic(hidden_states_out, 0)
    pypto.mark_dynamic(residual_out, 0)

    calc_dtype = pypto.DT_FP32
    input_dtype = x.dtype
    x_mean_coff = 1.0 / x.shape[-1]

    bs = x.shape[0]
    hidden_size = x.shape[1]
    view_shape = (8, hidden_size)

    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]

    # 实现kernel逻辑， 包在函数中实现变量自动回收
    for _ in pypto.loop(1, name="LOOP_RESHAPE_INPLACE", idx_name="_"):
        pypto.set_vec_tile_shapes(hidden_size)
        x_gamma_2d = pypto.reshape(x_gamma, [1, hidden_size], inplace=True)
        x_bias_2d = pypto.reshape(x_bias, [1, hidden_size], inplace=True)

    # 循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_RMS_NORM_L0", idx_name="bs_idx"):
        x_tile = pypto.view(x, view_shape, [bs_idx * view_shape[0], 0],
                            valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), hidden_size])
        residual_input_tile = pypto.view(residual_input, view_shape, [bs_idx * view_shape[0], 0],
                                            valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                        hidden_size])

        pypto.set_vec_tile_shapes(1, hidden_size)
        x_tile_fp32 = pypto.cast(x_tile, calc_dtype)

        # add
        residual_input_tile_fp32 = pypto.cast(residual_input_tile, calc_dtype)
        x_f32 = pypto.add(residual_input_tile_fp32, x_tile_fp32)

        # rms norm
        square = pypto.mul(x_f32, x_f32)
        mean_res = pypto.mul(square, x_mean_coff)
        reduce_asum = pypto.sum(mean_res, -1, True)
        reduce_sum = pypto.add(reduce_asum, eps)
        reduce_sqrt = pypto.sqrt(reduce_sum)
        res_div = pypto.div(x_f32, reduce_sqrt)

        hidden_bf16 = pypto.tensor([view_shape[0], hidden_size], pypto.DT_BF16, "hidden_bf16")
        residual_bf16_tmp = pypto.cast(x_f32, input_dtype)
        for tmp_idx in range(view_shape[0]):
            x_gamma_2d_fp32 = pypto.cast(x_gamma_2d, calc_dtype)
            x_bias_2d_fp32 = pypto.cast(x_bias_2d, calc_dtype)
            res_div_single = pypto.view(res_div, [1, hidden_size], [tmp_idx, 0])
            res = pypto.mul(res_div_single, x_gamma_2d_fp32)
            res_add = pypto.add(res, x_bias_2d_fp32)
            x_norm = pypto.cast(res_add, input_dtype)
            hidden_bf16[tmp_idx:tmp_idx + 1, 0:] = x_norm

        residual_out[bs_idx * pypto.symbolic_scalar(view_shape[0]):, 0:] = residual_bf16_tmp
        hidden_states_out[bs_idx * pypto.symbolic_scalar(view_shape[0]):, 0:] = hidden_bf16


def test_rms_norm_main():
    bs = 8
    h_num = 5120

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    eps = 1e-5

    for i in range(0, 3):
        if (i == 1):
            bs = 16
        if (i == 2):
            bs = 1037
        # 准备测试数据
        residual_tensor = torch.rand(
            (bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        hidden_states_tensor = torch.rand(
            (bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        weight_tensor = torch.rand(
            (h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        bias_input = torch.rand(
            (h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')

        output_hidden_states = torch.zeros(
            (bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        output_residual = torch.zeros(
            (bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')

        inputs = [hidden_states_tensor,
                  residual_tensor, weight_tensor, bias_input]
        outputs = [output_hidden_states, output_residual]

        pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
        pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
        add_rms_norm_custom(pto_inputs, pto_outputs, eps)
        pypto.runtime._device_synchronize()

        golden_hidden_states, golden_residual = add_rms_norm_golden(hidden_states_tensor,
                                                                    residual_tensor, weight_tensor, bias_input, eps)

        assert_allclose(np.array(output_residual.cpu().flatten().tolist()),
                        np.array(golden_residual.flatten().tolist()), rtol=1e-5, atol=1e-5)
        assert_allclose(np.array(output_hidden_states.cpu().flatten().tolist()),
                        np.array(golden_hidden_states.flatten().tolist()), rtol=8e-3, atol=8e-3)


if __name__ == "__main__":
    main()
