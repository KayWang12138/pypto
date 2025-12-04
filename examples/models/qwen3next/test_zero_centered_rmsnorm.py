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


def add_rms_norm_golden(residual, hidden_states, gamma, eps):
    x = residual + hidden_states
    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]
    
    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff
    
    reduce_sum = mean_res.sum(dim=-1, keepdim=True) + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt
    
    res = res_div * (gamma + 1)
    
    if x_dtype != torch.float32:
        res = res.to(x_dtype)
        x_out = x_f32.to(x_dtype)
    return res, x_out


@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_unaligned": True}
)
def cust_add_rms_norm(in_tensor, out_tensor):
    # 从入参拿到输入和输出tensor
    residual = in_tensor[0]
    hidden_states = in_tensor[1]
    weight = in_tensor[2]
    eps = 1e-6
    
    y = out_tensor[0]
    x = out_tensor[1]
    
    # 设置axis=0为动态shape
    pypto.mark_dynamic(residual, 0)
    pypto.mark_dynamic(hidden_states, 0)
    pypto.mark_dynamic(y, 0)
    pypto.mark_dynamic(x, 0)
    
    calc_dtype = pypto.DT_FP32
    input_dtype = hidden_states.dtype
    
    m = hidden_states.shape[0]
    n = hidden_states.shape[1]
    view_shape = (1, n)
    tile_shape = [128, 128]

    bs_loop = (m + view_shape[0] - 1) // view_shape[0]

    # 循环展开BS动态轴
    for idx_loop in pypto.loop(bs_loop, name="LOOP_RMS_NORM_L0", idx_name="idx_loop"):
        # 通过view得到输入
        tile_residual = pypto.view(residual, view_shape, [idx_loop * view_shape[0], 0],
                                    valid_shape=[(m - idx_loop * view_shape[0]).min(view_shape[0]), n])
        tile_hidden_states = pypto.view(hidden_states, view_shape, 
                                    [idx_loop * view_shape[0], 0],
                                    valid_shape=[(m - idx_loop * view_shape[0]).min(view_shape[0]), n])

        # 设置set_vec_tile_shape时 尽可能用满UB，但不要超过UB的大小
        pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
        
        mean_coff = 1.0 / tile_hidden_states.shape[-1]
        
        # 按照计算图实现逻辑
        # cast to calc_dtype
        tile_residual_fp32 = pypto.cast(tile_residual, calc_dtype)
        tile_hidden_states_fp32 = pypto.cast(tile_hidden_states, calc_dtype)
        
        weight_shape = [1] * len(tile_hidden_states_fp32.shape)
        weight_shape[-1] = weight.shape[0]
        weight_2d = pypto.reshape(weight, weight_shape)
        tile_weight_fp32 = pypto.cast(weight_2d, calc_dtype)
        
        x_f32 = pypto.add(tile_residual_fp32, tile_hidden_states_fp32) # tile_hidden_states
        square = pypto.mul(x_f32, x_f32) # square
        mean_res = pypto.mul(square, mean_coff) # mean_res = square * mean_coff
        reduce_asum = pypto.sum(mean_res, keepdim=True) # reduce_asum = mean_res.sum(dim=-1, keepdim=True)
        reduce_sum = pypto.add(reduce_asum, eps) # reduce_sum = reduce_asum + eps
        reduce_sqrt = pypto.sqrt(reduce_sum) # reduce_sqrt = torch.sqrt(reduce_sum)
        res_div = pypto.div(x_f32, reduce_sqrt) # res_div = x_f32 / reduce_sqrt
        res = pypto.mul(res_div, pypto.add(tile_weight_fp32, 1.0)) # res = res_div * weight
        
        y_output = pypto.cast(res, input_dtype)
        y[idx_loop * pypto.symbolic_scalar(view_shape[0]):, pypto.symbolic_scalar(0):] = y_output
        x_output = pypto.cast(x_f32, input_dtype)
        x[idx_loop * pypto.symbolic_scalar(view_shape[0]):, pypto.symbolic_scalar(0):] = x_output
    assert isinstance(y, pypto.tensor)
    assert isinstance(x, pypto.tensor)


def test_rms_norm():
    m = 5
    n = 2048
    
    device_id = int(os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    eps = 1e-6
    
    # 准备测试数据
    residual_tensor = torch.rand((m, n), dtype=torch.float16, device=f'npu:{device_id}')
    hidden_states_tensor = torch.rand((m, n), dtype=torch.float16, device=f'npu:{device_id}')
    weight_tensor = torch.full((n, ), 0, dtype=torch.int32, device=f'npu:{device_id}')
    output_hidden_states = torch.full((m, n), 9, dtype=torch.float16, device=f'npu:{device_id}')    
    output_residual = torch.full((m, n), 7, dtype=torch.float16, device=f'npu:{device_id}')
    
    inputs = [residual_tensor, hidden_states_tensor, weight_tensor]
    outputs = [output_hidden_states, output_residual]

    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    
    cust_add_rms_norm(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()
    
    golden_res, golden_x = add_rms_norm_golden(residual_tensor.cpu(), hidden_states_tensor.cpu(), 
                                weight_tensor.cpu(), eps)
    
    assert_allclose(np.array(output_residual.cpu().flatten().tolist()), np.array(golden_x.flatten().tolist()),
                    rtol=0.001, atol=0.001)
    assert_allclose(np.array(output_hidden_states.cpu().flatten().tolist()), np.array(golden_res.flatten().tolist()),
                    rtol=0.001, atol=0.001)

    
if __name__ == "__main__":
    test_rms_norm()