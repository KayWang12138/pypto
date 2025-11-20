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
import os
import pto
import torch
import numpy as np
from numpy.testing import assert_allclose


def main():
    test_rms_main()


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
    res = res_div * gamma
    
    if x_dtype != torch.float32:
        res = res.to(x_dtype)
        x_out = x_f32.to(x_dtype)
    return res, x_out


@pto.jit
def cust_add_rms_norm(in_tensor, out_tensor, eps):
    # 添加支持动态的config
    pto.set_codegen_option("support_dynamic_unaligned", True) 
    pto.set_host_option("only_codegen", True)
    
    # 从入参拿到输入和输出tensor
    residual = in_tensor[0]
    hidden_states = in_tensor[1]
    weight = in_tensor[2]
    
    y = out_tensor[0]
    x = out_tensor[1]
    
    # 设置axis=0为动态shape
    pto.mark_dynamic(residual, 0)
    pto.mark_dynamic(hidden_states, 0)
    pto.mark_dynamic(y, 0)
    pto.mark_dynamic(x, 0)
    
    calc_dtype = pto.DT_FP32
    input_dtype = hidden_states.dtype
    
    m = hidden_states.shape[0]
    n = hidden_states.shape[1]
    view_shape = (16, n)
    tile_shape = [16, 1024]

    bs_loop = (m + view_shape[0] - 1) // view_shape[0]
    # 定义动态函数
    with pto.function("ADD_RMS_NORM", [residual, hidden_states, weight], [y, x]):
        def rms_inside_func():
            # 实现kernel逻辑， 包在函数中实现变量自动回收
            # 循环展开BS动态轴
            for idx_loop in pto.loop(bs_loop, name="LOOP_RMS_NORM_L0", idx_name="idx_loop"):
                def bs_loop_func(idx_loop):
                    # 通过view得到输入
                    tile_residual = pto.view(residual, view_shape, 
                                             [idx_loop * view_shape[0], 0],
                                             valid_shape=[(m - idx_loop * view_shape[0]).min(view_shape[0]), n])
                    tile_hidden_states = pto.view(hidden_states, view_shape, 
                                             [idx_loop * view_shape[0], 0],
                                             valid_shape=[(m - idx_loop * view_shape[0]).min(view_shape[0]), n])

                    # 设置set_vec_tile_shape时 尽可能用满UB，但不要超过UB的大小
                    pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                    
                    mean_coff = 1.0 / tile_hidden_states.shape[-1]
                    
                    # 按照计算图实现逻辑
                    # cast to calc_dtype
                    tile_residual_fp32 = pto.cast(tile_residual, calc_dtype)
                    tile_hidden_states_fp32 = pto.cast(tile_hidden_states, calc_dtype)
                    
                    weight_shape = [1] * len(tile_hidden_states_fp32.shape)
                    weight_shape[-1] = weight.shape[0]
                    weight_2d = pto.reshape(weight, weight_shape)
                    tile_weight_fp32 = pto.cast(weight_2d, calc_dtype)
                    
                    x_f32 = pto.add(tile_residual_fp32, tile_hidden_states_fp32) # tile_hidden_states
                    square = pto.mul(x_f32, x_f32) # square
                    mean_res = pto.mul(square, mean_coff) # mean_res = square * mean_coff
                    reduce_asum = pto.sum(mean_res) # reduce_asum = mean_res.sum(dim=-1, keepdim=True)
                    reduce_sum = pto.add(reduce_asum, eps) # reduce_sum = reduce_asum + eps
                    reduce_sqrt = pto.sqrt(reduce_sum) # reduce_sqrt = torch.sqrt(reduce_sum)
                    res_div = pto.div(x_f32, reduce_sqrt) # res_div = x_f32 / reduce_sqrt
                    res = pto.mul(res_div, tile_weight_fp32) # res = res_div * weight
                    
                    y_output = pto.cast(res, input_dtype)
                    y[idx_loop * pto.symbolic_scalar(view_shape[0]):, pto.symbolic_scalar(0):] = y_output
                    x_output = pto.cast(x_f32, input_dtype)
                    x[idx_loop * pto.symbolic_scalar(view_shape[0]):, pto.symbolic_scalar(0):] = x_output
                bs_loop_func(idx_loop)
        rms_inside_func()
    assert isinstance(y, pto.tensor)
    assert isinstance(x, pto.tensor)

   
def test_rms_main():
    m = 5
    n = 2048
    
    device_id = int(os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    eps = 1e-6
    
    for i in range(0, 4):
        if (i == 2):
            m = 2
        # 准备测试数据
        residual_tensor = torch.rand((m, n), dtype=torch.float16, device=f'npu:{device_id}')
        hidden_states_tensor = torch.rand((m, n), dtype=torch.float16, device=f'npu:{device_id}')
        weight_tensor = torch.rand((n), dtype=torch.float16, device=f'npu:{device_id}')
        
        output_hidden_states = torch.full((m, n), 9, dtype=torch.float16, device=f'npu:{device_id}')    
        output_residual = torch.full((m, n), 7, dtype=torch.float16, device=f'npu:{device_id}')
        
        inputs = [residual_tensor, hidden_states_tensor, weight_tensor]
        outputs = [output_hidden_states, output_residual]
        
        cust_add_rms_norm(inputs, outputs, eps)
        pto.runtime._device_synchronize()
        
        golden_res, golden_x = add_rms_norm_golden(
            residual_tensor.cpu(), 
            hidden_states_tensor.cpu(), 
            weight_tensor.cpu(), 
            eps)
        
        assert_allclose(
            np.array(output_residual.cpu().flatten().tolist()), 
            np.array(golden_x.flatten().tolist()), rtol=0.001, atol=0.001)
        assert_allclose(
            np.array(output_hidden_states.cpu().flatten().tolist()),
            np.array(golden_res.flatten().tolist()), rtol=0.001, atol=0.001)


if __name__ == "__main__":
    main()