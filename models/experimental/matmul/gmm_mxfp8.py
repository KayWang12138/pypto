#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import os
import sys
import math
import time
from dataclasses import dataclass

import numpy as np
import pypto
import torch
import torch_npu
from numpy.testing import assert_allclose

def compute_golden_result(x, weight, scaled_x_golden, scaled_weight_golden, a_trans, b_trans):
    if a_trans:
        x = np.swapaxes(x, -1, -2)
        scaled_x_golden = np.swapaxes(scaled_x_golden, -1, -2)
        if len(scaled_x_golden.shape) == 3:
            scaled_x_golden = scaled_x_golden.reshape(scaled_x_golden.shape[0] * scaled_x_golden.shape[1], scaled_x_golden.shape[2])
        scaled_x_golden = np.swapaxes(scaled_x_golden, -1, -2)
    else:
        if len(scaled_x_golden.shape) == 3:
            scaled_x_golden = scaled_x_golden.reshape(scaled_x_golden.shape[0], scaled_x_golden.shape[1] * scaled_x_golden.shape[2])
    if b_trans:
        weight = np.swapaxes(weight, -1, -2)
        if len(scaled_weight_golden.shape) == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(scaled_weight_golden.shape[0], scaled_weight_golden.shape[1] * scaled_weight_golden.shape[2])
        scaled_weight_golden = np.swapaxes(scaled_weight_golden, -1, -2)
    else:
        scaled_weight_golden = np.swapaxes(scaled_weight_golden, -1, -2)
        if len(scaled_weight_golden.shape) == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(scaled_weight_golden.shape[0] * scaled_weight_golden.shape[1], scaled_weight_golden.shape[2])
    
    k_dim = x.shape[-1]
    if math.ceil(k_dim / 32) % 2 != 0:
        scaled_x_golden = scaled_x_golden[:, :-1]
        scaled_weight_golden = scaled_weight_golden[:-1, :]

    scaled_x_golden_broadcast = torch.repeat_interleave(scaled_x_golden, repeats=32, dim=-1)
    scaled_weight_golden_broadcast = torch.repeat_interleave(scaled_weight_golden, repeats=32, dim=-2)
    x1_dims = len(x.shape)
    x2_dims = len(weight.shape)
    x1_pad_len = scaled_x_golden_broadcast.shape[-1] - x.shape[-1]
    x2_pad_len = scaled_weight_golden_broadcast.shape[-2] - weight.shape[-2]
    x1_pad = [0, x1_pad_len]
    for _ in range(x1_dims - 1):
        x1_pad += [0,0]
    x1_golden = torch.nn.functional.pad(x, x1_pad, mode='constant', value = 0)

    weight_pad = [0,0]
    weight_pad += [0, x2_pad_len]
    for _ in range(x2_dims - 2):
        weight_pad += [0,0]
    weight_golden = torch.nn.functional.pad(weight, weight_pad, mode='constant', value = 0)

    x_fp32 = x.to(torch.float32)
    scaled_x_golden_broadcast_fp32 = scaled_x_golden_broadcast.to(torch.float32)
    x1_golden = x_fp32 * scaled_x_golden_broadcast_fp32

    weight_fp32 = weight.to(torch.float32)
    scaled_weight_golden_broadcast_fp32 = scaled_weight_golden_broadcast.to(torch.float32)
    weight_golden = weight_fp32 * scaled_weight_golden_broadcast_fp32

    golden = torch.matmul(x1_golden, weight_golden)

    return golden

def gen_golden(a, b, scaled_a, scaled_b, group_list, a_trans, b_trans):
    round = b.shape[0]
    result = []
    begin = 0
    end = 0

    for i in range(round):
        if group_list[i] <= 0:
            continue
        begin = end
        end = end + group_list[i]
        if a_trans:
            x = a[:,begin:end]
        else:
            x = a[begin:end,:]
        weight = b[i]
        if a_trans:
            scaled_x_golden = scaled_a[:,begin:end,:]
        else:
            scaled_x_golden = scaled_a[begin:end,:,:]
        scaled_weight_golden = scaled_b[i]

        golden_temp = compute_golden_result(
            x=x,
            weight=weight,
            scaled_x_golden = scaled_x_golden,
            scaled_weight_golden = scaled_weight_golden,
            a_trans = a_trans,
            b_trans = b_trans,
        )
        result.append(golden_temp)
    golden_result = torch.cat(result, dim=0)
    return golden_result

@dataclass
class ShapeConfig:
    ori_shape: list
    tile_size: int
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    vector_tile_shape: list
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    
@pypto.jit
def scaled_matmul_kernel(a: pypto.Tensor, b: pypto.Tensor, scaled_a: pypto.Tensor, scaled_b: pypto.Tensor, out: pypto.Tensor, group_list, tile_config) -> None:
    round = b.shape[0]
    n_size = b.shape[-1]
    begin = 0
    end = 0
    for i in range(round):
        begin = end
        end = end + group_list[i]
        
        x = a[begin:end, :]
        weight = b[i]
        scaled_x = scaled_a[begin:end,:,:]
        pypto.set_vec_tile_shapes(tile_config.vector_tile_shape[0], tile_config.vector_tile_shape[1], tile_config.vector_tile_shape[2], tile_config.vector_tile_shape[3])
        scaled_weight = scaled_b[i]
        
        pypto.set_cube_tile_shapes(tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape,
                                    enable_multi_data_load = True, enable_split_k = True)
        out[begin:end,:] = pypto.scaled_mm(x, weight, pypto.DT_FP32, scaled_x, scaled_weight)
        
def gen_mxfp8(a, b, scaled_a, scaled_b, group_list, tile_config):
    a = a.npu()
    b = b.npu()
    scaled_a = scaled_a.npu()
    scaled_b = scaled_b.npu()
    
    out_shape = (a.shape[0], b.shape[-1])
    out = torch.zeros(out_shape, dtype=torch.float32).npu()
    
    input_tensors = {
        a: [],
        b: [],
        scaled_a: [],
        scaled_b: [],
    }
    output_tensors = {
        out: [],
    }
    
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in input_tensors.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in output_tensors.items()]
    
    scaled_matmul_kernel(*pto_inputs, *pto_outputs, group_list, tile_config)
    out = out.to(torch.float32)
    return out
    
def test_gmm_mxfp8(params):
    m = params["m"]
    k = params["k"]
    n = params["n"]
    group_list = params["group_list"]
    a_trans = params["a_trans"]
    b_trans = params["b_trans"]
    tile_size = params["tile_size"]
    m_tile_shape = params["m_tile_shape"]
    k_tile_shape = params["k_tile_shape"]
    n_tile_shape = params["n_tile_shape"]
    vector_tile_shape = params["vector_tile_shape"]
    
    tile_config = ShapeConfig([m, k, n], tile_size, m_tile_shape, k_tile_shape, n_tile_shape, vector_tile_shape,
                                a_trans, b_trans, False, False, False)
                                
    a = torch.randn((m, k), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
    scaled_a = torch.randn((m, k // 64, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    
    if b_trans:
        b = torch.randn((len(group_list), n, k), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
    else:
        b = torch.randn((len(group_list), k, n), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
        
    if b_trans:
        scaled_b = torch.randn((len(group_list), n, k // 64, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    else:
        scaled_b = torch.randn((len(group_list), k // 64, n, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    
    golden = gen_golden(a, b, scaled_a, scaled_b, group_list, a_trans, b_trans)
    print("golden")
    print(golden)
    
    result = gen_mxfp8(a, b, scaled_a, scaled_b, group_list, tile_config)
    print("result")
    print(result)
    
    assert_allclose(golden.cpu().numpy(), result.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("Gmm mxfp8 completed successfully")
    
def get_params(case_name):
    if case_name == "testcase6":
        params = {
        "m": 16, "k": 512, "n": 7168, "group_list": [7, 9], "a_trans": False, "b_trans": False,
        "tile_size": 256, "m_tile_shape": [9, 9], "k_tile_shape": [256, 256], "n_tile_shape": [256, 256], "vector_tile_shape": [1, 8, 256, 32]    
        }
    else:
        raise Exception(f"Can't get params, Case({case_name})")
    return params

if __name__ == "__main__":
    params = get_params("testcase6")
    test_gmm_mxfp8(params)

