#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Grouped Matrix Multiplication with MXFP8 Quantization using PyPTO new frontend.
K轴均匀切分，每个group使用不同的K轴块进行量化矩阵乘法，结果累加到输出张量。
"""

import math
from dataclasses import dataclass

import numpy as np
import pypto
import torch
import torch_npu
from numpy.testing import assert_allclose


@dataclass
class GmmGoldenInputs:
    """Golden输入参数: a[K,M]或[M,K], b[K,N]或[N,K], scaled_a/scaled_b, y[g,M,N]"""
    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    y: torch.Tensor
    num_groups: int
    a_trans: bool = True
    b_trans: bool = False


@dataclass
class GmmMxfp8Inputs:
    """NPU输入参数，tile_config包含num_groups和切分配置"""
    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    y: torch.Tensor
    tile_config: 'ShapeConfig'


@dataclass
class ShapeConfig:
    """配置参数: ori_shape[M,K,N], num_groups(分组数), tile_shapes"""
    ori_shape: list
    num_groups: int
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    vector_tile_shape: list
    in_dtype: pypto.DataType = pypto.DT_FP8E4M3
    a_trans: bool = True
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    description: str = ""


@dataclass
class GoldenComputeInputs:
    """单组矩阵乘参数"""
    x: torch.Tensor
    weight: torch.Tensor
    scaled_x: torch.Tensor
    scaled_weight: torch.Tensor
    a_trans: bool
    b_trans: bool


def compute_golden_result(inputs: GoldenComputeInputs) -> torch.Tensor:
    """单组矩阵乘Golden计算: 应用scale后执行matmul"""
    x = inputs.x
    weight = inputs.weight
    scaled_x_golden = inputs.scaled_x
    scaled_weight_golden = inputs.scaled_weight
    a_trans = inputs.a_trans
    b_trans = inputs.b_trans

    if a_trans:
        x = torch.swapaxes(x, -1, -2)
        scaled_x_golden = torch.swapaxes(scaled_x_golden, -1, -2)
        if len(scaled_x_golden.shape) == 3:
            scaled_x_golden = scaled_x_golden.reshape(
                scaled_x_golden.shape[0] * scaled_x_golden.shape[1], scaled_x_golden.shape[2]
            )
        scaled_x_golden = torch.swapaxes(scaled_x_golden, -1, -2)
    else:
        if len(scaled_x_golden.shape) == 3:
            scaled_x_golden = scaled_x_golden.reshape(
                scaled_x_golden.shape[0], scaled_x_golden.shape[1] * scaled_x_golden.shape[2]
            )

    if b_trans:
        weight = torch.swapaxes(weight, -1, -2)
        if len(scaled_weight_golden.shape) == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(
                scaled_weight_golden.shape[0] * scaled_weight_golden.shape[1],
                scaled_weight_golden.shape[2]
            )
        scaled_weight_golden = torch.swapaxes(scaled_weight_golden, -1, -2)
    else:
        scaled_weight_golden = torch.swapaxes(scaled_weight_golden, -1, -2)
        if len(scaled_weight_golden.shape) == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(
                scaled_weight_golden.shape[0] * scaled_weight_golden.shape[1],
                scaled_weight_golden.shape[2]
            )

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
        x1_pad += [0, 0]
    x1_golden = torch.nn.functional.pad(x, x1_pad, mode='constant', value=0)

    weight_pad = [0, 0]
    weight_pad += [0, x2_pad_len]
    for _ in range(x2_dims - 2):
        weight_pad += [0, 0]
    weight_golden = torch.nn.functional.pad(weight, weight_pad, mode='constant', value=0)

    x_fp32 = x.to(torch.float32)
    scaled_x_golden_broadcast_fp32 = scaled_x_golden_broadcast.to(torch.float32)
    x1_golden = x_fp32 * scaled_x_golden_broadcast_fp32

    weight_fp32 = weight.to(torch.float32)
    scaled_weight_golden_broadcast_fp32 = scaled_weight_golden_broadcast.to(torch.float32)
    weight_golden = weight_fp32 * scaled_weight_golden_broadcast_fp32

    golden = torch.matmul(x1_golden, weight_golden)
    return golden


def gen_golden(inputs: GmmGoldenInputs) -> torch.Tensor:
    """分组矩阵乘Golden: K轴均匀切分，每组计算matmul后累加到y[i]
    
    数据格式:
    - a_trans=True: a[K,M], 切分第一维
    - a_trans=False: a[M,K], 切分第二维
    - b_trans=True: b[N,K], 切分第二维
    - b_trans=False: b[K,N], 切分第一维
    """
    a = inputs.a
    b = inputs.b
    scaled_a = inputs.scaled_a
    scaled_b = inputs.scaled_b
    y = inputs.y
    num_groups = inputs.num_groups
    a_trans = inputs.a_trans
    b_trans = inputs.b_trans

    k = a.shape[0] if a_trans else a.shape[1]
    k_block = k // num_groups
    golden_result = y.clone()
    
    for i in range(num_groups):
        begin = i * k_block
        end = (i + 1) * k_block
        scale_offset = begin // 64 + i  # MX量化格式: scale偏移 = begin/64 + i
        scale_length = k_block // 64

        if a_trans:
            x = a[begin:end, :]
            scaled_x_golden = scaled_a[scale_offset : scale_offset + scale_length, :, :]
        else:
            x = a[:, begin:end]
            scaled_x_golden = scaled_a[:, scale_offset : scale_offset + scale_length, :]
        
        if b_trans:
            weight = b[:, begin:end]  # b is [N, K], 切分K轴=切分第二维
            scaled_weight_golden = scaled_b[:, scale_offset : scale_offset + scale_length, :]  # scaled_b is [N, (K//64)+g, 2]
        else:
            weight = b[begin:end, :]  # b is [K, N], 切分K轴=切分第一维
            scaled_weight_golden = scaled_b[scale_offset : scale_offset + scale_length, :, :]  # scaled_b is [(K//64)+g, N, 2]

        golden_temp = compute_golden_result(
            GoldenComputeInputs(
                x=x, weight=weight, scaled_x=scaled_x_golden, scaled_weight=scaled_weight_golden,
                a_trans=a_trans, b_trans=b_trans,
            )
        )
        golden_result[i] = golden_result[i] + golden_temp

    return golden_result


@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1},
    pass_options={"cube_nbuffer_setting":{-1:2}}
)
def scaled_matmul_kernel(
    a: pypto.Tensor(),
    b: pypto.Tensor(),
    scaled_a: pypto.Tensor(),
    scaled_b: pypto.Tensor(),
    y: pypto.Tensor(),
    tile_config: ShapeConfig
):
    """PyPTO Kernel: K轴均匀切分，每组执行scaled_mm后累加到y
    
    数据格式:
    - a_trans=True: a[K,M], scaled_a[(K//64)+g,M,2]
    - a_trans=False: a[M,K], scaled_a[M,(K//64)+g,2]
    - b_trans=True: b[N,K], scaled_b[N,(K//64)+g,2]
    - b_trans=False: b[K,N], scaled_b[(K//64)+g,N,2]
    
    约束: K轴64对齐(MX量化)，K可被num_groups整除，内轴32字节对齐
    """
    num_groups = tile_config.num_groups
    m = tile_config.ori_shape[0]
    n = tile_config.ori_shape[2]
    k = tile_config.ori_shape[1]
    k_block = k // num_groups
    
    mm_result_tensor = pypto.tensor([num_groups, m, n], pypto.DT_FP32)
    a_trans = tile_config.a_trans
    b_trans = tile_config.b_trans

    for i in pypto.loop(num_groups):
        begin = i * k_block
        end = (i + 1) * k_block
        scale_offset = begin // 64 + i
        scale_length = k_block // 64

        if a_trans:
            x = a[begin:end, :]
            scaled_x = scaled_a[scale_offset : scale_offset + scale_length, :, :]
        else:
            x = a[:, begin:end]
            scaled_x = scaled_a[:, scale_offset : scale_offset + scale_length, :]
        
        if b_trans:
            weight = b[:, begin:end]
            scaled_weight = scaled_b[:, scale_offset : scale_offset + scale_length, :]
        else:
            weight = b[begin:end, :]
            scaled_weight = scaled_b[scale_offset : scale_offset + scale_length, :, :]

        pypto.set_vec_tile_shapes(
            tile_config.vector_tile_shape[0], tile_config.vector_tile_shape[1],
            tile_config.vector_tile_shape[2], tile_config.vector_tile_shape[3]
        )
        pypto.set_cube_tile_shapes(
            tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape
        )
        
        scale_a_trans = a_trans
        mm_result_tensor[i] = pypto.scaled_mm(
            x, weight, pypto.DT_FP32, scaled_x, scaled_weight,
            a_trans=a_trans, scale_a_trans=scale_a_trans, b_trans=b_trans
        )
    
    y[:,:,:] = pypto.add(y, mm_result_tensor)


def gen_mxfp8(inputs: GmmMxfp8Inputs) -> torch.Tensor:
    """执行PyPTO kernel并返回结果"""
    a = inputs.a.npu()
    b = inputs.b.npu()
    scaled_a = inputs.scaled_a.npu()
    scaled_b = inputs.scaled_b.npu()
    y = inputs.y.npu()

    scaled_matmul_kernel(a, b, scaled_a, scaled_b, y, inputs.tile_config)
    return y.to(torch.float32)


def test_gmm_mxfp8(tile_config: ShapeConfig):
    """测试函数: 生成数据、计算Golden和PyPTO结果、对比验证"""
    m = tile_config.ori_shape[0]
    k = tile_config.ori_shape[1]
    n = tile_config.ori_shape[2]
    num_groups = tile_config.num_groups
    in_dtype = tile_config.in_dtype
    a_trans = tile_config.a_trans
    b_trans = tile_config.b_trans

    torch_dtype_map = {
        pypto.DT_FP8E4M3: torch.float8_e4m3fn,
        pypto.DT_FP8E5M2: torch.float8_e5m2,
    }
    torch_dtype = torch_dtype_map.get(in_dtype, torch.float8_e4m3fn)

    if a_trans:
        a = torch.randn((k, m), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scaled_a = torch.randn((k // 64 + num_groups, m, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    else:
        a = torch.randn((m, k), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scaled_a = torch.randn((m, k // 64 + num_groups, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    if b_trans:
        b = torch.randn((n, k), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scaled_b = torch.randn((n, k // 64 + num_groups, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    else:
        b = torch.randn((k, n), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scaled_b = torch.randn((k // 64 + num_groups, n, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    y_init = torch.randn((num_groups, m, n), dtype=torch.float32)
    y_init_npu = y_init.clone().npu()

    golden = gen_golden(GmmGoldenInputs(
        a=a, b=b, scaled_a=scaled_a, scaled_b=scaled_b, y=y_init,
        num_groups=num_groups, a_trans=a_trans, b_trans=b_trans,
    ))
    result = gen_mxfp8(GmmMxfp8Inputs(
        a=a, b=b, scaled_a=scaled_a, scaled_b=scaled_b, y=y_init_npu, tile_config=tile_config,
    ))

    assert_allclose(golden.cpu().numpy(), result.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(tile_config.description, "PASSED")


if __name__ == "__main__":
    test_gmm_mxfp8(
        ShapeConfig(
            ori_shape=[768, 7168, 4096], num_groups=28,
            m_tile_shape=[32, 32], k_tile_shape=[64, 256], n_tile_shape=[512, 2048],
            vector_tile_shape=[1, 4, 4096, 2],
            in_dtype=pypto.DT_FP8E4M3, a_trans=True, b_trans=False,
            a_format_nz=False, b_format_nz=False, c_format_nz=False,
            description="Case1: FP8E4M3, M=768, K=7168, N=4096, num_groups=28"
        )
    )
    
    test_gmm_mxfp8(
        ShapeConfig(
            ori_shape=[64, 1024, 4096], num_groups=2,
            m_tile_shape=[64, 64], k_tile_shape=[64, 512], n_tile_shape=[512, 2048],
            vector_tile_shape=[1, 8, 2048, 2],
            in_dtype=pypto.DT_FP8E4M3, a_trans=True, b_trans=False,
            a_format_nz=False, b_format_nz=False, c_format_nz=False,
            description="Case2: FP8E4M3, M=64, K=1024, N=4096, num_groups=2"
        )
    )
    
    test_gmm_mxfp8(
        ShapeConfig(
            ori_shape=[32, 768, 2048], num_groups=3,
            m_tile_shape=[32, 32], k_tile_shape=[64, 256], n_tile_shape=[512, 1024],
            vector_tile_shape=[1, 4, 1024, 2],
            in_dtype=pypto.DT_FP8E4M3, a_trans=True, b_trans=False,
            a_format_nz=False, b_format_nz=False, c_format_nz=False,
            description="Case3: FP8E4M3, M=32, K=768, N=2048, num_groups=3"
        )
    )
    
    test_gmm_mxfp8(
        ShapeConfig(
            ori_shape=[32, 512, 1024], num_groups=2,
            m_tile_shape=[32, 32], k_tile_shape=[64, 256], n_tile_shape=[512, 1024],
            vector_tile_shape=[1, 4, 1024, 2],
            in_dtype=pypto.DT_FP8E4M3, a_trans=True, b_trans=False,
            a_format_nz=False, b_format_nz=False, c_format_nz=False,
            description="Case4: FP8E4M3, M=32, K=512, N=1024, num_groups=2"
        )
    )

    test_gmm_mxfp8(
        ShapeConfig(
            ori_shape=[32, 512, 1024], num_groups=2,
            m_tile_shape=[32, 32], k_tile_shape=[64, 256], n_tile_shape=[512, 1024],
            vector_tile_shape=[1, 4, 1024, 2],
            in_dtype=pypto.DT_FP8E5M2, a_trans=True, b_trans=False,
            a_format_nz=False, b_format_nz=False, c_format_nz=False,
            description="Case5: FP8E5M2, M=32, K=512, N=1024, num_groups=2"
        )
    )