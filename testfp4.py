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
"""
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import pypto
import pytest
from numpy.testing import assert_allclose


K_BLOCK_SIZE_64 = 64
K_BLOCK_SIZE_32 = 32
SHAPE_DIM_2 = 2


@dataclass
class ShapeConfig:
    ori_shape: list #[64, 64, 64]
    m_tile_shape: list #[64, 64]
    k_tile_shape: list #[64, 256]
    n_tile_shape: list #[256, 256]
    view_shape: list #[64, 64]
    in_dtype: pypto.DataType #fp4e2m1
    out_dtype: pypto.DataType #fp16
    a_trans: bool = False #False
    b_trans: bool = False #True
    scale_a_trans: bool = False #True
    scale_b_trans: bool = False #False
    a_format_nz: bool = False #False
    b_format_nz: bool = False #True
    c_format_nz: bool = False


def trans_nd_to_fractal_nz(data: torch.Tensor, keep_m_dim=False):
    def _gen_axes_for_transpose(offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def _ceil_div(a, b):
        return (a + b - 1) // b

    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = (0,) * batch_num
    m0 = 16
    n0 = 32 // data.dtype.itemsize
    if data.dtype == torch.int32:
        n0 = 16
    m1, n1 = _ceil_div(m_ori, m0), _ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    if not keep_m_dim:
        data = F.pad(data, (batch_padding + (0, padding_n, 0, padding_m)), "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
        data = data.reshape(batch_ori + (m1, m0, n1, n0)).permute(*array_trans).contiguous()
    else:
        data = F.pad(data, (batch_padding + (0, padding_n, 0, 0)), "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [1, 0, 2])
        data = data.reshape(batch_ori + (m_ori, n1, n0)).permute(*array_trans).contiguous()
    return data


def convert_pypto_dtype_to_torch(dtype):
    if dtype == pypto.DataType.DT_FP8E4M3:
        return torch.float8_e4m3fn
    elif dtype == pypto.DataType.DT_FP8E5M2:
        return torch.float8_e5m2
    elif dtype == pypto.DataType.DT_FP4_E2M1X2:
        return torch.uint8
    else:
        raise ValueError(f"Unsupported pypto DataType: {dtype}")


def unpack_fp4_to_float32(packed_uint8_tensor, tensor_shape):
    """
    将打包的 uint8 张量 (float4_e2m1fn_x2) 解包并转换为 float32
    假设 packed_uint8_tensor 形状为 [M, K//2]，即每行有 K//2 个字节
    """
    # 1. 分离高 4 位和低 4 位
    # 低 4 位 (第一个数)
    low_nibble = packed_uint8_tensor & 0x0F
    # 高 4 位 (第二个数)
    high_nibble = (packed_uint8_tensor >> 4) & 0x0F
    # 创建一个空张量来存放解包后的索引
    unpacked_indices = torch.empty(tensor_shape, dtype=torch.uint8)
    # 将高、低 4 位交错放入 (假设存储顺序是 [A, B] -> byte = (A<<4) | B)
    # 注意：具体顺序取决于你的数据源定义，这里假设偶数列是低4位，奇数列是高4位
    unpacked_indices[:, 0::2] = low_nibble
    unpacked_indices[:, 1::2] = high_nibble
    # 3. 查表法将 4-bit 索引转换为 float32 数值
    # E2M1 的查找表 (简化版，需根据具体标准如 NVIDIA FP8/FP4 标准调整)
    # 假设位模式: S(1) E(2) M(1)
    # 这里仅做演示，实际需根据 IEEE 或 NVIDIA 标准定义
    fp4_values = torch.tensor([
        0.0,   0.5,   1.0,   1.5,   # 0xxx (正数)
        2.0,   3.0,   4.0,   6.0,   # 1xxx (正数，指数增大)
        -0.0, -0.5,  -1.0,  -1.5,  # ... (负数)
        -2.0, -3.0,  -4.0,  -6.0
    ], dtype=torch.float32)
    # 将索引映射为数值
    float32_matrix = fp4_values[unpacked_indices.view(-1).to(torch.int)].view(tensor_shape)

    return float32_matrix


def create_scaled_mm_kernel_with_mn_split(tile_config: ShapeConfig):
    m, k, n = tile_config.ori_shape
    m_view, n_view = tile_config.view_shape
    a_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.a_format_nz else pypto.TileOpFormat.TILEOP_ND
    b_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.b_format_nz else pypto.TileOpFormat.TILEOP_ND
    a_shape = [k, m] if tile_config.a_trans else [m, k]
    b_shape = [n, k] if tile_config.b_trans else [k, n]
    bias_shape = [1, n]
    scale_a_shape = [k // K_BLOCK_SIZE_64, m, SHAPE_DIM_2] if tile_config.scale_a_trans else \
                    [m, k // K_BLOCK_SIZE_64, SHAPE_DIM_2]
    scale_b_shape = [n, k // K_BLOCK_SIZE_64, SHAPE_DIM_2] if tile_config.scale_b_trans else \
                    [k // K_BLOCK_SIZE_64, n, SHAPE_DIM_2]
    out_shape = [m, n]

    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0},
                        verify_options={"enable_pass_verify": True}
                        )
    def scaled_mm_pto(a_tensor: pypto.Tensor(a_shape, tile_config.in_dtype, format=a_format),
        a_scale: pypto.Tensor(scale_a_shape, pypto.DT_FP8E8M0),
        b_tensor: pypto.Tensor(b_shape, tile_config.in_dtype, format=b_format),
        b_scale: pypto.Tensor(scale_b_shape, pypto.DT_FP8E8M0),
        bias: pypto.Tensor(bias_shape, pypto.DT_FP16),
        out_tensor: pypto.Tensor(out_shape, tile_config.out_dtype),
    ):
        pypto.set_cube_tile_shapes(tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape)
        m_loop = (m + m_view - 1) // m_view
        n_loop = (n + n_view - 1) // n_view
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
                #Get the view tensor of mat_a
                if tile_config.a_trans:
                    a_view = a_tensor[:, m_idx * m_view: m_idx * m_view + m_view]
                else:
                    a_view = a_tensor[m_idx * m_view: m_idx * m_view + m_view, :]
                #Get the view tensor of scale_a
                if tile_config.scale_a_trans:
                    scale_a_view = a_scale[:, m_idx * m_view: m_idx * m_view + m_view, :]
                else:
                    scale_a_view = a_scale[m_idx * m_view: m_idx * m_view + m_view, :, :]
                #Get the view tensor of mat_b
                if tile_config.b_trans:
                    b_view = b_tensor[n_idx * n_view: n_idx * n_view + n_view, :]
                else:
                    b_view = b_tensor[:, n_idx * n_view: n_idx * n_view + n_view]
                #Get the view tensor of scale_b
                if tile_config.scale_b_trans:
                    scale_b_view = b_scale[n_idx * n_view: n_idx * n_view + n_view, :, :]
                else:
                    scale_b_view = b_scale[:, n_idx * n_view: n_idx * n_view + n_view, :]
                #Get the view tensor of bias
                bias_view = bias[:, n_idx * n_view: n_idx * n_view + n_view]
                extend_params = {'bias_tensor': bias_view}
                out_view = pypto.scaled_mm(a_view, b_view, tile_config.out_dtype, scale_a_view, scale_b_view,
                                        extend_params=extend_params, a_trans=tile_config.a_trans,
                                        b_trans=tile_config.b_trans, scale_a_trans=tile_config.scale_a_trans,
                                        scale_b_trans=tile_config.scale_b_trans)
                out_tensor[m_idx * m_view: m_idx * m_view + m_view,
                        n_idx * n_view: n_idx * n_view + n_view] = out_view
    return scaled_mm_pto



def create_scale_mm_with_bias(tile_config: ShapeConfig):
    m = tile_config.ori_shape[0] #64
    k = tile_config.ori_shape[1] #64
    n = tile_config.ori_shape[2] #64
    a_shape = [k // 2, m] if tile_config.a_trans else [m, k // 2] #[64, 32]
    a_shape_ori = [k, m] if tile_config.a_trans else [m, k] #[64, 64]
    b_shape = [n, k // 2] if tile_config.b_trans else [k // 2, n] #[64, 32]
    b_shape_ori = [n, k] if tile_config.b_trans else [k, n] #[64, 64]
    scale_a_shape = [k // K_BLOCK_SIZE_64, m, SHAPE_DIM_2] if tile_config.scale_a_trans else \
                    [m, k // K_BLOCK_SIZE_64, SHAPE_DIM_2] #[1, 64, 2]
    scale_b_shape = [n, k // K_BLOCK_SIZE_64, SHAPE_DIM_2] if tile_config.scale_b_trans else \
                    [k // K_BLOCK_SIZE_64, n, SHAPE_DIM_2] #[1, 64, 2]
    bias_shape = [1, n] # [1, 64]
    bias = torch.randn(bias_shape, dtype=torch.float16).uniform_(-3, 3)

    torch_in_dtype = convert_pypto_dtype_to_torch(tile_config.in_dtype) #fp4->uint8
    mat_a = torch.randn(a_shape, dtype=torch.float32).uniform_(-3, 3).to(torch_in_dtype)
    scale_a = torch.randn(scale_a_shape, dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    mat_b = torch.randn(b_shape, dtype=torch.float32).uniform_(-3, 3).to(torch_in_dtype)
    scale_b = torch.randn(scale_b_shape, dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    scale_a_tmp = scale_a.view(m, k // K_BLOCK_SIZE_32) if not tile_config.scale_a_trans else \
        torch.transpose(scale_a, -2, -1).reshape(k // K_BLOCK_SIZE_32, m).T
    scale_b_tmp = torch.transpose(scale_b, -2, -1).reshape(k // K_BLOCK_SIZE_32, n) if \
        not tile_config.scale_b_trans else scale_b.view(n, k // K_BLOCK_SIZE_32).T
    scale_a_tmp = np.repeat(scale_a_tmp.to(torch.float32), 32, axis=1)
    scale_b_tmp = np.repeat(scale_b_tmp.to(torch.float32), 32, axis=0)

    mat_a_fp4 = unpack_fp4_to_float32(mat_a, a_shape_ori) #fp32
    mat_a_tmp = mat_a_fp4.to(torch.float32).T if tile_config.a_trans else mat_a_fp4.to(torch.float32)
    mat_a_tmp = mat_a_tmp * scale_a_tmp.to(torch.float32)
    mat_b_fp4 = unpack_fp4_to_float32(mat_b, b_shape_ori)
    mat_b_tmp = mat_b_fp4.to(torch.float32).T if tile_config.b_trans else mat_b_fp4.to(torch.float32)
    mat_b_tmp = scale_b_tmp.to(torch.float32) * mat_b_tmp
    bias_tmp = np.repeat(bias, m, axis=0)
    golden = torch.matmul(mat_a_tmp.to(torch.float32), mat_b_tmp.to(torch.float32)) + bias_tmp
    if tile_config.a_format_nz:
        mat_a = trans_nd_to_fractal_nz(mat_a, True)
    if tile_config.b_format_nz:
        mat_b = trans_nd_to_fractal_nz(mat_b, True)
    out = torch.zeros([m, n], dtype=torch.float16).npu()
    print(mat_b.shape)
    mat_b = mat_b.squeeze(0)
    create_scaled_mm_kernel_with_mn_split(tile_config)(mat_a.npu(), scale_a.npu(), mat_b.npu(), scale_b.npu(),
                                                            bias.npu(), out)
    print(out.cpu())
    assert torch.allclose(out.cpu().to(torch.float32), golden, rtol=1e-3, atol=1e-3), "结果精度不匹配"


def test_scaled_mm_with_bias():
    tile_config = ShapeConfig([64, 64, 64], [64, 64], [64, 256], [256, 256], [64, 64], pypto.DataType.DT_FP4_E2M1X2,
                              pypto.DataType.DT_FP16, False, True, True, False, False, True)
    create_scale_mm_with_bias(tile_config)


if __name__ == "__main__":
    test_scaled_mm_with_bias()
