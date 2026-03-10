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
import os
import numpy as np
import torch
import torch_npu
import pypto
import pytest
from numpy.testing import assert_allclose
import torch.nn.functional as F

FP32 = pypto.DT_FP32
FP16 = pypto.DT_FP16
INT32 = pypto.DT_INT32
INT8 = pypto.DT_INT8
UINT64 = pypto.DT_UINT64
UINT32 = pypto.DT_UINT32


@dataclass
class ShapeConfig:
    ori_shape: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    view_shape: list
    in_dtype: pypto.DataType
    out_dtype: pypto.DataType
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    mdl_flag: bool = False
    gm_acc: bool = False


@dataclass
class ExtendParams:
    bias_shape: list = field(default_factory=list)
    bias_dtype: np.dtype = None
    scale_shape: list = field(default_factory=list)
    scale_dtype: np.dtype = None
    scale: int = None
    relu_type: int = None
    trans_mode: pypto.TransMode = pypto.TransMode.CAST_NONE


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


@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
)
def matmul_kernel_with_mn_split(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    shape_info: ShapeConfig,
):
    m = shape_info.ori_shape[0]
    n = shape_info.ori_shape[2]
    m_view = shape_info.view_shape[0]
    n_view = shape_info.view_shape[1]
    pypto.set_cube_tile_shapes(shape_info.m_tile_shape, shape_info.k_tile_shape, shape_info.n_tile_shape,
                                enable_multi_data_load=shape_info.mdl_flag, enable_split_k=shape_info.gm_acc)
    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            if shape_info.a_trans:
                a_view = a_tensor[:, m_idx * m_view: m_idx * m_view + m_view]
            else:
                a_view = a_tensor[m_idx * m_view: m_idx * m_view + m_view, :]
            if shape_info.b_trans:
                b_view = b_tensor[n_idx * n_view: n_idx * n_view + n_view, :]
            else:
                b_view = b_tensor[:, n_idx * n_view: n_idx * n_view + n_view]
            out_view = pypto.matmul(a_view, b_view, a_trans=shape_info.a_trans, b_trans=shape_info.b_trans,
                                    out_dtype=shape_info.out_dtype)
            out_tensor[m_idx * m_view: m_idx * m_view + m_view, n_idx * n_view: n_idx * n_view + n_view] = out_view


@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
)
def bmm_kernel_with_no_mn_split(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    shape_info: ShapeConfig,
):
    pypto.set_cube_tile_shapes(shape_info.m_tile_shape, shape_info.k_tile_shape, shape_info.n_tile_shape, 
                               enable_multi_data_load=shape_info.mdl_flag, enable_split_k=shape_info.gm_acc)
    result = pypto.matmul(a_tensor, b_tensor, a_trans=shape_info.a_trans, b_trans=shape_info.b_trans,
                          out_dtype=shape_info.out_dtype)
    out_tensor.move(result)


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_mm_with_mn_split():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch_npu.npu.config.allow_internal_format = True
    m = 69
    k = 99
    n = 129
    tile_m = 64
    tile_k = 64
    tile_n = 64
    m_view = 128
    n_view = 256
    shape_info = ShapeConfig([m, k, n], [tile_m, tile_m], [tile_k, tile_k], [tile_n, tile_n], [m_view, n_view], FP16,
                                FP32, True, True, False, False, False, False, False)
    a1_tensor = torch.rand([k, m], dtype=torch.float16, device=f"npu:{device_id}")
    b1_tensor = torch.rand([n, k], dtype=torch.float16, device=f"npu:{device_id}")
    c1_tensor = torch.zeros([m, n], dtype=torch.float32, device=f"npu:{device_id}")
    golden = torch.matmul(a1_tensor.to(torch.float32).T, b1_tensor.to(torch.float32).T)
    matmul_kernel_with_mn_split(
        a1_tensor, b1_tensor, c1_tensor,
        shape_info
    )
    assert torch.allclose(c1_tensor.cpu().to(torch.float32), golden.cpu().to(torch.float32), atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_mm_with_mn_split_nz():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch_npu.npu.config.allow_internal_format = True
    torch.npu.set_device(int(device_id))
    m = 64
    k = 128
    n = 128
    tile_m = 64
    tile_k = 64
    tile_n = 64
    m_view = 128
    n_view = 256
    shape_info = ShapeConfig([m, k, n], [tile_m, tile_m], [tile_k, tile_k], [tile_n, tile_n], [m_view, n_view], FP16,
                                FP32, True, True, True, True, False, False, False)
    a1_tensor = torch.rand([k, m], dtype=torch.float16, device=f'npu:{device_id}')
    b1_tensor = torch.rand([n, k], dtype=torch.float16, device=f'npu:{device_id}')
    c1_tensor = torch.zeros([m, n], dtype=torch.float32, device=f'npu:{device_id}')

    a1_tensor_nz = torch_npu.npu_format_cast(a1_tensor, 29) if shape_info.a_format_nz else a1_tensor
    b1_tensor_nz = torch_npu.npu_format_cast(b1_tensor, 29) if shape_info.b_format_nz else b1_tensor

    golden = torch.matmul(a1_tensor.to(torch.float32).T, b1_tensor.to(torch.float32).T)
    matmul_kernel_with_mn_split(
        a1_tensor_nz, b1_tensor_nz, c1_tensor,
        shape_info
    )
    assert torch.allclose(c1_tensor.cpu().to(torch.float32), golden.cpu().to(torch.float32), atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_bmm_with_mn_split():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    b = 3
    m = 63
    k = 127
    n = 129
    tile_m = 64
    tile_k = 64
    tile_n = 64
    shape_info = ShapeConfig([b, m, k, n], [tile_m, tile_m], [tile_k, tile_k], [tile_n, tile_n], [-1, -1], FP16, FP32,
                                True, False, False, False, False, False, False)
    a1_tensor = torch.rand([b, k, m], dtype=torch.float16, device=f'npu:{device_id}')
    b1_tensor = torch.rand([b, k, n], dtype=torch.float16, device=f'npu:{device_id}')
    c1_tensor = torch.zeros([b, m, n], dtype=torch.float32, device=f'npu:{device_id}')
    golden = torch.matmul(a1_tensor.to(torch.float32).transpose(-2, -1), b1_tensor.to(torch.float32))
    bmm_kernel_with_no_mn_split(
        a1_tensor, b1_tensor, c1_tensor,
        shape_info
    )
    assert torch.allclose(c1_tensor.cpu().to(torch.float32), golden.cpu().to(torch.float32), atol=1e-3, rtol=1e-3)


def fp32_to_tf32_modes(tensor: torch.Tensor):
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    
    sign = torch.sign(tensor)
    abs_tensor = torch.abs(tensor)
    bits = abs_tensor.view(torch.int32)
    truncate_mask = 0xFFFFE000
    half_ulp_mask = 0x00001000
    less_than_half_mask = 0x00000FFF
    increment_mask = 0x00002000
    last_bit_mask = 0x00002000
    tensor_trunc = torch.tensor(truncate_mask, dtype=torch.int64)
    tensor_half = torch.tensor(half_ulp_mask, dtype=torch.int64)
    tensor_less = torch.tensor(less_than_half_mask, dtype=torch.int64)
    tensor_inc = torch.tensor(increment_mask, dtype=torch.int64)
    tensor_last = torch.tensor(last_bit_mask, dtype=torch.int64)
    truncated = bits & tensor_trunc
    round_part = bits & (tensor_half | tensor_less)
    greater_than_half = (round_part > tensor_half)
    equal_to_half = (round_part == tensor_half)
    last_bit_is_one = (truncated & tensor_last) != 0
    ties_need_increment = equal_to_half & last_bit_is_one
    needs_increment_rne = greater_than_half | ties_need_increment
    res_bits_rne = torch.where(needs_increment_rne, truncated + tensor_inc, truncated)
    needs_increment_rafz = (round_part >= tensor_half)
    res_bits_rafz = torch.where(needs_increment_rafz, truncated + tensor_inc, truncated)
    res_abs_rne = res_bits_rne.view(torch.float32)
    res_abs_rafz = res_bits_rafz.view(torch.float32)
    is_special = ~torch.isfinite(tensor)
    res_rne = sign * res_abs_rne
    res_rafz = sign * res_abs_rafz
    res_rne = torch.where(is_special, tensor, res_rne)
    res_rafz = torch.where(is_special, tensor, res_rafz)
    diff = (res_rafz[0][0] - res_rne[0][0]).item()

    return res_rne, res_rafz


def create_mm_kernel_with_mn_split_with_param_tf32(tile_config, extend_config):
    m = tile_config.ori_shape[0]
    k = tile_config.ori_shape[1]
    n = tile_config.ori_shape[2]
    a_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.a_format_nz else pypto.TileOpFormat.TILEOP_ND
    b_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.b_format_nz else pypto.TileOpFormat.TILEOP_ND
    a_shape = (k, m) if tile_config.a_trans else (m, k)
    b_shape = (n, k) if tile_config.b_trans else (k, n)
    tile_shape = (tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape)

    @pypto.frontend.jit(runtime_options={"run_mode": 0})
    def matmul_kernel(
        a_tensor: pypto.Tensor(a_shape, tile_config.in_dtype, format=a_format),
        b_tensor: pypto.Tensor(b_shape, tile_config.in_dtype, format=b_format),
        out_tensor: pypto.Tensor((m, n), tile_config.out_dtype, format=pypto.TileOpFormat.TILEOP_ND)
    ):
        pypto.set_cube_tile_shapes(*tile_shape, True, False)
        pypto.set_vec_tile_shapes(tile_shape[0][0], tile_shape[2][0])
        tile_m = tile_config.view_shape[0]
        tile_n = tile_config.view_shape[1]
        m_loop = (m + tile_m - 1) // tile_m
        n_loop = (n + tile_n - 1) // tile_n
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                m_offset = m_idx * tile_m
                n_offset = n_idx * tile_n
                if tile_config.a_trans:
                    input_a_view = a_tensor[0:k, m_offset:m_offset + tile_m]
                else:
                    input_a_view = a_tensor[m_offset:m_offset + tile_m, 0:k]
                if tile_config.b_trans:
                    input_b_view = b_tensor[n_offset:n_offset + tile_n, 0:k]
                else:
                    input_b_view = b_tensor[0:k, n_offset:n_offset + tile_n]
                output_view = pypto.matmul(input_a_view, input_b_view, out_dtype=tile_config.out_dtype,
                                            a_trans=tile_config.a_trans, b_trans=tile_config.b_trans, c_matrix_nz=False,
                                            extend_params={"trans_mode": extend_config.trans_mode})
                out_tensor[m_offset:m_offset + tile_m, n_offset:n_offset + tile_n] = output_view
    return matmul_kernel


@pytest.mark.soc("950")
def test_tf32_rint():
    m = 255
    k = 127
    n = 513
    tile_m = 64
    tile_k = 64
    tile_n = 64
    m_view = 128
    n_view = 256
    tile_config = ShapeConfig([m, k, n], [tile_m, tile_m], [tile_k, tile_k], [tile_n, tile_n], [m_view, n_view], FP32,
                                FP32, True, True, False, False, False, False, False)
    extend_config = ExtendParams(trans_mode=pypto.TransMode.CAST_RINT)
    a1_tensor = torch.rand([k, m], dtype=torch.float32)
    b1_tensor = torch.rand([n, k], dtype=torch.float32)
    golden = torch.matmul(fp32_to_tf32_modes(a1_tensor.to(torch.float32).T)[0],
                            fp32_to_tf32_modes(b1_tensor.to(torch.float32).T)[0])
    c1 = torch.empty([m, n], dtype=torch.float32).npu()
    create_mm_kernel_with_mn_split_with_param_TF32(tile_config, extend_config)(a1_tensor.npu(), b1_tensor.npu(), c1)
    assert torch.allclose(c1.cpu().to(torch.float32), golden, atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950")
def test_tf32_round():
    m = 257
    k = 129
    n = 511
    tile_m = 128
    tile_k = 128
    tile_n = 128
    m_view = 256
    n_view = 128
    tile_config = ShapeConfig([m, k, n], [tile_m, tile_m], [tile_k, tile_k], [tile_n, tile_n], [m_view, n_view], FP32,
                                FP32, True, True, False, False, False, False, False)
    extend_config = ExtendParams(trans_mode=pypto.TransMode.CAST_ROUND)
    b1_tensor = torch.rand([n, k], dtype=torch.float32)
    a1_tensor = torch.rand([k, m], dtype=torch.float32)
    golden = torch.matmul(fp32_to_tf32_modes(a1_tensor.to(torch.float32).T)[1],
                            fp32_to_tf32_modes(b1_tensor.to(torch.float32).T)[1])
    c1 = torch.empty([m, n], dtype=torch.float32).npu()
    create_mm_kernel_with_mn_split_with_param_TF32(tile_config, extend_config)(a1_tensor.npu(), b1_tensor.npu(), c1)
    assert torch.allclose(c1.cpu().to(torch.float32), golden, atol=1e-3, rtol=1e-3)