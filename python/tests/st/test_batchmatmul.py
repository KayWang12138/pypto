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
import pypto
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
    gm_acc: bool = False


@dataclass
class ExtendParams:
    bias_shape: list = field(default_factory=list)
    bias_dtype: np.dtype = None
    scale_shape: list = field(default_factory=list)
    scale_dtype: np.dtype = None
    scale: int = None
    relu_type: int = None


def create_bmm3d_kernel_with_mn_split(tile_config):
    b = tile_config.ori_shape[0]
    m = tile_config.ori_shape[1]
    k = tile_config.ori_shape[2]
    n = tile_config.ori_shape[3]
    m_view = tile_config.view_shape[0]
    n_view = tile_config.view_shape[1]
    a_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.a_format_nz else pypto.TileOpFormat.TILEOP_ND
    b_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.b_format_nz else pypto.TileOpFormat.TILEOP_ND
    a_shape = [b, k, m] if tile_config.a_trans else [b, m, k]
    b_shape = [b, n, k] if tile_config.b_trans else [b, k, n]
    
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
    )
    def matmul_kernel(
        a_tensor: pypto.Tensor(a_shape, tile_config.in_dtype, format=a_format),
        b_tensor: pypto.Tensor(b_shape, tile_config.in_dtype, format=b_format),
    ) -> pypto.Tensor([b, m, n], tile_config.out_dtype):
        m_loop = (m + m_view - 1) // m_view
        n_loop = (n + n_view - 1) // n_view
        pypto.set_cube_tile_shapes(tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape,
                                    enable_split_k=tile_config.gm_acc)
        out_tensor = pypto.Tensor([b, m, n], tile_config.out_dtype)
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
                if tile_config.a_trans:
                    a_view = a_tensor[:, :, m_idx * m_view: m_idx * m_view + m_view]
                else:
                    a_view = a_tensor[:, m_idx * m_view: m_idx * m_view + m_view, :]
                if tile_config.b_trans:
                    b_view = b_tensor[:, n_idx * n_view: n_idx * n_view + n_view, :]
                else:
                    b_view = b_tensor[:, :, n_idx * n_view: n_idx * n_view + n_view]
                out_view = pypto.matmul(a_view, b_view, a_trans=tile_config.a_trans, b_trans=tile_config.b_trans,
                                        out_dtype=tile_config.out_dtype)
                out_tensor[:, 
                           m_idx * m_view: m_idx * m_view + m_view, 
                           n_idx * n_view: n_idx * n_view + n_view] = out_view
        return out_tensor
    return matmul_kernel


def create_bmm4d_kernel_with_mn_split(tile_config):
    b1 = tile_config.ori_shape[0]
    b2 = tile_config.ori_shape[1]
    m = tile_config.ori_shape[2]
    k = tile_config.ori_shape[3]
    n = tile_config.ori_shape[4]
    m_view = tile_config.view_shape[0]
    n_view = tile_config.view_shape[1]
    a_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.a_format_nz else pypto.TileOpFormat.TILEOP_ND
    b_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.b_format_nz else pypto.TileOpFormat.TILEOP_ND
    a_shape = [b1, b2, k, m] if tile_config.a_trans else [b1, b2, m, k]
    b_shape = [b1, b2, n, k] if tile_config.b_trans else [b1, b2, k, n]
    
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
    )
    def matmul_kernel(
        a_tensor: pypto.Tensor(a_shape, tile_config.in_dtype, format=a_format),
        b_tensor: pypto.Tensor(b_shape, tile_config.in_dtype, format=b_format),
    ) -> pypto.Tensor([b1, b2, m, n], tile_config.out_dtype):
        m_loop = (m + m_view - 1) // m_view
        n_loop = (n + n_view - 1) // n_view
        pypto.set_cube_tile_shapes(tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape, 
                                   enable_split_k=tile_config.gm_acc)
        out_tensor = pypto.Tensor([b1, b2, m, n], tile_config.out_dtype)
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
                if tile_config.a_trans:
                    a_view = a_tensor[:, :, :, m_idx * m_view: m_idx * m_view + m_view]
                else:
                    a_view = a_tensor[:, :, m_idx * m_view: m_idx * m_view + m_view, :]
                if tile_config.b_trans:
                    b_view = b_tensor[:, :, n_idx * n_view: n_idx * n_view + n_view, :]
                else:
                    b_view = b_tensor[:, :, :, n_idx * n_view: n_idx * n_view + n_view]
                out_view = pypto.matmul(a_view, b_view, a_trans=tile_config.a_trans, b_trans=tile_config.b_trans,
                                        out_dtype=tile_config.out_dtype)
                out_tensor[:, :, 
                           m_idx * m_view: m_idx * m_view + m_view, 
                           n_idx * n_view: n_idx * n_view + n_view] = out_view
        return out_tensor
    return matmul_kernel


@pytest.mark.soc("950", "910")
def test_bmm3d_with_mn_split():
    b = 3
    m = 128
    k = 256
    n = 512
    tile_m = 64
    tile_k = 64
    tile_n = 64
    m_view = 64
    n_view = 128
    shape_info = ShapeConfig([b, m, k, n], [tile_m, tile_m], [tile_k, 2 * tile_k], [tile_n, tile_n], [m_view, n_view],
                                FP16, FP32, False, False, False, False, False, False)
    a_tensor = torch.rand([b, m, k], dtype=torch.float16)
    b_tensor = torch.rand([b, k, n], dtype=torch.float16)
    golden = torch.matmul(a_tensor.to(torch.float32), b_tensor.to(torch.float32))
    c = create_bmm3d_kernel_with_mn_split(shape_info)(a_tensor.npu(), b_tensor.npu())
    assert torch.allclose(c.cpu().to(torch.float32), golden, atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950", "910")
def test_bmm4d_with_mn_split():
    b1 = 3
    b2 = 2
    m = 128
    k = 256
    n = 512
    tile_m = 64
    tile_k = 64
    tile_n = 64
    m_view = 64
    n_view = 128
    shape_info = ShapeConfig([b1, b2, m, k, n], [tile_m, tile_m], [tile_k, 2 * tile_k], [tile_n, tile_n], 
                            [m_view, n_view], FP16, FP32, False, False, False, False, False, False)
    a_tensor = torch.rand([b1, b2, m, k], dtype=torch.float16)
    b_tensor = torch.rand([b1, b2, k, n], dtype=torch.float16)
    golden = torch.matmul(a_tensor.to(torch.float32), b_tensor.to(torch.float32))
    c = create_bmm4d_kernel_with_mn_split(shape_info)(a_tensor.npu(), b_tensor.npu())
    assert torch.allclose(c.cpu().to(torch.float32), golden, atol=1e-3, rtol=1e-3)