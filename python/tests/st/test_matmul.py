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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import pypto
from numpy.testing import assert_allclose
import torch.nn.functional as F

FP32 = np.float32
FP16 = np.float16
INT32 = np.int32
INT8 = np.int8
UINT64 = np.uint64
UINT32 = np.uint32


@dataclass
class ShapeConfig:
    ori_shape: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    view_shape: list
    in_dtype: np.dtype
    out_dtype: np.dtype
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    mdl_flag: bool = False


@dataclass
class ExtendParams:
    bias_shape: list = field(default_factory=list)
    bias_dtype: np.dtype = None
    scale_shape: list = field(default_factory=list)
    scale_dtype: np.dtype = None
    scale: int = None
    relu_type: int = None


def create_mm_kernel_with_mn_split(tile_config, dynamic=True):
    m = tile_config.ori_shape[0]
    k = tile_config.ori_shape[1]
    n = tile_config.ori_shape[2]
    m_view = tile_config.view_shape[0]
    n_view = tile_config.view_shape[1]
    if dynamic:
        m = pypto.frontend.dynamic("M")
        n = pypto.frontend.dynamic("N")
    
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0}
    )
    def matmul_pto(
        a_tensor: pypto.Tensor([m, k], pypto.DT_FP16, format=pypto.TileOpFormat.TILEOP_ND),
        b_tensor: pypto.Tensor([k, n], pypto.DT_FP16, format=pypto.TileOpFormat.TILEOP_ND),
    ) -> pypto.Tensor([m, n], pypto.DT_FP16):
        pypto.set_cube_tile_shapes(tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape,
                                   enable_multi_data_load=True, enable_split_k=False)
        m_loop = (m + m_view - 1) // m_view
        n_loop = (n + n_view - 1) // n_view
        out_tensor = pypto.Tensor([m, n], pypto.DT_FP16)
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_LO_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_LO_nIdx", idx_name="n_idx"):
                a_view = a_tensor[m_idx * m_view: m_idx * m_view + m_view, :]
                b_view = b_tensor[:, n_idx * n_view: n_idx * n_view + n_view]
                out_view = pypto.matmul(a_view, b_view, out_dtype=pypto.DT_FP16)
                out_tensor[m_idx * m_view: m_idx * m_view + m_view, n_idx * n_view: n_idx * n_view + n_view] = out_view
        return out_tensor
    return matmul_pto


def create_bmm_kernel_with_mn_split(tile_config, dynamic=True):
    b = tile_config.ori_shape[0]
    m = tile_config.ori_shape[1]
    k = tile_config.ori_shape[2]
    n = tile_config.ori_shape[3]
    if dynamic:
        m = pypto.frontend.dynamic("M")
        n = pypto.frontend.dynamic("N")
    
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0}
    )
    def matmul_pto(
        a_tensor: pypto.Tensor([b, m, k], pypto.DT_FP16, format=pypto.TileOpFormat.TILEOP_ND),
        b_tensor: pypto.Tensor([b, k, n], pypto.DT_FP16, format=pypto.TileOpFormat.TILEOP_ND),
    ) -> pypto.Tensor([b, m, n], pypto.DT_FP16):
        pypto.set_cube_tile_shapes(tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape, 
                                   enable_multi_data_load=True, enable_split_k=False)
        out_tensor = pypto.Tensor([b, m, n], pypto.DT_FP16)
        out_tensor = pypto.matmul(a_tensor, b_tensor, out_dtype=pypto.DT_FP16)
        return out_tensor
    return matmul_pto


def create_mm_l0c2l1_fixpipe_kernel_with_no_split(tile_config1, tile_config2):
    m1 = tile_config1.ori_shape[0]
    k1 = tile_config1.ori_shape[1]
    n1 = tile_config1.ori_shape[2]
    m2 = tile_config1.ori_shape[2]
    k2 = tile_config2.ori_shape[2]

    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0}
    )
    def matmul_pto_l0c2l1(
        a1_tensor: pypto.Tensor([m1, k1], pypto.DT_INT8),
        b1_tensor: pypto.Tensor([k1, n1], pypto.DT_INT8),
        a2_tensor: pypto.Tensor([m2, k2], pypto.DT_FP16),
        scale_tensor: pypto.Tensor([1, n1], pypto.DT_UINT64),
    ) -> pypto.Tensor([m1, k2], pypto.DT_FP32):
        mm2_c = pypto.Tensor([m1, k2], pypto.DT_FP32)
        params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
        pypto.set_cube_tile_shapes(tile_config1.m_tile_shape, tile_config1.k_tile_shape, tile_config1.n_tile_shape,
                                    enable_multi_data_load=tile_config1.mdl_flag, enable_split_k=False)
        mm1_c = pypto.matmul(a1_tensor, b1_tensor, out_dtype=pypto.DT_FP16, extend_params=params)
        pypto.set_cube_tile_shapes(tile_config2.m_tile_shape, tile_config2.k_tile_shape, tile_config2.n_tile_shape,
                                    enable_multi_data_load=tile_config2.mdl_flag, enable_split_k=False)
        mm2_c = pypto.matmul(mm1_c, a2_tensor, out_dtype=pypto.DT_FP32)
        return mm2_c
    return matmul_pto_l0c2l1


def create_ub2l1_kernel_with_no_split(tile_config, vector_tile_shape, dynamic=True):
    m1 = tile_config.ori_shape[0]
    k1 = tile_config.ori_shape[1]
    m2 = tile_config.ori_shape[2]

    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0}
    )
    def matmul_pto(
        a1_tensor: pypto.Tensor([m1, k1], pypto.DT_FP16, format=pypto.TileOpFormat.TILEOP_ND),
        a2_tensor: pypto.Tensor([m2, m1], pypto.DT_FP16, format=pypto.TileOpFormat.TILEOP_ND),
    ) -> pypto.Tensor([m2, k1], pypto.DT_FP16):
        pypto.set_vec_tile_shapes(vector_tile_shape[0], vector_tile_shape[1])
        mm1_c = pypto.add(a1_tensor, a1_tensor)
        pypto.set_cube_tile_shapes(tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape,
                                    enable_multi_data_load=tile_config.mdl_flag, enable_split_k=False)
        mm2_c = pypto.matmul(a2_tensor, mm1_c, out_dtype=pypto.DT_FP16)
        return mm2_c
    return matmul_pto


def create_l0c2ub_kernel_with_no_split(tile_config, vector_tile_shape, dynamic=True):
    k1 = tile_config.ori_shape[1]
    m2 = tile_config.ori_shape[2]
    m1 = tile_config.ori_shape[0]

    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0}
    )
    def matmul_pto(
        a1_tensor: pypto.Tensor([m1, k1], pypto.DT_FP16, format=pypto.TileOpFormat.TILEOP_ND),
        a2_tensor: pypto.Tensor([m2, m1], pypto.DT_FP16, format=pypto.TileOpFormat.TILEOP_ND),
    ) -> pypto.Tensor([m2, k1], pypto.DT_FP16):
        pypto.set_cube_tile_shapes(tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape,
                                    enable_multi_data_load=tile_config.mdl_flag, enable_split_k=False)
        mm2_c = pypto.matmul(a1_tensor, a2_tensor, out_dtype=pypto.DT_FP16)
        pypto.set_vec_tile_shapes(vector_tile_shape[0], vector_tile_shape[1])
        mm2_c = pypto.add(a1_tensor, mm2_c)
        return mm2_c
    return matmul_pto


def test_mm_with_mn_split(dynamic=True):
    m = 255
    k = 127
    n = 513
    tile_m = 64
    tile_k = 64
    tile_n = 64
    m_view = 128
    n_view = 256
    tile_config = ShapeConfig([m, k, n], [tile_m, tile_m], [tile_k, tile_k], [tile_n, tile_n], [m_view, n_view], FP16,
                                FP16, False, False, False, False, False)
    a1_tensor = torch.rand([m, k], dtype=torch.float16)
    b1_tensor = torch.rand([k, n], dtype=torch.float16)
    golden = torch.matmul(a1_tensor.to(torch.float32), b1_tensor.to(torch.float32))
    c1 = create_mm_kernel_with_mn_split(tile_config, dynamic)(a1_tensor.npu(), b1_tensor.npu())
    assert torch.allclose(c1.cpu().to(torch.float32), golden, atol=1e-3, rtol=1e-3)


def test_bmm_with_mn_split(dynamic=False):
    b = 3
    m = 63
    k = 127
    n = 129
    tile_m = 64
    tile_k = 64
    tile_n = 64
    tile_config = ShapeConfig([b, m, k, n], [tile_m, tile_m], [tile_k, tile_k], [tile_n, tile_n], [-1, -1], FP16, FP16,
                                False, False, False, False, False)
    a1_tensor = torch.rand([b, m, k], dtype=torch.float16)
    b1_tensor = torch.rand([b, k, n], dtype=torch.float16)
    golden = torch.matmul(a1_tensor.to(torch.float32), b1_tensor.to(torch.float32))
    c1 = create_bmm_kernel_with_mn_split(tile_config, dynamic)(a1_tensor.npu(), b1_tensor.npu())
    assert torch.allclose(c1.cpu().to(torch.float32), golden, atol=1e-3, rtol=1e-3)


def test_l0c2l1_fixpipe_with_no_split():
    torch.npu.set_device(0)
    m1 = 320
    k1 = 512
    n1 = 258
    m2 = n1
    k2 = 128
    a1_tensor = torch.randint(low=-2, high=2, size=[m1, k1], dtype=torch.int8)
    b1_tensor = torch.randint(low=-2, high=2, size=[k1, n1], dtype=torch.int8)
    a2_tensor = torch.rand([m2, k2], dtype=torch.float16)
    tile_config1 = ShapeConfig([m1, k1, n1], [128, 128], [128, 128], [128, 128], [-1, -1], INT8, INT8,
                               False, False, False, False, False)
    tile_config2 = ShapeConfig([m1, m2, k2], [64, 64], [64, 64], [128, 128], [-1, -1], FP16, FP16,
                               False, False, False, False, False)
    scale_tensor = np.random.uniform(-2, 2, [1, n1]).astype(np.float32)
    tensor_data = scale_tensor.view(np.uint32)
    mask = 0xFFFFE000
    tensor_data = tensor_data & mask
    fp32_modified = tensor_data.view(np.float32)
    fp32_modified_tensor = torch.from_numpy(fp32_modified).to(torch.float32)
    scale_uint64 = tensor_data.astype(np.uint64)
    scale_tensor_uint64 = torch.from_numpy(scale_uint64)
    mm2_c = create_mm_l0c2l1_fixpipe_kernel_with_no_split(tile_config1, tile_config2)(a1_tensor.npu(), b1_tensor.npu(),
                                                                        a2_tensor.npu(), scale_tensor_uint64.npu())
    torch.npu.synchronize()
    tensor_res_mm1_asc = torch.matmul(a1_tensor.to(torch.float32), b1_tensor.to(torch.float32)).to(torch.float32)
    tensor_res_mm1_asc = F.relu(tensor_res_mm1_asc)
    tensor_res_mm1_asc = (tensor_res_mm1_asc * fp32_modified_tensor).to(torch.float32)
    tensor_res_mm1_asc = tensor_res_mm1_asc.to(torch.float16)
    tensor_res_mm2_asc = torch.matmul(tensor_res_mm1_asc.to(torch.float32), a2_tensor.cpu().to(torch.float32))
    assert torch.allclose(mm2_c.cpu().to(torch.float32), tensor_res_mm2_asc.to(torch.float32), atol=1e-3, rtol=1e-3)


def test_ub2l1_with_no_split():
    torch.npu.set_device(0)
    m1 = 128
    k1 = 128
    n1 = 128
    m2 = 128
    tile_m1 = 64
    tile_k1 = 64
    tile_n1 = 64
    mm1_a_shape = (m1, k1)
    mm2_a_shape = (m2, m1)
    mm2_c_shape = (m2, k1)
    a1_tensor = torch.rand(mm1_a_shape, dtype=torch.half)
    a2_tensor = torch.rand(mm2_a_shape, dtype=torch.half)
    mm2_c = torch.zeros(mm2_c_shape, dtype=torch.half)
    tile_config = ShapeConfig([m1, k1, n1], [tile_m1, tile_m1], [tile_k1, tile_k1], [tile_n1, tile_n1], [-1, -1], FP16,
                                FP16, False, False, False, False, False)
    vector_tile_shape = (128, 64)
    mm2_c = create_ub2l1_kernel_with_no_split(tile_config, vector_tile_shape)(a1_tensor.npu(), a2_tensor.npu())
    torch.npu.synchronize()
    tensor_res_mm1_asc = torch.add(a1_tensor, a1_tensor)
    tensor_res_mm2_asc = torch.matmul(a2_tensor, tensor_res_mm1_asc)
    assert torch.allclose(mm2_c.cpu(), tensor_res_mm2_asc.cpu(), atol=1e-3, rtol=1e-3)


def test_l0c2ub_with_no_split():
    torch.npu.set_device(0)
    m1 = 128
    k1 = 128
    n1 = 128
    tile_m1 = 64
    tile_k1 = 64
    tile_n1 = 64
    mm1_a_shape = (m1, k1)
    mm2_a_shape = (k1, n1)
    mm2_c_shape = (m1, n1)
    a1_tensor = torch.rand(mm1_a_shape, dtype=torch.half)
    a2_tensor = torch.rand(mm2_a_shape, dtype=torch.half)
    mm2_c = torch.zeros(mm2_c_shape, dtype=torch.half)
    tile_config = ShapeConfig([m1, k1, n1], [tile_m1, tile_m1], [tile_k1, tile_k1], [tile_n1, tile_n1], [-1, -1], FP16,
                                FP16, False, False, False, False, False)
    vector_tile_shape = (64, 64)
    mm2_c = create_l0c2ub_kernel_with_no_split(tile_config, vector_tile_shape)(a1_tensor.npu(), a2_tensor.npu())
    torch.npu.synchronize()
    tensor_res_mm1_asc = torch.matmul(a1_tensor, a2_tensor)
    tensor_res_mm2_asc = torch.add(a1_tensor, tensor_res_mm1_asc)
    assert torch.allclose(mm2_c.cpu(), tensor_res_mm2_asc.cpu(), atol=1e-3, rtol=1e-3)