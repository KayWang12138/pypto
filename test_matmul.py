#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR KIND, EITHER EXPRESS OR IMPLIED,
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
    mdl_flag: bool = True
    gm_acc: bool = False


@dataclass
class ExtendParams:
    bias_shape: list = field(default_factory=list)
    bias_dtype: np.dtype = None
    scale_shape: list = field(default_factory=list)
    scale_dtype: np.dtype = None
    scale: int = None
    relu_type: int = None


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}, runtime_options={"run_mode": pypto.RunMode.SIM})
def matmul_pto(a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    tile_config):
    m = tile_config.ori_shape[0]
    n = tile_config.ori_shape[2]
    m_view = tile_config.view_shape[0]
    n_view = tile_config.view_shape[1]
    pypto.set_cube_tile_shapes(tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape,
                                    enable_split_k=tile_config.gm_acc)
    pypto.set_vec_tile_shapes(128, 128)
    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
                #Get the view tensor of mat_a
                if tile_config.a_trans:
                    a_view = a_tensor[:, m_idx * m_view: m_idx * m_view + m_view]
                else:
                    a_view = a_tensor[m_idx * m_view: m_idx * m_view + m_view, :]
                #Get the view tensor of mat_b
                if tile_config.b_trans:
                    b_view = b_tensor[n_idx * n_view: n_idx * n_view + n_view, :]
                else:
                    b_view = b_tensor[:, n_idx * n_view: n_idx * n_view + n_view]
                #a_view = pypto.add(a_view, a_view)
                #b_view = pypto.add(b_view, b_view)
                out_view = pypto.matmul(a_view, b_view, tile_config.out_dtype, 
                                        a_trans=tile_config.a_trans, 
                                        b_trans=tile_config.b_trans)
                out_tensor[m_idx * m_view: m_idx * m_view + m_view, 
                        n_idx * n_view: n_idx * n_view + n_view] = out_view


def test_ub2l1_float(tile_config: ShapeConfig):
    m = tile_config.ori_shape[0]
    k = tile_config.ori_shape[1]
    n = tile_config.ori_shape[2]
    a_shape = [k, m] if tile_config.a_trans else [m, k]
    b_shape = [n, k] if tile_config.b_trans else [k, n]

    # mat_a = torch.randint(-3, 3, a_shape).to(torch.int8)
    # mat_b = torch.randint(-3, 3, b_shape).to(torch.int8)
    # values = torch.arange(1, k * m + 1)
    # mat_a = values.view(a_shape).to(torch.float16)
    mat_a = torch.randn(a_shape, dtype=torch.float32).uniform_(1, 1).to(torch.float16)
    mat_b = torch.randn(b_shape, dtype=torch.float32).uniform_(1, 1).to(torch.float16)
    mat_a_tmp = mat_a.T if tile_config.a_trans else mat_a
    mat_b_tmp = mat_b.T if tile_config.b_trans else mat_b
    #mat_a_tmp = mat_a_tmp + mat_a_tmp
    #mat_b_tmp = mat_b_tmp + mat_b_tmp
    golden = torch.matmul(mat_a_tmp.to(torch.float32), mat_b_tmp.to(torch.float32))
    out = torch.zeros([m, n], dtype=torch.float32)
    print("golden:", golden)
    matmul_pto(mat_a, mat_b, out, tile_config)
    print("out:", out)
    assert torch.allclose(out.cpu().to(torch.float32), golden.cpu(), atol=1e-3, rtol=1e-3), "结果精度不匹配"


# def test_ub2l1_int8(tile_config: ShapeConfig):
#     m = tile_config.ori_shape[0]
#     k = tile_config.ori_shape[1]
#     n = tile_config.ori_shape[2]
#     a_shape = [k, m] if tile_config.a_trans else [m, k]
#     b_shape = [n, k] if tile_config.b_trans else [k, n]
#
#     mat_a = torch.randint(1, 2, a_shape).to(torch.int8)
#     mat_b = torch.randint(1, 2, b_shape).to(torch.int8)
#     mat_a_tmp = mat_a.T if tile_config.a_trans else mat_a
#     mat_b_tmp = mat_b.T if tile_config.b_trans else mat_b
#     mat_a_tmp = mat_a_tmp + mat_a_tmp
#     mat_b_tmp = mat_b_tmp + mat_b_tmp
#     golden = torch.matmul(mat_a_tmp.to(torch.int32), mat_b_tmp.to(torch.int32))
#     print(golden.cpu())
#     out = matmul_pto(tile_config)(mat_a.cpu(), mat_b.cpu(), tile_config)
#     print(out.cpu())
#     assert torch.allclose(out.cpu().to(torch.int32), golden, atol=1e-3, rtol=1e-3), "结果精度不匹配"


def main():
    tile_config = ShapeConfig([32, 32, 16], [128, 128], [128, 128], [128, 128], [32, 16], 
                               pypto.DataType.DT_FP16, pypto.DataType.DT_FP32,
                               False, False)
    test_ub2l1_float(tile_config)



if __name__ == "__main__":
    main()
