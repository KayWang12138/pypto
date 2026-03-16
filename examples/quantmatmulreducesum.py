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
import pypto
import torch
import torch.nn.functional as F


def trans_nd_to_fractal_nz(data: torch.Tensor, keep_m_dim=False):
    def _gen_axes_for_transpose(offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def _ceil_div(a, b):
        return (a + b - 1) // b

    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    m0 = 16
    n0 = 32 // data.dtype.itemsize
    if data.dtype == torch.int32:
        n0 = 16
    m1, n1 = _ceil_div(m_ori, m0), _ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    if not keep_m_dim:
        pad_list = [0, padding_n, 0, padding_m] + [0, 0] * batch_num
        data = F.pad(data, pad_list, "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
        data = data.reshape(batch_ori + (m1, m0, n1, n0)).permute(*array_trans).contiguous()
    else:
        pad_list = [0, padding_n, 0, 0] + [0, 0] * batch_num
        data = F.pad(data, pad_list, "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [1, 0, 2])
        data = data.reshape(batch_ori + (m_ori, n1, n0)).permute(*array_trans).contiguous()
    return data


def quant_matmul_reduce_sum_pypto(batch, m, k, n):
    x1_shape = [batch, m, k]
    x2_shape = [batch, k, n]
    x1_scale_shape = [batch, m]
    x2_scale_shape = [n]
    out_shape = [m, n]
    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1})
    def quant_matmul_reduce_sum_impl(
        x1: pypto.Tensor(x1_shape, pypto.DT_INT8),
        x2: pypto.Tensor(x2_shape, pypto.DT_INT8, format=pypto.TileOpFormat.TILEOP_NZ),
        x1_scale: pypto.Tensor(x1_scale_shape, pypto.DT_FP32),
        x2_scale: pypto.Tensor(x2_scale_shape, pypto.DT_BF16)
    ) -> pypto.Tensor(out_shape, pypto.DT_BF16):
        pypto.set_cube_tile_shapes([16, 32], [32, 32], [32, 32])
        pypto.set_vec_tile_shapes(1, 4, 32)
        out_tensor = pypto.Tensor([m, n], pypto.DT_FP32)
        
        for batch_idx in pypto.loop(0, batch, 1, name="LOOP_L0_batchIdx", idx_name="batch_idx"):
            x1_view = x1[batch_idx, :, :]
            x2_view = x2[batch_idx, :, :]
            x1_scale_view = x1_scale[batch_idx, :]
            x1_scale_2d = pypto.unsqueeze(x1_scale_view, 1)
            x1_scale_broadcast = pypto.expand_clone(x1_scale_2d, [m, n])
            x2_scale_2d = pypto.unsqueeze(x2_scale, 0)
            x2_scale_broadcast = pypto.expand_clone(x2_scale_2d, [m, n])
            x2_scale_broadcast_fp32 = pypto.cast(x2_scale_broadcast, pypto.DT_FP32)
            matmul_result = pypto.matmul(x1_view, x2_view, pypto.DT_FP32)
            scaled_result = matmul_result * x1_scale_broadcast * x2_scale_broadcast_fp32
            out_tensor = out_tensor + scaled_result
        out_bf16 = pypto.cast(out_tensor, pypto.DT_BF16)
        return out_bf16
    return quant_matmul_reduce_sum_impl


def test_quant_matmul_reduce_sum_pypto():
    b, m, k, n = (2, 16, 32, 32)
    
    x1 = torch.randint(-10, 10, (b, m, k), dtype=torch.int8).npu()
    x2_nd = torch.randint(-10, 10, (b, k, n), dtype=torch.int8).npu()
    x2 = trans_nd_to_fractal_nz(x2_nd).npu()
    x1_scale = torch.randn((b, m), dtype=torch.float32).uniform_(0, 2).npu()
    x2_scale = torch.randn((n,), dtype=torch.bfloat16).uniform_(0, 2).npu()
    
    pypto_out = quant_matmul_reduce_sum_pypto(b, m, k, n)(x1, x2, x1_scale, x2_scale)
    
    print("PyPTO output shape:", pypto_out.shape)
    print("PyPTO output dtype:", pypto_out.dtype)
    print("PyPTO output:", pypto_out)
    
    return pypto_out


if __name__ == "__main__":
    test_quant_matmul_reduce_sum_pypto()
