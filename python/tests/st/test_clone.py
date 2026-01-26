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
import torch_npu
import numpy as np
from numpy.testing import assert_allclose

b = 3
s = 4
n1 = 64
d = 64


def create_kernel(shape_out):
    @pypto.frontend.jit()
    def kernel_func(
        in_tensor: pypto.Tensor((b, s, n1, d), pypto.DT_FP32), 
    ) -> pypto.Tensor(shape_out, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(1, 1, 64, 64)
        out_tensor = pypto.Tensor(shape_out, pypto.DT_FP32)
        for b_idx in pypto.loop(b, name="b_loop", idx_name="b_idx"):
            for s_idx in pypto.loop(s, name="s_loop", idx_name="s_idx"):
                a0 = pypto.view(in_tensor, [1, 1, n1, d], [b_idx, s_idx, 0, 0])
                a1 = pypto.add(a0, 1.0)
                a2 = pypto.reshape(a1, [1, 1, n1 * d])
                a3 = a2.clone()
                pypto.assemble(a3, [b_idx, s_idx, 0], out_tensor)
        return out_tensor
    return kernel_func


def test_clone():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.manual_seed(42)
    # prepare data
    input_data = torch.rand((b, s, n1, d), dtype=torch.float32, device=f'npu:{device_id}')
    
    output_shape = (b, s, n1 * d)
    kernel_func = create_kernel(output_shape)
    # compute on npu
    output_result = kernel_func(input_data)
    torch_npu.npu.synchronize()

    output_cpu = output_result.cpu()

    ## golden
    golden = input_data.cpu().reshape((b, s, n1 * d)) + 1

    assert torch.allclose(output_cpu, golden, atol=1e-3, rtol=1e-3)

