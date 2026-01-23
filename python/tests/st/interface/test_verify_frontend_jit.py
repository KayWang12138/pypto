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
Test cases for pypto.frontend.jit with verification options.
"""
import os
import pytest
import torch
import torch_npu
import pypto
import numpy as np

verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
}


@pypto.frontend.jit(verify_options=verify_options)
def add(
    a: pypto.Tensor((64, 64), pypto.DT_FP32),
    b: pypto.Tensor((64, 64), pypto.DT_FP32),
) -> pypto.Tensor((64, 64), pypto.DT_FP32):
    """Add two tensors element-wise."""
    result = pypto.tensor((64, 64), pypto.DT_FP32)
    for _ in pypto.loop(1):
        pypto.set_vec_tile_shapes(16, 16)
        result = pypto.add(a, b)
    return result


def test_verify_frontend_jit_full_options():
    """Test pypto.frontend.jit with full verify options."""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    a = torch.ones((64, 64), dtype=torch.float32)
    b = torch.ones((64, 64), dtype=torch.float32)

    # Move tensors to NPU device
    a = a.to(f"npu:{device_id}")
    b = b.to(f"npu:{device_id}")

    # Expected result
    golden = a + b

    # Call the JIT-compiled function directly with torch tensors
    result = add(a, b)

    # Verify the result
    assert torch.allclose(result.cpu(), golden.cpu())


@pypto.frontend.jit(
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_pass_filter": ["RemoveRedundantReshape", "ExpandFunction", "DuplicateOp1"]
    }
)
def add_dyn(
    x: pypto.Tensor((72, 144), pypto.DT_FP16),
    y: pypto.Tensor((72, 144), pypto.DT_FP16),
) -> pypto.Tensor((72, 144), pypto.DT_FP16):
    """Add two tensors with dynamic shape handling."""
    first_dim, second_dim = x.shape
    view_shape, tile_shape = (64, 64), (32, 32)

    first_view_shape, second_view_shape = view_shape

    out = pypto.tensor((72, 144), pypto.DT_FP16)

    for b_idx in pypto.loop(int(np.ceil(first_dim / view_shape[0])), name="LOOP_L0", idx_name="b_idx"):
        for s_idx in pypto.loop(int(np.ceil(second_dim / view_shape[1])), name="LOOP_L1", idx_name="s_idx"):
            tile_tensor_0 = pypto.view(
                x, view_shape,
                [b_idx * first_view_shape, s_idx * second_view_shape]
            )
            tile_tensor_1 = pypto.view(
                y, view_shape,
                [b_idx * first_view_shape, s_idx * second_view_shape]
            )
            pypto.set_vec_tile_shapes(*tile_shape)  # 32*32
            res = pypto.tensor()
            if pypto.cond(b_idx < 2):
                res = ((tile_tensor_0 * (tile_tensor_0 + tile_tensor_1)) - tile_tensor_1) * tile_tensor_1
            else:
                res = tile_tensor_0
            pypto.assemble(
                res,
                [b_idx * first_view_shape, s_idx * second_view_shape],
                out,
            )
            del res, tile_tensor_0, tile_tensor_1
    return out


def test_verify_frontend_jit_dyn():
    """Test pypto.frontend.jit with dynamic shapes and verify options."""
    shape = [72, 144]
    a = torch.rand(shape, dtype=torch.float16)
    b = torch.rand(shape, dtype=torch.float16)

    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    # Move tensors to NPU device
    a = a.to(f'npu:{device_id}')
    b = b.to(f'npu:{device_id}')

    # Expected result
    golden = ((a * (a + b)) - b) * b
    pypto.set_verify_golden_data(goldens=[None, None, golden.cpu()])

    # Call the JIT-compiled function directly with torch tensors
    result = add_dyn(a, b)

    # Verify the result
    assert torch.allclose(result.cpu(), golden.cpu(), rtol=1e-3, atol=1e-3)
