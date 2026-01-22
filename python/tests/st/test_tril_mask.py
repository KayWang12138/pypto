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
ST tests for tril_mask operation.
"""
import os
import pypto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose
import torch_npu


def test_tril_mask_basic():
    """Test basic lower triangular mask (q_idx == k_idx)"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    length = 4
    q_idx = 0
    k_idx = 0

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def tril_mask_kernel() -> pypto.Tensor((length, length), pypto.DT_UINT8):
        pypto.set_vec_tile_shapes(2, 16)
        output = pypto.tril_mask(q_idx, k_idx, length)
        return output

    out_torch = tril_mask_kernel()
    torch_npu.npu.synchronize()

    expected = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]
    ], dtype=torch.uint8)

    assert_allclose(out_torch.cpu().numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)


def test_tril_mask_positive_offset():
    """Test shifted mask with positive offset (q_idx > k_idx)"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    length = 4
    q_idx = 2
    k_idx = 0

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def tril_mask_kernel() -> pypto.Tensor((length, length), pypto.DT_UINT8):
        pypto.set_vec_tile_shapes(2, 16)
        output = pypto.tril_mask(q_idx, k_idx, length)
        return output

    out_torch = tril_mask_kernel()
    torch_npu.npu.synchronize()

    # offset = 2, mask[i][j] = 1 when j <= i + 2
    expected = torch.tensor([
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ], dtype=torch.uint8)

    assert_allclose(out_torch.cpu().numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)


def test_tril_mask_negative_offset():
    """Test shifted mask with negative offset (q_idx < k_idx)"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    length = 4
    q_idx = 0
    k_idx = 2

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def tril_mask_kernel() -> pypto.Tensor((length, length), pypto.DT_UINT8):
        pypto.set_vec_tile_shapes(2, 16)
        output = pypto.tril_mask(q_idx, k_idx, length)
        return output

    out_torch = tril_mask_kernel()
    torch_npu.npu.synchronize()

    # offset = -2, mask[i][j] = 1 when j <= i - 2
    expected = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0]
    ], dtype=torch.uint8)

    assert_allclose(out_torch.cpu().numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)


def test_tril_mask_small_size():
    """Test tril_mask with small size (2x2)"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    length = 2
    q_idx = 0
    k_idx = 0

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def tril_mask_kernel() -> pypto.Tensor((length, length), pypto.DT_UINT8):
        pypto.set_vec_tile_shapes(2, 16)
        output = pypto.tril_mask(q_idx, k_idx, length)
        return output

    out_torch = tril_mask_kernel()
    torch_npu.npu.synchronize()

    expected = torch.tensor([
        [1, 0],
        [1, 1]
    ], dtype=torch.uint8)

    assert_allclose(out_torch.cpu().numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)


def test_tril_mask_larger_size():
    """Test tril_mask with larger size (8x8)"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    length = 8
    q_idx = 0
    k_idx = 0

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def tril_mask_kernel() -> pypto.Tensor((length, length), pypto.DT_UINT8):
        pypto.set_vec_tile_shapes(2, 16)
        output = pypto.tril_mask(q_idx, k_idx, length)
        return output

    out_torch = tril_mask_kernel()
    torch_npu.npu.synchronize()

    expected = torch.tril(torch.ones(length, length, dtype=torch.uint8))

    assert_allclose(out_torch.cpu().numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)


def test_tril_mask_offset_one():
    """Test tril_mask with offset = 1"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    length = 4
    q_idx = 1
    k_idx = 0

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def tril_mask_kernel() -> pypto.Tensor((length, length), pypto.DT_UINT8):
        pypto.set_vec_tile_shapes(2, 16)
        output = pypto.tril_mask(q_idx, k_idx, length)
        return output

    out_torch = tril_mask_kernel()
    torch_npu.npu.synchronize()

    # offset = 1, mask[i][j] = 1 when j <= i + 1
    expected = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ], dtype=torch.uint8)

    assert_allclose(out_torch.cpu().numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)


def test_tril_mask_offset_negative_one():
    """Test tril_mask with offset = -1"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    length = 4
    q_idx = 0
    k_idx = 1

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def tril_mask_kernel() -> pypto.Tensor((length, length), pypto.DT_UINT8):
        pypto.set_vec_tile_shapes(2, 16)
        output = pypto.tril_mask(q_idx, k_idx, length)
        return output

    out_torch = tril_mask_kernel()
    torch_npu.npu.synchronize()

    # offset = -1, mask[i][j] = 1 when j <= i - 1
    expected = torch.tensor([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0]
    ], dtype=torch.uint8)

    assert_allclose(out_torch.cpu().numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)


def test_tril_mask_large_positive_offset():
    """Test tril_mask with large positive offset (all ones)"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    length = 4
    q_idx = 10
    k_idx = 0

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def tril_mask_kernel() -> pypto.Tensor((length, length), pypto.DT_UINT8):
        pypto.set_vec_tile_shapes(2, 16)
        output = pypto.tril_mask(q_idx, k_idx, length)
        return output

    out_torch = tril_mask_kernel()
    torch_npu.npu.synchronize()

    # Large positive offset results in all ones
    expected = torch.ones(length, length, dtype=torch.uint8)

    assert_allclose(out_torch.cpu().numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)


def test_tril_mask_large_negative_offset():
    """Test tril_mask with large negative offset (all zeros)"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    length = 4
    q_idx = 0
    k_idx = 10

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def tril_mask_kernel() -> pypto.Tensor((length, length), pypto.DT_UINT8):
        pypto.set_vec_tile_shapes(2, 16)
        output = pypto.tril_mask(q_idx, k_idx, length)
        return output

    out_torch = tril_mask_kernel()
    torch_npu.npu.synchronize()

    # Large negative offset results in all zeros
    expected = torch.zeros(length, length, dtype=torch.uint8)

    assert_allclose(out_torch.cpu().numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)
