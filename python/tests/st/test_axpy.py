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
Test Axpy operation on board
"""
import os
import math
import torch
import pypto
import pytest
from numpy.testing import assert_allclose
import torch_npu


def test_axpy_onboard_fp32():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (64, 64)
    view_shape = (32, 32)
    tile_shape = (32, 32)
    alpha = 2.0

    pypto.runtime._device_init()

    y_tensor = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_y")
    x_tensor = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_x")
    output = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_output")

    b_loop_num = math.ceil(shape[0] / view_shape[0])
    s_loop_num = math.ceil(shape[1] / view_shape[1])

    with pypto.function("MAIN", y_tensor, x_tensor, output):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                offset_x = b_idx * view_shape[0]
                offset_y = s_idx * view_shape[1]

                valid_shape_x = pypto.min(pypto.symbolic_scalar(shape[0]) - offset_x,
                                          pypto.symbolic_scalar(view_shape[0]))
                valid_shape_y = pypto.min(pypto.symbolic_scalar(shape[1]) - offset_y,
                                          pypto.symbolic_scalar(view_shape[1]))
                view_tensor_y = pypto.view(y_tensor, view_shape,
                                           [offset_x, offset_y],
                                           valid_shape=[valid_shape_x, valid_shape_y])
                view_tensor_x = pypto.view(x_tensor, view_shape,
                                           [offset_x, offset_y],
                                           valid_shape=[valid_shape_x, valid_shape_y])

                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                res = pypto.axpy(view_tensor_y, view_tensor_x, alpha)
                pypto.assemble(res, [offset_x, offset_y], output)

    assert isinstance(output, pypto.tensor)
    y_data = torch.randn(size=[shape[0], shape[1]], dtype=torch.float32)
    x_data = torch.randn(size=[shape[0], shape[1]], dtype=torch.float32)
    out_data = torch.zeros(shape[0], shape[1], dtype=torch.float32)

    pto_y_tensor = pypto.from_torch(y_data, "PTO_TENSOR_y")
    pto_x_tensor = pypto.from_torch(x_data, "PTO_TENSOR_x")
    pto_out_tensor = pypto.from_torch(out_data, "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_y_tensor, pto_x_tensor, pto_out_tensor)

    golden = alpha * x_data + y_data

    assert_allclose(out_data.flatten(), golden.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


def test_axpy_onboard_fp16():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (32, 32)
    view_shape = (16, 16)
    tile_shape = (16, 16)
    alpha = 1.5

    pypto.runtime._device_init()

    y_tensor = pypto.tensor(shape, pypto.DT_FP16, "PTO_TENSOR_y")
    x_tensor = pypto.tensor(shape, pypto.DT_FP16, "PTO_TENSOR_x")
    output = pypto.tensor(shape, pypto.DT_FP16, "PTO_TENSOR_output")

    b_loop_num = math.ceil(shape[0] / view_shape[0])
    s_loop_num = math.ceil(shape[1] / view_shape[1])

    with pypto.function("MAIN", y_tensor, x_tensor, output):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                offset_x = b_idx * view_shape[0]
                offset_y = s_idx * view_shape[1]

                valid_shape_x = pypto.min(pypto.symbolic_scalar(shape[0]) - offset_x,
                                          pypto.symbolic_scalar(view_shape[0]))
                valid_shape_y = pypto.min(pypto.symbolic_scalar(shape[1]) - offset_y,
                                          pypto.symbolic_scalar(view_shape[1]))
                view_tensor_y = pypto.view(y_tensor, view_shape,
                                           [offset_x, offset_y],
                                           valid_shape=[valid_shape_x, valid_shape_y])
                view_tensor_x = pypto.view(x_tensor, view_shape,
                                           [offset_x, offset_y],
                                           valid_shape=[valid_shape_x, valid_shape_y])

                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                res = pypto.axpy(view_tensor_y, view_tensor_x, alpha)
                pypto.assemble(res, [offset_x, offset_y], output)

    assert isinstance(output, pypto.tensor)
    y_data = torch.randn(size=[shape[0], shape[1]], dtype=torch.float16)
    x_data = torch.randn(size=[shape[0], shape[1]], dtype=torch.float16)
    out_data = torch.zeros(shape[0], shape[1], dtype=torch.float16)

    pto_y_tensor = pypto.from_torch(y_data, "PTO_TENSOR_y")
    pto_x_tensor = pypto.from_torch(x_data, "PTO_TENSOR_x")
    pto_out_tensor = pypto.from_torch(out_data, "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_y_tensor, pto_x_tensor, pto_out_tensor)

    golden = alpha * x_data + y_data

    assert_allclose(out_data.flatten(), golden.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()


def test_axpy_onboard_bf16():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (32, 32)
    view_shape = (16, 16)
    tile_shape = (16, 16)
    alpha = 2.0

    pypto.runtime._device_init()

    y_tensor = pypto.tensor(shape, pypto.DT_BF16, "PTO_TENSOR_y")
    x_tensor = pypto.tensor(shape, pypto.DT_BF16, "PTO_TENSOR_x")
    output = pypto.tensor(shape, pypto.DT_BF16, "PTO_TENSOR_output")

    b_loop_num = math.ceil(shape[0] / view_shape[0])
    s_loop_num = math.ceil(shape[1] / view_shape[1])

    with pypto.function("MAIN", y_tensor, x_tensor, output):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                offset_x = b_idx * view_shape[0]
                offset_y = s_idx * view_shape[1]

                valid_shape_x = pypto.min(pypto.symbolic_scalar(shape[0]) - offset_x,
                                          pypto.symbolic_scalar(view_shape[0]))
                valid_shape_y = pypto.min(pypto.symbolic_scalar(shape[1]) - offset_y,
                                          pypto.symbolic_scalar(view_shape[1]))
                view_tensor_y = pypto.view(y_tensor, view_shape,
                                           [offset_x, offset_y],
                                           valid_shape=[valid_shape_x, valid_shape_y])
                view_tensor_x = pypto.view(x_tensor, view_shape,
                                           [offset_x, offset_y],
                                           valid_shape=[valid_shape_x, valid_shape_y])

                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                res = pypto.axpy(view_tensor_y, view_tensor_x, alpha)
                pypto.assemble(res, [offset_x, offset_y], output)

    assert isinstance(output, pypto.tensor)
    y_data = torch.randn(size=[shape[0], shape[1]], dtype=torch.bfloat16)
    x_data = torch.randn(size=[shape[0], shape[1]], dtype=torch.bfloat16)
    out_data = torch.zeros(shape[0], shape[1], dtype=torch.bfloat16)

    pto_y_tensor = pypto.from_torch(y_data, "PTO_TENSOR_y")
    pto_x_tensor = pypto.from_torch(x_data, "PTO_TENSOR_x")
    pto_out_tensor = pypto.from_torch(out_data, "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_y_tensor, pto_x_tensor, pto_out_tensor)

    # bf16: compute in fp32 precision (same as NPU PASS handling)
    y_fp32 = y_data.float()
    x_fp32 = x_data.float()
    golden_fp32 = alpha * x_fp32 + y_fp32
    golden = golden_fp32.bfloat16()

    assert_allclose(out_data.float().flatten(), golden.float().flatten(), rtol=1e-2, atol=1e-2)

    pypto.runtime._device_fini()


def test_axpy_onboard_broadcast():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    y_shape = (64, 64)
    x_shape = (64, 1)
    view_shape = (32, 32)
    tile_shape = (32, 32)
    alpha = 2.0

    pypto.runtime._device_init()

    y_tensor = pypto.tensor(y_shape, pypto.DT_FP32, "PTO_TENSOR_y")
    x_tensor = pypto.tensor(x_shape, pypto.DT_FP32, "PTO_TENSOR_x")
    output = pypto.tensor(y_shape, pypto.DT_FP32, "PTO_TENSOR_output")

    b_loop_num = math.ceil(y_shape[0] / view_shape[0])
    s_loop_num = math.ceil(y_shape[1] / view_shape[1])

    with pypto.function("MAIN", y_tensor, x_tensor, output):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                offset_x = b_idx * view_shape[0]
                offset_y = s_idx * view_shape[1]

                valid_shape_x = pypto.min(pypto.symbolic_scalar(y_shape[0]) - offset_x,
                                          pypto.symbolic_scalar(view_shape[0]))
                valid_shape_y = pypto.min(pypto.symbolic_scalar(y_shape[1]) - offset_y,
                                          pypto.symbolic_scalar(view_shape[1]))
                view_tensor_y = pypto.view(y_tensor, view_shape,
                                           [offset_x, offset_y],
                                           valid_shape=[valid_shape_x, valid_shape_y])
                # x broadcast: x_shape = (64, 1), so offset for dim 1 is always 0
                view_tensor_x = pypto.view(x_tensor, view_shape,
                                           [offset_x, 0],
                                           valid_shape=[valid_shape_x, 1])

                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                res = pypto.axpy(view_tensor_y, view_tensor_x, alpha)
                pypto.assemble(res, [offset_x, offset_y], output)

    assert isinstance(output, pypto.tensor)
    y_data = torch.randn(size=[y_shape[0], y_shape[1]], dtype=torch.float32)
    x_data = torch.randn(size=[x_shape[0], x_shape[1]], dtype=torch.float32)
    out_data = torch.zeros(y_shape[0], y_shape[1], dtype=torch.float32)

    pto_y_tensor = pypto.from_torch(y_data, "PTO_TENSOR_y")
    pto_x_tensor = pypto.from_torch(x_data, "PTO_TENSOR_x")
    pto_out_tensor = pypto.from_torch(out_data, "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_y_tensor, pto_x_tensor, pto_out_tensor)

    golden = alpha * x_data + y_data

    assert_allclose(out_data.flatten(), golden.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()