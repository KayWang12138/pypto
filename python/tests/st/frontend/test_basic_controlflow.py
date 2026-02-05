#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Frontend control-flow test cases."""

import os
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


def create_cond_loop_kernel(shape: tuple[int, int]):
    w, h = shape

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def cond_loop_kernel(
        a: pypto.Tensor((w, h), pypto.DT_FP32),
        b: pypto.Tensor((w, h), pypto.DT_FP32),
    ) -> pypto.Tensor((w, h), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(1, h)
        out = pypto.tensor((w, h), pypto.DT_FP32)
        for i in pypto.loop(0, w, 1, name="row_loop", idx_name="row"):
            a_view = a[i:i + 1, :]
            b_view = b[i:i + 1, :]
            if i < 1:
                out[i:i + 1, :] = a_view + b_view
            else:
                out[i:i + 1, :] = a_view - b_view
        return out

    return cond_loop_kernel


def create_begin_end_kernel(shape: tuple[int, int]):
    w, h = shape

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def begin_end_kernel(
        x: pypto.Tensor((w, h), pypto.DT_FP32),
    ) -> pypto.Tensor((w, h), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(1, h)
        out = pypto.tensor((w, h), pypto.DT_FP32)
        for i in pypto.loop(0, w, 1, name="row_begin_end", idx_name="row"):
            x_view = x[i:i + 1, :]
            if pypto.is_loop_begin(i):
                out[i:i + 1, :] = x_view + 1.0
            elif pypto.is_loop_end(i):
                out[i:i + 1, :] = x_view + 3.0
            else:
                out[i:i + 1, :] = x_view + 2.0
        return out

    return begin_end_kernel


def create_loop_unroll_kernel(shape: tuple[int, int]):
    w, h = shape

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def loop_unroll_kernel(
        x: pypto.Tensor((w, h), pypto.DT_FP32),
    ) -> pypto.Tensor((w, h), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(1, h)
        out = pypto.tensor((w, h), pypto.DT_FP32)
        for i, tile_rows in pypto.loop_unroll(
            0,
            w,
            1,
            name="row_unroll",
            idx_name="row",
            unroll_list=[1, 2],
        ):
            x_view = x[i:i + tile_rows, :]
            out[i:i + tile_rows, :] = x_view * 2.0 + 1.0
        return out

    return loop_unroll_kernel


def test_cond_with_loop_index():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    shape = (3, 4)
    kernel = create_cond_loop_kernel(shape)

    torch.manual_seed(2026)
    a = torch.rand(shape, dtype=torch.float, device=f"npu:{device_id}")
    b = torch.rand(shape, dtype=torch.float, device=f"npu:{device_id}")
    out = kernel(a, b)
    torch.npu.synchronize()

    expected = a - b
    expected[0:1, :] = a[0:1, :] + b[0:1, :]
    assert_allclose(np.array(out.cpu()), np.array(expected.cpu()), rtol=3e-3, atol=3e-3)


def test_loop_begin_end_branches():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    shape = (4, 4)
    kernel = create_begin_end_kernel(shape)

    torch.manual_seed(2026)
    x = torch.rand(shape, dtype=torch.float, device=f"npu:{device_id}")
    out = kernel(x)
    torch.npu.synchronize()

    expected = x + 2.0
    expected[0] = x[0] + 1.0
    expected[-1] = x[-1] + 3.0
    assert_allclose(np.array(out.cpu()), np.array(expected.cpu()), rtol=3e-3, atol=3e-3)


def test_loop_unroll_updates_tiles():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    shape = (4, 4)
    kernel = create_loop_unroll_kernel(shape)

    torch.manual_seed(2026)
    x = torch.rand(shape, dtype=torch.float, device=f"npu:{device_id}")
    out = kernel(x)
    torch.npu.synchronize()

    expected = x * 2.0 + 1.0
    assert_allclose(np.array(out.cpu()), np.array(expected.cpu()), rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    test_cond_with_loop_index()
    test_loop_begin_end_branches()
    test_loop_unroll_updates_tiles()
