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
"""Frontend arithmetic test cases."""

import os
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


def create_elementwise_kernel(shape: tuple[int, ...]):

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def elementwise_kernel(
        a: pypto.Tensor(shape, pypto.DT_FP32),
        b: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> pypto.Tensor(shape, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(2, 4)
        c = (a + b) * 2.0 - b
        return c

    return elementwise_kernel


def create_broadcast_add_kernel(a_shape: tuple[int, int], b_shape: tuple[int, ...]):

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def broadcast_add_kernel(
        a: pypto.Tensor(a_shape, pypto.DT_FP32),
        b: pypto.Tensor(b_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(a_shape, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(2, 4)
        c = pypto.add(a, b)
        return c

    return broadcast_add_kernel


def test_elementwise_add_mul_sub():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    # 创建算子
    shape = (2, 4)
    kernel = create_elementwise_kernel(shape)

    # 调用算子
    torch.manual_seed(2026)
    a = torch.rand(shape, dtype=torch.float, device=f"npu:{device_id}")
    b = torch.rand(shape, dtype=torch.float, device=f"npu:{device_id}")
    out = kernel(a, b)
    torch.npu.synchronize()

    # 验证算子
    expected = (a + b) * 2.0 - b
    assert_allclose(np.array(out.cpu()), np.array(expected.cpu()), rtol=3e-3, atol=3e-3)


def test_broadcast_add_vector():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    a_shape = (2, 4)
    b_shape = (4,)
    kernel = create_broadcast_add_kernel(a_shape, b_shape)

    torch.manual_seed(2026)
    a = torch.rand(a_shape, dtype=torch.float, device=f"npu:{device_id}")
    b = torch.rand(b_shape, dtype=torch.float, device=f"npu:{device_id}")
    out = kernel(a, b)
    torch.npu.synchronize()

    expected = a + b
    assert_allclose(np.array(out.cpu()), np.array(expected.cpu()), rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    test_elementwise_add_mul_sub()
    test_broadcast_add_vector()