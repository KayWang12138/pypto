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
"""
Test for permute operation.
"""
import os
import pypto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose
import torch_npu


def test_permute_2d():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32

    n, m = 32, 48
    perm = [1, 0]

    out_shape = (m, n)

    view_shape = (16, 16)
    tile_shape = (8, 8)

    pypto.runtime._device_init()

    a = pypto.tensor((n, m), dtype, "PERMUTE_TENSOR_a")
    b = pypto.tensor(out_shape, dtype, "PERMUTE_TENSOR_b")

    with pypto.function("PERMUTE", a, b):
        n_tiles_row = int(np.ceil(n / view_shape[0]))
        n_tiles_col = int(np.ceil(m / view_shape[1]))
        for b_idx in pypto.loop(n_tiles_row, name="LOOP_PERMUTE_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(n_tiles_col, name="LOOP_PERMUTE_L1", idx_name="s_idx"):

                offset_in = [b_idx * view_shape[0], s_idx * view_shape[1]]
                valid_shape = [
                    (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                    (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1]))
                ]
                tile_a = pypto.view(a, view_shape, offset_in, valid_shape=valid_shape)
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_permuted = pypto.permute(tile_a, perm)
                offset_out = [
                    offset_in[perm[0]],
                    offset_in[perm[1]]
                ]

                pypto.assemble(tile_permuted, offset_out, b)

    a_tensor = torch.rand(n, m, dtype=torch.float32) * 4 - 2
    b_tensor = torch.zeros(m, n, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = a_tensor.permute(*perm)

    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()