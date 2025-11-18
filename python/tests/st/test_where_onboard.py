#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import os
import numpy as np
import torch
import pytest
import torch_npu
from numpy.testing import assert_allclose
import pto


def test_vector_operation_where():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = int(tiling * 2.3), int(tiling * 2.7)
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.runtime._device_init()
    condition = pto.tensor(shape, pto.DT_BOOL, "WHERE_TENSOR_cond")
    input_base = pto.tensor(shape, dtype, "WHERE_TENSOR_input")
    other_base = pto.tensor(shape, dtype, "WHERE_TENSOR_other")
    out = pto.tensor(shape, dtype, "WHERE_TENSOR_out")

    with pto.function("WHERE", [condition, input_base, other_base], [out]):
        pto.set_codegen_options(support_dynamic_unaligned=True)
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_ADD_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ADD_L1", idx_name="s_idx"):
                tile_cond = pto.view(condition, view_shape,
                                     [b_idx * view_shape[0], s_idx * view_shape[1]],
                                     valid_shape=[(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(
                                         pto.symbolic_scalar(view_shape[0])),
                                         (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(
                                         pto.symbolic_scalar(view_shape[1]))])
                tile_input = pto.view(input_base, view_shape,
                                      [b_idx * view_shape[0],
                                          s_idx * view_shape[1]],
                                      valid_shape=[(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(
                                          pto.symbolic_scalar(view_shape[0])),
                                          (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(
                                          pto.symbolic_scalar(view_shape[1]))])
                tile_other = pto.view(other_base, view_shape,
                                      [b_idx * view_shape[0],
                                          s_idx * view_shape[1]],
                                      valid_shape=[(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(
                                          pto.symbolic_scalar(view_shape[0])),
                                          (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(
                                          pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_input.move(pto.where(tile_cond, tile_input, tile_other))
                pto.assemble(
                    tile_input, [b_idx * view_shape[0], s_idx * view_shape[1]], out)
                del tile_cond, tile_input, tile_other
    cond_tensor = torch.randint(0, 2, (n, m), dtype=torch.bool)
    input_tensor = torch.rand(n, m, dtype=torch.float32)
    other_tensor = torch.zeros(n, m, dtype=torch.float32)
    out_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host(
        [cond_tensor, input_tensor, other_tensor], [out_tensor])

    expected = torch.where(cond_tensor, input_tensor, other_tensor)
    assert_allclose(out_tensor.flatten(),
                    expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()
