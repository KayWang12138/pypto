# -----------------------------------------------------------------------------------------------------------
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
import pto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose
import torch_npu


def test_vector_operation_greater():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.runtime._device_init()
    a = pto.tensor(shape, dtype, "Greater_TENSOR_a")
    b = pto.tensor(shape, dtype, "Greater_TENSOR_b")
    c = pto.tensor(shape, pto.DT_BOOL, "Greater_TENSOR_c")
    with pto.function("Greater", [a, b], [c]):
        pto.set_codegen_options(support_dynamic_unaligned=True)

        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_GREATER_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_GREATER_L1", idx_name="s_idx"):
                tile_a = pto.view(a, view_shape,
                                  [b_idx * view_shape[0],
                                      s_idx * view_shape[1]],
                                  valid_shape=[
                                      pto.min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                              pto.symbolic_scalar(view_shape[0])),
                                      pto.min(pto.symbolic_scalar(m) - s_idx * view_shape[1],
                                              pto.symbolic_scalar(view_shape[1]))])
                tile_b = pto.view(b, view_shape,
                                  [b_idx * view_shape[0],
                                      s_idx * view_shape[1]],
                                  valid_shape=[(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(
                                      pto.symbolic_scalar(view_shape[0])),
                                      (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(
                                      pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pto.greater(tile_a, tile_b))
                pto.assemble(
                    tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                del tile_a, tile_b
    a_tensor = torch.from_numpy(
        np.random.uniform(-100, 100, [n, m]).astype(np.float32))
    b_tensor = torch.from_numpy(
        np.random.uniform(-100, 100, [n, m]).astype(np.float32))
    c_tensor = torch.zeros(n, m, dtype=torch.bool)

    pto.runtime._device_run_once_data_from_host(
        [a_tensor, b_tensor], [c_tensor])
    expected = torch.greater(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(),
                    rtol=1e-3, atol=1e-3)

    pto.runtime._device_fini()
