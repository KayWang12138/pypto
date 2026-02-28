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
"""
import os
import math
import pypto
import pytest
from numpy.testing import assert_allclose
import torch
import torch_npu


def test_prod_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (72, 71)
    view_shape = (32, 32)
    tile_shape = (32, 32)
    pypto.runtime._device_init()

    # 使用 DT_FLOAT 对应 float32
    input1 = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_input1")
    output = pypto.tensor((1, shape[1]), pypto.DT_FP32, "PTO_TENSOR_output")  # dim=0, keepdim=True

    b_loop_num = math.ceil(shape[0] / view_shape[0])
    s_loop_num = math.ceil(shape[1] / view_shape[1])

    # 初始化输出为全1.0（prod 的单位元）
    with pypto.function("INIT_OUTPUT", output):
        pypto.assign(pypto.full_like(output, pypto.scalar(2.0, pypto.DT_FP32)), output)

    with pypto.function("MAIN", input1, output):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                view_tensor_a = pypto.view(input1, view_shape,
                                         [b_idx * view_shape[0],
                                             s_idx * view_shape[1]],
                                         valid_shape=[
                                             pypto.min(pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0],
                                                     pypto.symbolic_scalar(view_shape[0])),
                                             pypto.min(pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1],
                                                     pypto.symbolic_scalar(view_shape[1])),
                                         ],
                                         )
                out_valid_cols = pypto.min(pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1],
                                           pypto.symbolic_scalar(view_shape[1]))
                view_output = pypto.view(output, (1, view_shape[1]),
                                        [0, s_idx * view_shape[1]],
                                        valid_shape=[1, out_valid_cols])

                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                # 执行 prod 沿 dim=0（在当前 view 内）
                partial_prod = pypto.prod(view_tensor_a, dim=0, keepdim=True)
                # updated_prod = pypto.mul(view_output, partial_prod)
                pypto.assemble(partial_prod, [0, s_idx * view_shape[1]], output)

    assert isinstance(output, pypto.tensor)

    # 使用 float32 的 torch tensor
    a_tensor = torch.rand(shape[0], shape[1], dtype=torch.float32) + 0.5  # 避免接近0导致数值不稳定
    b_tensor = torch.ones(1, shape[1], dtype=torch.float32)  # 初始值为1.0
    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    golden = torch.prod(a_tensor, dim=0, keepdim=True)
    assert_allclose(b_tensor.flatten(), golden.flatten(), rtol=3e-3, atol=3e-3)
    pypto.runtime._device_fini()