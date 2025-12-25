# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import torch
import pypto
import numpy as np
from numpy.testing import assert_allclose

CASE_NAME = "add_scalar_loop_dynamic_axis"

VAL = 1

H = pypto.frontend.dynamic("H")
W = 32
N = 1
C = 256


@pypto.frontend.jit()
def add_scalar_loop_dynamic_axis(
    input0: pypto.Tensor((H, W, N, C), pypto.DT_FP32),
    input1: pypto.Tensor((H, W, N, C), pypto.DT_FP32),
) -> pypto.Tensor((H, W, N, C), pypto.DT_FP32):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    val = VAL

    #calculate the loop parameters
    b = H
    tile_b = 1
    b_loop = b // tile_b

    output = pypto.tensor((H, W, N, C), pypto.DT_FP32)
    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = pypto.min((idx + 1) * tile_b, b)
        t0_sub = pypto.view(input0, [tile_b, W, N, C], [b_offset, 0, 0, 0], valid_shape=[b_offset_end - b_offset, W, N, C])
        t1_sub = pypto.view(input1, [tile_b, W, N, C], [b_offset, 0, 0, 0], valid_shape=[b_offset_end - b_offset, W, N, C])
        t3_sub = t0_sub + t1_sub
        t3_sub = t3_sub + val
        pypto.assemble(t3_sub, [b_offset, 0, 0, 0], output)
    return output


def test_add_scalar_loop_dynamic_axis():
    shape = (32, W, N, C)
    val = VAL

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    #prepare data
    input_data0 = torch.rand(shape, dtype=torch.float, device='npu')
    input_data1 = torch.rand(shape, dtype=torch.float, device='npu')

    output_data = add_scalar_loop_dynamic_axis(input_data0, input_data1).cpu()

    torch_add = torch.add(input_data0, input_data1).cpu() + val

    try:
        assert_allclose(np.array(output_data), np.array(torch_add), rtol=3e-3, atol=3e-3)
    except AssertionError as e:
        print(f"\033[91m✗ {CASE_NAME} FAILED !!!\033[0m")
        raise e
    else:
        print(f"\033[92m✓ {CASE_NAME} PASSED !!!\033[0m")


if __name__ == "__main__":
    print("Running examples that require NPU hardware...")
    print("(Make sure CANN environment is configured and NPU is available)\n")
    test_add_scalar_loop_dynamic_axis()
