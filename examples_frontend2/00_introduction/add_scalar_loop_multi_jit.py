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
from numpy.testing import assert_allclose\

CASE_NAME = "add_scalar_loop_multi_jit"

SHAPE = (32, 32, 1, 256)
VAL = 1

def add_core(input0: pypto.Tensor, input1: pypto.Tensor, add1_flag: bool = False):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out = pypto.tensor(SHAPE, pypto.DT_FP32)
    if add1_flag:
        t3 = input0 + input1
        out[:] = t3 + VAL
    else:
        out[:] = input0 + input1
    return out


@pypto.frontend.jit()
def add_true(
    input0: pypto.Tensor(SHAPE, pypto.DT_FP32),
    input1: pypto.Tensor(SHAPE, pypto.DT_FP32),
) -> pypto.Tensor(SHAPE, pypto.DT_FP32):
    out = add_core(input0, input1, True)
    return out


@pypto.frontend.jit()
def add_false(
    input0: pypto.Tensor(SHAPE, pypto.DT_FP32),
    input1: pypto.Tensor(SHAPE, pypto.DT_FP32),
) -> pypto.Tensor(SHAPE, pypto.DT_FP32):
    out = add_core(input0, input1, False)
    return out


def test_add_scalar_multi_jit():
    shape = SHAPE
    val = VAL

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    #prepare data
    input_data0 = torch.rand(shape, dtype=torch.float, device='npu')
    input_data1 = torch.rand(shape, dtype=torch.float, device='npu')
    output_data = add_false(input_data0, input_data1).cpu()

    ## test add1_flag is False
    torch_add = torch.add(input_data0, input_data1).cpu()
    assert_allclose(np.array(output_data), np.array(torch_add), rtol=3e-3, atol=3e-3)
    print("test add1_flag False is OK!")

    ## test add1_flag is True
    output_data1 = add_true(input_data0, input_data1).cpu()

    torch_add1 = torch.add(input_data0, input_data1).cpu() + val

    try:
        assert_allclose(np.array(output_data1), np.array(torch_add1), rtol=3e-3, atol=3e-3)
    except AssertionError as e:
        print(f"\033[91m✗ {CASE_NAME} FAILED !!!\033[0m")
        raise e
    else:
        print(f"\033[92m✓ {CASE_NAME} PASSED !!!\033[0m")


if __name__ == "__main__":
    print("Running examples that require NPU hardware...")
    print("(Make sure CANN environment is configured and NPU is available)\n")
    test_add_scalar_multi_jit()
