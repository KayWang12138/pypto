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

CASE_NAME = "add_direct"

SHAPE = (1, 4, 1, 64)


@pypto.frontend.jit()
def add(
    input_data0: pypto.Tensor(SHAPE, pypto.DT_FP32),
    input_data1: pypto.Tensor(SHAPE, pypto.DT_FP32),
) -> pypto.Tensor(SHAPE, pypto.DT_FP32):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output_data = pypto.add(input_data0, input_data1)
    return output_data


def test_add():
    shape = SHAPE

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    
    #prepare data
    input_data0 = torch.rand(shape, dtype=torch.float, device='npu')
    input_data1 = torch.rand(shape, dtype=torch.float, device='npu')

    output_data = add(input_data0, input_data1).cpu()

    torch_add = torch.add(input_data0, input_data1).cpu()

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
    test_add()
