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
import inspect
import torch
import pytest
import numpy as np
from numpy.testing import assert_allclose
import pypto


@pypto.jit
def add_kernel(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int):
    tensor_shape = input0.shape
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b = pypto.symbolic_scalar(tensor_shape[0])
    tile_b = pypto.symbolic_scalar(1)
    b_loop = b / tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = input0[b_offset:b_offset_end, :, :, :]
        t1_sub = input1[b_offset:b_offset_end, :, :, :]
        t3_sub = t0_sub + t1_sub
        if pypto.cond(idx < 2):
            output[b_offset:b_offset_end, :, :, :] = t3_sub + val
        else:
            output[b_offset:b_offset_end, :, :, :] = t3_sub


def add_scalar_loop_dyn_axis_dynamic_cond(input_data0, input_data1, output_data, val=0, dynamic_axis=False):
    if dynamic_axis == True:
        print("tensor is dyamic shape tensor")
        pto_input0 = pypto.from_torch(input_data0, "IN_0", dynamic_axis=[0])
        pto_input1 = pypto.from_torch(input_data1, "IN_1", dynamic_axis=[0])
        pto_output = pypto.from_torch(output_data, "OUT_0", dynamic_axis=[0])
    else:
        print("tensor is static shape tensor")
        pto_input0 = pypto.from_torch(input_data0, "IN_0")
        pto_input1 = pypto.from_torch(input_data1, "IN_1")
        pto_output = pypto.from_torch(output_data, "OUT_0")
    add_kernel(pto_input0, pto_input1, pto_output, val)


def test_add_scalar_loop_dynamic_axis_dynamic_cond(device_id=-1):
    if (device_id == -1):
        device_id = torch.npu.get_device_id()
    else:
        torch.npu.set_device(device_id)

    shape = (32, 32, 1, 256)
    #prepare data
    val = 1
    input_data0 = torch.rand(shape, dtype=torch.float, device=f'npu:{device_id}')
    input_data1 = torch.rand(shape, dtype=torch.float, device=f'npu:{device_id}')
    output_data = torch.zeros(shape, dtype=torch.float, device=f'npu:{device_id}')
    add_scalar_loop_dyn_axis_dynamic_cond(input_data0, input_data1, output_data, val, True)

    torch_add = torch.add(input_data0, input_data1)
    torch_add[0:2, :, :, :] = torch_add[0:2, :, :, :] + val
    npu_data = output_data.cpu()
    torch_data = torch_add.cpu()
    assert_allclose(np.array(npu_data), np.array(torch_data), rtol=3e-3, atol=3e-3)
    print(inspect.currentframe().f_code.co_name + " is OK!")


if __name__ == "__main__":
    device_id = 5
    test_add_scalar_loop_dynamic_axis_dynamic_cond(device_id)
