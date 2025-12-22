# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import pypto


@pypto.jit
def add_kernel(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int):
    tensor_shape = input0.shape
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b = tensor_shape[0]
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = input0[b_offset:b_offset_end, ...]
        t1_sub = input1[b_offset:b_offset_end, ...]
        t3_sub = t0_sub + t1_sub
        output[b_offset:b_offset_end, ...] = t3_sub + val


def add_scalar_loop(input_data0, input_data1, output_data, val):
    pto_input0 = pypto.from_torch(input_data0, "IN_0")
    pto_input1 = pypto.from_torch(input_data1, "IN_1")
    pto_output = pypto.from_torch(output_data, "OUT_0")

    add_kernel(pto_input0, pto_input1, pto_output, val)


def test_add_scalar_loop():
    shape = (32, 32, 1, 256)
    #prepare data
    val = 1
    input_data0 = torch.rand(shape, dtype=torch.float, device='npu')
    input_data1 = torch.rand(shape, dtype=torch.float, device='npu')
    output_data = torch.zeros(shape, dtype=torch.float, device='npu')
    add_scalar_loop(input_data0, input_data1, output_data, val)

    torch_add = torch.add(input_data0, input_data1) + val
    torch.allclose(output_data, torch_add, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    print("Running examples that require NPU hardware...")
    print("(Make sure CANN environment is configured and NPU is available)\n")
    test_add_scalar_loop()
