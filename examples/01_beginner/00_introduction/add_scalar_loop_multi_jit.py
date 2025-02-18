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


def add_core(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int, add1_flag: bool = False):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    if add1_flag:
        t3 = input0 + input1
        output[:] = t3 + val
    else:
        output[:] = input0 + input1


@pypto.jit
def add_kernel_true(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int):
    add_core(input0, input1, output, val, True)


@pypto.jit
def add_kernel_false(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int):
    add_core(input0, input1, output, val, False)


def add_add1flag_false(input_data0, input_data1, output_data, val=0):
    pto_input0 = pypto.from_torch(input_data0, "IN_0")
    pto_input1 = pypto.from_torch(input_data1, "IN_1")
    pto_output = pypto.from_torch(output_data, "OUT_0")
    add_kernel_false(pto_input0, pto_input1, pto_output, val)


def add_add1flag_true(input_data0, input_data1, output_data, val=0):
    pto_input0 = pypto.from_torch(input_data0, "IN_0")
    pto_input1 = pypto.from_torch(input_data1, "IN_1")
    pto_output = pypto.from_torch(output_data, "OUT_0")
    add_kernel_true(pto_input0, pto_input1, pto_output, val)


def test_add_scalar_multi_jit():
    shape = (32, 32, 1, 256)
    #prepare data
    val = 1
    input_data0 = torch.rand(shape, dtype=torch.float, device='npu')
    input_data1 = torch.rand(shape, dtype=torch.float, device='npu')
    output_data = torch.zeros(shape, dtype=torch.float, device='npu')
    add_add1flag_false(input_data0, input_data1, output_data, val)

    ## test add1_flag is False
    torch_add = torch.add(input_data0, input_data1)
    torch.allclose(output_data, torch_add, rtol=3e-3, atol=3e-3)
    print("test add1_flag False is OK!")

    ## test add1_flag is True
    output_data1 = torch.zeros(shape, dtype=torch.float, device='npu')
    add_add1flag_true(input_data0, input_data1, output_data1, val)

    torch_add1 = torch.add(input_data0, input_data1) + val
    torch.allclose(output_data1, torch_add1, rtol=3e-3, atol=3e-3)
    print("test add1_flag True is OK!")


if __name__ == "__main__":
    print("Running examples that require NPU hardware...")
    print("(Make sure CANN environment is configured and NPU is available)\n")
    test_add_scalar_multi_jit()
