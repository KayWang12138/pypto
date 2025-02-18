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
def add_kernel(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output[:] = input0 + input1


def add(input_data0, input_data1, output_data):
    pto_input0 = pypto.from_torch(input_data0, "IN_0")
    pto_input1 = pypto.from_torch(input_data1, "IN_1")
    pto_output = pypto.from_torch(output_data, "OUT_0")

    add_kernel(pto_input0, pto_input1, pto_output)


def test_add():
    shape = (1, 4, 1, 64)
    #prepare data
    input_data0 = torch.rand(shape, dtype=torch.float, device='npu')
    input_data1 = torch.rand(shape, dtype=torch.float, device='npu')
    output_data = torch.zeros(shape, dtype=torch.float, device='npu')
    add(input_data0, input_data1, output_data)

    torch_add = torch.add(input_data0, input_data1)
    torch.allclose(output_data, torch_add, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    test_add()
