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

VAL = 1

H = 32
W = 32
N = 1
C = 256

def add_wrapper(val: int, add1_flag: bool = False):

    @pypto.frontend.jit()
    def add_core(
        input0: pypto.Tensor((H, W, N, C), pypto.DT_FP32),
        input1: pypto.Tensor((H, W, N, C), pypto.DT_FP32),
    ) -> pypto.Tensor((H, W, N, C), pypto.DT_FP32):
        """
        Core computation logic that can be specialized at compile time.

        This function demonstrates static branching where the condition
        is evaluated at compile time, resulting in different compiled kernels.
        """
        pypto.set_vec_tile_shapes(1, 4, 1, 64)

        # Calculate the loop parameters
        b = H
        tile_b = 1
        b_loop = b // tile_b

        output = pypto.tensor((H, W, N, C), pypto.DT_FP32)
        for idx in pypto.loop(b_loop):
            b_offset = idx * tile_b
            b_offset_end = (idx + 1) * tile_b
            t0_sub = input0[b_offset:b_offset_end, ...]
            t1_sub = input1[b_offset:b_offset_end, ...]
            t3_sub = t0_sub + t1_sub
            if add1_flag:
                result = t3_sub + val
            else:
                result = t3_sub
            pypto.assemble(result, [b_offset, 0, 0, 0], output)

        return output

    return add_core


def add_kernel_default(
    input0,
    input1,
):
    val = VAL
    return add_wrapper(val)(input0, input1)


def add_kernel_true(
    input0,
    input1,
):
    val = VAL
    return add_wrapper(val, True)(input0, input1)


def add_kernel_false(
    input0,
    input1,
):
    val = VAL
    return add_wrapper(val, False)(input0, input1)


def test_add_scalar_loop_dynamic_axis_static_cond():
    shape = (32, W, N, C)

    # Prepare data
    input_data0 = torch.rand(shape, dtype=torch.float, device="npu")
    input_data1 = torch.rand(shape, dtype=torch.float, device="npu")

    # Test 1: default (add1_flag is False)
    output_data = add_kernel_default(input_data0, input_data1).cpu()
    torch_add = torch.add(input_data0, input_data1).cpu()
    torch.allclose(output_data, torch_add, rtol=3e-3, atol=3e-3)
    print("test add1_flag default False is OK!")

    # Test 2: add1_flag is False
    output_data1 = add_kernel_false(input_data0, input_data1).cpu()
    torch.allclose(output_data1, torch_add, rtol=3e-3, atol=3e-3)
    print("test add1_flag False is OK!")

    # Test 3: add1_flag is True
    output_data2 = add_kernel_true(input_data0, input_data1).cpu()
    torch_add1 = torch.add(input_data0, input_data1).cpu() + VAL
    torch.allclose(output_data2, torch_add1, rtol=3e-3, atol=3e-3)
    print("test add1_flag True is OK!")


if __name__ == "__main__":
    print("Running examples that require NPU hardware...")
    print("(Make sure CANN environment is configured and NPU is available)\n")
    test_add_scalar_loop_dynamic_axis_static_cond()

