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

H = pypto.frontend.dynamic("H")
W = 32
N = 1
C = 256


@pypto.frontend.jit()
def add_kernel(
    input0: pypto.Tensor((H, W, N, C), pypto.DT_FP32),
    input1: pypto.Tensor((H, W, N, C), pypto.DT_FP32),
) -> pypto.Tensor((H, W, N, C), pypto.DT_FP32):
    """
    Example demonstrating dynamic conditional branching using loop boundary detection.

    This function adds two tensors and conditionally adds extra values
    to the first and last batch elements based on loop position.
    """
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    val = VAL

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
        if pypto.is_loop_begin(idx):
            result = t3_sub + val
        elif pypto.is_loop_end(idx):
            result = t3_sub + val + 1
        else:
            result = t3_sub
        pypto.assemble(result, [b_offset, 0, 0, 0], output)

    return output


def test_add_scalar_loop_dynamic_axis_dynamic_loop_cond():
    shape = (32, W, N, C)

    # Prepare data
    input_data0 = torch.rand(shape, dtype=torch.float, device="npu")
    input_data1 = torch.rand(shape, dtype=torch.float, device="npu")

    output_data = add_kernel(input_data0, input_data1).cpu()

    torch_add = torch.add(input_data0, input_data1).cpu()
    torch_add[0:1, ...] = torch_add[0:1, ...] + VAL
    torch_add[31:32, ...] = torch_add[31:32, ...] + VAL + 1
    torch.allclose(output_data, torch_add, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    print("Running examples that require NPU hardware...")
    print("(Make sure CANN environment is configured and NPU is available)\n")
    test_add_scalar_loop_dynamic_axis_dynamic_loop_cond()

