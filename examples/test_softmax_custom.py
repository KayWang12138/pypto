#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
import pypto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose


def softmax_core(input_tensor):
    row_max = pypto.amax(input_tensor)
    sub = input_tensor - row_max
    exp = pypto.exp(sub)
    esum = pypto.sum(exp)
    return exp / esum


# enalbe jit for softmax_custom
@pypto.jit
def softmax_custom(inputs, outputs):
    input_tensor = inputs[0]
    output_tensor = outputs[0]

    # setting of dynamic axis, the actual size of the axis can be any integer number during runtime
    # the dynamic axis of input_tensor/output_tensor will be marked as symbolic_scalar
    pypto.mark_dynamic(input_tensor, 0)
    pypto.mark_dynamic(output_tensor, 0)

    # after the dynamic axis of tensor is marked, get the tensor shape accordingly
    tensor_shape = input_tensor.shape
    b = tensor_shape[0]
    n1, n2, dim = tensor_shape[1:]
    tile_b = pypto.symbolic_scalar(1)
    b_loop = b / tile_b

    # tiling shape setting
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    with pypto.function("SOFTMAX", [input_tensor], [output_tensor]):
        for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
            b_offset = idx * tile_b
            b_offset_end = (idx + 1) * tile_b
            input_view = input_tensor[b_offset:b_offset_end, :n1, :n2, :dim]
            softmax_out = softmax_core(input_view)
            pypto.assemble(softmax_out, [b_offset, 0, 0, 0], output_tensor)


def test_softmax_custom():
    # shape for verification, NCHW, N can be any interger number as it is defined as dynamic axis
    shape = (32, 32, 1, 256)

    # setting of which device to be used
    device_id = 0
    torch.npu.set_device(device_id)

    # prepare data
    input_data = torch.rand(shape, dtype=torch.float, device=f'npu:{device_id}')
    output_data = torch.zeros(shape, dtype=torch.float, device=f'npu:{device_id}')

    inputs = [input_data]
    outputs = [output_data]

    # launch the kernel
    softmax_custom(inputs, outputs)

    torch_softmax = torch.softmax(input_data, dim=3)
    npu_data = output_data.cpu()
    torch_data = torch_softmax.cpu()

    assert_allclose(np.array(npu_data), np.array(torch_data), rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    test_softmax_custom()
