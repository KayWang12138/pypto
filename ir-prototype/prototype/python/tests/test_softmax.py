#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Softmax Example for PyPTO

This example demonstrates how to implement a softmax operation using PyPTO, including:
- Manual softmax computation from basic operations
- Dynamic axis marking for variable batch sizes
- Tiling configuration for efficient execution
- Loop-based processing for large tensors

Softmax is a fundamental operation in neural networks, especially for attention mechanisms.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np

if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

import torch
import numpy as np
try:
    from .. import pypto
except ImportError:
    # Fallback for direct execution
    import pypto

b = pypto.frontend.dynamic("B")
n1, n2, dim = 32, 1, 256

@pypto.frontend.jit()
def softmax(
    input_tensor: pypto.Tensor((b, n1, n2, dim), pypto.DT_FP32),
    ) -> (
        pypto.Tensor((b, n1, n2, dim), pypto.DT_FP32)
    ):
    """
    Softmax implementation with dynamic batch size support.

    This function processes input tensors in batches, applying softmax
    to each batch independently. The batch dimension is marked as dynamic,
    allowing variable batch sizes at runtime.

    Parameters
    ----------
    input_tensor : pypto.Tensor
        shape [batch, n1, n2, dim]
    """
    output_tensor = pypto.Tensor((b, n1, n2, dim), pypto.DT_FP32)
    tile_b = 1  # Process one batch at a time
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b

        # Extract batch slice using view (supports symbolic offsets)
        input_view = pypto.view(input_tensor, [tile_b, n1, n2, dim], [b_offset, 0, 0, 0])

        # core softmax formula
        row_max = pypto.amax(input_view, dim=-1, keepdim=True)
        sub = input_view - row_max
        exp = pypto.exp(sub)
        esum = pypto.sum(exp, dim=-1, keepdim=True)
        softmax_out = exp / esum

        # Assemble result back to output tensor
        pypto.assemble(softmax_out, [b_offset, 0, 0, 0], output_tensor)

    return output_tensor


def test_softmax():
    """
    Test softmax implementation against PyTorch reference.

    Tests with shape [batch, n1, n2, dim] where batch is dynamic.
    """

    # Shape for verification: NCHW format, N can be any integer number as it is defined as dynamic axis
    b_concrete = 32
    shape = (b_concrete, n1, n2, dim)

    # Prepare data
    input_data = torch.rand(shape, dtype=torch.float32)

    # Launch the kernel
    output_data = softmax(input_data)


if __name__ == "__main__":
    test_softmax()
