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
import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path for direct execution
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

import numpy as np
try:
    from .. import pypto
except ImportError:
    # Fallback for direct execution
    import pypto

@pypto.frontend.function
def inner_add(
    x: pypto.Tensor((8,), pypto.DT_FP32),
    bias: pypto.Tensor((8,), pypto.DT_FP32),
) -> (
    pypto.Tensor((8,), pypto.DT_FP32),
):
    tmp = pypto.add(x, bias)
    return tmp


@pypto.frontend.jit
def outer_kernel(
    a: pypto.Tensor((8,), pypto.DT_FP32),
    b: pypto.Tensor((8,), pypto.DT_FP32),
) -> (
    pypto.Tensor((8,), pypto.DT_FP32),
):
    res = inner_add(a, b)
    return res


def test_nested_function_kernel_inline_expansion():
    shape = (8,)

    # Generate test data
    a = torch.rand(shape, dtype=torch.float32)
    b = torch.rand(shape, dtype=torch.float32)

    # Execute nested function kernel
    result = outer_kernel(a, b)

if __name__ == "__main__":
    test_nested_function_kernel_inline_expansion()