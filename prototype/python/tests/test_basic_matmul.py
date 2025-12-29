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
import torch
import numpy as np

# Add parent directory to path for direct execution
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
try:
    from .. import pypto
except ImportError:
    # Fallback for direct execution
    import pypto

N = 128
M = 128

@pypto.frontend.jit()
def basic_matmul(
    a: pypto.Tensor((N, M), pypto.DT_FP16),
    b: pypto.Tensor((N, M), pypto.DT_FP16),
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
    pypto.Tensor((N, M), pypto.DT_FP32),
):

    c = pypto.Tensor((N, M), pypto.DT_FP32)
    d = pypto.Tensor((N, M), pypto.DT_FP32)

    c[:] = pypto.matmul(a, b, out_dtype=pypto.DT_FP32)
    d[:] = pypto.matmul(a, b, out_dtype=pypto.DT_FP32, b_trans=True)
    return c, d


def test_basic_matmul_run():
    n, m = 128, 128
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float16)
    b = torch.rand((n, m), dtype=torch.float16)

    c, d = basic_matmul(a, b)

if __name__ == "__main__":
    test_basic_matmul_run()
