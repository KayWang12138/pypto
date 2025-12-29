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

N = pypto.frontend.dynamic("N")
M = 1024

FLAG = False

@pypto.frontend.jit()
def basic_dynamic(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32),
    c: pypto.Tensor((N, M), pypto.DT_FP32),
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
    pypto.Tensor((N, M), pypto.DT_FP32),
):
    d = pypto.Tensor((N, M), pypto.DT_FP32)
    e = pypto.Tensor((N, M), pypto.DT_FP32)

    for bs_idx in pypto.loop(N//256):
        tile_a = pypto.view(a, (256, M), [bs_idx * 256, 0])
        tile_b = pypto.view(b, (256, M), [bs_idx * 256, 0])
        tile_c = pypto.view(c, (256, M), [bs_idx * 256, 0])
        if FLAG:
            tile_a[:] = pypto.add(tile_a, tile_c)
        else:
            tile_a[:] = pypto.sub(tile_a, tile_c)
        tile_b[:] = pypto.sub(tile_b, tile_c)
        d[bs_idx * 256: (bs_idx + 1) * 256, :M] = pypto.add(tile_a, tile_a)
        e[bs_idx * 256: (bs_idx + 1) * 256, :M] = pypto.add(tile_b, tile_b)
    return d, e


def test_basic_dynamic_run():
    n, m = 1024, 1024
    np.random.seed(0)
    a = torch.zeros((n, m), dtype=torch.float32)
    b = torch.zeros((n, m), dtype=torch.float32)
    c = torch.rand((n, m), dtype=torch.float32)

    d, e = basic_dynamic(a, b, c)

if __name__ == "__main__":
    test_basic_dynamic_run()
