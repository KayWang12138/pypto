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
from typing import List

import pypto
import torch
import torch_npu

from pypto import Tensor as PTensor, loop, ceildiv, SymInt
from pypto.frontend import jit


@jit
def isfinite_2d_impl(x: PTensor, view_shape: List[SymInt], tile_shape: List[int]) -> PTensor:
    out = PTensor(x.shape, pypto.DT_BOOL)
    pypto.set_vec_tile_shapes(tile_shape)
    for i in loop(ceildiv(x.shape[0], view_shape[0])):
        for j in loop(ceildiv(x.shape[1], view_shape[1])):
            tile = pypto.view(x, view_shape)
            result = pypto.isfinite(tile)
            pypto.assemble(result, [i * view_shape[0], j * view_shape[1]], out)
            del tile
    return out


def test_is_finite():
    view_shape = [32, 128]
    tile_shape = [32, 32]
    x_pt = torch.rand(32, 128)
    ids = torch.randint(0, 32 * 128, 30)
    x_pt.view(-1, 1)[ids] = torch.nan
    ids = torch.randint(0, 32 * 128, 30)
    x_pt.view(-1, 1)[ids] = torch.inf
    ids = torch.randint(0, 32 * 128, 30)
    x_pt.view(-1, 1)[ids] = -torch.inf

    torch_npu.npu.set_device(0)
    x = pypto.from_torch(x_pt)
    golden = torch.isfinite(x_pt)
    out = isfinite_2d_impl(x, view_shape, tile_shape)
    assert torch.allclose(golden, out)


if __name__ == "__main__":
    test_is_finite()
