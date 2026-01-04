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
from dataclasses import dataclass
from typing import List, Tuple
import pypto


@dataclass
class HcPostTileConfig: 
    def __init__(self): 
        self.tile_b = 8
        

@dataclass
class MlaQuantInputs: 
    x: pypto.tensor = None
    residual: pypto.tensor = None
    post: pypto.tensor = None
    comb: pypto.tensor = None


def hc_post_compute(
    x: pypto.tensor,
    residual: pypto.tensor, 
    post: pypto.tensor, 
    comb: pypto.tensor,
    y: pypto.tensor,
    tile_config: HcPostTileConfig):
    assert len(x.shape) == 2 and len(residual.shape) == 3 and len(post.shape) == 2 and len(comb.shape) == 3

    dtype = x.dtype
    t = x.shape[0]
    hc = residual.shape[1]
    d = residual.shape[2]

    post_reshape = pypto.reshape(post, [t, hc, 1], inplace=True)
    x_reshape = pypto.reshape(x, [t, 1, d], inplace=True)
    comb_reshape = pypto.reshape(comb, [t, hc, hc, 1], inplace=True)
    residual_reshape = pypto.reshape(residual, [t, hc, 1, d], inplace=True)

    assert hc == 4 or d == 512
    for t_idx in pypto.loop(0, t, 1, name="LI_LOOP_BATCH", idx_name="t_idx"):
        pypto.set_vec_tile_shapes(1, 4, 128)
        post_slice = pypto.view(post_reshape, [1, hc, 1], [t_idx, 0, 0])
        x_slice = pypto.view(x_reshape, [1, 1, d], [t_idx, 0, 0])
        x_slice_fp32 = pypto.cast(x_slice, pypto.DT_FP32)
        post_res = post_slice * x_slice_fp32

        pypto.set_vec_tile_shapes(1, 4, 4, 128)
        residual_slice = pypto.view(residual_reshape, [1, hc, 1, d], [t_idx, 0, 0, 0])
        comb_slice = pypto.view(comb_reshape, [1, hc, hc, 1], [t_idx, 0, 0, 0])

        residual_res = residual_slice * comb_slice
        residual_reduce = pypto.sum(residual_res, 1)
        pypto.set_vec_tile_shapes(1, 4, 128)
        y_tmp = pypto.add(post_res, residual_reduce)
        y_dtype = pypto.cast(y_tmp, pypto.DT_BF16)
        pypto.assemble(y_dtype, [t_idx, 0, 0], y)


@pypto.jit(
    host_options={"only_codegen": True}
)
def hc_post_kernel(
    x: pypto.tensor,
    residual: pypto.tensor,
    post: pypto.tensor,
    comb: pypto.tensor,
    y: pypto.tensor,
    tile_config: HcPostTileConfig):
    hc_post_compute(x, residual, post, comb, y, tile_config)


@allow_in_graph
def hc_post_torch_graph(
    x: torch.tensor,
    residual: torch.tensor,
    post: torch.tensor,
    comb: torch.tensor,
    y: torch.tensor,
    tile_config: HcPostTileConfig):

    inputs = {
        x: [0],
        residual: [0],
        post: [0],
        comb: [0],
    }
    outputs = {
        y: [0]
    }

    if not isinstance(x, FakeTensor):
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        hc_post_kernel(*pto_inputs, *pto_outputs, tile_config)
        pypto.runtime._device_synchronize()#内部接口，不推荐使用