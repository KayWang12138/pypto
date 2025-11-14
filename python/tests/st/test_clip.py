#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import enum
import math
from itertools import product

import pytest
import torch
import numpy as np

import pto
from pto import (
    tensor, view, function,
    set_vec_tile_shapes,
)


MAPPING = {
    np.int8: pto.DT_INT8,
    np.int16: pto.DT_INT16,
    np.int32: pto.DT_INT32,
    np.float16: pto.DT_FP16,
    np.float32: pto.DT_FP32,
}


class TestType(int, enum.Enum):
    NotDefault2D = 0
    NotDefault3D = 1
    NotDefault4D = 2
    ElementDefaultMinDefaultMax = 3
    ElementDefaultMinNotDefaultMax = 4
    ElementNotDefaultMinDefaultMax = 5
    TensorDefaultMinDefaultMax = 6
    TensorNotDefaultMinDefaultMax = 7
    TensorDefaultMinNotDefaultMax = 8
    NoValue = 10


class ClipArgs:
    tile_shape = None
    view_shape = None
    min_ = None
    max_ = None


def get_broadcast_view_shape(self: pto.Tensor, other, view_shape):
    results = []
    for i, (self_dim, other_dim) in enumerate(zip(self.GetShape(), other.GetShape())):
        if self_dim != other_dim and self_dim == 1 and other_dim != 1:
            results.append(1)
        else:
            results.append(min(self_dim, view_shape[i]))
    return results


def get_broadcast_offset_ratio(self, other, view_shape):
    results = []
    for _, (self_dim, other_dim) in enumerate(zip(self.GetShape(), other.GetShape())):
        if self_dim != other_dim and self_dim == 1 and other_dim != 1:
            results.append(0)
        else:
            results.append(1)
    return results


def get_valid_shape(origin_shapes, view_shapes, loop_vars):
    if len(loop_vars) != len(origin_shapes) or len(origin_shapes) != len(view_shapes):
        raise ValueError("Length of `origin_shapes`/`view_shapes` should be the same as `loop_vars`")
    valid_shapes = []
    for origin_shape, view_shape, loop_var in zip(origin_shapes, view_shapes, loop_vars):
        valid_shape = min(origin_shape - loop_var * view_shape, view_shape)
        valid_shapes.append(valid_shape)
    return valid_shapes


def get_offsets(view_shapes, loop_vars, ratios=None):
    if len(loop_vars) != len(view_shapes):
        raise ValueError("Length of `view_shapes` should be the same as `loop_vars`")
    ratios = ratios or [1] * len(view_shapes)
    
    offsets = []
    for loop_var, view_shape, ratio in zip(loop_vars, view_shapes, ratios):
        offsets.append(loop_var * view_shape * ratio)
    return offsets


def broadcast_view(need_broadcast, broadcasted, view_shapes, loop_vars):
    tile_view_shape = get_broadcast_view_shape(need_broadcast, broadcasted, view_shapes)
    tile_offset_ratio = get_broadcast_offset_ratio(need_broadcast, broadcasted, view_shapes)
    valid_shapes = get_valid_shape(broadcasted.GetShape(), tile_view_shape, loop_vars)
    offsets = get_offsets(tile_view_shape, loop_vars, tile_offset_ratio)
    result = view(need_broadcast, tile_view_shape, valid_shapes, offsets)
    return result


def process_element_mode(tile_tensor_0, args):
    result = tensor()
    if args.type in [TestType.NotDefault2D, TestType.NotDefault3D, TestType.NotDefault4D]:
        result = pto.clip(tile_tensor_0, args.min_, args.max_)
    elif args.type == TestType.ElementDefaultMinDefaultMax:
        result = pto.clip(tile_tensor_0)
    elif args.type == TestType.ElementDefaultMinNotDefaultMax:
        result = pto.clip(tile_tensor_0, max_=args.max_)
    elif args.type == TestType.ElementNotDefaultMinDefaultMax:
        result = pto.clip(tile_tensor_0, min_=args.min_)
    return result


def process_tensor_mode(tile_tensor_0, inputs, args, loop_vars):
    result = tensor()
    if args.type in [TestType.NotDefault2D, TestType.NotDefault3D, TestType.NotDefault4D]:
        min_ = broadcast_view(inputs[1], inputs[0], args.view_shape, loop_vars)
        max_ = broadcast_view(inputs[2], inputs[0], args.view_shape, loop_vars)
        result = pto.clip(tile_tensor_0, min_, max_)
    elif args.type == TestType.TensorDefaultMinDefaultMax:
        result = pto.clip(tile_tensor_0)
    elif args.type == TestType.TensorDefaultMinNotDefaultMax:
        max_ = broadcast_view(inputs[2], inputs[0], args.view_shape, loop_vars)
        result = pto.clip(tile_tensor_0, max_=max_)
    elif args.type == TestType.TensorNotDefaultMinDefaultMax:
        min_ = broadcast_view(inputs[1], inputs[0], args.view_shape, loop_vars)
        result = pto.clip(tile_tensor_0, min_=min_)
    return result


def build_clip(inputs, outputs, view_shape, tile_shape, args):
    shape = inputs[0].GetShape()
    view_shape = [min(v, self_dim) for v, self_dim in zip(view_shape, shape)]
    with function("Clip", inputs, outputs):
        for loop_vars in product(*[pto.loop(math.ceil(s / v)) for s, v in zip(shape, view_shape)]):
            offsets = get_offsets(view_shape, loop_vars)
            valid_shape = get_valid_shape(inputs[0].GetShape(), view_shape, loop_vars)
            tile_tensor_0 = view(inputs[0], view_shape, valid_shape, offsets)
            set_vec_tile_shapes(*tile_shape)

            res = tensor()
            if args.is_element:
                res.move(process_element_mode(tile_tensor_0, args))
            else:
                res.move(process_tensor_mode(tile_tensor_0, inputs, args, loop_vars))
            pto.assemble(res, offsets, outputs[0])


def run_clip(inputs, outputs, args):
    inputs = [tensor(x, MAPPING[x.dtype]) for x in inputs]
    outputs = [tensor(x, MAPPING[x.dtype]) for x in outputs]
    build_clip(inputs, outputs, args.view_shape, args.tile_shape, args)
    pto.runtime._device_run_once_data_from_host(
        [x.flatten().tolist() for x in inputs],
        [y.flatten().tolist() for y in outputs])
    result = [np.array(y).reshape(y.size()) for y in outputs]
    pto.runtime._device_fini()
    return result


@pytest.mark.skip(reason="Not Passed.")
def test_clip_1():
    inputs = [np.random.rand(3, 2), np.random.rand(3, 2), np.random.rand(3, 2)]
    outputs = [np.random.rand(3, 2)]
    args = ClipArgs(
        view_shape=[2, 2],
        tile_shape=[2, 2],
    )

    outputs = run_clip(inputs, outputs, args)
    golden = torch.clip(inputs[0], inputs[1], inputs[2])
    assert torch.allclose(outputs, golden, rtol=1e-9, atol=1e-10)
