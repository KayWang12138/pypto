#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Frontend transform operator basic test cases."""

import os
import numpy as np
import pypto
import torch
from numpy.testing import assert_allclose


def create_view_kernel(
    input_shape: tuple[int, ...],
    view_shape: list[int],
    offsets: list[int],
    valid_shape: list[int] = None,
):
    out_shape = tuple(view_shape)
    tile_shapes = [8 for _ in range(len(input_shape))]

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def view_kernel(
        x: pypto.Tensor(input_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(out_shape, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(*tile_shapes)
        if valid_shape is None:
            out = pypto.view(x, view_shape, offsets)
        else:
            out = pypto.view(x, view_shape, offsets, valid_shape=valid_shape)
        return out

    return view_kernel


def create_assemble_kernel(src_shape: tuple, dst_shape: tuple, offsets: list):
    tile_shapes = [8 for _ in range(len(src_shape))]

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def assemble_kernel(
        src: pypto.Tensor(src_shape, pypto.DT_FP32),
        dst: pypto.Tensor(dst_shape, pypto.DT_FP32),
    ) -> None:
        pypto.set_vec_tile_shapes(*tile_shapes)
        pypto.assemble(src, offsets, dst)

    return assemble_kernel


def create_gather_kernel(
    input_shape: tuple[int, ...],
    index_shape: tuple[int, ...],
    dim: int,
):
    tile_shapes = [8 for _ in range(len(input_shape))]

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def gather_kernel(
        x: pypto.Tensor(input_shape, pypto.DT_INT32),
        index: pypto.Tensor(index_shape, pypto.DT_INT64),
    ) -> pypto.Tensor(index_shape, pypto.DT_INT32):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out = pypto.gather(x, dim, index)
        return out

    return gather_kernel


def create_concat_kernel(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    dim: int,
):
    dim_pos = dim if dim >= 0 else dim + len(a_shape)
    out_shape = list(a_shape)
    out_shape[dim_pos] = a_shape[dim_pos] + b_shape[dim_pos]
    out_shape_tuple = tuple(out_shape)
    tile_shapes = [8 for _ in range(len(a_shape))]

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def concat_kernel(
        a: pypto.Tensor(a_shape, pypto.DT_FP32),
        b: pypto.Tensor(b_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(out_shape_tuple, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out = pypto.concat([a, b], dim=dim)
        return out

    return concat_kernel


def create_transpose_kernel(
    shape: tuple[int, ...],
    dim0: int,
    dim1: int,
):
    dim0_pos = dim0 if dim0 >= 0 else dim0 + len(shape)
    dim1_pos = dim1 if dim1 >= 0 else dim1 + len(shape)
    out_shape = list(shape)
    out_shape[dim0_pos], out_shape[dim1_pos] = out_shape[dim1_pos], out_shape[dim0_pos]
    out_shape_tuple = tuple(out_shape)
    tile_shapes = [8 for _ in range(len(shape))]

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def transpose_kernel(
        x: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> pypto.Tensor(out_shape_tuple, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out = pypto.transpose(x, dim0, dim1)
        return out

    return transpose_kernel


def test_view():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    x_shape = (4, 6)
    view_shape = [2, 3]
    offsets = [1, 2]
    kernel = create_view_kernel(x_shape, view_shape, offsets)

    x = torch.arange(24, dtype=torch.float32, device=f"npu:{device_id}").reshape(4, 6)
    out = kernel(x)
    torch.npu.synchronize()

    expected = x[1:3, 2:5]
    assert_allclose(np.array(out.cpu()), np.array(expected.cpu()), rtol=3e-3, atol=3e-3)


def test_assemble():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    src_shape = (2, 3)
    dst_shape = (5, 5)
    offsets = [1, 1]
    kernel = create_assemble_kernel(src_shape, dst_shape, offsets)

    src = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32, device=f"npu:{device_id}")
    dst = torch.zeros(dst_shape, dtype=torch.float32, device=f"npu:{device_id}")
    kernel(src, dst)
    torch.npu.synchronize()
    
    expected = torch.tensor([[0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0],
                             [0, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]], dtype=torch.float32, device=f"npu:{device_id}")
    assert_allclose(dst.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)


def test_gather():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    input_tensor = torch.tensor(
        [[0, 1, 2, 3], [10, 11, 12, 13]],
        dtype=torch.int32,
        device=f"npu:{device_id}",
    )
    index_tensor = torch.tensor(
        [[3, 0, 1], [2, 1, 0]],
        dtype=torch.int64,
        device=f"npu:{device_id}",
    )
    kernel = create_gather_kernel(input_tensor.shape, index_tensor.shape, dim=1)

    out = kernel(input_tensor, index_tensor)
    torch.npu.synchronize()

    expected = torch.gather(input_tensor, dim=1, index=index_tensor)
    assert_allclose(np.array(out.cpu()), np.array(expected.cpu()), rtol=0.0, atol=0.0)


def test_concat():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    a = torch.tensor([[1.0], [1.0]], dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32, device=f"npu:{device_id}")
    kernel = create_concat_kernel(a.shape, b.shape, dim=-1)

    out = kernel(a, b)
    torch.npu.synchronize()

    expected = torch.cat([a, b], dim=-1)
    assert_allclose(np.array(out.cpu()), np.array(expected.cpu()), rtol=3e-3, atol=3e-3)


def test_transpose():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 1))
    torch.npu.set_device(device_id)

    x = torch.arange(24, dtype=torch.float32, device=f"npu:{device_id}").reshape(2, 3, 4)
    kernel = create_transpose_kernel(x.shape, dim0=-1, dim1=-2)

    out = kernel(x)
    torch.npu.synchronize()

    expected = x.transpose(-1, -2)
    assert_allclose(np.array(out.cpu()), np.array(expected.cpu()), rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    test_view()
    test_assemble()
    test_gather()
    test_concat()
    test_transpose()