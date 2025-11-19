#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import os
from typing import Union, Tuple, List, Callable
import pytest
import torch
import torch_npu
import pto
import torch.nn.functional as F


class Shape:
    def __init__(self, *dims):
        self.dims = tuple(dims)

    def __getitem__(self, idx):
        return self.dims[idx]

    def __len__(self):
        return len(self.dims)

    def __iter__(self):
        return iter(self.dims)

    def __repr__(self):
        return f"Shape{self.dims}"

    def __mul__(self, other):
        """Element-wise multiplication"""
        return self._apply_binary_op(other, lambda a, b: a * b)

    def __add__(self, other):
        """Element-wise addition"""
        return self._apply_binary_op(other, lambda a, b: a + b)

    def __sub__(self, other):
        """Element-wise subtraction"""
        return self._apply_binary_op(other, lambda a, b: a - b)

    def __floordiv__(self, other):
        """Element-wise floor division"""
        return self._apply_binary_op(other, lambda a, b: a // b)

    @property
    def rank(self):
        """Number of dimensions"""
        return len(self.dims)

    @staticmethod
    def min(shape0, shape1):
        """Element-wise min"""
        assert isinstance(shape1, Shape)
        assert shape0.rank == shape1.rank
        dims0 = shape0._symbolize_dims(shape1)
        return Shape(*[
            (pto.min(a, b) if isinstance(a, pto.symbolic_scalar) else min(a, b))
            for a, b in zip(dims0, shape1.dims)
        ])

    def _symbolize_dims(self, other):
        """Convert dimensions to pto symbolic scalars when needed."""
        if isinstance(other, Shape):
            assert self.rank == other.rank
            return [
                (pto.symbolic_scalar(a) if isinstance(b, pto.symbolic_scalar) else a)
                for a, b in zip(self.dims, other.dims)
            ]
        elif isinstance(other, pto.symbolic_scalar):
            return [pto.symbolic_scalar(d) for d in self.dims]
        elif isinstance(other, int):
            return self.dims
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

    def _apply_binary_op(self, other, op: Callable):
        """Apply binary operation element-wise."""
        dims = self._symbolize_dims(other)
        if isinstance(other, Shape):
            assert self.rank == other.rank
            return Shape(*[op(a, b) for a, b in zip(dims, other.dims)])
        elif isinstance(other, (int,)):
            return Shape(*[op(d, other) for d in dims])
        else:
            return NotImplemented


class TilePartitioner:
    def __init__(self, shape: Union[Shape, Tuple, List], tile_shape: Union[Shape, Tuple, List]):
        # Convert to Shape objects if needed
        if isinstance(shape, (tuple, list)):
            shape = Shape(*shape)
        if isinstance(tile_shape, (tuple, list)):
            tile_shape = Shape(*tile_shape)

        assert shape.rank == tile_shape.rank

        self.shape = shape
        self.tile_shape = tile_shape
        # Calculate grid shape: ceil(shape / tile_shape)
        self.grid_shape = (shape + tile_shape - 1) // tile_shape

    def __repr__(self):
        return f"TilePartitioner(shape={self.shape}, tile_shape={self.tile_shape}, grid_shape={self.grid_shape})"

    def grid_dims(self):
        return list(self.grid_shape)

    def get_tile_shape(self, tile_coord: Union[Tuple, List]) -> Shape:
        assert len(tile_coord) == self.shape.rank
        # assert tile_coord < self.grid_shape
        remaining = self.shape - (Shape(*tile_coord) * self.tile_shape)
        return Shape.min(self.tile_shape, remaining)

    def get_tile_offset(self, tile_coord: Union[Tuple, List]) -> Shape:
        assert len(tile_coord) == self.shape.rank
        return Shape(*tile_coord) * self.tile_shape


def make_tile_partitioner(shape, tile_shape):
    return TilePartitioner(shape, tile_shape)


def golden_one_hot(input_tensor: torch.Tensor, num_classes: int):
    original_dtype = input_tensor.dtype
    input_long = input_tensor.to(torch.long)
    output = F.one_hot(input_long, num_classes)
    return output.to(original_dtype)


def pto_nested_loop(ranges: List, names: List[str], idx_names: List[str]):
    if not ranges:
        yield ()
        return

    for i in pto.loop(ranges[0], name=names[0], idx_name=idx_names[0]):
        for rest in pto_nested_loop(ranges[1:], names[1:], idx_names[1:]):
            yield (i,) + rest


pto_dtype = pto.DT_INT64
torch_dtype = torch.int64


def generate_names(prefixes: List[str], rank: int) -> Tuple[List[str], List[str]]:
    loop_names = [f"{prefix}0" for prefix in prefixes[:rank]]
    index_names = [f"{prefix}idx" for prefix in prefixes[:rank]]
    return loop_names, index_names


def pto_one_hot(src_tensor: torch.Tensor,
                num_classes: int,
                view_shape: Tuple[int, ...],
                tile_shape: Tuple[int, ...]) -> torch.Tensor:
    src_shape = tuple(src_tensor.shape)
    dst_shape = src_shape + (num_classes, )

    src_view_shape = view_shape[:-1]  # Exclude class dimension
    dst_view_shape = view_shape

    src_partitioner = make_tile_partitioner(src_shape, src_view_shape)
    dst_partitioner = make_tile_partitioner(dst_shape, dst_view_shape)

    src_pto_tensor = pto.tensor(src_shape, pto_dtype, "PTO_TENSOR_SRC")
    dst_pto_tensor = pto.tensor(dst_shape, pto_dtype, "PTO_TENSOR_DST")

    assert len(dst_shape) <= 4, "Currently only support rank <= 4"
    loop_names, loop_idx_names = generate_names(["b", "s", "n", "d"], len(dst_shape))

    pto.set_codegen_options(support_dynamic_unaligned=True)
    with pto.function("MAIN", [src_pto_tensor], [dst_pto_tensor]):
        for dst_coord in pto_nested_loop(dst_partitioner.grid_dims(), loop_names, loop_idx_names):
            src_coord = dst_coord[:-1]
            src_view_tensor = pto.view(src_pto_tensor,
                                       src_view_shape,
                                       list(src_partitioner.get_tile_offset(src_coord)),
                                       valid_shape=list(src_partitioner.get_tile_shape(src_coord)))

            dst_view_tensor = pto.tensor(dst_view_shape, pto_dtype, "PTO_TENSOR_TMP")

            pto.set_vec_tile_shapes(*tile_shape)
            dst_view_tensor.move(pto.one_hot(src_view_tensor, num_classes))
            pto.assemble(dst_view_tensor, dst_partitioner.get_tile_offset(dst_coord), dst_pto_tensor)

            del src_view_tensor, dst_view_tensor

    assert isinstance(dst_pto_tensor, pto.tensor)

    dst_tensor = torch.zeros(dst_shape, dtype=torch_dtype)
    pto.runtime._device_run_once_data_from_host([src_tensor], [dst_tensor])
    return dst_tensor


def one_hot_onboard(src_shape, view_shape, tile_shape, num_classes: int):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    src_tensor = torch.randint(0, num_classes, src_shape, dtype=torch_dtype)

    golden_dst = golden_one_hot(src_tensor, num_classes)

    pto.runtime._device_init()
    try:
        pto_dst = pto_one_hot(src_tensor, num_classes, view_shape, tile_shape)
        assert torch.equal(golden_dst, pto_dst)
    finally:
        pto.runtime._device_fini()


def test_one_hot_onboard():
    num_classes = 10
    one_hot_onboard(src_shape=(64,),
                    view_shape=(16, num_classes),
                    tile_shape=(8, num_classes),
                    num_classes=num_classes)
    one_hot_onboard(src_shape=(50, 70),
                    view_shape=(10, 10, num_classes),
                    tile_shape=(5, 5, num_classes),
                    num_classes=num_classes)
    one_hot_onboard(src_shape=(20, 30, 40),
                    view_shape=(5, 10, 10, num_classes),
                    tile_shape=(2, 2, 5, num_classes),
                    num_classes=num_classes)