#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import pto


def test_tile_shape_reset():
    tile_shape = pto.tile_shape()
    tile_shape.reset()

    assert tile_shape is not None


def test_tile_shape_dump():
    tile_shape = pto.tile_shape()
    tile_shape.to_string(pto.tile_type.VEC)
    tile_shape.to_string()

    assert tile_shape is not None


def test_tile_shape_specify_static_rank_id():
    tile_shape = pto.tile_shape()
    rank_id = 1
    tile_shape.set_dist_rank_id(rank_id)

    assert tile_shape is not None


def test_tile_shape_set_vec_tiles_shape_2d():
    tile_shape = pto.tile_shape()
    expected = (8, 16)
    tile_shape.set_vec_tile_shapes(*expected)
    actual = tile_shape.get_vec_tile_shapes()
    assert tuple(actual) == expected


def test_tile_shape_set_vec_tiles_shape_3d():
    tile_shape = pto.tile_shape()
    expected = (1, 2, 3)
    tile_shape.set_vec_tile_shapes(*expected)
    actual = tile_shape.get_vec_tile_shapes()
    assert tuple(actual) == expected
