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


def test_tile_shape_set_vec_tiles_shape_2d():
    expected = (8, 16)
    pto.set_vec_tile_shapes(*expected)
    actual = pto.get_vec_tile_shapes()
    assert tuple(actual) == expected


def test_tile_shape_set_vec_tiles_shape_3d():
    expected = (1, 2, 3)
    pto.set_vec_tile_shapes(*expected)
    actual = pto.get_vec_tile_shapes()
    assert tuple(actual) == expected


def test_tile_shape_set_vec_tiles_shape_4d():
    expected = (1, 2, 3, 8)
    pto.set_vec_tile_shapes(*expected)
    actual = pto.get_vec_tile_shapes()
    assert tuple(actual) == expected


def test_cube_tile_shapes():
    expected = ([16, 16], [256, 512, 512], [128, 128], True)
    pto.set_cube_tile_shapes(*expected[:3], expected[3])
    actual = pto.get_cube_tile_shapes()
    assert actual == expected


def test_cube_tile_shapes_l1():
    expected = ([16, 16], [256, 512, 512], [128, 128], False)
    pto.set_cube_tile_shapes(*expected[:3], expected[3])
    actual = pto.get_cube_tile_shapes()
    assert actual == expected
