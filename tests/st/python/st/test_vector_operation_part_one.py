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
import os
import pto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose


def test_vector_operation_add():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.DeviceInit()
    a = pto.tensor(dtype, shape, "ADD_TENSOR_a")
    b = pto.tensor(dtype, shape, "ADD_TENSOR_b")
    c = pto.tensor(dtype, shape, "ADD_TENSOR_c")

    with pto.dyn_function("ADD", [a, b], [c]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_ADD_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_ADD_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, view_shape, [min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                          pto.symbolic_scalar(n)), min(pto.symbolic_scalar(m) - b_idx * view_shape[1],
                                          pto.symbolic_scalar(m))], [b_idx * view_shape[0], s_idx * view_shape[1]])
                        tile_b = pto.view(b, view_shape, [min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                          pto.symbolic_scalar(n)), min(pto.symbolic_scalar(m) - b_idx * view_shape[1],
                                          pto.symbolic_scalar(m))], [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.add(tile_a, tile_b))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                        del tile_a
    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    b_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = b_tensor.flatten().tolist()
    c_data = list([0] * n * m)

    pto.DeviceRunOnceDataFromHost([a_data, b_data], [c_data])

    assert c_data == torch.add(torch.tensor(a_tensor), torch.tensor(b_tensor)).flatten().tolist()
    pto.DeviceFini()


def test_vector_operation_div():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.DeviceInit()
    a = pto.tensor(dtype, shape, "DIV_TENSOR_a")
    b = pto.tensor(dtype, shape, "DIV_TENSOR_b")
    c = pto.tensor(dtype, shape, "DIV_TENSOR_c")

    with pto.dyn_function("DIV", [a, b], [c]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_DIV_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_DIV_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, view_shape, [min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                          pto.symbolic_scalar(n)), min(pto.symbolic_scalar(m) - b_idx * view_shape[1],
                                          pto.symbolic_scalar(m))], [b_idx * view_shape[0], s_idx * view_shape[1]])
                        tile_b = pto.view(b, view_shape, [min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                          pto.symbolic_scalar(n)), min(pto.symbolic_scalar(m) - b_idx * view_shape[1],
                                          pto.symbolic_scalar(m))], [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.div(tile_a, tile_b))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                        del tile_a
    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    b_tensor = np.random.uniform(1, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = b_tensor.flatten().tolist()
    c_data = list([0] * n * m)

    pto.DeviceRunOnceDataFromHost([a_data, b_data], [c_data])

    assert_allclose(np.array(c_data),
                    np.array(torch.div(torch.tensor(a_tensor), torch.tensor(b_tensor)).flatten().tolist()),
                    rtol=1e-3, atol=1e-3)
    pto.DeviceFini()


def test_vector_operation_mul():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.DeviceInit()
    a = pto.tensor(dtype, shape, "MUL_TENSOR_a")
    b = pto.tensor(dtype, shape, "MUL_TENSOR_b")
    c = pto.tensor(dtype, shape, "MUL_TENSOR_c")

    with pto.dyn_function("MUL", [a, b], [c]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_MUL_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_MUL_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, view_shape, [min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                          pto.symbolic_scalar(n)), min(pto.symbolic_scalar(m) - b_idx * view_shape[1],
                                          pto.symbolic_scalar(m))], [b_idx * view_shape[0], s_idx * view_shape[1]])
                        tile_b = pto.view(b, view_shape, [min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                          pto.symbolic_scalar(n)), min(pto.symbolic_scalar(m) - b_idx * view_shape[1],
                                          pto.symbolic_scalar(m))], [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.mul(tile_a, tile_b))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                        del tile_a
    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    b_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = b_tensor.flatten().tolist()
    c_data = list([0] * n * m)

    pto.DeviceRunOnceDataFromHost([a_data, b_data], [c_data])

    assert c_data == torch.mul(torch.tensor(a_tensor), torch.tensor(b_tensor)).flatten().tolist()
    pto.DeviceFini()


def test_vector_operation_sub():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.DeviceInit()
    a = pto.tensor(dtype, shape, "SUB_TENSOR_a")
    b = pto.tensor(dtype, shape, "SUB_TENSOR_b")
    c = pto.tensor(dtype, shape, "SUB_TENSOR_c")

    with pto.dyn_function("SUB", [a, b], [c]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_SUB_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_SUB_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, view_shape, [min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                          pto.symbolic_scalar(n)), min(pto.symbolic_scalar(m) - b_idx * view_shape[1],
                                          pto.symbolic_scalar(m))], [b_idx * view_shape[0], s_idx * view_shape[1]])
                        tile_b = pto.view(b, view_shape, [min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                          pto.symbolic_scalar(n)), min(pto.symbolic_scalar(m) - b_idx * view_shape[1],
                                          pto.symbolic_scalar(m))], [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.sub(tile_a, tile_b))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                        del tile_a
    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    b_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = b_tensor.flatten().tolist()
    c_data = list([0] * n * m)

    pto.DeviceRunOnceDataFromHost([a_data, b_data], [c_data])

    assert c_data == torch.sub(torch.tensor(a_tensor), torch.tensor(b_tensor)).flatten().tolist()
    pto.DeviceFini()


def test_vector_operation_abs():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.DeviceInit()
    a = pto.tensor(dtype, shape, "ABS_TENSOR_a")
    b = pto.tensor(dtype, shape, "ABS_TENSOR_b")

    with pto.dyn_function("ABS", [a], [b]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_ABS_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_ABS_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, view_shape, [min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                          pto.symbolic_scalar(n)), min(pto.symbolic_scalar(m) - b_idx * view_shape[1],
                                          pto.symbolic_scalar(m))], [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.abs(tile_a))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                        del tile_a
    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = list([0] * n * m)

    pto.DeviceRunOnceDataFromHost([a_data], [b_data])

    assert b_data == torch.abs(torch.tensor(a_tensor)).flatten().tolist()
    pto.DeviceFini()


def test_vector_operation_sqrt():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.DeviceInit()
    a = pto.tensor(pto.data_type.DT_FP32, shape, "SQRT_TENSOR_a")
    b = pto.tensor(pto.data_type.DT_FP32, shape, "SQRT_TENSOR_b")

    with pto.dyn_function("SQRT", [a], [b]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_SQRT_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_SQRT_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, view_shape, [min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                          pto.symbolic_scalar(n)), min(pto.symbolic_scalar(m) - b_idx * view_shape[1],
                                          pto.symbolic_scalar(m))], [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.sqrt(tile_a))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                        del tile_a
    a_tensor = np.random.uniform(0, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = list([0] * n * m)
    pto.DeviceRunOnceDataFromHost([a_data], [b_data])
    assert_allclose(np.array(b_data),
                    np.array(torch.sqrt(torch.tensor(a_tensor)).flatten().tolist()), rtol=1e-3, atol=1e-3)
    pto.DeviceFini()


def test_vector_operation_neg():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)

    pto.DeviceInit()

    a = pto.tensor(pto.data_type.DT_FP32, (n, m), "NEG_TENSOR_a")
    b = pto.tensor(pto.data_type.DT_FP32, (n, m), "NEG_TENSOR_b")

    with pto.dyn_function("NEG", [a], [b]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_NEG_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_NEG_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, view_shape, [min(pto.symbolic_scalar(n) - b_idx * view_shape[0],
                                          pto.symbolic_scalar(n)), min(pto.symbolic_scalar(m) - b_idx * view_shape[1],
                                          pto.symbolic_scalar(m))], [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.neg(tile_a))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                        del tile_a
    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = list([0] * n * m)

    pto.DeviceRunOnceDataFromHost([a_data], [b_data])

    assert b_data == torch.negative(torch.tensor(a_tensor)).flatten().tolist()
    pto.DeviceFini()