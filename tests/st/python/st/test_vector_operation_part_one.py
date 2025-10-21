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
    pto.set_codegen_config("support_dynamic_unaligned", True)
    pto.device_init()
    a = pto.tensor(shape, dtype, "ADD_TENSOR_a")
    b = pto.tensor(shape, dtype, "ADD_TENSOR_b")
    c = pto.tensor(shape, dtype, "ADD_TENSOR_c")

    with pto.dyn_function("ADD", [a, b], [c]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_ADD_L0", "b_idx", loop_range_b) as bloop_add:
            with pto.loop_function("LOOP_ADD_L1", "s_idx", loop_range_s) as sloop_add:
                for b_idx in bloop_add:
                    for s_idx in sloop_add:
                        tile_a = pto.view(a, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [b_idx * view_shape[0], s_idx * view_shape[1]])
                        tile_b = pto.view(b, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.add(tile_a, tile_b))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                        del tile_a, tile_b
    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    b_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = b_tensor.flatten().tolist()
    c_data = list([0] * n * m)

    pto.device_run_once_data_from_host([a_data, b_data], [c_data])

    assert c_data == torch.add(torch.tensor(a_tensor), torch.tensor(b_tensor)).flatten().tolist()
    pto.device_fini()


def test_vector_operation_div():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_config("support_dynamic_unaligned", True)
    pto.device_init()
    a = pto.tensor(shape, dtype, "DIV_TENSOR_a")
    b = pto.tensor(shape, dtype, "DIV_TENSOR_b")
    c = pto.tensor(shape, dtype, "DIV_TENSOR_c")

    with pto.dyn_function("DIV", [a, b], [c]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_DIV_L0", "b_idx", loop_range_b) as bloop_div:
            with pto.loop_function("LOOP_DIV_L1", "s_idx", loop_range_s) as sloop_div:
                for b_idx in bloop_div:
                    for s_idx in sloop_div:
                        tile_a = pto.view(a, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [b_idx * view_shape[0], s_idx * view_shape[1]])
                        tile_b = pto.view(b, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.div(tile_a, tile_b))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                        del tile_a, tile_b
    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    b_tensor = np.random.uniform(1, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = b_tensor.flatten().tolist()
    c_data = list([0] * n * m)

    pto.device_run_once_data_from_host([a_data, b_data], [c_data])

    assert_allclose(np.array(c_data),
                    np.array(torch.div(torch.tensor(a_tensor), torch.tensor(b_tensor)).flatten().tolist()),
                    rtol=1e-3, atol=1e-3)
    pto.device_fini()


def test_vector_operation_mul():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_config("support_dynamic_unaligned", True)
    pto.device_init()
    a = pto.tensor(shape, dtype, "MUL_TENSOR_a")
    b = pto.tensor(shape, dtype, "MUL_TENSOR_b")
    c = pto.tensor(shape, dtype, "MUL_TENSOR_c")

    with pto.dyn_function("MUL", [a, b], [c]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_MUL_L0", "b_idx", loop_range_b) as bloop_mul:
            with pto.loop_function("LOOP_MUL_L1", "s_idx", loop_range_s) as sloop_mul:
                for b_idx in bloop_mul:
                    for s_idx in sloop_mul:
                        tile_a = pto.view(a, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [b_idx * view_shape[0], s_idx * view_shape[1]])
                        tile_b = pto.view(b, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.mul(tile_a, tile_b))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                        del tile_a, tile_b
    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    b_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = b_tensor.flatten().tolist()
    c_data = list([0] * n * m)

    pto.device_run_once_data_from_host([a_data, b_data], [c_data])

    assert c_data == torch.mul(torch.tensor(a_tensor), torch.tensor(b_tensor)).flatten().tolist()
    pto.device_fini()


def test_vector_operation_sub():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_config("support_dynamic_unaligned", True)
    pto.device_init()
    a = pto.tensor(shape, dtype, "SUB_TENSOR_a")
    b = pto.tensor(shape, dtype, "SUB_TENSOR_b")
    c = pto.tensor(shape, dtype, "SUB_TENSOR_c")

    with pto.dyn_function("SUB", [a, b], [c]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_SUB_L0", "b_idx", loop_range_b) as bloop_sub:
            with pto.loop_function("LOOP_SUB_L1", "s_idx", loop_range_s) as sloop_sub:
                for b_idx in bloop_sub:
                    for s_idx in sloop_sub:
                        tile_a = pto.view(a, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [b_idx * view_shape[0], s_idx * view_shape[1]])
                        tile_b = pto.view(b, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.sub(tile_a, tile_b))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                        del tile_a, tile_b
    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    b_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = b_tensor.flatten().tolist()
    c_data = list([0] * n * m)

    pto.device_run_once_data_from_host([a_data, b_data], [c_data])

    assert c_data == torch.sub(torch.tensor(a_tensor), torch.tensor(b_tensor)).flatten().tolist()
    pto.device_fini()


def test_vector_operation_abs():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_config("support_dynamic_unaligned", True)
    pto.device_init()
    a = pto.tensor(shape, dtype, "ABS_TENSOR_a")
    b = pto.tensor(shape, dtype, "ABS_TENSOR_b")

    with pto.dyn_function("ABS", [a], [b]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_ABS_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_ABS_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.abs(tile_a))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                        del tile_a
    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = list([0] * n * m)

    pto.device_run_once_data_from_host([a_data], [b_data])

    assert b_data == torch.abs(torch.tensor(a_tensor)).flatten().tolist()
    pto.device_fini()


def test_vector_operation_sqrt():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_config("support_dynamic_unaligned", True)
    pto.device_init()
    a = pto.tensor(shape, dtype, "SQRT_TENSOR_a")
    b = pto.tensor(shape, dtype, "SQRT_TENSOR_b")

    with pto.dyn_function("SQRT", [a], [b]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_SQRT_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_SQRT_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.sqrt(tile_a))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                        del tile_a
    a_tensor = np.random.uniform(0, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = list([0] * n * m)
    pto.device_run_once_data_from_host([a_data], [b_data])
    assert_allclose(np.array(b_data),
                    np.array(torch.sqrt(torch.tensor(a_tensor)).flatten().tolist()), rtol=1e-3, atol=1e-3)
    pto.device_fini()


def test_vector_operation_neg():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_config("support_dynamic_unaligned", True)
    pto.device_init()

    a = pto.tensor((n, m), dtype, "NEG_TENSOR_a")
    b = pto.tensor((n, m), dtype, "NEG_TENSOR_b")

    with pto.dyn_function("NEG", [a], [b]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_NEG_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_NEG_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tile_a.move(pto.neg(tile_a))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                        del tile_a
    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = list([0] * n * m)

    pto.device_run_once_data_from_host([a_data], [b_data])

    assert b_data == torch.negative(torch.tensor(a_tensor)).flatten().tolist()
    pto.device_fini()


def test_vector_operation_vec_dup():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_config("support_dynamic_unaligned", True)
    pto.device_init()

    a = pto.tensor((n, m), dtype, "VEC_DUP_TENSOR_a")
    b = pto.element(dtype, 2.0)

    with pto.dyn_function("VEC_DUP", [], [a]):
        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_VEC_DUP_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_VEC_DUP_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.tensor()
                        tile_a.move(pto.vector_duplicate(b, dtype, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))]))
                        pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], a)
                        del tile_a
    a_data = list([0] * n * m)
    pto.device_run_once_data_from_host([], [a_data])
    assert a_data == list([2] * n * m)
    pto.device_fini()


def test_vector_operation_logical_not():
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.device_init()
    pto.set_codegen_config("support_dynamic_unaligned", True)

    a = pto.tensor((n, m), pto.data_type.DT_FP32, "LOGICALNOT_TENSOR_a")
    b = pto.tensor((n, m), pto.data_type.DT_BOOL, "LOGICALNOT_TENSOR_b")

    with pto.dyn_function("LOGICALNOT", [a], [b]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_LOGICALNOT_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_LOGICALNOT_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [b_idx * view_shape[0], s_idx * view_shape[1]])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tmp_a = pto.tensor()
                        tmp_a.move(pto.logical_not(tile_a))
                        pto.assemble(tmp_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                        del tile_a, tmp_a
    a_tensor = np.random.uniform(-3, 3, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = list([True] * n * m)

    pto.device_run_once_data_from_host([a_data], [b_data])

    assert b_data == torch.logical_not(torch.tensor(a_tensor)).flatten().tolist()
    pto.device_fini()


def test_vector_operation_expand():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_config("support_dynamic_unaligned", True)
    pto.device_init()

    a = pto.tensor((n, 1), dtype, "EXPAND_TENSOR_a")
    b = pto.tensor((n, m), dtype, "EXPAND_TENSOR_b")

    with pto.dyn_function("EXPAND", [a], [b]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_EXPAND_L0", "b_idx", loop_range_b) as bloop:
            with pto.loop_function("LOOP_EXPAND_L1", "s_idx", loop_range_s) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tile_a = pto.view(a, [16, 1],
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            1],
                            [b_idx * view_shape[0], 0])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tmp_a = pto.tensor()
                        tmp_a.move(pto.expand(tile_a, view_shape,
                            [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))]))
                        pto.assemble(tmp_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                        del tmp_a, tile_a
    a_data = list([-16] * n * 1)
    b_data = list([0] * n * m)

    pto.device_run_once_data_from_host([a_data], [b_data])

    assert b_data == list([-16] * n * m)
    pto.device_fini()


def test_vector_operation_concat():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 32)
    tile_shape = (8, 8)
    pto.set_codegen_config("support_dynamic_unaligned", True)
    pto.device_init()

    a = pto.tensor(shape, dtype, "CONCAT_TENSOR_a")
    b = pto.tensor(shape, dtype, "CONCAT_TENSOR_b")
    c = pto.tensor([n, m * 2], dtype, "CONCAT_TENSOR_c")

    with pto.dyn_function("CONCAT", [a, b], [c]):
        loop_range_b = pto.loop_range(int(np.ceil(n / view_shape[0])))
        with pto.loop_function("LOOP_CONCAT_L0", "b_idx", loop_range_b) as bloop:
            for b_idx in bloop:
                tile_a = pto.view(a, view_shape,
                    [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    n],
                    [b_idx * view_shape[0], 0])
                tile_b = pto.view(b, view_shape,
                    [(pto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    n],
                    [b_idx * view_shape[0], 0])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_c = pto.tensor()
                tmp_c.move(pto.concat([tile_a, tile_b], -1))
                pto.assemble(tmp_c, [b_idx * view_shape[0], 0], c)
                del tile_a, tile_b, tmp_c

    a_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_tensor = np.random.uniform(-100, 100, [n, m]).astype(np.float32)
    b_data = b_tensor.flatten().tolist()
    c_data = list([0] * 2 * n * m)
    pto.device_run_once_data_from_host([a_data, b_data], [c_data])
    assert c_data == torch.cat([torch.tensor(a_tensor), torch.tensor(b_tensor)], dim=-1).flatten().tolist()
    pto.device_fini()


def test_vector_operation_rowmaxsingle():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    output_shape = (1, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.device_init()
    pto.set_codegen_config("support_dynamic_unaligned", True)
    a = pto.tensor(shape, dtype, "ROWMAXSINGLE_TENSOR_a")
    b = pto.tensor(output_shape, dtype, "ROWMAXSINGLE_TENSOR_b")
    dim = 0

    with pto.dyn_function("ROWMAXSINGLE", [a], [b]):
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_ROWMAXSINGLE_L1", "s_idx", loop_range_s) as sloop:
            for s_idx in sloop:
                tile_a = pto.view(a, [32, view_shape[1]],
                    [pto.symbolic_scalar(n),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                    [0, s_idx * view_shape[1]])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_a = pto.tensor()
                tmp_a.move(pto.row_max_single(tile_a, dim))
                pto.assemble(tmp_a, [0, s_idx * view_shape[1]], b)
                del tile_a, tmp_a
    a_tensor = np.random.uniform(0, 100, shape).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = list([0] * output_shape[0] * output_shape[1])
    pto.device_run_once_data_from_host([a_data], [b_data])
    assert_allclose(np.array(b_data),
        np.array(a_tensor.max(axis=dim, keepdims=True).reshape(output_shape[0] * output_shape[1]).tolist()),
        rtol=1e-3, atol=1e-3)
    pto.device_fini()


def test_vector_operation_rowsumsingle():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    output_shape = (1, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.device_init()
    pto.set_codegen_config("support_dynamic_unaligned", True)
    a = pto.tensor(shape, dtype, "ROWSUMSINGLE_TENSOR_a")
    b = pto.tensor(output_shape, dtype, "ROWSUMSINGLE_TENSOR_b")
    dim = 0

    with pto.dyn_function("ROWSUMSINGLE", [a], [b]):
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_ROWSUMSINGLE_L1", "s_idx", loop_range_s) as sloop:
            for s_idx in sloop:
                tile_a = pto.view(a, [32, view_shape[1]],
                    [pto.symbolic_scalar(n),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                    [0, s_idx * view_shape[1]])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_a = pto.tensor()
                tmp_a.move(pto.row_sum_single(tile_a, dim))
                pto.assemble(tmp_a, [0, s_idx * view_shape[1]], b)
                del tile_a, tmp_a
    a_tensor = np.random.uniform(0, 100, shape).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = list([0] * output_shape[0] * output_shape[1])
    pto.device_run_once_data_from_host([a_data], [b_data])
    assert_allclose(np.array(b_data),
        np.array(a_tensor.sum(axis=dim, keepdims=True).reshape(output_shape[0] * output_shape[1]).tolist()),
        rtol=1e-3, atol=1e-3)
    pto.device_fini()


def test_vector_operation_rowminsingle():
    dtype = pto.data_type.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    output_shape = (1, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.device_init()
    pto.set_codegen_config("support_dynamic_unaligned", True)
    a = pto.tensor(shape, dtype, "ROWMINSINGLE_TENSOR_a")
    b = pto.tensor(output_shape, dtype, "ROWMINSINGLE_TENSOR_b")
    dim = 0

    with pto.dyn_function("ROWMINSINGLE", [a], [b]):
        loop_range_s = pto.loop_range(int(np.ceil(m / view_shape[1])))
        with pto.loop_function("LOOP_ROWMINSINGLE_L1", "s_idx", loop_range_s) as sloop:
            for s_idx in sloop:
                tile_a = pto.view(a, [32, view_shape[1]],
                    [pto.symbolic_scalar(n),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                    [0, s_idx * view_shape[1]])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_a = pto.tensor()
                tmp_a.move(pto.row_min_single(tile_a, dim))
                pto.assemble(tmp_a, [0, s_idx * view_shape[1]], b)
                del tile_a, tmp_a
    a_tensor = np.random.uniform(0, 100, shape).astype(np.float32)
    a_data = a_tensor.flatten().tolist()
    b_data = list([0] * output_shape[0] * output_shape[1])
    pto.device_run_once_data_from_host([a_data], [b_data])
    assert_allclose(np.array(b_data),
        np.array(a_tensor.min(axis=dim, keepdims=True).reshape(output_shape[0] * output_shape[1]).tolist()),
        rtol=1e-3, atol=1e-3)
    pto.device_fini()
