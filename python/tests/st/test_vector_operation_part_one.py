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
import pto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose
import torch_npu


def test_vector_operation_add():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.runtime._device_init()
    a = pto.tensor(shape, dtype, "ADD_TENSOR_a")
    b = pto.tensor(shape, dtype, "ADD_TENSOR_b")
    c = pto.tensor(shape, dtype, "ADD_TENSOR_c")

    with pto.function("ADD", [a, b], [c]):
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_ADD_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ADD_L1", idx_name="s_idx"):
                tile_a = pto.view(a, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                tile_b = pto.view(b, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pto.add(tile_a, tile_b))
                pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                del tile_a, tile_b
    a_tensor = torch.rand(n, m, dtype=torch.float32) * 100
    b_tensor = torch.rand(n, m, dtype=torch.float32) * 100
    c_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor, b_tensor], [c_tensor])

    expected = a_tensor + b_tensor
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)


def test_vector_operation_div():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.runtime._device_init()
    a = pto.tensor(shape, dtype, "DIV_TENSOR_a")
    b = pto.tensor(shape, dtype, "DIV_TENSOR_b")
    c = pto.tensor(shape, dtype, "DIV_TENSOR_c")

    with pto.function("DIV", [a, b], [c]):
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_DIV_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_DIV_L1", idx_name="s_idx"):
                tile_a = pto.view(a, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                tile_b = pto.view(b, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pto.div(tile_a, tile_b))
                pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                del tile_a, tile_b
    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    b_tensor = torch.rand(n, m, dtype=torch.float32) * 99 + 1
    c_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor, b_tensor], [c_tensor])

    expected = torch.div(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_mul():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.runtime._device_init()
    a = pto.tensor(shape, dtype, "MUL_TENSOR_a")
    b = pto.tensor(shape, dtype, "MUL_TENSOR_b")
    c = pto.tensor(shape, dtype, "MUL_TENSOR_c")

    with pto.function("MUL", [a, b], [c]):
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_MUL_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_MUL_L1", idx_name="s_idx"):
                tile_a = pto.view(a, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                tile_b = pto.view(b, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pto.mul(tile_a, tile_b))
                pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                del tile_a, tile_b
    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    b_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    c_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor, b_tensor], [c_tensor])

    expected = torch.mul(a_tensor, b_tensor)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_sub():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.runtime._device_init()
    a = pto.tensor(shape, dtype, "SUB_TENSOR_a")
    b = pto.tensor(shape, dtype, "SUB_TENSOR_b")
    c = pto.tensor(shape, dtype, "SUB_TENSOR_c")

    with pto.function("SUB", [a, b], [c]):
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_SUB_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_SUB_L1", idx_name="s_idx"):
                tile_a = pto.view(a, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                tile_b = pto.view(b, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pto.sub(tile_a, tile_b))
                pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], c)
                del tile_a, tile_b
    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    b_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    c_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor, b_tensor], [c_tensor])

    expected = a_tensor - b_tensor
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_abs():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.runtime._device_init()
    a = pto.tensor(shape, dtype, "ABS_TENSOR_a")
    b = pto.tensor(shape, dtype, "ABS_TENSOR_b")

    with pto.function("ABS", [a], [b]):
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_ABS_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ABS_L1", idx_name="s_idx"):
                tile_a = pto.view(a, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pto.abs(tile_a))
                pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                del tile_a
    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    expected = torch.abs(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_sqrt():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.runtime._device_init()
    a = pto.tensor(shape, dtype, "SQRT_TENSOR_a")
    b = pto.tensor(shape, dtype, "SQRT_TENSOR_b")

    with pto.function("SQRT", [a], [b]):
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_SQRT_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_SQRT_L1", idx_name="s_idx"):
                tile_a = pto.view(a, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pto.sqrt(tile_a))
                pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                del tile_a
    a_tensor = torch.rand(n, m, dtype=torch.float32) * 100
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    expected = torch.sqrt(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_exp():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.runtime._device_init()
    a = pto.tensor(shape, dtype, "SQRT_TENSOR_a")
    b = pto.tensor(shape, dtype, "SQRT_TENSOR_b")

    with pto.function("EXP", [a], [b]):
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_SQRT_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_SQRT_L1", idx_name="s_idx"):
                tile_a = pto.view(a, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pto.exp(tile_a))
                pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                del tile_a
    a_tensor = torch.rand(n, m, dtype=torch.float32) * 100
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    expected = torch.exp(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_neg():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.runtime._device_init()

    a = pto.tensor((n, m), dtype, "NEG_TENSOR_a")
    b = pto.tensor((n, m), dtype, "NEG_TENSOR_b")

    with pto.function("NEG", [a], [b]):
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_NEG_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_NEG_L1", idx_name="s_idx"):
                tile_a = pto.view(a, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(pto.neg(tile_a))
                pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                del tile_a
    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    expected = -a_tensor
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_full():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.runtime._device_init()

    a = pto.tensor((n, m), dtype, "VEC_DUP_TENSOR_a")
    b = pto.element(dtype, 2.0)

    with pto.function("VEC_DUP", [], [a]):
        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_VEC_DUP_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_VEC_DUP_L1", idx_name="s_idx"):
                tile_a = pto.tensor()
                tile_a.move(pto.full(view_shape, b, dtype,
                valid_shape=[(pto.symbolic_scalar(n) -
                b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                (pto.symbolic_scalar(m) -
                s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))]))
                pto.assemble(
                    tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], a)
                del tile_a
    a_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([], [a_tensor])

    expected = torch.full((n, m), 2, dtype=torch.float32)
    assert_allclose(a_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_logical_not():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.runtime._device_init()
    pto.set_codegen_options(support_dynamic_unaligned=True)

    a = pto.tensor((n, m), pto.DT_FP32, "LOGICALNOT_TENSOR_a")
    b = pto.tensor((n, m), pto.DT_BOOL, "LOGICALNOT_TENSOR_b")

    with pto.function("LOGICALNOT", [a], [b]):
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_LOGICALNOT_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_LOGICALNOT_L1", idx_name="s_idx"):
                tile_a = pto.view(a, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_a = pto.tensor(view_shape, pto.DT_BOOL)
                tmp_a.move(pto.logical_not(tile_a))
                pto.assemble(tmp_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                del tile_a, tmp_a
    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 6 - 1.5  # 生成 [-3, 3] 范围
    b_tensor = torch.ones(n, m, dtype=torch.bool)  # 使用 torch.bool 类型

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    expected = torch.logical_not(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_expand():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.runtime._device_init()

    a = pto.tensor((n, 1), dtype, "EXPAND_TENSOR_a")
    b = pto.tensor((n, m), dtype, "EXPAND_TENSOR_b")

    with pto.function("EXPAND", [a], [b]):
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EXPAND_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_EXPAND_L1", idx_name="s_idx"):
                tile_a = pto.view(a, [16, 1],
                                  [b_idx * view_shape[0], 0],
                                  valid_shape=[(pto.symbolic_scalar(n) -
                                                b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                                               1])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_a = pto.tensor()
                tmp_a.move(pto.expand_clone(tile_a, view_shape,
                                            valid_shape=[(pto.symbolic_scalar(n) - b_idx * view_shape[0]).
                                                         min(pto.symbolic_scalar(
                                                             view_shape[0])),
                                                         (pto.symbolic_scalar(m) - s_idx * view_shape[1]).
                                                         min(pto.symbolic_scalar(view_shape[1]))]))
                pto.assemble(
                    tmp_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                del tmp_a, tile_a
    a_tensor = torch.full((n, 1), -16, dtype=torch.float32)
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    expected = torch.full((n, m), -16, dtype=torch.float32)
    assert_allclose(b_tensor.flatten(), expected.flatten(),
                    rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_concat():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 32)
    tile_shape = (8, 8)
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.runtime._device_init()

    a = pto.tensor(shape, dtype, "CONCAT_TENSOR_a")
    b = pto.tensor(shape, dtype, "CONCAT_TENSOR_b")
    c = pto.tensor([n, m * 2], dtype, "CONCAT_TENSOR_c")

    with pto.function("CONCAT", [a, b], [c]):
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_CONCAT_L0", idx_name="b_idx"):
            tile_a = pto.view(a, view_shape,
                [b_idx * view_shape[0], 0],
                valid_shape=[(pto.symbolic_scalar(n) -
                b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                n])
            tile_b = pto.view(b, view_shape,
                [b_idx * view_shape[0], 0],
                valid_shape=[(pto.symbolic_scalar(n) -
                b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                n])
            pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            tmp_c = pto.tensor([16, 64], dtype)
            tmp_c.move(pto.concat([tile_a, tile_b], -1))
            pto.assemble(tmp_c, [b_idx * view_shape[0], 0], c)
            del tile_a, tile_b, tmp_c

    a_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    b_tensor = (torch.rand(n, m, dtype=torch.float32) - 0.5) * 200
    c_tensor = torch.zeros(n, 2 * m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor, b_tensor], [c_tensor])

    expected = torch.cat([a_tensor, b_tensor], dim=-1)
    assert_allclose(c_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_rowmaxsingle():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    output_shape = (1, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.runtime._device_init()
    pto.set_codegen_options(support_dynamic_unaligned=True)
    a = pto.tensor(shape, dtype, "ROWMAXSINGLE_TENSOR_a")
    b = pto.tensor(output_shape, dtype, "ROWMAXSINGLE_TENSOR_b")
    dim = 0

    with pto.function("ROWMAXSINGLE", [a], [b]):
        for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ROWMAXSINGLE_L1", idx_name="s_idx"):
            tile_a = pto.view(a, [32, view_shape[1]],
                [0, s_idx * view_shape[1]],
                valid_shape=[pto.symbolic_scalar(n),
                (pto.symbolic_scalar(m) -
                s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
            pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            tmp_a = pto.tensor([1, view_shape[1]], dtype)
            tmp_a.move(pto.amax(tile_a, dim, True))
            pto.assemble(tmp_a, [0, s_idx * view_shape[1]], b)
            del tile_a, tmp_a
    a_tensor = torch.rand(shape, dtype=torch.float32) * 100
    b_tensor = torch.zeros(output_shape, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    expected = a_tensor.max(dim=dim, keepdim=True)[0].reshape(output_shape)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_rowsumsingle():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    output_shape = (1, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.runtime._device_init()
    pto.set_codegen_options(support_dynamic_unaligned=True)
    a = pto.tensor(shape, dtype, "ROWSUMSINGLE_TENSOR_a")
    b = pto.tensor(output_shape, dtype, "ROWSUMSINGLE_TENSOR_b")
    dim = 0

    with pto.function("ROWSUMSINGLE", [a], [b]):
        for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ROWSUMSINGLE_L1", idx_name="s_idx"):
            tile_a = pto.view(a, [32, view_shape[1]],
                [0, s_idx * view_shape[1]],
                valid_shape=[pto.symbolic_scalar(n),
                (pto.symbolic_scalar(m) -
                s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
            pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            tmp_a = pto.tensor([1, view_shape[1]], dtype)
            tmp_a.move(pto.sum(tile_a, dim, True))
            pto.assemble(tmp_a, [0, s_idx * view_shape[1]], b)
            del tile_a, tmp_a
    a_tensor = torch.rand(shape, dtype=torch.float32) * 100
    b_tensor = torch.zeros(output_shape, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    expected = a_tensor.sum(dim=dim, keepdim=True).reshape(output_shape)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_vector_operation_rowminsingle():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    output_shape = (1, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.runtime._device_init()
    pto.set_codegen_options(support_dynamic_unaligned=True)
    a = pto.tensor(shape, dtype, "ROWMINSINGLE_TENSOR_a")
    b = pto.tensor(output_shape, dtype, "ROWMINSINGLE_TENSOR_b")
    dim = 0

    with pto.function("ROWMINSINGLE", [a], [b]):
        for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ROWMINSINGLE_L1", idx_name="s_idx"):
            tile_a = pto.view(a, [32, view_shape[1]],
                [0, s_idx * view_shape[1]],
                valid_shape=[pto.symbolic_scalar(n),
                (pto.symbolic_scalar(m) -
                s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
            pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
            tmp_a = pto.tensor([1, view_shape[1]], dtype)
            tmp_a.move(pto.amin(tile_a, dim, True))
            pto.assemble(tmp_a, [0, s_idx * view_shape[1]], b)
            del tile_a, tmp_a
    a_tensor = torch.rand(shape, dtype=torch.float32) * 100
    b_tensor = torch.zeros(output_shape, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    expected = a_tensor.min(dim=dim, keepdim=True)[0].reshape(output_shape)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_tensor_operation_expand():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.runtime._device_init()

    a = pto.tensor((n, 1), dtype, "EXPAND_TENSOR_a")
    b = pto.tensor((n, m), dtype, "EXPAND_TENSOR_b")

    with pto.function("EXPAND", [a], [b]):
        for b_idx in pto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_EXPAND_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_EXPAND_L1", idx_name="s_idx"):
                tile_a = pto.view(a, [16, 1],
                                  [b_idx * view_shape[0], 0],
                                  valid_shape=[(pto.symbolic_scalar(n) -
                                                b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                                               1])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_a = pto.tensor()
                tmp_a.move(tile_a.expand_clone(view_shape,
                                               valid_shape=[(pto.symbolic_scalar(n) - b_idx * view_shape[0]).
                                                            min(pto.symbolic_scalar(
                                                                view_shape[0])),
                                                            (pto.symbolic_scalar(m) - s_idx * view_shape[1]).
                                                            min(pto.symbolic_scalar(view_shape[1]))]))
                pto.assemble(
                    tmp_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                del tmp_a, tile_a
    a_tensor = torch.full((n, 1), -16, dtype=torch.float32)
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    expected = torch.full((n, m), -16, dtype=torch.float32)
    assert_allclose(b_tensor.flatten(), expected.flatten(),
                    rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()
