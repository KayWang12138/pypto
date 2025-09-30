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

import os
from dataclasses import dataclass

import numpy as np
import pto
from numpy.testing import assert_allclose

FP32 = np.float32
FP16 = np.float16
INT32 = np.int32
INT8 = np.int8


@dataclass
class ShapeConfig:
    ori_shape: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    view_shape: list
    in_dtype: np.dtype
    out_dtype: np.dtype
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False


def test_matmul_fp16_with_m_split():
    input_config = ShapeConfig([128, 256, 512], [64, 64], [128, 128], [128, 128], [96, -1], FP16, FP32,
                               True, False, False, False, True)
    dynamic_matmul_onboard_util(input_config)


def test_matmul_bf16_with_n_split():
    input_config = ShapeConfig([16, 32, 512], [128, 128], [128, 128], [128, 128], [-1, 100], FP16, FP32,
                               True, True, False, False, False)
    dynamic_matmul_onboard_util(input_config)


def test_matmul_bf16_nz_with_m_n_split():
    input_config = ShapeConfig([32, 64, 64], [64, 128], [64, 128], [64, 128], [96, 96], FP16, FP32, True, True,
                               True, False, True)
    dynamic_matmul_onboard_util(input_config)


def test_matmul_bf16_nd_with_no_split():
    input_config = ShapeConfig([127, 255, 511], [128, 128], [128, 128], [128, 128], [-1, -1], FP16, FP32, True, True,
                               False, False, False)
    dynamic_matmul_onboard_util(input_config)


def dynamic_matmul_onboard_util(input_config: ShapeConfig):
    # onboard prepare
    device_id = os.environ.get("TILE_FWK_STEST_DEVICE_ID", 0)
    pto.DeviceInit()
    pto.set_codegen_config("SUPPORT_DYNAMIC_UNALIGNED", True)
    pto.set_host_config("ONLY_CODEGEN", True)
    pto.set_cube_tile_shapes(input_config.m_tile_shape, input_config.k_tile_shape, input_config.n_tile_shape)

    tensor_a = create_tensor(input_config.in_dtype, input_config.ori_shape, "tensor_a", input_config.a_format_nz,
                             input_config.a_trans)
    tensor_b = create_tensor(input_config.in_dtype, input_config.ori_shape, "tensor_b", input_config.b_format_nz,
                             input_config.b_trans)
    tensor_c = create_tensor(input_config.out_dtype, input_config.ori_shape, "tensor_c", input_config.c_format_nz)

    view_m = input_config.view_shape[0]
    view_n = input_config.view_shape[1]
    if view_m > 0 and view_n > 0:
        split_m_n_axis(tensor_a, tensor_b, tensor_c, input_config)
    elif view_m > 0:
        split_m_axis(tensor_a, tensor_b, tensor_c, input_config)
    elif view_n > 0:
        split_n_axis(tensor_a, tensor_b, tensor_c, input_config)
    else:
        no_split_m_n(tensor_a, tensor_b, tensor_c, input_config)

    # gen golden
    a_data, b_data, c_data, c_device_data = gen_matmul_golden_data(input_config)

    # onboard execute
    pto.DeviceRunOnceDataFromHost([a_data, b_data], [c_device_data])

    # compare golden with onboard data
    assert_allclose(c_data, c_device_data, rtol=0.001, atol=0.001)

    # onboard finish--clean env
    pto.DeviceFini()


def no_split_m_n(tensor_a, tensor_b, tensor_c, input_config):
    shape_a = tensor_a.get_shape()
    shape_b = tensor_b.get_shape()
    valid_shape_a = [shape_a[0], shape_a[1]]
    valid_shape_b = [shape_b[0], shape_b[1]]
    dtype = convert_np_dtype_to_pto_dtype(input_config.out_dtype)
    with pto.dyn_function("test_no_split", [tensor_a, tensor_b], [tensor_c]):
        with pto.loop_function("loop", "idx", pto.loop_range(1)) as idx_loop:
            for idx in idx_loop:
                dyn_a = pto.view(tensor_a, shape_a, valid_shape_a, [idx, 0])
                dyn_b = pto.view(tensor_b, shape_b, valid_shape_b, [0, 0])
                tensor_c.move(pto.matmul(dtype, dyn_a, dyn_b, input_config.a_trans, input_config.b_trans,
                                input_config.c_format_nz))
                del dyn_a
                del dyn_b


def split_m_axis(tensor_a, tensor_b, tensor_c, input_config):
    shape_a = tensor_a.get_shape()
    view_shape = input_config.view_shape
    a_trans = input_config.a_trans

    m_axis = shape_a[1] if a_trans else shape_a[0]
    loop_end = ceil_div_util(m_axis, view_shape[0])

    dtype = convert_np_dtype_to_pto_dtype(input_config.out_dtype)
    with pto.dyn_function("test_m_split", [tensor_a, tensor_b], [tensor_c]):
        with pto.loop_function("m_loop", "m_idx", pto.loop_range(0, loop_end, 1)) as m_idx_loop:
            for m_idx in m_idx_loop:
                matmul_split_m_utils(tensor_a, tensor_b, tensor_c, input_config, m_idx)


def matmul_split_m_utils(tensor_a, tensor_b, tensor_c, input_config, m_idx):
    shape_a = tensor_a.get_shape()
    shape_b = tensor_b.get_shape()
    view_shape = input_config.view_shape
    a_trans = input_config.a_trans
    dtype = convert_np_dtype_to_pto_dtype(input_config.out_dtype)
    if a_trans:
        dyn_a = pto.view(tensor_a, [shape_a[0], view_shape[0]],
                    [shape_a[0], (shape_a[1] - m_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0]))],
                    [0, m_idx * view_shape[0]])
    else:
        dyn_a = pto.view(tensor_a, [view_shape[0], shape_a[1]],
                    [(shape_a[0] - m_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])), shape_a[1]],
                    [0, m_idx * view_shape[0]])

    dyn_b = pto.view(tensor_b, shape_b, [shape_b[0], shape_b[1]], [0, 0])
    res = pto.matmul(dtype, dyn_a, dyn_b, input_config.a_trans, input_config.b_trans,
                                            input_config.c_format_nz)

    pto.assemble(res, [m_idx * view_shape[0], 0], tensor_c)
    del dyn_a
    del dyn_b
    del res


def split_n_axis(tensor_a, tensor_b, tensor_c, input_config):
    shape_b = tensor_b.get_shape()
    view_shape = input_config.view_shape
    b_trans = input_config.b_trans

    n_axis = shape_b[0] if b_trans else shape_b[1]
    loop_end = ceil_div_util(n_axis, view_shape[1])

    dtype = convert_np_dtype_to_pto_dtype(input_config.out_dtype)
    with pto.dyn_function("test_n_split", [tensor_a, tensor_b], [tensor_c]):
        with pto.loop_function("n_loop", "n_idx", pto.loop_range(0, loop_end, 1)) as n_idx_loop:
            for n_idx in n_idx_loop:
                matmul_split_n_utils(tensor_a, tensor_b, tensor_c, input_config, n_idx)



def matmul_split_n_utils(tensor_a, tensor_b, tensor_c, input_config, n_idx):
    shape_a = tensor_a.get_shape()
    shape_b = tensor_b.get_shape()
    view_shape = input_config.view_shape
    b_trans = input_config.b_trans
    dtype = convert_np_dtype_to_pto_dtype(input_config.out_dtype)
    dyn_a = pto.view(tensor_a, shape_a, [shape_a[0], shape_a[1]], [0, 0])
    if b_trans:
        dyn_b = pto.view(tensor_b, [view_shape[1], shape_b[1]],
        [(shape_b[0] - n_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1])), shape_b[1]],
        [n_idx * view_shape[1], 0])
    else:
        dyn_b = pto.view(tensor_b, [shape_b[0], view_shape[1]],
        [(shape_b[0], shape_b[1] - n_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
        [0, n_idx * view_shape[1]])

    res = pto.matmul(dtype, dyn_a, dyn_b, input_config.a_trans, input_config.b_trans,
                                input_config.c_format_nz)

    pto.assemble(res, [0, n_idx * view_shape[1]], tensor_c)
    del dyn_a
    del dyn_b
    del res



def split_m_n_axis(tensor_a, tensor_b, tensor_c, input_config):
    shape_a = tensor_a.get_shape()
    view_shape = input_config.view_shape
    a_trans = input_config.a_trans

    m_axis = shape_a[1] if a_trans else shape_a[0]
    m_loop_end = ceil_div_util(m_axis, view_shape[0])

    with pto.dyn_function("test_m_n_split", [tensor_a, tensor_b], [tensor_c]):
        with pto.loop_function("m_loop", "m_idx", pto.loop_range(0, m_loop_end, 1)) as m_idx_loop:
            for m_idx in m_idx_loop:
                matmul_split_m_n_util(tensor_a, tensor_b, tensor_c, input_config, m_idx)


def matmul_split_m_n_util(tensor_a, tensor_b, tensor_c, input_config, m_idx):
    shape_a = tensor_a.get_shape()
    shape_b = tensor_b.get_shape()
    view_shape = input_config.view_shape
    b_trans = input_config.b_trans
    a_trans = input_config.a_trans

    n_axis = shape_b[0] if b_trans else shape_b[1]
    n_loop_end = ceil_div_util(n_axis, view_shape[1])

    dtype = convert_np_dtype_to_pto_dtype(input_config.out_dtype)

    with pto.loop_function("n_loop", "n_idx", pto.loop_range(0, n_loop_end, 1)) as n_idx_loop:
        for n_idx in n_idx_loop:
            if a_trans:
                dyn_a = pto.view(tensor_a, [shape_a[0], view_shape[0]],
                            [shape_a[0], (shape_a[1] - m_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0]))],
                            [0, m_idx * view_shape[0]])
            else:
                dyn_a = pto.view(tensor_a, [view_shape[0], shape_a[1]],
                            [(shape_a[0] - m_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])), shape_a[1]],
                            [0, m_idx * view_shape[0]])
            if not b_trans:
                dyn_b = pto.view(tensor_b, [shape_b[0], view_shape[1]],
                            [(shape_b[0], shape_b[1] - n_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))],
                            [0, n_idx * view_shape[1]])
            else:
                dyn_b = pto.view(tensor_b, [view_shape[1], shape_b[1]],
                            [(shape_b[0] - n_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1])), shape_b[1]],
                            [n_idx * view_shape[1], 0])
            res = pto.matmul(dtype, dyn_a, dyn_b, input_config.a_trans, input_config.b_trans,
                                                input_config.c_format_nz)

            pto.assemble(res, [m_idx * view_shape[0], n_idx * view_shape[1]], tensor_c)
            del dyn_a
            del dyn_b
            del res


def convert_np_dtype_to_pto_dtype(dtype):
    if dtype == INT8:
        return pto.data_type.DT_INT8
    elif dtype == FP16:
        return pto.data_type.DT_FP16
    elif dtype == INT32:
        return pto.data_type.DT_INT32
    elif dtype == FP32:
        return pto.data_type.DT_FP32
    else:
        assert False, "pto dtype not found in matmul"


def create_tensor(dtype, ori_shape, tensor_name, format_nz, transposed=None):
    m = ori_shape[0]
    k = ori_shape[1]
    n = ori_shape[2]
    if tensor_name == "tensor_c":
        shape = (m, n)
    elif tensor_name == "tensor_b":
        shape = (n, k) if transposed else (k, n)
    elif tensor_name == "tensor_a":
        shape = (k, m) if transposed else (m, k)
    else:
        assert False, "tensor name not found in matmul"
    if format_nz:
        return pto.tensor(convert_np_dtype_to_pto_dtype(dtype), shape, tensor_name, pto.tile_op_format.TILEOP_NZ)
    else:
        return pto.tensor(convert_np_dtype_to_pto_dtype(dtype), shape, tensor_name)


def ceil_div_util(a, b):
    return a if b == 0 else (a + b - 1) // b


def nd_trans_to_fractal_nz(data: np.ndarray):
    ori_shape = data.shape
    ori_m, ori_n = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_padding = ((0, 0),) * len(batch_ori)
    if data.dtype == INT8:
        m0, n0 = 16, 32
    elif data.dtype == FP16 or data.dtype == INT32:
        m0, n0 = 16, 16
    elif data.dtype == FP32:
        m0, n0 = 16, 8

    m_data = ceil_div_util(ori_m, m0)
    n_data = ceil_div_util(ori_n, n0)
    padding_m = m_data * m0 - ori_m
    padding_n = n_data * n0 - ori_n
    data = np.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), "constant")
    offset = len(data.shape) - 2
    array_trans = [x for x in range(offset)] + [x + offset for x in [2, 0, 1, 3]]
    data = data.reshape(batch_ori + (m_data, m0, n_data, n0)).transpose(*array_trans)
    return data


def gen_matmul_golden_data(input_config: ShapeConfig):
    ori_shape = input_config.ori_shape
    shape_a = [ori_shape[0], ori_shape[1]]
    shape_b = [ori_shape[1], ori_shape[2]]

    if input_config.in_dtype == INT8:
        a = np.random.randint(-4, 5, shape_a).astype(INT8)
        b = np.random.randint(-4, 5, shape_b).astype(INT8)
        c = np.matmul(a.astype(INT32), b.astype(INT32)).astype(INT32)
    elif input_config.in_dtype == FP16 or input_config.in_dtype == FP32:
        a = np.random.uniform(-1, 1, shape_a).astype(input_config.in_dtype)
        b = np.random.uniform(-1, 1, shape_b).astype(input_config.in_dtype)
        c = np.matmul(a.astype(FP32), b.astype(FP32)).astype(input_config.out_dtype)
    else:
        assert False, "golden dtype not found"

    if input_config.a_trans:
        a = a.transpose(1, 0)
    if input_config.a_format_nz:
        a = nd_trans_to_fractal_nz(a)

    if input_config.b_trans:
        b = b.transpose(1, 0)
    if input_config.b_format_nz:
        b = nd_trans_to_fractal_nz(b)

    if input_config.c_format_nz:
        c = nd_trans_to_fractal_nz(c)

    a_data = a.flatten().tolist()
    b_data = b.flatten().tolist()
    c_data = c.flatten().tolist()
    c_device_data = [0] * c.size

    return a_data, b_data, c_data, c_device_data
