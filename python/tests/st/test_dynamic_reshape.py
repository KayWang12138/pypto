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
import pto
import pytest
import numpy as np
import torch
from numpy.testing import assert_allclose


GRAPG_T = pto.GraphType.TENSOR_GRAPH
FUNC_T = pto.FunctionType.STATIC


def test_reshape_shape():
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    s = 16
    d = 32
    shape = [s, d]
    dst_shape = [d, s]
    q = pto.tensor(shape, dtype)

    with pto.pto_function("Reshape1", GRAPG_T, FUNC_T, q):
        pto.set_vec_tile_shapes(16, 16)
        res = pto.reshape(q, dst_shape)
    
    assert res.shape == dst_shape 
    pto.runtime._device_fini() 


def test_reshape_equal():
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    s = 16
    d = 32
    shape = [s, d]
    dst_shape = [s * 2, d // 2]
    q = pto.tensor(shape, dtype)
    out = pto.tensor(dst_shape, dtype)

    with pto.function("Reshape2", [q], [out]):
        with pto.loop_function("Reshape2Loop", "batchId", pto.loop_range(1)) as loop:
            for _ in loop:
                pto.set_vec_tile_shapes(16, 16)
                q0 = q.reshape(dst_shape)
                out.move(q0)
                del q0
                del out
    
    q_tensor = np.arange(s * 32, dtype=np.float32).reshape(s, 32)
    q_data = q_tensor.flatten().tolist()
    out_data = list([0] * s * 32)

    pto.runtime._device_run_once_data_from_host([q_data], [out_data])
    assert out_data == torch.tensor(q_tensor).flatten().tolist()
    pto.runtime._device_fini() 


def test_reshape_equal2():
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    s = 16
    d = 32
    shape = [s, d]
    dst_shape = [s * 2, d // 2]
    q = pto.tensor(shape, dtype)
    t = pto.tensor(shape, dtype)
    out = pto.tensor(dst_shape, dtype)

    with pto.function("Reshape3", [q, t], [out]):
        with pto.loop_function("Reshape3Loop", "batchId", pto.loop_range(1)) as loop:
            for _ in loop:
                pto.set_vec_tile_shapes(16, 16)
                q0 = pto.reshape(q, dst_shape, [16, 16])
                t0 = pto.reshape(t, dst_shape, [16, 16])
                out.move(pto.add(q0, t0))
                del q0
                del t0
                del out
    
    q_tensor = np.arange(s * 32, dtype=np.float32).reshape(s, 32)
    q_data = q_tensor.flatten().tolist()
    tmp_tensor = np.arange(s * 32, dtype=np.float32).reshape(s, 32)
    tmp_data = q_tensor.flatten().tolist()
    out_data = list([0] * s * 32)

    pto.runtime._device_run_once_data_from_host([q_data, tmp_data], [out_data])
    assert out_data == torch.add(torch.tensor(q_tensor), torch.tensor(tmp_tensor)).flatten().tolist()
    pto.runtime._device_fini() 


def test_reshape_validshape():
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    s = 16
    d = 32
    shape = [s * d]
    dst_shape = [s, d]
    q = pto.tensor(shape, dtype)
    out = pto.tensor(dst_shape, dtype)

    with pto.function("Reshape4", [q], [out]):
        with pto.loop_function("Reshape4Loop", "batchId", pto.loop_range(1)) as loop:
            for _ in loop:
                pto.set_vec_tile_shapes(16, 16)
                q0 = pto.reshape(q, dst_shape, [8, 32])
                qp = pto.add(q0, 1.0)
                out.move(q0)
                del q0
                del out
    
    q_tensor = np.arange(s * 32, dtype=np.float32)
    q_data = q_tensor.flatten().tolist()
    out_data = list([0] * s * 32)

    pto.runtime._device_run_once_data_from_host([q_data], [out_data])
    assert out_data[:32] == torch.tensor(q_tensor)[:32].flatten().tolist()
    pto.runtime._device_fini() 


def test_reshape_validshape2():
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    s = 16
    d = 32
    shape = [s, d]
    dst_shape = [s * d]
    q = pto.tensor(shape, dtype)
    out = pto.tensor(dst_shape, dtype)

    with pto.function("Reshape5", [q], [out]):
        with pto.loop_function("Reshape5Loop", "batchId", pto.loop_range(1)) as loop:
            for _ in loop:
                pto.set_vec_tile_shapes(16, 16)
                q0 = pto.reshape(q, dst_shape)
                q0 = q0 + 1.0
                out.move(q0)
                del q0
                del out
    
    q_tensor = np.arange(s * 32, dtype=np.float32).reshape(16, 32)
    q_data = q_tensor.flatten().tolist()
    scalar_tensor = np.ones(s * 32, dtype=np.float32).reshape(16, 32)
    out_data = list([0] * s * 32)

    pto.runtime._device_run_once_data_from_host([q_data], [out_data])
    assert out_data[:64] == torch.add(torch.tensor(q_tensor), torch.tensor(scalar_tensor))[:2, :].flatten().tolist()
    pto.runtime._device_fini() 