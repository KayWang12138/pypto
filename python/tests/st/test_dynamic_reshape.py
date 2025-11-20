# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import os
import pypto
import pytest
import numpy as np
import torch
from numpy.testing import assert_allclose
import torch_npu


def test_reshape_shape():
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    s = 16
    d = 32
    shape = [s, d]
    dst_shape = [d, s]
    q = pypto.tensor(shape, dtype)

    with pypto.function("Reshape1", q, static=True):
        pypto.set_vec_tile_shapes(16, 16)
        res = pypto.reshape(q, dst_shape)

    assert res.shape == dst_shape
    pypto.runtime._device_fini()


def test_reshape_equal():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    s = 16
    d = 32
    shape = [s, d]
    dst_shape = [s * 2, d // 2]
    q = pypto.tensor(shape, dtype)
    out = pypto.tensor(dst_shape, dtype)

    with pypto.function("Reshape2", [q], [out]):
        for _ in pypto.loop(1, name="Reshape2Loop", idx_name="batchId"):
            pypto.set_vec_tile_shapes(16, 16)
            q0 = q.reshape(dst_shape)
            out.move(q0)
            del q0
            del out

    q_tensor = torch.arange(s * 32, dtype=torch.float32).reshape(s, 32)
    out_tensor = torch.zeros_like(q_tensor)

    pypto.runtime._device_run_once_data_from_host([q_tensor], [out_tensor])
    assert torch.equal(out_tensor.flatten(), q_tensor.flatten())
    pypto.runtime._device_fini()


def test_reshape_equal2():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    s = 16
    d = 32
    shape = [s, d]
    dst_shape = [s * 2, d // 2]
    q = pypto.tensor(shape, dtype)
    t = pypto.tensor(shape, dtype)
    out = pypto.tensor(dst_shape, dtype)

    with pypto.function("Reshape3", [q, t], [out]):
        for _ in pypto.loop(1, name="Reshape3Loop", idx_name="batchId"):
            pypto.set_vec_tile_shapes(16, 16)
            q0 = pypto.reshape(q, dst_shape, valid_shape=[16, 16])
            t0 = pypto.reshape(t, dst_shape, valid_shape=[16, 16])
            out.move(pypto.add(q0, t0))
            del q0
            del t0
            del out

    q_tensor = torch.arange(s * 32, dtype=torch.float32).reshape(s, 32)
    tmp_tensor = torch.arange(s * 32, dtype=torch.float32).reshape(s, 32)
    out_tensor = torch.zeros_like(q_tensor)

    pypto.runtime._device_run_once_data_from_host([q_tensor, tmp_tensor], [out_tensor])
    assert torch.equal(out_tensor.flatten(), torch.add(q_tensor, tmp_tensor).flatten())
    pypto.runtime._device_fini()


def test_reshape_validshape():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    s = 16
    d = 32
    shape = [s * d]
    dst_shape = [s, d]
    q = pypto.tensor(shape, dtype)
    out = pypto.tensor(dst_shape, dtype)

    with pypto.function("Reshape4", [q], [out]):
        for _ in pypto.loop(1, name="Reshape4Loop", idx_name="batchId"):
            pypto.set_vec_tile_shapes(16, 16)
            q0 = pypto.reshape(q, dst_shape, valid_shape=[8, 32])
            qp = pypto.add(q0, 1.0)
            out.move(q0)
            del q0
            del out

    q_tensor = torch.arange(s * 32, dtype=torch.float32)
    out_tensor = torch.zeros_like(q_tensor)

    pypto.runtime._device_run_once_data_from_host([q_tensor], [out_tensor])
    assert torch.equal(out_tensor[:32], q_tensor[:32])
    pypto.runtime._device_fini()


def test_reshape_validshape2():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    pypto.runtime._device_init()
    s = 16
    d = 32
    shape = [s, d]
    dst_shape = [s * d]
    q = pypto.tensor(shape, dtype)
    out = pypto.tensor(dst_shape, dtype)

    with pypto.function("Reshape5", [q], [out]):
        for _ in pypto.loop(1, name="Reshape5Loop", idx_name="batchId"):
            pypto.set_vec_tile_shapes(16, 16)
            q0 = pypto.reshape(q, dst_shape)
            q0 = q0 + 1.0
            out.move(q0)
            del q0
            del out

    q_tensor = torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32)
    scalar_tensor = torch.ones(16 * 32, dtype=torch.float32).reshape(16, 32)
    out_tensor = torch.zeros(16 * 32, dtype=torch.float32)

    pypto.runtime._device_run_once_data_from_host([q_tensor], [out_tensor])

    expected = (q_tensor + scalar_tensor).flatten()
    assert torch.equal(out_tensor[:64], expected[:64])
    pypto.runtime._device_fini()