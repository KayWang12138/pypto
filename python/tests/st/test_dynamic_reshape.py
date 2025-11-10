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
import torch_npu

def test_reshape_shape():
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    s = 16
    d = 32
    shape = [s, d]
    dst_shape = [d, s]
    q = pto.tensor(shape, dtype)

    with pto.function("Reshape1", q, static=True):
        pto.set_vec_tile_shapes(16, 16)
        res = pto.reshape(q, dst_shape)

    assert res.shape == dst_shape
    pto.runtime._device_fini()


def test_reshape_equal():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    s = 16
    d = 32
    shape = [s, d]
    dst_shape = [s * 2, d // 2]
    q = pto.tensor(shape, dtype)
    out = pto.tensor(dst_shape, dtype)

    with pto.function("Reshape2", [q], [out]):
        for _ in pto.loop(1, name="Reshape2Loop", idx_name="batchId"):
            pto.set_vec_tile_shapes(16, 16)
            q0 = q.reshape(dst_shape)
            out.move(q0)
            del q0
            del out

    q_tensor = torch.arange(s * 32, dtype=torch.float32).reshape(s, 32)
    out_tensor = torch.zeros_like(q_tensor)

    pto.runtime._device_run_once_data_from_host([q_tensor], [out_tensor])
    assert torch.equal(out_tensor.flatten(), q_tensor.flatten())
    pto.runtime._device_fini()


def test_reshape_equal2():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
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
        for _ in pto.loop(1, name="Reshape3Loop", idx_name="batchId"):
            pto.set_vec_tile_shapes(16, 16)
            q0 = pto.reshape(q, dst_shape, valid_shape=[16, 16])
            t0 = pto.reshape(t, dst_shape, valid_shape=[16, 16])
            out.move(pto.add(q0, t0))
            del q0
            del t0
            del out

    q_tensor = torch.arange(s * 32, dtype=torch.float32).reshape(s, 32)
    tmp_tensor = torch.arange(s * 32, dtype=torch.float32).reshape(s, 32)
    out_tensor = torch.zeros_like(q_tensor)

    pto.runtime._device_run_once_data_from_host([q_tensor, tmp_tensor], [out_tensor])
    assert torch.equal(out_tensor.flatten(), torch.add(q_tensor, tmp_tensor).flatten())
    pto.runtime._device_fini()


def test_reshape_validshape():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    s = 16
    d = 32
    shape = [s * d]
    dst_shape = [s, d]
    q = pto.tensor(shape, dtype)
    out = pto.tensor(dst_shape, dtype)

    with pto.function("Reshape4", [q], [out]):
        for _ in pto.loop(1, name="Reshape4Loop", idx_name="batchId"):
            pto.set_vec_tile_shapes(16, 16)
            q0 = pto.reshape(q, dst_shape, valid_shape=[8, 32])
            qp = pto.add(q0, 1.0)
            out.move(q0)
            del q0
            del out

    q_tensor = torch.arange(s * 32, dtype=torch.float32)
    out_tensor = torch.zeros_like(q_tensor)

    pto.runtime._device_run_once_data_from_host([q_tensor], [out_tensor])
    assert torch.equal(out_tensor[:32], q_tensor[:32])
    pto.runtime._device_fini()


def test_reshape_validshape2():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    pto.runtime._device_init()
    s = 16
    d = 32
    shape = [s, d]
    dst_shape = [s * d]
    q = pto.tensor(shape, dtype)
    out = pto.tensor(dst_shape, dtype)

    with pto.function("Reshape5", [q], [out]):
        for _ in pto.loop(1, name="Reshape5Loop", idx_name="batchId"):
            pto.set_vec_tile_shapes(16, 16)
            q0 = pto.reshape(q, dst_shape)
            q0 = q0 + 1.0
            out.move(q0)
            del q0
            del out

    q_tensor = torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32)
    scalar_tensor = torch.ones(16 * 32, dtype=torch.float32).reshape(16, 32)
    out_tensor = torch.zeros(16 * 32, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([q_tensor], [out_tensor])

    expected = (q_tensor + scalar_tensor).flatten()
    assert torch.equal(out_tensor[:64], expected[:64])
    pto.runtime._device_fini()
