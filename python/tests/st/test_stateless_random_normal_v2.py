#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
import torch
import numpy as np
from numpy.testing import assert_allclose

import tensorflow as tf
from tensorflow.python.ops import gen_stateless_random_ops_v2
tf.compat.v1.disable_eager_execution()


def test_normal_FP32():
    """Test whether the output of FP32 is correct"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()

    shape = torch.tensor([4, 4], dtype=torch.int32)
    key = torch.tensor([1234], dtype=torch.uint64)
    counter = torch.tensor([0, 1], dtype=torch.uint64)
    alg = torch.tensor([1], dtype=torch.int32)

    pto_shape_tensor = pypto.from_torch(shape)
    pto_key_tensor = pypto.from_torch(key)
    pto_counter_tensor = pypto.from_torch(counter)
    pto_alg_tensor = pypto.from_torch(alg)
    dtype = pypto.DT_FP32
    res = pypto.tensor([4, 4], dtype)

    with pypto.function("NORMAL_CONTENT_FP32", shape, key, counter, alg, res):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(4, 4)
            res.move(
                pypto.stateless_random_normal_v2(
                    pto_shape_tensor, pto_key_tensor, pto_counter_tensor, pto_alg_tensor, dtype))

    shape_tf = tf.constant([4, 4])
    key_tf = [1234]
    counter_tf = [0, 1]
    rnd = gen_stateless_random_ops_v2.stateless_random_normal_v2(
        shape_tf,
        key=tf.constant(key_tf, dtype=tf.uint64),
        counter=tf.constant(counter_tf, dtype=tf.uint64),
        dtype=tf.float32,
        alg=1
    )

    with tf.compat.v1.Session():
        sess.run(tf.compat.v1.global_variables_initializer())
        expected = sess.run(rnd)

    res_tensor = torch.zeros(4, 4, dtype=torch.float32)
    pto_res_tensor = pypto.from_torch(res_tensor, "res_tensor")
    pypto.runtime._device_run_once_data_from_host(
        pto_shape_tensor, pto_key_tensor, pto_counter_tensor, pto_alg_tensor, pto_res_tensor)
    assert_allclose(res_tensor.flatten(), np.array(expected).flatten(), atol=1e-3, verbose=True)
    pypto.runtime._device_fini()
