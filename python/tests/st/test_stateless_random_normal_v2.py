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
import math
import os
import pypto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose

import tensorflow as tf
from tensorflow.python.ops import gen_stateless_random_ops_v2
tf.compat.v1.disable_eager_execution()


def stateless_random_normal_v2_golden(shape, key, counter, alg, dtype):
    dtype_tf = tf.float32
    if dtype == pypto.DT_FP16:
        dtype_tf = tf.float16
    elif dtype == pypto.DT_BF16:
        dtype_tf = tf.bfloat16

    rnd = gen_stateless_random_ops_v2.stateless_random_normal_v2(
        shape=tf.constant(shape, dtype=tf.int64),
        key=tf.constant(key, dtype=tf.uint64),
        counter=tf.constant(counter, dtype=tf.uint64),
        dtype=dtype_tf,
        alg=alg[0]
    )

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        expected = sess.run(rnd)

    return expected


@pytest.mark.soc("950")
def test_normal_FP32():
    """Test whether the output of FP32 is correct"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()

    view_shape = [32,]
    tile_shape = [32,]

    shape = [32,]
    key = [1234]
    counter = [0, 1]
    alg = [1]
    dtype = pypto.DT_FP32
    output = pypto.tensor(shape, dtype)

    loop_num = math.ceil(shape[0] / view_shape[0])
    with pypto.function("NORMAL_CONTENT_FP32", output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            valid_shape = pypto.min(pypto.symbolic_scalar(shape[0]) - offset, pypto.symbolic_scalar(view_shape[0]))
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.stateless_random_normal_v2(shape, key, counter, alg, dtype)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    out_data = np.zeros(shape, dtype=np.float32)
    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    golden = stateless_random_normal_v2_golden(shape, key, counter, alg, dtype)
    assert_allclose(out_data.flatten(), golden.flatten(), rtol=1e-4, atol=1e-4)

    pypto.runtime._device_fini()
