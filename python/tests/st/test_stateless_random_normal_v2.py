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


def stateless_random_normal_v2_numpy_golden(shape, key, counter, alg, dtype):
    """
    Numpy implementation of stateless_random_normal_v2 based on TensorFlow's Philox algorithm.
    Replicates TensorFlow's behavior using the Philox 4x32_10 algorithm with Box-Muller transform.
    
    Args:
        shape: Output shape (list of integers)
        key: Random seed key (list with one uint64 value)
        counter: Counter value (list with two uint64 values)
        alg: Algorithm identifier (list with one int, 1=Philox)
        dtype: Data type (pypto.DT_FP32, pypto.DT_FP16, pypto.DT_BF16)
    
    Returns:
        Random normal array with values from standard normal distribution (mean=0, std=1)
    """
    import math
    
    if alg[0] != 1:
        raise ValueError(f"Only Philox algorithm (alg=1) is supported, got alg={alg[0]}")
    
    if dtype == pypto.DT_FP16:
        np_dtype = np.float16
    elif dtype == pypto.DT_BF16:
        np_dtype = np.float32
    else:
        np_dtype = np.float32
    
    def uint64_to_uint32_pair(val):
        arr = np.array([val], dtype=np.uint64)
        return arr.view(np.uint32).copy()
    
    key_arr = uint64_to_uint32_pair(np.uint64(key[0]))
    counter_arr = np.concatenate([
        uint64_to_uint32_pair(np.uint64(counter[0])),
        uint64_to_uint32_pair(np.uint64(counter[1]))
    ])
    
    PHILOX_W32A = 0x9E3779B9
    PHILOX_W32B = 0xBB67AE85
    PHILOX_M4X32A = 0xD2511F53
    PHILOX_M4X32B = 0xCD9E8D57
    
    def multiply_high_low(a, b):
        product = int(a) * int(b)
        return (product & 0xFFFFFFFF), ((product >> 32) & 0xFFFFFFFF)
    
    def compute_single_round(counter, key):
        lo0, hi0 = multiply_high_low(PHILOX_M4X32A, int(counter[0]))
        lo1, hi1 = multiply_high_low(PHILOX_M4X32B, int(counter[2]))
        
        result = np.zeros(4, dtype=np.uint32)
        result[0] = np.uint32(hi1 ^ int(counter[1]) ^ int(key[0]))
        result[1] = np.uint32(lo1)
        result[2] = np.uint32(hi0 ^ int(counter[3]) ^ int(key[1]))
        result[3] = np.uint32(lo0)
        return result
    
    def raise_key(key):
        key[0] = np.uint32((int(key[0]) + PHILOX_W32A) & 0xFFFFFFFF)
        key[1] = np.uint32((int(key[1]) + PHILOX_W32B) & 0xFFFFFFFF)
    
    def philox_next(counter, key):
        c = counter.copy()
        k = key.copy()
        
        for _ in range(10):
            c = compute_single_round(c, k)
            raise_key(k)
        
        counter[0] = np.uint32((int(counter[0]) + 1) & 0xFFFFFFFF)
        if counter[0] == 0:
            counter[1] = np.uint32((int(counter[1]) + 1) & 0xFFFFFFFF)
            if counter[1] == 0:
                counter[2] = np.uint32((int(counter[2]) + 1) & 0xFFFFFFFF)
                if counter[2] == 0:
                    counter[3] = np.uint32((int(counter[3]) + 1) & 0xFFFFFFFF)
        
        return c
    
    def uint32_to_float(uint_val):
        man = int(uint_val) & 0x7fffff
        exp = 127
        val = (exp << 23) | man
        result = np.frombuffer(np.array([val], dtype=np.uint32).tobytes(), dtype=np.float32)[0]
        return float(result - 1.0)
    
    def box_muller_float(x0, x1):
        epsilon = 1.0e-7
        u1 = uint32_to_float(x0)
        if u1 < epsilon:
            u1 = epsilon
        
        v1 = 2.0 * math.pi * uint32_to_float(x1)
        u2 = math.sqrt(-2.0 * math.log(u1))
        
        f0 = u2 * math.sin(v1)
        f1 = u2 * math.cos(v1)
        
        return f0, f1
    
    def convert_normal_pair(uint0, uint1):
        f0, f1 = box_muller_float(uint0, uint1)
        
        if dtype == pypto.DT_FP16:
            return float(np.float16(f0)), float(np.float16(f1))
        elif dtype == pypto.DT_BF16:
            return f0, f1
        else:
            return f0, f1
    
    total_elements = np.prod(shape)
    num_pairs = (total_elements + 1) // 2
    num_rounds = (num_pairs + 1) // 2
    
    counter_state = counter_arr.copy()
    key_state = key_arr.copy()
    
    all_random_vals = []
    for _ in range(num_rounds):
        random_uints = philox_next(counter_state, key_state)
        for i in range(0, len(random_uints), 2):
            if i + 1 < len(random_uints):
                f0, f1 = convert_normal_pair(random_uints[i], random_uints[i + 1])
                all_random_vals.extend([f0, f1])
    
    result = np.array(all_random_vals[:total_elements], dtype=np_dtype)
    return result.reshape(shape)


def test_compare_golden_implementations():
    """Compare TensorFlow and Numpy golden implementations for Normal distribution"""
    shape = [32]
    key = [1234]
    counter = [0, 1]
    alg = [1]
    
    print("\\n" + "="*80)
    print("Testing Normal Distribution Golden Implementations")
    print("="*80)
    
    for dtype, rtol, atol, dtype_name in [(pypto.DT_FP32, 1e-5, 1e-5, "FP32"), 
                                            (pypto.DT_FP16, 1e-3, 1e-3, "FP16"),
                                            (pypto.DT_BF16, 1e-2, 1e-2, "BF16")]:
        print("\\n" + "="*80)
        print("Testing " + dtype_name)
        print("="*80)
        
        tf_result = stateless_random_normal_v2_golden(shape, key, counter, alg, dtype)
        np_result = stateless_random_normal_v2_numpy_golden(shape, key, counter, alg, dtype)
        
        print(f"TensorFlow result shape: {tf_result.shape}, dtype: {tf_result.dtype}")
        print(f"Numpy result shape: {np_result.shape}, dtype: {np_result.dtype}")
        
        if dtype == pypto.DT_BF16:
            tf_min = float(tf_result.min())
            tf_max = float(tf_result.max())
            tf_mean = float(tf_result.mean())
            tf_std = float(tf_result.std())
        else:
            tf_min = tf_result.min()
            tf_max = tf_result.max()
            tf_mean = tf_result.mean()
            tf_std = tf_result.std()
        
        print(f"TensorFlow range: [{tf_min:.6f}, {tf_max:.6f}]")
        print(f"Numpy range: [{np_result.min():.6f}, {np_result.max():.6f}]")
        print(f"TensorFlow mean: {tf_mean:.6f}, std: {tf_std:.6f}")
        print(f"Numpy mean: {np_result.mean():.6f}, std: {np_result.std():.6f}")
        
        # Show first few values
        if dtype == pypto.DT_BF16:
            tf_first3 = [float(x) for x in tf_result.flatten()[:5]]
        else:
            tf_first3 = tf_result.flatten()[:5].tolist()
        np_first3 = np_result.flatten()[:5].tolist()
        
        print("\\nFirst 5 TensorFlow values: " + str(tf_first3))
        print("First 5 Numpy values:      " + str(np_first3))
        
        try:
            if dtype == pypto.DT_BF16:
                tf_flat = [float(x) for x in tf_result.flatten()]
                np_flat = np_result.flatten().tolist()
                assert_allclose(tf_flat, np_flat, rtol=rtol, atol=atol)
            else:
                assert_allclose(tf_result.flatten(), np_result.flatten(), rtol=rtol, atol=atol)
            print("\\n✓ " + dtype_name + " Results match within tolerance (rtol=" + str(rtol) + ", atol=" + str(atol) + ")")
            
            # Verify statistical properties
            if dtype == pypto.DT_BF16:
                tf_mean_val = float(tf_result.mean())
                tf_std_val = float(tf_result.std())
            else:
                tf_mean_val = tf_result.mean()
                tf_std_val = tf_result.std()
            
            np_mean_val = np_result.mean()
            np_std_val = np_result.std()
            
            assert_allclose([tf_mean_val], [np_mean_val], rtol=0.1, atol=0.1)
            assert_allclose([tf_std_val], [np_std_val], rtol=0.1, atol=0.1)
            print("✓ " + dtype_name + " Statistical properties (mean/std) match")
            
        except AssertionError as e:
            print("\\n✗ " + dtype_name + " Results do NOT match!")
            if dtype == pypto.DT_BF16:
                tf_flat = [float(x) for x in tf_result.flatten()]
                np_flat = np_result.flatten().tolist()
            else:
                tf_flat = tf_result.flatten().tolist()
                np_flat = np_result.flatten().tolist()
            
            diff = np.abs(np.array(tf_flat) - np.array(np_flat))
            print(f"Max absolute difference: {diff.max():.6e}")
            print(f"Mean absolute difference: {diff.mean():.6e}")
            print("\\nDifference statistics:")
            print(f"  Min difference: {diff.min():.6e}")
            print(f"  Max difference: {diff.max():.6e}")
            print(f"  Mean difference: {diff.mean():.6e}")
            print(f"  Std difference: {diff.std():.6e}")
            raise
    
    print("\\n" + "="*80)
    print("SUCCESS: All Normal Distribution Golden Implementations Match!")
    print("="*80)


@pytest.mark.soc("950")
def test_normal_fp32():
    """Test whether the output of FP32 is correct"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()

    view_shape = [32]
    tile_shape = [32]

    shape = [32]
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


@pytest.mark.soc("950")
def test_normal_fp16():
    """Test whether the output of FP16 is correct"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()

    view_shape = [32]
    tile_shape = [32]

    shape = [32]
    key = [1234]
    counter = [0, 1]
    alg = [1]
    dtype = pypto.DT_FP16
    output = pypto.tensor(shape, dtype)

    loop_num = math.ceil(shape[0] / view_shape[0])
    with pypto.function("NORMAL_CONTENT_FP16", output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            valid_shape = pypto.min(pypto.symbolic_scalar(shape[0]) - offset, pypto.symbolic_scalar(view_shape[0]))
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.stateless_random_normal_v2(shape, key, counter, alg, dtype)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    out_data = np.zeros(shape, dtype=np.float16)
    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    golden = stateless_random_normal_v2_numpy_golden(shape, key, counter, alg, dtype)
    assert_allclose(out_data.flatten(), golden.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()
