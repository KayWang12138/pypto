#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Test Random operation on board
"""
import os
import math
import torch
import pypto
import pytest
import numpy as np
from numpy.testing import assert_allclose
import torch_npu

bfloat16 = np.dtype('bfloat16')


def random_golden(key, counter0, counter1, shape, rounds, dtype=np.float32):
    def uint32(x):
        return x & 0xFFFFFFFF

    def multiply_high_low(a, b):
        product = a * b
        hi = uint32(product >> 32)
        lo = uint32(product)
        return lo, hi
    
    def philox_single_round(counter, key0, key1):
        lo0, hi0 = multiply_high_low(0xD2511F53, counter[0])
        lo1, hi1 = multiply_high_low(0xCD9E8D57, counter[2])
        
        return [
            uint32(hi1 ^ counter[1] ^ key0),
            uint32(lo1),
            uint32(hi0 ^ counter[3] ^ key1),
            uint32(lo0)
        ]
    
    def raise_key(key0, key1):
        return (
            uint32(key0 + 0x9E3779B9),
            uint32(key1 + 0xBB67AE85)
        )
    
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    result = np.zeros(total_elements, dtype=np.uint32)

    init_key0 = uint32(key)
    init_key1 = uint32(key >> 32)
    
    original_counter = [
        uint32(counter0),
        uint32(counter0 >> 32),
        uint32(counter1),
        uint32(counter1 >> 32)
    ]
    
    for i in range(0, total_elements, 4):
        key0, key1 = init_key0, init_key1
        current_counter = original_counter.copy()

        for _ in range(rounds):
            current_counter = philox_single_round(current_counter, key0, key1)
            key0, key1 = raise_key(key0, key1)
        
        for j in range(min(4, total_elements - i)):
            result[i + j] = current_counter[j]
        
        original_counter[0] = uint32(original_counter[0] + 1)
        if original_counter[0] == 0:
            original_counter[1] = uint32(original_counter[1] + 1)
            if original_counter[1] == 0:
                original_counter[2] = uint32(original_counter[2] + 1)
                if original_counter[2] == 0:
                    original_counter[3] = uint32(original_counter[3] + 1)
    
    if dtype == np.float32:
        result_float = np.zeros(total_elements, dtype=np.float32)
        for i in range(total_elements):
            x = result[i]
            man = x & 0x7fffff
            exp = 127
            val = (exp << 23) | man
            result_float[i] = np.frombuffer(np.array([val], dtype=np.uint32).tobytes(), dtype=np.float32)[0] - 1.0
        return result_float.reshape(shape)
    elif dtype == np.float16:
        result_half = np.zeros(total_elements, dtype=np.float16)
        for i in range(total_elements):
            x = result[i]
            x_uint16 = np.uint16(x & 0xFFFF)
            man = x_uint16 & 0x3ff
            exp = np.uint16(15)
            val = (exp << 10) | man
            result_half[i] = np.frombuffer(np.array([val], dtype=np.uint16).tobytes(), dtype=np.float16)[0] - np.float16(1.0)
        return result_half.reshape(shape)
    elif dtype == bfloat16:
        result_bfloat16 = np.zeros(total_elements, dtype=bfloat16)
        for i in range(total_elements):
            x = result[i]
            x_uint16 = np.uint16(x & 0xFFFF)
            man = x_uint16 & 0x7f
            exp = np.uint16(127)
            val = (exp << 7) | man
            result_bfloat16[i] = np.frombuffer(np.array([val], dtype=np.uint16).tobytes(), dtype=bfloat16)[0] - bfloat16.type(1.0)
        return result_bfloat16.reshape(shape)
    else:
        result_float = np.zeros(total_elements, dtype=np.float32)
        for i in range(total_elements):
            x = result[i]
            man = x & 0x7fffff
            exp = 127
            val = (exp << 23) | man
            result_float[i] = np.frombuffer(np.array([val], dtype=np.uint32).tobytes(), dtype=np.float32)[0] - 1.0
        return result_float.reshape(shape)


def test_random_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    output_shape = (64,)
    view_shape = (32,)
    tile_shape = (16,)

    pypto.runtime._device_init()

    output = pypto.tensor(output_shape, pypto.DT_FP32, "PTO_TENSOR_output")

    loop_num = math.ceil(output_shape[0] / view_shape[0])
    
    key = 12345678901234
    counter0 = 0
    counter1 = 0
    rounds = 10
    
    with pypto.function("MAIN", output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            
            valid_shape = pypto.min(pypto.symbolic_scalar(output_shape[0]) - offset, 
                                    pypto.symbolic_scalar(view_shape[0]))
            
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.random(key, counter0, counter1, view_shape, rounds, pypto.DataType.DT_FP32)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    
    out_data = np.zeros(output_shape, dtype=np.float32)

    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    
    golden = random_golden(key, counter0, counter1, output_shape, rounds, np.float32)
    
    assert_allclose(out_data.flatten(), golden.flatten(), rtol=1e-5, atol=1e-5)
    
    pypto.runtime._device_fini()


def test_random_onboard_large():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    output_shape = (256,)
    view_shape = (128,)
    tile_shape = (64,)

    pypto.runtime._device_init()

    output = pypto.tensor(output_shape, pypto.DT_FP32, "PTO_TENSOR_output")

    loop_num = math.ceil(output_shape[0] / view_shape[0])
    
    key = 99999999999999
    counter0 = 100
    counter1 = 200
    rounds = 10
    
    with pypto.function("MAIN", output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            
            valid_shape = pypto.min(pypto.symbolic_scalar(output_shape[0]) - offset, 
                                    pypto.symbolic_scalar(view_shape[0]))
            
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.random(key, counter0, counter1, view_shape, rounds, pypto.DataType.DT_FP32)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    
    out_data = np.zeros(output_shape, dtype=np.float32)

    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    
    golden = random_golden(key, counter0, counter1, output_shape, rounds, np.float32)
    
    assert_allclose(out_data.flatten(), golden.flatten(), rtol=1e-5, atol=1e-5)
    
    pypto.runtime._device_fini()


def test_random_onboard_fp16():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    output_shape = (64,)
    view_shape = (32,)
    tile_shape = (16,)

    pypto.runtime._device_init()

    output = pypto.tensor(output_shape, pypto.DT_FP16, "PTO_TENSOR_output")

    loop_num = math.ceil(output_shape[0] / view_shape[0])
    
    key = 12345678901234
    counter0 = 0
    counter1 = 0
    rounds = 10
    
    with pypto.function("MAIN", output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            
            valid_shape = pypto.min(pypto.symbolic_scalar(output_shape[0]) - offset, 
                                    pypto.symbolic_scalar(view_shape[0]))
            
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.random(key, counter0, counter1, view_shape, rounds, pypto.DataType.DT_FP16)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    
    out_data = np.zeros(output_shape, dtype=np.float16)

    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    
    golden = random_golden(key, counter0, counter1, output_shape, rounds, np.float16)
    
    assert_allclose(out_data.flatten().astype(np.float32), golden.flatten().astype(np.float32), rtol=1e-3, atol=1e-3)
    
    pypto.runtime._device_fini()


def test_random_onboard_bf16():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    output_shape = (64,)
    view_shape = (32,)
    tile_shape = (16,)

    pypto.runtime._device_init()

    output = pypto.tensor(output_shape, pypto.DT_BF16, "PTO_TENSOR_output")

    loop_num = math.ceil(output_shape[0] / view_shape[0])
    
    key = 12345678901234
    counter0 = 0
    counter1 = 0
    rounds = 10
    
    with pypto.function("MAIN", output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            
            valid_shape = pypto.min(pypto.symbolic_scalar(output_shape[0]) - offset, 
                                    pypto.symbolic_scalar(view_shape[0]))
            
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.random(key, counter0, counter1, view_shape, rounds, pypto.DataType.DT_BF16)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    
    out_data = np.zeros(output_shape, dtype=bfloat16)

    pto_out = pypto.from_torch(torch.from_numpy(out_data.astype(np.float32)).to(torch.bfloat16), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    
    golden = random_golden(key, counter0, counter1, output_shape, rounds, bfloat16)
    
    assert_allclose(out_data.flatten().astype(np.float32), golden.flatten().astype(np.float32), rtol=1e-2, atol=1e-2)
    
    pypto.runtime._device_fini()
