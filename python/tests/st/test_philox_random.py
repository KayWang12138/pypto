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
Test PhiloxRandom operation on board
"""
import os
import math
import torch
import pypto
import pytest
import numpy as np
from numpy.testing import assert_allclose
import torch_npu


def philox_random_golden(key, counter, shape, rounds):
    def multiply_high_low(a, b):
        product = a * b
        hi = (product >> 32) & 0xFFFFFFFF
        lo = product & 0xFFFFFFFF
        return lo, hi
    
    def philox_single_round(counter, key):
        lo0, hi0 = multiply_high_low(0xD2511F53, counter[0])
        lo1, hi1 = multiply_high_low(0xCD9E8D57, counter[2])
        
        return [
            hi1 ^ counter[1] ^ key[0],
            lo1,
            hi0 ^ counter[3] ^ key[1],
            lo0
        ]
    
    def raise_key(key):
        return [
            (key[0] + 0x9E3779B9) & 0xFFFFFFFF,
            (key[1] + 0xBB67AE85) & 0xFFFFFFFF
        ]
    
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    result = np.zeros(total_elements, dtype=np.uint32)
    current_key = list(key)
    current_counter = list(counter)
    
    for i in range(0, total_elements, 4):
        for _ in range(rounds):
            current_counter = philox_single_round(current_counter, current_key)
            current_key = raise_key(current_key)
        
        for j in range(min(4, total_elements - i)):
            result[i + j] = current_counter[j]
        
        current_counter[0] = (current_counter[0] + 1) & 0xFFFFFFFF
        if current_counter[0] == 0:
            current_counter[1] = (current_counter[1] + 1) & 0xFFFFFFFF
            if current_counter[1] == 0:
                current_counter[2] = (current_counter[2] + 1) & 0xFFFFFFFF
                if current_counter[2] == 0:
                    current_counter[3] = (current_counter[3] + 1) & 0xFFFFFFFF
    
    return result.reshape(shape)


def test_philox_random_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    output_shape = (64,)
    view_shape = (32,)
    tile_shape = (16,)

    pypto.runtime._device_init()

    key_tensor = pypto.tensor((2,), pypto.DT_UINT32, "PTO_TENSOR_key")
    counter_tensor = pypto.tensor((4,), pypto.DT_UINT32, "PTO_TENSOR_counter")
    output = pypto.tensor(output_shape, pypto.DT_UINT32, "PTO_TENSOR_output")

    loop_num = math.ceil(output_shape[0] / view_shape[0])
    
    with pypto.function("MAIN", key_tensor, counter_tensor, output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            
            valid_shape = pypto.min(pypto.symbolic_scalar(output_shape[0]) - offset, 
                                    pypto.symbolic_scalar(view_shape[0]))
            view_key = pypto.view(key_tensor, (2,), [0], valid_shape=[2])
            view_counter = pypto.view(counter_tensor, (4,), [0], valid_shape=[4])
            
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.philox_random(view_key, view_counter, view_shape, pypto.DT_UINT32, rounds=10)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    
    key_data = np.array([12345, 67890], dtype=np.uint32)
    counter_data = np.array([0, 0, 0, 0], dtype=np.uint32)
    out_data = np.zeros(output_shape, dtype=np.uint32)

    pto_key = pypto.from_torch(torch.from_numpy(key_data), "PTO_TENSOR_key")
    pto_counter = pypto.from_torch(torch.from_numpy(counter_data), "PTO_TENSOR_counter")
    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_key, pto_counter, pto_out)
    
    golden = philox_random_golden(key_data.tolist(), counter_data.tolist(), output_shape, 10)
    
    assert_allclose(out_data.flatten(), golden.flatten())
    
    pypto.runtime._device_fini()


def test_philox_random_onboard_2d():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    output_shape = (4, 8)
    view_shape = (2, 4)
    tile_shape = (2, 4)

    pypto.runtime._device_init()

    key_tensor = pypto.tensor((2,), pypto.DT_UINT32, "PTO_TENSOR_key")
    counter_tensor = pypto.tensor((4,), pypto.DT_UINT32, "PTO_TENSOR_counter")
    output = pypto.tensor(output_shape, pypto.DT_UINT32, "PTO_TENSOR_output")

    loop_num_0 = math.ceil(output_shape[0] / view_shape[0])
    loop_num_1 = math.ceil(output_shape[1] / view_shape[1])
    
    with pypto.function("MAIN", key_tensor, counter_tensor, output):
        for idx0 in pypto.loop(loop_num_0, name="loop0", idx_name="idx0"):
            for idx1 in pypto.loop(loop_num_1, name="loop1", idx_name="idx1"):
                offset_0 = idx0 * view_shape[0]
                offset_1 = idx1 * view_shape[1]
                
                valid_shape_0 = pypto.min(pypto.symbolic_scalar(output_shape[0]) - offset_0, 
                                          pypto.symbolic_scalar(view_shape[0]))
                valid_shape_1 = pypto.min(pypto.symbolic_scalar(output_shape[1]) - offset_1, 
                                          pypto.symbolic_scalar(view_shape[1]))
                view_key = pypto.view(key_tensor, (2,), [0], valid_shape=[2])
                view_counter = pypto.view(counter_tensor, (4,), [0], valid_shape=[4])
                
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                res = pypto.philox_random(view_key, view_counter, view_shape, pypto.DT_UINT32, rounds=10)
                pypto.assemble(res, [offset_0, offset_1], output)

    assert isinstance(output, pypto.tensor)
    
    key_data = np.array([11111, 22222], dtype=np.uint32)
    counter_data = np.array([1, 2, 3, 4], dtype=np.uint32)
    out_data = np.zeros(output_shape, dtype=np.uint32)

    pto_key = pypto.from_torch(torch.from_numpy(key_data), "PTO_TENSOR_key")
    pto_counter = pypto.from_torch(torch.from_numpy(counter_data), "PTO_TENSOR_counter")
    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_key, pto_counter, pto_out)
    
    golden = philox_random_golden(key_data.tolist(), counter_data.tolist(), output_shape, 10)
    
    assert_allclose(out_data.flatten(), golden.flatten())
    
    pypto.runtime._device_fini()
