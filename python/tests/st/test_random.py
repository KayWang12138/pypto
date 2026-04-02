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


def random_golden(key, counter0, counter1, shape, rounds):
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
    
    return result.reshape(shape)


def test_random_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    output_shape = (64,)
    view_shape = (32,)
    tile_shape = (16,)

    pypto.runtime._device_init()

    output = pypto.tensor(output_shape, pypto.DT_UINT32, "PTO_TENSOR_output")

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
            res = pypto.random(key, counter0, counter1, view_shape, rounds)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    
    out_data = np.zeros(output_shape, dtype=np.uint32)

    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    
    golden = random_golden(key, counter0, counter1, output_shape, rounds)
    
    assert_allclose(out_data.flatten(), golden.flatten())
    
    pypto.runtime._device_fini()


def test_random_onboard_large():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    output_shape = (256,)
    view_shape = (128,)
    tile_shape = (64,)

    pypto.runtime._device_init()

    output = pypto.tensor(output_shape, pypto.DT_UINT32, "PTO_TENSOR_output")

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
            res = pypto.random(key, counter0, counter1, view_shape, rounds)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    
    out_data = np.zeros(output_shape, dtype=np.uint32)

    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    
    golden = random_golden(key, counter0, counter1, output_shape, rounds)
    
    assert_allclose(out_data.flatten(), golden.flatten())
    
    pypto.runtime._device_fini()
