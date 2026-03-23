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


def random_golden(key, counter, shape, rounds):
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
    
    key_lo = key & 0xFFFFFFFF
    key_hi = (key >> 32) & 0xFFFFFFFF
    current_key = [key_lo, key_hi]
    
    counter_lo0 = counter[0] & 0xFFFFFFFF
    counter_hi0 = (counter[0] >> 32) & 0xFFFFFFFF
    counter_lo1 = counter[1] & 0xFFFFFFFF
    counter_hi1 = (counter[1] >> 32) & 0xFFFFFFFF
    current_counter = [counter_lo0, counter_hi0, counter_lo1, counter_hi1]
    
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
    counter = [0, 0]
    rounds = 10
    
    with pypto.function("MAIN", output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            
            valid_shape = pypto.min(pypto.symbolic_scalar(output_shape[0]) - offset, 
                                    pypto.symbolic_scalar(view_shape[0]))
            
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.random(key, counter, view_shape, rounds)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    
    out_data = np.zeros(output_shape, dtype=np.uint32)

    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    
    golden = random_golden(key, counter, output_shape, rounds)
    
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
    counter = [100, 200]
    rounds = 10
    
    with pypto.function("MAIN", output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            
            valid_shape = pypto.min(pypto.symbolic_scalar(output_shape[0]) - offset, 
                                    pypto.symbolic_scalar(view_shape[0]))
            
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.random(key, counter, view_shape, rounds)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    
    out_data = np.zeros(output_shape, dtype=np.uint32)

    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    
    golden = random_golden(key, counter, output_shape, rounds)
    
    assert_allclose(out_data.flatten(), golden.flatten())
    
    pypto.runtime._device_fini()
