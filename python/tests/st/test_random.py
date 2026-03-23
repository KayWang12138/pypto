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


def skip_counter(counter, offset):
    result = list(counter)
    
    offset_lo = offset & 0xFFFFFFFF
    offset_hi = (offset >> 32) & 0xFFFFFFFF
    
    c0 = result[0] & 0xFFFFFFFF
    c1 = result[1] & 0xFFFFFFFF
    c2 = result[2] & 0xFFFFFFFF
    c3 = result[3] & 0xFFFFFFFF
    
    c0 = (c0 + offset_lo) & 0xFFFFFFFF
    carry = 1 if c0 < offset_lo else 0
    
    sum1 = (c1 + offset_hi + carry) & 0xFFFFFFFF
    carry = 1 if sum1 < c1 or (carry and sum1 == c1) else 0
    c1 = sum1
    
    if carry:
        c2 = (c2 + 1) & 0xFFFFFFFF
        if c2 == 0:
            c3 = (c3 + 1) & 0xFFFFFFFF
    
    result[0] = c0
    result[1] = c1
    result[2] = c2
    result[3] = c3
    
    return result


def calculate_linear_offset(offset, shape):
    linear_offset = 0
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        linear_offset += offset[i] * stride
        stride *= shape[i]
    return linear_offset


def random_golden(key, counter, shape, rounds):
    def multiply_high_low(a, b):
        product = a * b
        hi = (product >> 32) & 0xFFFFFFFF
        lo = product & 0xFFFFFFFF
        return lo, hi
    
    def philox_single_round(counter, key_hi, key_lo):
        lo0, hi0 = multiply_high_low(0xD2511F53, counter[0])
        lo1, hi1 = multiply_high_low(0xCD9E8D57, counter[2])
        
        return [
            hi1 ^ counter[1] ^ key_hi,
            lo1,
            hi0 ^ counter[3] ^ key_lo,
            lo0
        ]
    
    def raise_key(key_hi, key_lo):
        return (
            (key_hi + 0x9E3779B9) & 0xFFFFFFFF,
            (key_lo + 0xBB67AE85) & 0xFFFFFFFF
        )
    
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    result = np.zeros(total_elements, dtype=np.uint32)
    key_hi = (key >> 32) & 0xFFFFFFFF
    key_lo = key & 0xFFFFFFFF
    current_counter = [c & 0xFFFFFFFF for c in counter]
    
    for i in range(0, total_elements, 4):
        for _ in range(rounds):
            current_counter = philox_single_round(current_counter, key_hi, key_lo)
            key_hi, key_lo = raise_key(key_hi, key_lo)
        
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
    counter = [0, 0, 0, 0]
    
    with pypto.function("MAIN", output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            
            valid_shape = pypto.min(pypto.symbolic_scalar(output_shape[0]) - offset, 
                                    pypto.symbolic_scalar(view_shape[0]))
            
            tile_offset = [idx * view_shape[0]]
            linear_offset = calculate_linear_offset(tile_offset, list(output_shape))
            tile_counter = skip_counter(counter, linear_offset)
            
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.random(key, tile_counter, view_shape, pypto.DT_UINT32, rounds=10)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    
    out_data = np.zeros(output_shape, dtype=np.uint32)

    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    
    golden = random_golden(key, counter, output_shape, 10)
    
    assert_allclose(out_data.flatten(), golden.flatten())
    
    pypto.runtime._device_fini()


def test_random_onboard_2d():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    output_shape = (4, 8)
    view_shape = (2, 4)
    tile_shape = (2, 4)

    pypto.runtime._device_init()

    output = pypto.tensor(output_shape, pypto.DT_UINT32, "PTO_TENSOR_output")

    loop_num_0 = math.ceil(output_shape[0] / view_shape[0])
    loop_num_1 = math.ceil(output_shape[1] / view_shape[1])
    
    key = 11111222223333
    counter = [1, 2, 3, 4]
    
    with pypto.function("MAIN", output):
        for idx0 in pypto.loop(loop_num_0, name="loop0", idx_name="idx0"):
            for idx1 in pypto.loop(loop_num_1, name="loop1", idx_name="idx1"):
                offset_0 = idx0 * view_shape[0]
                offset_1 = idx1 * view_shape[1]
                
                valid_shape_0 = pypto.min(pypto.symbolic_scalar(output_shape[0]) - offset_0, 
                                          pypto.symbolic_scalar(view_shape[0]))
                valid_shape_1 = pypto.min(pypto.symbolic_scalar(output_shape[1]) - offset_1, 
                                          pypto.symbolic_scalar(view_shape[1]))
                
                tile_offset = [idx0 * view_shape[0], idx1 * view_shape[1]]
                linear_offset = calculate_linear_offset(tile_offset, list(output_shape))
                tile_counter = skip_counter(counter, linear_offset)
                
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                res = pypto.random(key, tile_counter, view_shape, pypto.DT_UINT32, rounds=10)
                pypto.assemble(res, [offset_0, offset_1], output)

    assert isinstance(output, pypto.tensor)
    
    out_data = np.zeros(output_shape, dtype=np.uint32)

    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    
    golden = random_golden(key, counter, output_shape, 10)
    
    assert_allclose(out_data.flatten(), golden.flatten())
    
    pypto.runtime._device_fini()
