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
    c1 = (result[0] >> 32) & 0xFFFFFFFF
    c2 = result[1] & 0xFFFFFFFF
    c3 = (result[1] >> 32) & 0xFFFFFFFF
    
    c0 = (c0 + offset_lo) & 0xFFFFFFFF
    if c0 < offset_lo:
        offset_hi = (offset_hi + 1) & 0xFFFFFFFF
    
    c1 = (c1 + offset_hi) & 0xFFFFFFFF
    if c1 < offset_hi:
        c2 = (c2 + 1) & 0xFFFFFFFF
        if c2 == 0:
            c3 = (c3 + 1) & 0xFFFFFFFF
    
    result[0] = c0 | (c1 << 32)
    result[1] = c2 | (c3 << 32)
    
    return result


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
    
    total_elements = shape[0]
    
    result = np.zeros(total_elements, dtype=np.uint32)
    key_hi = (key >> 32) & 0xFFFFFFFF
    key_lo = key & 0xFFFFFFFF
    
    current_counter = [
        counter[0] & 0xFFFFFFFF,
        (counter[0] >> 32) & 0xFFFFFFFF,
        counter[1] & 0xFFFFFFFF,
        (counter[1] >> 32) & 0xFFFFFFFF
    ]
    
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
    
    return result


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
    
    with pypto.function("MAIN", output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            
            valid_shape = pypto.min(pypto.symbolic_scalar(output_shape[0]) - offset, 
                                    pypto.symbolic_scalar(view_shape[0]))
            
            tile_offset = idx * view_shape[0]
            tile_counter = skip_counter(counter, tile_offset)
            
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.random(key, tile_counter, [valid_shape], rounds=10)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    
    out_data = np.zeros(output_shape, dtype=np.uint32)

    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    
    golden = random_golden(key, counter, output_shape, 10)
    
    assert_allclose(out_data.flatten(), golden.flatten())
    
    pypto.runtime._device_fini()
