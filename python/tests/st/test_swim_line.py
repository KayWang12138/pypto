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
profiling of aicpu pref  test for PyPTO
"""
import json
from typing import List, Dict
import os
import sys
import argparse
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.
    
    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("ERROR: Environment variable TILE_FWK_DEVICE_ID is not set.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    
    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None

SHAPE = (32, 32, 1, 256)
VAL = 1


def add_core(input0: pypto.Tensor, input1: pypto.Tensor, add1_flag: bool = False):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out = pypto.tensor(SHAPE, pypto.DT_FP32)
    if add1_flag:
        t3 = input0 + input1
        out[:] = t3 + VAL
    else:
        out[:] = input0 + input1
    return out


def create_add_kernel( add1_flag: bool = True):
    
    mode = pypto.RunMode.NPU
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode},
                        debug_options={"runtime_debug_mode": 1})
    def add_kernel(
        input0: pypto.Tensor(SHAPE, pypto.DT_FP32),
        input1: pypto.Tensor(SHAPE, pypto.DT_FP32),
    ) -> pypto.Tensor(SHAPE, pypto.DT_FP32):
        out = add_core(input0, input1, add1_flag)
        return out
    
    return add_kernel


def test_aicpu_perf() -> None:
    device_id = get_device_id()
    if device_id is None:
        return
    device = f'npu:{device_id}'
    import torch_npu
    torch.npu.set_device(device_id)

    shape = SHAPE
    #prepare data
    val = VAL
    
    input_data0 = torch.rand(shape, dtype=torch.float, device=device)
    input_data1 = torch.rand(shape, dtype=torch.float, device=device)
    print(f"Input0 shape: {input_data0.shape}")
    print(f"Input1 shape: {input_data1.shape}")
    golden = torch.add(input_data0, input_data1)

    golden2 = torch.add(input_data0, input_data1) + val
    output_data2 = create_add_kernel(True)(input_data0, input_data1)
    max_diff = np.abs(output_data2.cpu().numpy() - golden2.cpu().numpy()).max()
    print(f"Output shape: {output_data2.shape}")
    print(f"Max difference: {max_diff:.6f}")
    assert_allclose(np.array(output_data2.cpu()), np.array(golden2.cpu()), rtol=3e-3, atol=3e-3)

    aicpu_json_path = pypto.pypto_impl.LogTopFolder() + "/aicpu_dev_pref.json"
    print("aicpu json path: ", aicpu_json_path)
    
    with open(aicpu_json_path, 'r', encoding='utf-8') as f:
            core_list: List[Dict] = json.load(f)
            for core in core_list:
                tasks = core.get("tasks", [])
                assert len(tasks) > 0, "Could not Get aicpu perf"
    print("✓ profiling of aicpu pref  test passed")
    print()