#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Example demonstrating the pypto.min() and pypto.max() functionality.

This example shows how to use the extended min/max functions for
SymbolicScalar and integer values.
"""

import pypto


def example_two_args():
    print("=== Example: Two Arguments (Backward Compatible) ===")
    a = pypto.symbolic_scalar(10)
    b = pypto.symbolic_scalar(20)
    
    min_result = pypto.min(a, b)
    max_result = pypto.max(a, b)
    
    print(f"min(10, 20) = {min_result.concrete()}")
    print(f"max(10, 20) = {max_result.concrete()}")
    print()


def example_multiple_args():
    print("=== Example: Multiple Arguments ===")
    values = [30, 10, 20, 5]
    
    min_result = pypto.min(*values)
    max_result = pypto.max(*values)
    
    print(f"min(30, 10, 20, 5) = {min_result.concrete()}")
    print(f"max(10, 30, 5, 20) = {max_result.concrete()}")
    print()


def example_iterable():
    print("=== Example: Iterable Support ===")
    values = [pypto.symbolic_scalar(30), 10, pypto.symbolic_scalar(20), 5]
    
    min_result = pypto.min(values)
    max_result = pypto.max(values)
    
    print(f"min([30, 10, 20, 5]) = {min_result.concrete()}")
    print(f"max([10, 30, 5, 20]) = {max_result.concrete()}")
    print()


def example_single_value():
    print("=== Example: Single Value ===")
    a = pypto.symbolic_scalar(10)
    
    min_result = pypto.min(a)
    max_result = pypto.max(a)
    
    print(f"min(10) = {min_result.concrete()}")
    print(f"max(10) = {max_result.concrete()}")
    print()


def example_in_kernel():
    print("=== Example: Usage in Kernel ===")
    
    # Simulating dynamic shape scenario
    batch_size = pypto.symbolic_scalar(128)
    tile_size = 32
    offset = pypto.symbolic_scalar(96)
    
    # Calculate valid boundary
    valid_size = pypto.min(batch_size - offset, tile_size)
    print(f"valid_size = min(128 - 96, 32) = {valid_size.concrete()}")
    
    # Multiple dynamic values
    dim1 = pypto.symbolic_scalar(100)
    dim2 = pypto.symbolic_scalar(80)
    dim3 = pypto.symbolic_scalar(90)
    min_dim = pypto.min(dim1, dim2, dim3)
    print(f"min_dim = min(100, 80, 90) = {min_dim.concrete()}")
    print()


def example_error_handling():
    print("=== Example: Error Handling ===")
    
    try:
        pypto.min([])
    except ValueError as e:
        print(f"min([]) raises ValueError: {e}")
    
    try:
        pypto.max([])
    except ValueError as e:
        print(f"max([]) raises ValueError: {e}")
    
    try:
        pypto.min()
    except TypeError as e:
        print(f"min() raises TypeError: {e}")
    
    try:
        pypto.max()
    except TypeError as e:
        print(f"max() raises TypeError: {e}")
    print()


if __name__ == "__main__":
    example_two_args()
    example_multiple_args()
    example_iterable()
    example_single_value()
    example_in_kernel()
    example_error_handling()
    
    print("All examples completed successfully!")