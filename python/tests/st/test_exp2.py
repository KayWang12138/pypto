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
Test exp2 operation
"""

import numpy as np
import pytest

import pypto


def test_exp2_basic():
    """Test exp2 with basic input"""
    # Test with 1D tensor
    x = pypto.tensor([3], pypto.DT_FP32)
    x.fill_([0.0, 1.0, 2.0])
    y = pypto.exp2(x)
    
    # JIT compile and run
    jit_func = pypto.jit(lambda x: pypto.exp2(x))
    result = jit_func(x)
    
    # Expected result: 2^0=1, 2^1=2, 2^2=4
    expected = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    
    # Verify the result
    assert result.shape == expected.shape
    assert np.allclose(result.numpy(), expected, atol=1e-6)


def test_exp2_tensor_method():
    """Test exp2 as a tensor method"""
    # Test with 2D tensor
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    x.fill_([[0.0, 1.0], [2.0, 3.0]])
    y = x.exp2()
    
    # JIT compile and run
    jit_func = pypto.jit(lambda x: x.exp2())
    result = jit_func(x)
    
    # Expected result
    expected = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    
    # Verify the result
    assert result.shape == expected.shape
    assert np.allclose(result.numpy(), expected, atol=1e-6)


def test_exp2_negative():
    """Test exp2 with negative input"""
    # Test with negative values
    x = pypto.tensor([3], pypto.DT_FP32)
    x.fill_([-1.0, -2.0, -3.0])
    y = pypto.exp2(x)
    
    # JIT compile and run
    jit_func = pypto.jit(lambda x: pypto.exp2(x))
    result = jit_func(x)
    
    # Expected result: 2^-1=0.5, 2^-2=0.25, 2^-3=0.125
    expected = np.array([0.5, 0.25, 0.125], dtype=np.float32)
    
    # Verify the result
    assert result.shape == expected.shape
    assert np.allclose(result.numpy(), expected, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
