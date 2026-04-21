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
import pytest
import pypto


def test_min_two_symbolic_scalars():
    a = pypto.symbolic_scalar(10)
    b = pypto.symbolic_scalar(20)
    result = pypto.min(a, b)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 10


def test_max_two_symbolic_scalars():
    a = pypto.symbolic_scalar(10)
    b = pypto.symbolic_scalar(20)
    result = pypto.max(a, b)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 20


def test_min_two_ints():
    result = pypto.min(10, 20)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 10


def test_max_two_ints():
    result = pypto.max(10, 20)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 20


def test_min_mixed_int_and_symbolic():
    a = pypto.symbolic_scalar(10)
    result = pypto.min(a, 20)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 10

    result = pypto.min(20, a)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 10


def test_max_mixed_int_and_symbolic():
    a = pypto.symbolic_scalar(20)
    result = pypto.max(a, 10)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 20

    result = pypto.max(10, a)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 20


def test_min_multiple_args():
    result = pypto.min(30, 10, 20, 5)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 5


def test_max_multiple_args():
    result = pypto.max(10, 30, 5, 20)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 30


def test_min_multiple_symbolic_scalars():
    a = pypto.symbolic_scalar(30)
    b = pypto.symbolic_scalar(10)
    c = pypto.symbolic_scalar(20)
    d = pypto.symbolic_scalar(5)
    result = pypto.min(a, b, c, d)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 5


def test_max_multiple_symbolic_scalars():
    a = pypto.symbolic_scalar(10)
    b = pypto.symbolic_scalar(30)
    c = pypto.symbolic_scalar(5)
    d = pypto.symbolic_scalar(20)
    result = pypto.max(a, b, c, d)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 30


def test_min_iterable():
    items = [30, 10, 20, 5]
    result = pypto.min(items)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 5


def test_max_iterable():
    items = [10, 30, 5, 20]
    result = pypto.max(items)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 30


def test_min_iterable_symbolic_scalars():
    items = [
        pypto.symbolic_scalar(30),
        pypto.symbolic_scalar(10),
        pypto.symbolic_scalar(20),
        pypto.symbolic_scalar(5),
    ]
    result = pypto.min(items)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 5


def test_max_iterable_symbolic_scalars():
    items = [
        pypto.symbolic_scalar(10),
        pypto.symbolic_scalar(30),
        pypto.symbolic_scalar(5),
        pypto.symbolic_scalar(20),
    ]
    result = pypto.max(items)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 30


def test_min_single_symbolic_scalar():
    a = pypto.symbolic_scalar(10)
    result = pypto.min(a)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 10


def test_max_single_symbolic_scalar():
    a = pypto.symbolic_scalar(20)
    result = pypto.max(a)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 20


def test_min_single_int():
    result = pypto.min(10)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 10


def test_max_single_int():
    result = pypto.max(20)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 20


def test_min_empty_iterable():
    with pytest.raises(ValueError, match="min.*arg is an empty sequence"):
        pypto.min([])


def test_max_empty_iterable():
    with pytest.raises(ValueError, match="max.*arg is an empty sequence"):
        pypto.max([])


def test_min_no_args():
    with pytest.raises(TypeError, match="min expected at least 1 argument"):
        pypto.min()


def test_max_no_args():
    with pytest.raises(TypeError, match="max expected at least 1 argument"):
        pypto.max()


def test_min_non_iterable():
    with pytest.raises(TypeError, match="'int' object is not iterable"):
        pypto.min(10)


def test_max_non_iterable():
    with pytest.raises(TypeError, match="'int' object is not iterable"):
        pypto.max(10)


def test_min_tuple_iterable():
    items = (30, 10, 20, 5)
    result = pypto.min(items)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 5


def test_max_tuple_iterable():
    items = (10, 30, 5, 20)
    result = pypto.max(items)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 30


def test_min_mixed_iterable():
    items = [30, pypto.symbolic_scalar(10), 20, pypto.symbolic_scalar(5)]
    result = pypto.min(items)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 5


def test_max_mixed_iterable():
    items = [pypto.symbolic_scalar(10), 30, pypto.symbolic_scalar(5), 20]
    result = pypto.max(items)
    assert isinstance(result, pypto.SymbolicScalar)
    assert result.concrete() == 30