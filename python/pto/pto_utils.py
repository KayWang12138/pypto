#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import inspect
import struct
from typing import Sequence, Union, List

from . import pto_impl

from .enum import DataType
from .symbolic_scalar import SymbolicScalar, SymInt


def to_sym(value) -> pto_impl.SymbolicScalar:
    if isinstance(value, int):
        return pto_impl.SymbolicScalar(value)
    if isinstance(value, pto_impl.SymbolicScalar):
        return value
    if isinstance(value, SymbolicScalar):
        return value.base()
    raise ValueError("Invalid value type")


def to_syms(value: Union[Sequence[int], Sequence[SymbolicScalar]]) -> List[pto_impl.SymbolicScalar]:
    return [to_sym(v) for v in value]


def ceil(a: SymInt, b: SymInt) -> SymInt:
    return (a + b - 1) // b


def set_source_location(level: int = 1):
    pto_impl.SetLocation(inspect.stack()[level + 1].filename, inspect.stack()[level + 1].lineno)


def clear_source_location():
    pto_impl.ClearLocation()


def convert_matmul_extend_params(extend_params) -> dict:
    extend_params.setdefault('bias_tensor', pto_impl.Tensor())
    extend_params.setdefault('scale_tensor', pto_impl.Tensor())
    extend_params.setdefault('relu_type', pto_impl.ReLuType.NoReLu)
    # scale: float trans to uint64
    if 'scale' not in extend_params:
        extend_params['scale'] = 0
    else:
        scale_value = extend_params['scale']
        if not isinstance(scale_value, float):
            raise RuntimeError("scale must float type")
        pakced_float: bytes = struct.pack('f', scale_value)
        scale_trans_val: int = struct.unpack('<I', pakced_float)[0]
        extend_params['scale'] = scale_trans_val
    return extend_params


def bytes_of(dtype: DataType) -> int:
    ''' return the number of bytes of the current datatype

    Parameters
    ----------
    dtype: pto.DataType
        datatype to be determined the number of bytes

    Returns
    -------
    int: the size of bytes the datatype contains

    Examples
    --------
    >>> print(pto.bytes_of(pto.DataType.DT_FP32))
        4
    '''
    # implementation
    return pto_impl.BytesOf(dtype)
