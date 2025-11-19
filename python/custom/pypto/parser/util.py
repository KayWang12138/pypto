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
from typing import Union, Sequence
from ..utils import CodeHelper, Instruction, CustStruct, Tensor, \
    Vector, Var, ConfigMap, Shape, get_vartype_str, get_obj_dtype


def get_var_str(v: Union[Tensor, Var, Vector, Shape, CustStruct, ConfigMap, float, int, str]):
    if isinstance(v, Tensor):
        return f'tsr{v.idx}'
    elif isinstance(v, Shape):
        return f'shape{v.idx}'
    elif isinstance(v, (CustStruct, ConfigMap)):
        return f'stru{v.idx}'
    elif isinstance(v, Var):
        return f'v{v.idx}'
    elif isinstance(v, Vector):
        return f'vec{v.idx}'
    elif isinstance(v, bool):
        return str(int(v))
    elif isinstance(v, int):
        if v < 0:
            return f"({v})"
        return str(v)
    elif isinstance(v, float):
        if v == float('-inf'):
            return '-3.4e+38f'
        return str(v)
    elif isinstance(v, str):
        return v
    else:
        raise NotImplementedError()


def get_curlybrace_list(v) -> str:
    if isinstance(v, (list, Sequence)):
        body = ', '.join([get_curlybrace_list(item) for item in v])
        return '{' + body + '}'
    else:
        return get_var_str(v)


def get_scalar_dtype(v: Union[Var, int, float]):
    dtype_mapper = {
        'int8_t': 'DataType::DT_INT8',
        'int16_t': 'DataType::DT_INT16',
        'int32_t': 'DataType::DT_INT32',
        'int64_t': 'DataType::DT_INT64',
        'npu::tile_fwk::float16': 'DataType::DT_FP16',
        'float': 'DataType::DT_FP32',
        'double': 'DataType::DT_DOUBLE',
        'npu::tile_fwk::bfloat16': 'DataType::DT_BF16',
        'uint8_t': 'DataType::DT_UINT8',
        'uint16_t': 'DataType::DT_UINT16',
        'uint32_t': 'DataType::DT_UINT32',
        'uint64_t': 'DataType::DT_UINT64',
        'int': 'DataType::DT_INT32',
        'str': 'DataType::DT_FP32' # to be modified
    }
    dtype = "UNKNOWN"
    if isinstance(v, Var):
        if v.dtype in dtype_mapper:
            dtype = dtype_mapper.get(v.dtype)
        else:
            dtype = dtype_mapper.get(type(v).__name__, "ILLEGAL_DTYPE"),
    return dtype
