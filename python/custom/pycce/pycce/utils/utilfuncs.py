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
from .datatype import DTYPE


def sizeof(dtype: DTYPE):
    assert isinstance(dtype, DTYPE)
    if dtype.ctype in ['half', 'bfloat16_t', 'uint16_t', 'int16_t']:
        return 2
    elif dtype.ctype in ['float', 'int', 'uint32_t', 'int32_t']:
        return 4
    elif dtype.ctype in ['int8_t', 'uint8_t']:
        return 1
    elif dtype.ctype in ['int64_t', 'uint64_t']:
        return 8
    elif dtype.ctype in ['void']:
        return 1
    else:
        raise NotImplementedError(f'Unknown dtype {dtype.ctype}')

