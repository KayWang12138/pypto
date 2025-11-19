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
class DATATYPE:
    fp16 = 'DataType::DT_FP16'
    fp32 = 'DataType::DT_FP32'
    bf16 = 'DataType::DT_BF16'
    bf32 = 'DataType::DT_BF32'
    int32 = 'DataType::DT_INT32'
    int = 'int'
    int32 = 'DataType::DT_INT32'


class ReduceMode:
    atomic_add = 'ReduceMode::ATOMIC_ADD'


class CastMode:
    NONE = "CAST_NONE"
    RINT = "CAST_RINT"
    ROUND = "CAST_ROUND"
    FLOOR = "CAST_FLOOR"
    CEIL = "CAST_CEIL"
    TRUNC = "CAST_TRUNC"
    ODD = "CAST_ODD"


class ScatterMode:
    NONE = "NONE"
    ADD = "ADD"
    MULTIPLY = "MULTIPLY"
