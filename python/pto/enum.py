#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""

from pto import pto_impl

DataType = pto_impl.DataType
TileOpFormat = pto_impl.TileOpFormat
CachePolicy = pto_impl.CachePolicy
ReduceMode = pto_impl.ReduceMode
MemoryType = pto_impl.MemoryType
FunctionType = pto_impl.FunctionType
GraphType = pto_impl.GraphType
CastMode = pto_impl.CastMode
OpType = pto_impl.OpType
OutType = pto_impl.OutType

DT_INT4 = pto_impl.DT_INT4
DT_INT8 = pto_impl.DT_INT8
DT_INT16 = pto_impl.DT_INT16
DT_INT32 = pto_impl.DT_INT32
DT_INT64 = pto_impl.DT_INT64
DT_FP8 = pto_impl.DT_FP8
DT_FP16 = pto_impl.DT_FP16
DT_FP32 = pto_impl.DT_FP32
DT_BF16 = pto_impl.DT_BF16
DT_HF4 = pto_impl.DT_HF4
DT_HF8 = pto_impl.DT_HF8
DT_UINT8 = pto_impl.DT_UINT8
DT_UINT16 = pto_impl.DT_UINT16
DT_UINT32 = pto_impl.DT_UINT32
DT_UINT64 = pto_impl.DT_UINT64
DT_BOOL = pto_impl.DT_BOOL
DT_DOUBLE = pto_impl.DT_DOUBLE
