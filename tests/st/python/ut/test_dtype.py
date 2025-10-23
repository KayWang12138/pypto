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
import pto


def test_mem_type():
    # Make sure all data types are defined
    assert isinstance(pto.DT_INT4, pto.DataType)
    assert isinstance(pto.DT_INT8, pto.DataType)
    assert isinstance(pto.DT_INT16, pto.DataType)
    assert isinstance(pto.DT_INT32, pto.DataType)
    assert isinstance(pto.DT_INT64, pto.DataType)
    assert isinstance(pto.DT_FP8, pto.DataType)
    assert isinstance(pto.DT_FP16, pto.DataType)
    assert isinstance(pto.DT_FP32, pto.DataType)
    assert isinstance(pto.DT_BF16, pto.DataType)
    assert isinstance(pto.DT_HF4, pto.DataType)
    assert isinstance(pto.DT_HF8, pto.DataType)
    assert isinstance(pto.DT_UINT8, pto.DataType)
    assert isinstance(pto.DT_UINT16, pto.DataType)
    assert isinstance(pto.DT_UINT32, pto.DataType)
    assert isinstance(pto.DT_UINT64, pto.DataType)
    assert isinstance(pto.DT_BOOL, pto.DataType)
    assert isinstance(pto.DT_DOUBLE, pto.DataType)
    assert isinstance(pto.DT_BOTTOM, pto.DataType)
