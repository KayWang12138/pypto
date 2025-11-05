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
    assert pto.bytes_of(pto.DT_INT4) == 1
    assert pto.bytes_of(pto.DT_INT8) == 1
    assert pto.bytes_of(pto.DT_INT16) == 2
    assert pto.bytes_of(pto.DT_INT32) == 4
    assert pto.bytes_of(pto.DT_INT64) == 8
    assert pto.bytes_of(pto.DT_FP8) == 1
    assert pto.bytes_of(pto.DT_FP16) == 2
    assert pto.bytes_of(pto.DT_FP32) == 4
    assert pto.bytes_of(pto.DT_BF16) == 2
    assert pto.bytes_of(pto.DT_HF4) == 1
    assert pto.bytes_of(pto.DT_HF8) == 1
    assert pto.bytes_of(pto.DT_UINT8) == 1
    assert pto.bytes_of(pto.DT_UINT16) == 2
    assert pto.bytes_of(pto.DT_UINT32) == 4
    assert pto.bytes_of(pto.DT_UINT64) == 8
    assert pto.bytes_of(pto.DT_BOOL) == 1
    assert pto.bytes_of(pto.DT_DOUBLE) == 8

