# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import pto


def test_dtype():
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

    assert str(pto.DT_INT4) == "DataType.DT_INT4"


def test_tile_op_format():
    assert str(pto.TileOpFormat.TILEOP_ND) == "TileOpFormat.TILEOP_ND"
    assert str(pto.TileOpFormat.TILEOP_NZ) == "TileOpFormat.TILEOP_NZ"


def test_cache_policy():
    assert str(pto.CachePolicy.PREFETCH) == "CachePolicy.PREFETCH"
    assert str(pto.CachePolicy.NONE_CACHEABLE) == "CachePolicy.NONE_CACHEABLE"


def test_reduce_mode():
    assert str(pto.ReduceMode.ATOMIC_ADD) == "ReduceMode.ATOMIC_ADD"


def test_cast_mode():
    assert str(pto.CastMode.CAST_RINT) == "CastMode.CAST_RINT"
    assert str(pto.CastMode.CAST_ROUND) == "CastMode.CAST_ROUND"
    assert str(pto.CastMode.CAST_FLOOR) == "CastMode.CAST_FLOOR"
    assert str(pto.CastMode.CAST_CEIL) == "CastMode.CAST_CEIL"
    assert str(pto.CastMode.CAST_TRUNC) == "CastMode.CAST_TRUNC"
    assert str(pto.CastMode.CAST_ODD) == "CastMode.CAST_ODD"


def test_op_type():
    assert str(pto.OpType.EQ) == "OpType.EQ"
    assert str(pto.OpType.NE) == "OpType.NE"
    assert str(pto.OpType.LT) == "OpType.LT"
    assert str(pto.OpType.LE) == "OpType.LE"
    assert str(pto.OpType.GT) == "OpType.GT"
    assert str(pto.OpType.GE) == "OpType.GE"


def test_out_type():
    assert str(pto.OutType.BOOL) == "OutType.BOOL"
    assert str(pto.OutType.BIT) == "OutType.BIT"
