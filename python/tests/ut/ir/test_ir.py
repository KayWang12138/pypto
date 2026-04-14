# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pytest
import pypto
from pypto import ir


def test_error_types():
    with pytest.raises(ValueError):
        pypto.raise_error("ValueError", "test value error")
    with pytest.raises(TypeError):
        pypto.raise_error("TypeError", "test type error")
    with pytest.raises(RuntimeError):
        pypto.raise_error("RuntimeError", "test runtime error")
    with pytest.raises(NotImplementedError):
        pypto.raise_error("NotImplementedError", "test not implemented error")
    with pytest.raises(IndexError):
        pypto.raise_error("IndexError", "test index error")
    with pytest.raises(AssertionError):
        pypto.raise_error("AssertionError", "test assertion error")
    with pytest.raises(pypto.InternalError):
        pypto.raise_error("InternalError", "test internal error")


def test_dtypes():
    dtypes = [
        # bits, is_signed, is_unsigned, is_float, name, c_type
        (ir.DT_BOOL, 8, False, False, False, "bool", "bool"),
        (ir.DT_INT4, 4, True, False, False, "int4", "unknown"),
        (ir.DT_INT8, 8, True, False, False, "int8", "int8_t"),
        (ir.DT_INT16, 16, True, False, False, "int16", "int16_t"),
        (ir.DT_INT32, 32, True, False, False, "int32", "int32_t"),
        (ir.DT_INT64, 64, True, False, False, "int64", "int64_t"),
        (ir.DT_INDEX, 64, True, False, False, "index", "int64_t"),
        (ir.DT_UINT4, 4, False, True, False, "uint4", "unknown"),
        (ir.DT_UINT8, 8, False, True, False, "uint8", "uint8_t"),
        (ir.DT_UINT16, 16, False, True, False, "uint16", "uint16_t"),
        (ir.DT_UINT32, 32, False, True, False, "uint32", "uint32_t"),
        (ir.DT_UINT64, 64, False, True, False, "uint64", "uint64_t"),
        (ir.DT_FP4, 4, False, False, True, "fp4", "unknown"),
        (ir.DT_FP8E4M3FN, 8, False, False, True, "fp8e4m3fn", "unknown"),
        (ir.DT_FP8E5M2, 8, False, False, True, "fp8e5m2", "unknown"),
        (ir.DT_FP16, 16, False, False, True, "fp16", "half"),
        (ir.DT_FP32, 32, False, False, True, "fp32", "float"),
        (ir.DT_BF16, 16, False, False, True, "bfloat16", "bfloat16"),
        (ir.DT_HF4, 4, False, False, True, "hf4", "unknown"),
        (ir.DT_HF8, 8, False, False, True, "hf8", "unknown"),
    ]
    for dtype, *info in dtypes:
        assert info == [dtype.bits(), dtype.is_signed(),
                        dtype.is_unsigned(), dtype.is_float(),
                        str(dtype), dtype.c_type()]


def test_span():
    span = ir.Span("span", 1, 2, 3, 4)
    assert span.filename == "span"
    assert span.begin_line == 1
    assert span.begin_column == 2
    assert span.end_line == 3
    assert span.end_column == 4
    assert not ir.Span.is_unknown(span)

    span = ir.Span.unknown()
    span1 = ir.Span.unknown()
    print(id(span), id(span1))
    # assert ir.Span.is_unknown(span)

    span = ir.Span("span", 1, 2)
    assert span.filename == "span"
    assert span.begin_line == 1
    assert span.begin_column == 2
    assert span.end_line == -1
    assert span.end_column == -1
    assert not ir.Span.is_unknown(span)


def test_logging(capfd):
    pypto.set_log_level(pypto.LogLevel.INFO)
    assert pypto.get_log_level() == pypto.LogLevel.INFO

    pypto.log_debug("test debug message")
    pypto.log_info("test info message")
    pypto.log_warn("test warn message")
    pypto.log_error("test error message")
    pypto.log_event("test event message")
    pypto.log_fatal("test fatal message")

    captured = capfd.readouterr()
    assert "event" in captured.err
    assert "fatal" in captured.err
    assert "error" in captured.err
    assert "warn" in captured.err
    assert "info" in captured.err
    assert "debug" not in captured.err


def test_check():
    with pytest.raises(ValueError, match="test check message"):
        pypto.check(False, "test check message")
    pypto.check(True, "test check message")

    with pytest.raises(pypto.InternalError, match="test internal check message"):
        pypto.internal_check(False, "test internal check message")
    pypto.internal_check(True, "test internal check message")
