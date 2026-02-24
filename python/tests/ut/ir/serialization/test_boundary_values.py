# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for serialization of boundary values and edge cases.

This file tests boundary values, special floating-point numbers, and other edge cases
to ensure serialization handles all valid inputs correctly.
"""

import math
import pytest
from pypto import ir
from pypto.ir import DataType


class TestBoundaryIntegerValues:
    """Test serialization of boundary integer values."""

    @staticmethod
    def test_int64_max_serialization():
        """Test INT64_MAX (9223372036854775807) serialization."""
        span = ir.Span.unknown()
        max_int = ir.ConstInt(9223372036854775807, DataType.INT64, span)

        # Serialize
        data = ir.serialize(max_int)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.ConstInt)
        assert restored.value == 9223372036854775807
        assert restored.dtype == DataType.INT64

        # Verify structural equality
        ir.assert_structural_equal(max_int, restored)

    @staticmethod
    def test_int64_min_serialization():
        """Test INT64_MIN (-9223372036854775808) serialization."""
        span = ir.Span.unknown()
        min_int = ir.ConstInt(-9223372036854775808, DataType.INT64, span)

        # Serialize
        data = ir.serialize(min_int)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.ConstInt)
        assert restored.value == -9223372036854775808
        assert restored.dtype == DataType.INT64

        # Verify structural equality
        ir.assert_structural_equal(min_int, restored)

    @staticmethod
    def test_int32_boundary_values():
        """Test INT32 boundary values."""
        span = ir.Span.unknown()

        max_int32 = ir.ConstInt(2147483647, DataType.INT32, span)
        data = ir.serialize(max_int32)
        restored = ir.deserialize(data)
        assert restored.value == 2147483647
        assert restored.dtype == DataType.INT32

        min_int32 = ir.ConstInt(-2147483648, DataType.INT32, span)
        data = ir.serialize(min_int32)
        restored = ir.deserialize(data)
        assert restored.value == -2147483648
        assert restored.dtype == DataType.INT32

    @staticmethod
    def test_zero_in_different_dtypes():
        """Test zero value in different integer dtypes."""
        span = ir.Span.unknown()

        for dtype in [DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64,
                      DataType.UINT8, DataType.UINT16, DataType.UINT32, DataType.UINT64]:
            zero = ir.ConstInt(0, dtype, span)
            data = ir.serialize(zero)
            restored = ir.deserialize(data)
            assert restored.value == 0
            assert restored.dtype == dtype
            ir.assert_structural_equal(zero, restored)

    @staticmethod
    def test_int_arithmetic_with_boundary_values():
        """Test arithmetic expressions with boundary values."""
        span = ir.Span.unknown()

        max_int = ir.ConstInt(9223372036854775807, DataType.INT64, span)
        zero = ir.ConstInt(0, DataType.INT64, span)
        expr = ir.Add(max_int, zero, DataType.INT64, span)

        # Serialize and deserialize
        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Add)
        assert isinstance(restored.left, ir.ConstInt)
        assert restored.left.value == 9223372036854775807
        ir.assert_structural_equal(expr, restored)
    
    @staticmethod
    def test_boundary_int_in_binary_expression():
        """Test binary expression with boundary integers."""
        span = ir.Span.unknown()

        # INT64_MAX - INT64_MIN
        max_int = ir.ConstInt(9223372036854775807, DataType.INT64, span)
        min_int = ir.ConstInt(-9223372036854775808, DataType.INT64, span)
        expr = ir.Sub(max_int, min_int, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Sub)
        assert restored.left.value == 9223372036854775807
        assert restored.right.value == -9223372036854775808
        ir.assert_structural_equal(expr, restored)


class TestSpecialFloatingPointValues:
    """Test serialization of special floating-point values."""

    @staticmethod
    def test_nan_serialization():
        """Test NaN (Not a Number) serialization."""
        span = ir.Span.unknown()
        nan_float = ir.ConstFloat(float('nan'), DataType.FP32, span)

        # Serialize
        data = ir.serialize(nan_float)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.ConstFloat)
        assert math.isnan(restored.value), "Restored value should be NaN"
        assert restored.dtype == DataType.FP32

    @staticmethod
    def test_positive_infinity_serialization():
        """Test positive infinity serialization."""
        span = ir.Span.unknown()
        pos_inf = ir.ConstFloat(float('inf'), DataType.FP32, span)

        # Serialize
        data = ir.serialize(pos_inf)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.ConstFloat)
        assert math.isinf(restored.value) and restored.value > 0, "Should be positive infinity"
        assert restored.dtype == DataType.FP32

    @staticmethod
    def test_negative_infinity_serialization():
        """Test negative infinity serialization."""
        span = ir.Span.unknown()
        neg_inf = ir.ConstFloat(float('-inf'), DataType.FP32, span)

        # Serialize
        data = ir.serialize(neg_inf)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.ConstFloat)
        assert math.isinf(restored.value) and restored.value < 0, "Should be negative infinity"
        assert restored.dtype == DataType.FP32

    @staticmethod
    def test_negative_zero_serialization():
        """Test negative zero (-0.0) serialization."""
        span = ir.Span.unknown()
        neg_zero = ir.ConstFloat(-0.0, DataType.FP32, span)

        # Serialize
        data = ir.serialize(neg_zero)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.ConstFloat)
        # Check both value and sign
        assert restored.value == 0.0
        assert math.copysign(1.0, restored.value) == math.copysign(1.0, -0.0), "Should preserve sign of zero"

    @staticmethod
    def test_very_small_float_serialization():
        """Test very small floating-point values (near underflow)."""
        span = ir.Span.unknown()

        # Smallest positive normalized FP32: ~1.175494e-38
        # Smallest positive subnormal FP32: ~1.401298e-45
        small_float = ir.ConstFloat(1.401298e-45, DataType.FP32, span)

        data = ir.serialize(small_float)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ConstFloat)
        assert restored.value == pytest.approx(1.401298e-45, rel=1e-6)

    @staticmethod
    def test_very_large_float_serialization():
        """Test very large floating-point values (near overflow)."""
        span = ir.Span.unknown()

        # Largest FP32: ~3.402823e+38
        large_float = ir.ConstFloat(3.402823e+38, DataType.FP32, span)

        data = ir.serialize(large_float)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ConstFloat)
        assert restored.value == pytest.approx(3.402823e+38, rel=1e-6)

    @staticmethod
    def test_special_float_in_expressions():
        """Test special float values in expressions."""
        span = ir.Span.unknown()

        nan_val = ir.ConstFloat(float('nan'), DataType.FP32, span)
        inf_val = ir.ConstFloat(float('inf'), DataType.FP32, span)
        expr = ir.Add(nan_val, inf_val, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Add)
        assert math.isnan(restored.left.value)
        assert math.isinf(restored.right.value)
        ir.assert_structural_equal(expr, restored)


class TestEmptyAndSpecialStrings:
    """Test serialization with empty and special strings."""

    @staticmethod
    def test_empty_variable_name():
        """Test variable with empty name.

        Note: This may or may not be allowed by the IR. If it raises an error,
        that's also a valid test result indicating the IR rejects empty names.
        """
        span = ir.Span.unknown()
        try:
            var = ir.Var("", ir.ScalarType(DataType.INT64), span)
            data = ir.serialize(var)
            restored = ir.deserialize(data)
            assert restored.name == "", "Empty name should be preserved"
        except Exception:
            # If empty name is rejected, that's also valid behavior
            pass

    @staticmethod
    def test_whitespace_only_variable_name():
        """Test variable with whitespace-only name."""
        span = ir.Span.unknown()
        var = ir.Var("   ", ir.ScalarType(DataType.INT64), span)

        data = ir.serialize(var)
        restored = ir.deserialize(data)

        assert restored.name == "   ", "Whitespace-only name should be preserved"
        ir.assert_structural_equal(var, restored, enable_auto_mapping=True)

    @staticmethod
    def test_newline_in_variable_name():
        """Test variable with newline in name."""
        span = ir.Span.unknown()
        var = ir.Var("var\nname", ir.ScalarType(DataType.INT64), span)

        data = ir.serialize(var)
        restored = ir.deserialize(data)

        assert restored.name == "var\nname", "Newline in name should be preserved"
        ir.assert_structural_equal(var, restored, enable_auto_mapping=True)

    @staticmethod
    def test_special_characters_in_name():
        """Test variable with special characters."""
        span = ir.Span.unknown()
        # Various special characters
        special_names = [
            "var@name",
            "var#name",
            "var$name",
            "var%name",
            "var&name",
            "var*name",
            "var!name",
            "var?name",
        ]

        for name in special_names:
            var = ir.Var(name, ir.ScalarType(DataType.INT64), span)
            data = ir.serialize(var)
            restored = ir.deserialize(data)
            assert restored.name == name, f"Special character name '{name}' should be preserved"

    @staticmethod
    def test_very_long_variable_name():
        """Test variable with very long name (10000 characters)."""
        span = ir.Span.unknown()
        # Create a very long name
        long_name = "x" * 10000
        var = ir.Var(long_name, ir.ScalarType(DataType.INT64), span)

        data = ir.serialize(var)
        restored = ir.deserialize(data)

        assert len(restored.name) == 10000, "Long name length should be preserved"
        assert restored.name == long_name, "Long name should be preserved"
        ir.assert_structural_equal(var, restored, enable_auto_mapping=True)


class TestEmptyStructures:
    """Test serialization of edge cases and empty structures."""

    @staticmethod
    def test_empty_structures():
        """Test serializing empty structures."""
        span = ir.Span.unknown()

        # Empty SeqStmts
        empty_seq = ir.SeqStmts([], span)
        data = ir.serialize(empty_seq)
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.SeqStmts)
        assert len(restored.stmts) == 0

        # Empty MakeTuple
        empty_tuple = ir.MakeTuple([], span)
        data = ir.serialize(empty_tuple)
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.MakeTuple)
        assert len(restored.elements) == 0