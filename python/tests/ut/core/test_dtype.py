# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the DataType enum and related utility functions."""

import pytest
from pypto import ir
from pypto.ir import DataType


class TestDataTypeEnum:
    """Test DataType enumeration values and access patterns."""

    def test_enum_values_exist(self):
        """Test that all expected enum values are defined."""
        # Signed integers
        assert hasattr(DataType, "INT4")
        assert hasattr(DataType, "INT8")
        assert hasattr(DataType, "INT16")
        assert hasattr(DataType, "INT32")
        assert hasattr(DataType, "INT64")

        # Floating point
        assert hasattr(DataType, "FP8E4M3FN")
        assert hasattr(DataType, "FP8E5M2")
        assert hasattr(DataType, "FP16")
        assert hasattr(DataType, "FP32")
        assert hasattr(DataType, "BF16")

        # Hisilicon float
        assert hasattr(DataType, "HF4")
        assert hasattr(DataType, "HF8")

        # Unsigned integers
        assert hasattr(DataType, "UINT4")
        assert hasattr(DataType, "UINT8")
        assert hasattr(DataType, "UINT16")
        assert hasattr(DataType, "UINT32")
        assert hasattr(DataType, "UINT64")

        # Boolean
        assert hasattr(DataType, "BOOL")

    def test_enum_values_are_unique(self):
        """Test that all enum values have unique integer values."""
        values = [
            DataType.INT4,
            DataType.INT8,
            DataType.INT16,
            DataType.INT32,
            DataType.INT64,
            DataType.UINT4,
            DataType.UINT8,
            DataType.UINT16,
            DataType.UINT32,
            DataType.UINT64,
            DataType.FP4,
            DataType.FP8E4M3FN,
            DataType.FP8E5M2,
            DataType.FP16,
            DataType.FP32,
            DataType.BF16,
            DataType.HF4,
            DataType.HF8,
            DataType.BOOL,
        ]
        # Convert to int to compare underlying values
        int_values = [v.code() for v in values]
        assert len(int_values) == len(set(int_values)), "Enum values must be unique"

    def test_convenience_constants(self):
        """Test that convenience constants match DataType enum values."""
        assert ir.INT4 == DataType.INT4
        assert ir.INT8 == DataType.INT8
        assert ir.INT16 == DataType.INT16
        assert ir.INT32 == DataType.INT32
        assert ir.INT64 == DataType.INT64
        assert ir.UINT4 == DataType.UINT4
        assert ir.UINT8 == DataType.UINT8
        assert ir.UINT16 == DataType.UINT16
        assert ir.UINT32 == DataType.UINT32
        assert ir.UINT64 == DataType.UINT64
        assert ir.FP4 == DataType.FP4
        assert ir.FP8E4M3FN == DataType.FP8E4M3FN
        assert ir.FP8E5M2 == DataType.FP8E5M2
        assert ir.FP16 == DataType.FP16
        assert ir.FP32 == DataType.FP32
        assert ir.BF16 == DataType.BF16
        assert ir.HF4 == DataType.HF4
        assert ir.HF8 == DataType.HF8
        assert ir.BOOL == DataType.BOOL

    def test_convenience_constants_in_ir_namespace(self):
        """Test that convenience constants are accessible from pypto.ir module."""
        assert hasattr(ir, "INT4")
        assert hasattr(ir, "INT8")
        assert hasattr(ir, "INT16")
        assert hasattr(ir, "INT32")
        assert hasattr(ir, "INT64")
        assert hasattr(ir, "UINT4")
        assert hasattr(ir, "UINT8")
        assert hasattr(ir, "UINT16")
        assert hasattr(ir, "UINT32")
        assert hasattr(ir, "UINT64")
        assert hasattr(ir, "FP4")
        assert hasattr(ir, "FP8E4M3FN")
        assert hasattr(ir, "FP8E5M2")
        assert hasattr(ir, "FP16")
        assert hasattr(ir, "FP32")
        assert hasattr(ir, "BF16")
        assert hasattr(ir, "HF4")
        assert hasattr(ir, "HF8")
        assert hasattr(ir, "BOOL")
        assert ir.INT32 == DataType.INT32


class TestDataTypeBit:
    """Test GetBit() method."""

    def test_1bit_types(self):
        """Test data types that are 1 bit."""
        assert DataType.BOOL.get_bit() == 1

    def test_4bit_types(self):
        """Test data types that are 4 bits."""
        assert DataType.INT4.get_bit() == 4
        assert DataType.UINT4.get_bit() == 4
        assert DataType.FP4.get_bit() == 4
        assert DataType.HF4.get_bit() == 4

    def test_8bit_types(self):
        """Test data types that are 8 bits."""
        assert DataType.INT8.get_bit() == 8
        assert DataType.UINT8.get_bit() == 8
        assert DataType.FP8E4M3FN.get_bit() == 8
        assert DataType.FP8E5M2.get_bit() == 8
        assert DataType.HF8.get_bit() == 8

    def test_16bit_types(self):
        """Test data types that are 16 bits."""
        assert DataType.INT16.get_bit() == 16
        assert DataType.UINT16.get_bit() == 16
        assert DataType.FP16.get_bit() == 16
        assert DataType.BF16.get_bit() == 16

    def test_32bit_types(self):
        """Test data types that are 32 bits."""
        assert DataType.INT32.get_bit() == 32
        assert DataType.UINT32.get_bit() == 32
        assert DataType.FP32.get_bit() == 32

    def test_64bit_types(self):
        """Test data types that are 64 bits."""
        assert DataType.INT64.get_bit() == 64
        assert DataType.UINT64.get_bit() == 64


class TestDataTypeString:
    """Test ToString() method."""

    def test_signed_integer_strings(self):
        """Test string representation of signed integer types."""
        assert DataType.INT4.to_string() == "int4"
        assert DataType.INT8.to_string() == "int8"
        assert DataType.INT16.to_string() == "int16"
        assert DataType.INT32.to_string() == "int32"
        assert DataType.INT64.to_string() == "int64"

    def test_unsigned_integer_strings(self):
        """Test string representation of unsigned integer types."""
        assert DataType.UINT4.to_string() == "uint4"
        assert DataType.UINT8.to_string() == "uint8"
        assert DataType.UINT16.to_string() == "uint16"
        assert DataType.UINT32.to_string() == "uint32"
        assert DataType.UINT64.to_string() == "uint64"

    def test_floating_point_strings(self):
        """Test string representation of floating point types."""
        assert DataType.FP4.to_string() == "fp4"
        assert DataType.FP8E4M3FN.to_string() == "fp8e4m3fn"
        assert DataType.FP8E5M2.to_string() == "fp8e5m2"
        assert DataType.FP16.to_string() == "fp16"
        assert DataType.FP32.to_string() == "fp32"
        assert DataType.BF16.to_string() == "bfloat16"

    def test_hybrid_float_strings(self):
        """Test string representation of Hisilicon float types."""
        assert DataType.HF4.to_string() == "hf4"
        assert DataType.HF8.to_string() == "hf8"

    def test_bool_string(self):
        """Test string representation of boolean type."""
        assert DataType.BOOL.to_string() == "bool"


class TestDataTypePredicates:
    """Test type checking predicate methods."""

    def test_is_float(self):
        """Test is_float() correctly identifies floating point types."""
        # Floating point types
        assert DataType.FP4.is_float() is True
        assert DataType.FP8E4M3FN.is_float() is True
        assert DataType.FP8E5M2.is_float() is True
        assert DataType.FP16.is_float() is True
        assert DataType.FP32.is_float() is True
        assert DataType.BF16.is_float() is True
        assert DataType.HF4.is_float() is True
        assert DataType.HF8.is_float() is True

        # Non-floating point types
        assert DataType.INT8.is_float() is False
        assert DataType.INT32.is_float() is False
        assert DataType.UINT8.is_float() is False
        assert DataType.BOOL.is_float() is False

    def test_is_signed_int(self):
        """Test is_signed_int() correctly identifies signed integer types."""
        # Signed integer types
        assert DataType.INT4.is_signed_int() is True
        assert DataType.INT8.is_signed_int() is True
        assert DataType.INT16.is_signed_int() is True
        assert DataType.INT32.is_signed_int() is True
        assert DataType.INT64.is_signed_int() is True

        # Non-signed integer types
        assert DataType.UINT8.is_signed_int() is False
        assert DataType.FP32.is_signed_int() is False
        assert DataType.BOOL.is_signed_int() is False

    def test_is_unsigned_int(self):
        """Test is_unsigned_int() correctly identifies unsigned integer types."""
        # Unsigned integer types
        assert DataType.UINT4.is_unsigned_int() is True
        assert DataType.UINT8.is_unsigned_int() is True
        assert DataType.UINT16.is_unsigned_int() is True
        assert DataType.UINT32.is_unsigned_int() is True
        assert DataType.UINT64.is_unsigned_int() is True

        # Non-unsigned integer types
        assert DataType.INT8.is_unsigned_int() is False
        assert DataType.FP32.is_unsigned_int() is False
        assert DataType.BOOL.is_unsigned_int() is False

    def test_is_int(self):
        """Test is_int() correctly identifies any integer types."""
        # Integer types (both signed and unsigned)
        assert DataType.INT4.is_int() is True
        assert DataType.INT8.is_int() is True
        assert DataType.INT16.is_int() is True
        assert DataType.INT32.is_int() is True
        assert DataType.INT64.is_int() is True
        assert DataType.UINT4.is_int() is True
        assert DataType.UINT8.is_int() is True
        assert DataType.UINT16.is_int() is True
        assert DataType.UINT32.is_int() is True
        assert DataType.UINT64.is_int() is True

        # Non-integer types
        assert DataType.FP4.is_int() is False
        assert DataType.FP8E4M3FN.is_int() is False
        assert DataType.FP8E5M2.is_int() is False
        assert DataType.FP16.is_int() is False
        assert DataType.FP32.is_int() is False
        assert DataType.BF16.is_int() is False
        assert DataType.HF4.is_int() is False
        assert DataType.HF8.is_int() is False
        assert DataType.BOOL.is_int() is False

    def test_type_predicates_mutual_exclusion(self):
        """Test that signed, unsigned, and floating point are mutually exclusive."""
        all_types = [
            DataType.INT4,
            DataType.INT8,
            DataType.INT16,
            DataType.INT32,
            DataType.INT64,
            DataType.FP4,
            DataType.FP8E4M3FN,
            DataType.FP8E5M2,
            DataType.FP16,
            DataType.FP32,
            DataType.BF16,
            DataType.HF4,
            DataType.HF8,
            DataType.UINT4,
            DataType.UINT8,
            DataType.UINT16,
            DataType.UINT32,
            DataType.UINT64,
            DataType.BOOL,
        ]

        for dtype in all_types:
            # A type should not be both signed integer and unsigned integer
            if dtype.is_signed_int():
                assert not dtype.is_unsigned_int()

            # A type should not be both integer and floating point
            if dtype.is_int():
                assert not dtype.is_float()


class TestDataTypeIntegration:
    """Integration tests for DataType system."""

    all_types: list = [
        DataType.INT4,
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
        DataType.INT64,
        DataType.UINT4,
        DataType.UINT8,
        DataType.UINT16,
        DataType.UINT32,
        DataType.UINT64,
        DataType.FP4,
        DataType.FP8E4M3FN,
        DataType.FP8E5M2,
        DataType.FP16,
        DataType.FP32,
        DataType.BF16,
        DataType.HF4,
        DataType.HF8,
        DataType.BOOL,
    ]

    def test_all_types_have_bit_size(self):
        """Test that all data types have a valid bit size."""

        for dtype in self.all_types:
            bit_size = dtype.get_bit()
            assert bit_size > 0, f"Type {dtype.to_string()} should have positive bit size"
            assert bit_size in [1, 4, 8, 16, 32, 64], f"Type {dtype.to_string()} should have valid bit size"

    def test_all_types_have_string_representation(self):
        """Test that all data types have a valid string representation."""

        for dtype in self.all_types:
            string_repr = dtype.to_string()
            assert string_repr != "unknown", f"Type {dtype} should have valid string representation"
            assert len(string_repr) > 0, f"Type {dtype} should have non-empty string representation"

    def test_all_types_classified(self):
        """Test that all data types are classified as either integer, float, or bool."""

        for dtype in self.all_types:
            is_integer = dtype.is_int()
            is_floating = dtype.is_float()
            is_boolean = dtype == DataType.BOOL

            # Each type should be classified as at least one category
            # (bool is a special case that's neither int nor float in this classification)
            assert is_integer or is_floating or is_boolean, (
                f"Type {dtype.to_string()} should be classified as int, float, or bool"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
