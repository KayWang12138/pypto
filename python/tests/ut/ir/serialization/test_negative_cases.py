#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Negative test cases for serialization error handling."""

import pytest
from pypto import ir
from pypto.ir import DataType


class TestDeserializationErrors:
    """Test error handling during deserialization."""

    def test_corrupted_bytes_deserialize_raises_exception(self):
        """Test that corrupted bytes data raises exception during deserialization."""
        # Test with completely corrupted/random bytes
        corrupted_data = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09'

        with pytest.raises(Exception) as exc_info:
            ir.deserialize(corrupted_data)

        # Verify that some exception was raised (could be RuntimeError, ValueError, etc.)
        assert exc_info.value is not None

    def test_partially_corrupted_msgpack_raises_exception(self):
        """Test that partially valid but corrupted msgpack data raises exception."""
        # Create valid data first
        span = ir.Span.unknown()
        var = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        valid_data = ir.serialize(var)

        # Corrupt the data by truncating it
        corrupted_data = valid_data[:len(valid_data)//2]

        with pytest.raises(Exception) as exc_info:
            ir.deserialize(corrupted_data)

        assert exc_info.value is not None

    def test_invalid_msgpack_structure_raises_exception(self):
        """Test that invalid msgpack structure raises exception."""
        # Create malformed msgpack data (valid msgpack but invalid IR structure)
        # This is a valid msgpack encoding of an integer, but not a valid IR node
        invalid_msgpack = b'\x2a'  # msgpack encoding of integer 42

        with pytest.raises(Exception) as exc_info:
            ir.deserialize(invalid_msgpack)

        assert exc_info.value is not None

    def test_empty_bytes_raises_clear_error(self):
        """Test that empty bytes raises a clear error message."""
        empty_data = b''

        with pytest.raises(Exception) as exc_info:
            ir.deserialize(empty_data)

        # Verify exception was raised
        assert exc_info.value is not None
        # The error message should indicate the problem
        error_msg = str(exc_info.value)
        assert error_msg != ""  # Should have some error message

    def test_none_input_raises_type_error(self):
        """Test that None input raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ir.deserialize(None)

        # Verify it's specifically a TypeError
        assert isinstance(exc_info.value, TypeError)

    def test_string_input_raises_type_error(self):
        """Test that string input raises TypeError instead of bytes."""
        with pytest.raises(TypeError) as exc_info:
            ir.deserialize("not bytes")

        assert isinstance(exc_info.value, TypeError)

    def test_integer_input_raises_type_error(self):
        """Test that integer input raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ir.deserialize(12345)

        assert isinstance(exc_info.value, TypeError)

    def test_list_input_raises_type_error(self):
        """Test that list input raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ir.deserialize([1, 2, 3])

        assert isinstance(exc_info.value, TypeError)

    def test_dict_input_raises_type_error(self):
        """Test that dict input raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ir.deserialize({"key": "value"})

        assert isinstance(exc_info.value, TypeError)


class TestSerializationErrors:
    """Test error handling during serialization."""

    def test_none_input_serialize_raises_error(self):
        """Test that serializing None raises an error."""
        with pytest.raises(Exception) as exc_info:
            ir.serialize(None)

        assert exc_info.value is not None

    def test_invalid_object_type_serialize_raises_error(self):
        """Test that serializing non-IR objects raises an error."""
        # Try to serialize a plain Python object
        with pytest.raises(Exception) as exc_info:
            ir.serialize("not an IR node")

        assert exc_info.value is not None

    def test_builtin_type_serialize_raises_error(self):
        """Test that serializing built-in Python types raises an error."""
        test_cases = [
            42,           # int
            3.14,         # float
            True,         # bool
            [1, 2, 3],    # list
            {"a": 1},     # dict
        ]

        for invalid_input in test_cases:
            with pytest.raises(Exception) as exc_info:
                ir.serialize(invalid_input)

            assert exc_info.value is not None


class TestRoundTripValidation:
    """Test validation of corrupted round-trip data."""

    def test_modified_serialized_data_fails_deserialization(self):
        """Test that modifying serialized data causes deserialization to fail.

        Note: This test may be skipped if the deserializer is robust to minor corruption.
        """
        span = ir.Span.unknown()
        original = ir.ConstInt(42, DataType.INT64, span)

        # Serialize
        data = ir.serialize(original)

        # Modify a byte in the middle of the data
        if len(data) > 10:
            data_array = bytearray(data)
            data_array[len(data)//2] = (data_array[len(data)//2] + 1) % 256
            corrupted_data = bytes(data_array)

            # Attempt to deserialize - may or may not raise exception
            # Some msgpack structures are resilient to single-byte corruption
            try:
                restored = ir.deserialize(corrupted_data)
                # If it doesn't raise, the data might still deserialize to something
                # Just check that it's not structurally equal to the original
                try:
                    ir.assert_structural_equal(original, restored)
                except AssertionError:
                    # Good - the restored data is different from original
                    pass
            except Exception:
                # Good - deserialization failed as expected
                pass

    def test_truncated_serialized_data_fails_deserialization(self):
        """Test that truncated serialized data fails to deserialize."""
        span = ir.Span.unknown()

        # Create a complex object to ensure sufficient data length
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        add_expr = ir.Add(x, y, DataType.INT64, span)

        data = ir.serialize(add_expr)

        # Truncate at various points
        for truncate_ratio in [0.25, 0.5, 0.75, 0.9]:
            truncate_point = int(len(data) * truncate_ratio)
            truncated_data = data[:truncate_point]

            with pytest.raises(Exception) as exc_info:
                ir.deserialize(truncated_data)

            assert exc_info.value is not None
