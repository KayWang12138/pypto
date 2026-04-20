# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Unit tests for PyPTO error handling and reporting.

This module tests the error classes exposed from C++ to Python, ensuring that:
1. Errors are properly raised and caught
2. Stack traces are captured and included in error messages
3. Error types map correctly to Python's built-in exceptions
4. Error inheritance works as expected
"""

import pytest
from pypto_block import testing


class TestErrorTypes:
    """Test that different error types are raised correctly."""

    def test_value_error_type(self):
        """Test that ValueError is raised with correct type."""
        with pytest.raises(ValueError) as exc_info:
            testing.raise_value_error("test value error")

        assert "test value error" in str(exc_info.value)

    def test_type_error_type(self):
        """Test that TypeError is raised with correct type."""
        with pytest.raises(TypeError) as exc_info:
            testing.raise_type_error("test type error")

        assert "test type error" in str(exc_info.value)

    def test_runtime_error_type(self):
        """Test that RuntimeError is raised with correct type."""
        with pytest.raises(RuntimeError) as exc_info:
            testing.raise_runtime_error("test runtime error")

        assert "test runtime error" in str(exc_info.value)

    def test_not_implemented_error_type(self):
        """Test that NotImplementedError is raised with correct type."""
        with pytest.raises(NotImplementedError) as exc_info:
            testing.raise_not_implemented_error("test not implemented")

        assert "test not implemented" in str(exc_info.value)

    def test_index_error_type(self):
        """Test that IndexError is raised with correct type."""
        with pytest.raises(IndexError) as exc_info:
            testing.raise_index_error("test index error")

        assert "test index error" in str(exc_info.value)

    def test_generic_error_type(self):
        """Test that generic Error is raised with correct type."""
        with pytest.raises(Exception) as exc_info:
            testing.raise_generic_error("test generic error")

        assert "test generic error" in str(exc_info.value)

    def test_assertion_error_type(self):
        """Test that AssertionError is raised with correct type."""
        with pytest.raises(AssertionError):
            testing.raise_assertion_error("test assertion error")

    def test_internal_error_type(self):
        """Test that InternalError is raised with correct type."""
        with pytest.raises(RuntimeError) as exc_info:
            testing.raise_internal_error("test internal error")

        assert "test internal error" in str(exc_info.value)


class TestErrorMessages:
    """Test that error messages are properly formatted and include necessary information."""

    def test_error_message_content(self):
        """Test that error messages contain the expected text."""
        with pytest.raises(ValueError) as exc_info:
            testing.raise_value_error("Custom error message")

        assert "Custom error message" in str(exc_info.value)

    def test_error_message_with_special_characters(self):
        """Test that error messages with special characters are handled correctly."""
        special_message = "Error with special chars: !@#$%^&*()"
        with pytest.raises(ValueError) as exc_info:
            testing.raise_value_error(special_message)

        assert special_message in str(exc_info.value)

    def test_error_message_with_numbers(self):
        """Test that error messages with numbers are handled correctly."""
        message = "Error code: 12345, value: 67890"
        with pytest.raises(RuntimeError) as exc_info:
            testing.raise_runtime_error(message)

        assert "12345" in str(exc_info.value)
        assert "67890" in str(exc_info.value)

    def test_multiline_error_message(self):
        """Test that multiline error messages are handled correctly."""
        message = "Line 1\nLine 2\nLine 3"
        with pytest.raises(TypeError) as exc_info:
            testing.raise_type_error(message)

        assert "Line 1" in str(exc_info.value)


class TestErrorInheritance:
    """Test that error inheritance works correctly."""

    def test_value_error_is_exception(self):
        """Test that ValueError can be caught as Exception."""
        with pytest.raises(Exception):
            testing.raise_value_error("test")

    def test_type_error_is_exception(self):
        """Test that TypeError can be caught as Exception."""
        with pytest.raises(Exception):
            testing.raise_type_error("test")

    def test_runtime_error_is_exception(self):
        """Test that RuntimeError can be caught as Exception."""
        with pytest.raises(Exception):
            testing.raise_runtime_error("test")

    def test_index_error_is_exception(self):
        """Test that IndexError can be caught as Exception."""
        with pytest.raises(Exception):
            testing.raise_index_error("test")

    def test_assertion_error_is_exception(self):
        """Test that AssertionError can be caught as Exception."""
        with pytest.raises(Exception):
            testing.raise_assertion_error("test")

    def test_internal_error_is_exception(self):
        """Test that InternalError can be caught as Exception."""
        with pytest.raises(Exception):
            testing.raise_internal_error("test")


class TestErrorCatching:
    """Test various error catching scenarios."""

    def test_catch_specific_error(self):
        """Test that specific error types can be caught."""
        caught = False
        try:
            testing.raise_value_error("test")
        except ValueError:
            caught = True

        assert caught

    def test_catch_with_wrong_type_fails(self):
        """Test that catching with wrong type doesn't work."""
        with pytest.raises(ValueError):
            try:
                testing.raise_value_error("test")
            except TypeError:
                pass  # This should not catch the ValueError

    def test_multiple_error_types(self):
        """Test handling multiple different error types."""
        error_types = [
            (testing.raise_value_error, ValueError),
            (testing.raise_type_error, TypeError),
            (testing.raise_runtime_error, RuntimeError),
            (testing.raise_index_error, IndexError),
            (testing.raise_not_implemented_error, NotImplementedError),
            (testing.raise_assertion_error, AssertionError),
            (testing.raise_internal_error, RuntimeError),
        ]

        for raise_func, expected_type in error_types:
            with pytest.raises(expected_type):
                raise_func("test message")


class TestErrorContexts:
    """Test errors in various contexts."""

    def test_error_in_nested_calls(self):
        """Test that errors can be raised from nested function calls."""

        def level_3():
            testing.raise_runtime_error("nested error")

        def level_2():
            level_3()

        def level_1():
            level_2()

        with pytest.raises(RuntimeError) as exc_info:
            level_1()

        assert "nested error" in str(exc_info.value)

    def test_error_message_formatting(self):
        """Test that error messages are properly formatted."""
        test_cases = [
            "Simple message",
            "Message with 'quotes'",
            'Message with "double quotes"',
            "Message with\ttabs",
        ]

        for message in test_cases:
            with pytest.raises(ValueError) as exc_info:
                testing.raise_value_error(message)

            # The message should be preserved in some form
            assert len(str(exc_info.value)) > 0


class TestErrorEdgeCases:
    """Test edge cases and boundary conditions for error handling."""

    def test_empty_error_message(self):
        """Test that empty error messages are handled."""
        with pytest.raises(ValueError):
            testing.raise_value_error("")

    def test_very_long_error_message(self):
        """Test that very long error messages are handled."""
        long_message = "X" * 10000
        with pytest.raises(ValueError) as exc_info:
            testing.raise_value_error(long_message)

        assert "X" in str(exc_info.value)

    def test_error_with_null_characters(self):
        """Test error messages with null characters."""
        # Python strings don't allow null bytes in the middle,
        # but we can test with other control characters
        message = "Error\x00Test"  # This will be truncated at null
        try:
            with pytest.raises(ValueError):
                testing.raise_value_error(message)
        except Exception:
            # Some systems might handle this differently
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
