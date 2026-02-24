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
"""Tests for serialization of basic IR types."""

import pytest
from pypto import ir
from pypto.ir import DataType


class TestBasicTypesSerialization:
    """Test serialization of basic IR types."""

    def test_scalar_type_serialization(self):
        """Test serializing ScalarType through a Var node."""
        span = ir.Span.unknown()
        scalar_type = ir.ScalarType(DataType.INT64)
        var = ir.Var("x", scalar_type, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None
        assert isinstance(data, bytes)

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.ScalarType)
        assert restored.type.dtype == DataType.INT64

        # Verify structural equality
        ir.assert_structural_equal(var, restored, enable_auto_mapping = True)

    def test_tensor_type_serialization(self):
        """Test serializing TensorType."""
        span = ir.Span.unknown()
        dim1 = ir.ConstInt(10, DataType.INT64, span)
        dim2 = ir.ConstInt(20, DataType.INT64, span)
        tensor_type = ir.TensorType([dim1, dim2], DataType.FP32)
        var = ir.Var("tensor", tensor_type, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.TensorType)
        assert restored.type.dtype == DataType.FP32
        assert len(restored.type.shape) == 2

        # Verify structural equality
        ir.assert_structural_equal(var, restored, enable_auto_mapping = True)

    def test_tile_type_serialization(self):
        """Test serializing TileType."""
        span = ir.Span.unknown()
        dim1 = ir.ConstInt(8, DataType.INT64, span)
        dim2 = ir.ConstInt(16, DataType.INT64, span)
        tile_type = ir.TileType([dim1, dim2], DataType.FP16)
        var = ir.Var("tile", tile_type, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.TileType)
        assert restored.type.dtype == DataType.FP16

        # Verify structural equality
        ir.assert_structural_equal(var, restored, enable_auto_mapping = True)

    def test_tuple_type_serialization(self):
        """Test serializing TupleType."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([
            ir.ScalarType(DataType.INT64),
            ir.ScalarType(DataType.FP32),
            ir.TensorType([], DataType.FP16)
        ])
        var = ir.Var("tuple_var", tuple_type, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.TupleType)
        assert len(restored.type.types) == 3

        # Verify structural equality
        ir.assert_structural_equal(var, restored, enable_auto_mapping = True)

    def test_unknown_type_serialization(self):
        """Test serializing UnknownType."""
        span = ir.Span.unknown()
        unknown_type = ir.UnknownType()
        var = ir.Var("unknown_var", unknown_type, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.UnknownType)

        # Verify structural equality
        ir.assert_structural_equal(var, restored, enable_auto_mapping = True)

    def test_multiple_dtypes_serialization(self):
        """Test serializing different data types."""
        span = ir.Span.unknown()
        dtypes = [
            DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64,
            DataType.UINT8, DataType.UINT16, DataType.UINT32, DataType.UINT64,
            DataType.FP16, DataType.FP32, DataType.BF16,
            DataType.BOOL
        ]

        for dtype in dtypes:
            scalar_type = ir.ScalarType(dtype)
            var = ir.Var(f"var_{dtype.to_string()}", scalar_type, span)

            # Serialize
            data = ir.serialize(var)
            assert data is not None

            # Deserialize
            restored = ir.deserialize(data)
            assert restored is not None
            assert isinstance(restored, ir.Var)
            assert restored.type.dtype == dtype

            # Verify structural equality
            ir.assert_structural_equal(var, restored, enable_auto_mapping = True)


class TestSpanSerialization:
    """Test serialization of Span information."""

    def test_span_serialization(self):
        """Test that Span information is preserved through serialization."""
        span = ir.Span("test.py", 10, 5, 10, 20)
        var = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert restored.span.filename == "test.py"
        assert restored.span.begin_line == 10
        assert restored.span.begin_column == 5
        assert restored.span.end_line == 10
        assert restored.span.end_column == 20

    def test_unknown_span_serialization(self):
        """Test serialization with unknown span."""
        span = ir.Span.unknown()
        var = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert restored.span.is_valid() is False
        assert restored.span.filename == ""


class TestCallOpSerialization:
    """Test serialization of IR operators."""

    def test_call_op_serialization(self):
        """Test serializing Op through a Call expression."""
        span = ir.Span.unknown()
        op = ir.Op("add")
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        call = ir.Call(op, [x, y], span)

        # Serialize
        data = ir.serialize(call)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.Call)
        assert restored.op.name == "add"

        # Verify structural equality
        ir.assert_structural_equal(call, restored, enable_auto_mapping = True)

    def test_multiple_call_ops_serialization(self):
        """Test serializing different ops."""
        span = ir.Span.unknown()
        op_names = ["add", "mul", "sub", "div", "matmul", "relu", "softmax"]

        for op_name in op_names:
            op = ir.Op(op_name)
            x = ir.Var("x", ir.ScalarType(DataType.FP32), span)
            call = ir.Call(op, [x], span)

            # Serialize
            data = ir.serialize(call)
            assert data is not None

            # Deserialize
            restored = ir.deserialize(data)
            assert restored is not None
            assert isinstance(restored, ir.Call)
            assert restored.op.name == op_name

            # Verify structural equality
            ir.assert_structural_equal(call, restored, enable_auto_mapping = True)
    
    def test_call_with_kwarg_serialization(self):
        """Test Call with single keyword argument.

        Example: reduce_sum(x, axis=1)
        """
        span = ir.Span.unknown()
        op = ir.Op("reduce_sum")
        x = ir.Var("x", ir.TensorType([2, 3], DataType.FP32), span)

        # Call with kwargs: reduce_sum(x, axis=1)
        # Note: kwargs should be primitive Python values, not Expr objects
        call = ir.Call(op, [x], {"axis": 1}, span)

        # Serialize and deserialize
        data = ir.serialize(call)
        restored = ir.deserialize(data)

        # Verify structure
        assert isinstance(restored, ir.Call)
        assert restored.op.name == "reduce_sum"
        assert len(restored.args) == 1

        # Verify kwargs
        assert hasattr(restored, 'kwargs'), "Call should have kwargs attribute"
        assert isinstance(restored.kwargs, dict), "kwargs should be a dict"
        assert "axis" in restored.kwargs, "axis should be in kwargs"
        assert restored.kwargs["axis"] == 1, "axis value should be 1"

        ir.assert_structural_equal(call, restored, enable_auto_mapping = True)

    def test_call_with_args_and_kwargs_serialization(self):
        """Test Call with both positional and keyword arguments.

        Example: conv2d(input, weight, stride=1, padding=0)
        """
        span = ir.Span.unknown()
        op = ir.Op("conv2d")

        # Positional arguments
        input_var = ir.Var("input", ir.TensorType([1, 3, 224, 224], DataType.FP32), span)
        weight_var = ir.Var("weight", ir.TensorType([64, 3, 3, 3], DataType.FP32), span)

        # Keyword arguments (use primitive Python values, not Expr objects)
        call = ir.Call(
            op,
            [input_var, weight_var],
            {"stride": 1, "padding": 0},
            span
        )

        # Serialize and deserialize
        data = ir.serialize(call)
        restored = ir.deserialize(data)

        # Verify structure
        assert isinstance(restored, ir.Call)
        assert restored.op.name == "conv2d"
        assert len(restored.args) == 2
        assert len(restored.kwargs) == 2

        # Verify args
        assert isinstance(restored.args[0], ir.Var)
        assert isinstance(restored.args[1], ir.Var)
        assert restored.args[0].name == "input"
        assert restored.args[1].name == "weight"

        # Verify kwargs
        assert "stride" in restored.kwargs
        assert "padding" in restored.kwargs
        assert restored.kwargs["stride"] == 1, "stride should be 1"
        assert restored.kwargs["padding"] == 0, "padding should be 0"

        ir.assert_structural_equal(call, restored, enable_auto_mapping = True)
