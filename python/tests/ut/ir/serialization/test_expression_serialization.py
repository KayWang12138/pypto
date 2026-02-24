# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for serialization of IR expression nodes.

This file tests all expression operators and tuple operations:
- Binary arithmetic operators (Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Pow)
- Binary comparison operators (Min, Max, Eq, Ne, Lt, Le, Gt, Ge)
- Binary logical operators (And, Or, Xor)
- Binary bitwise operators (BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight)
- Unary operators (Abs, Neg, Not, BitNot, Cast)
- Tuple operations (MakeTuple, TupleGetItemExpr)

Source: Extracted from test_IR_node_serialization.py to eliminate redundancy.
"""

import pytest
from pypto import ir
from pypto.ir import DataType


class TestBinaryArithmeticOperators:
    """Test serialization of binary arithmetic expression operators."""

    @staticmethod
    def test_add_serialization():
        """Test Add expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Add(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Add)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_sub_serialization():
        """Test Sub expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.FP32), span)
        y = ir.Var("y", ir.ScalarType(DataType.FP32), span)
        expr = ir.Sub(x, y, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Sub)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_mul_serialization():
        """Test Mul expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        expr = ir.Mul(x, y, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Mul)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_floor_div_serialization():
        """Test FloorDiv expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.FloorDiv(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.FloorDiv)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_floor_mod_serialization():
        """Test FloorMod expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.FloorMod(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.FloorMod)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_float_div_serialization():
        """Test FloatDiv expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.FP32), span)
        y = ir.Var("y", ir.ScalarType(DataType.FP32), span)
        expr = ir.FloatDiv(x, y, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.FloatDiv)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_pow_serialization():
        """Test Pow expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.FP32), span)
        y = ir.Var("y", ir.ScalarType(DataType.FP32), span)
        expr = ir.Pow(x, y, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Pow)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)


class TestBinaryComparisonOperators:
    """Test serialization of binary comparison operators."""

    @staticmethod
    def test_min_serialization():
        """Test Min expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Min(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Min)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_max_serialization():
        """Test Max expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Max(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Max)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_eq_serialization():
        """Test Eq expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Eq(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Eq)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_ne_serialization():
        """Test Ne expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Ne(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Ne)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_lt_serialization():
        """Test Lt expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.FP32), span)
        y = ir.Var("y", ir.ScalarType(DataType.FP32), span)
        expr = ir.Lt(x, y, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Lt)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_le_serialization():
        """Test Le expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Le(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Le)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_gt_serialization():
        """Test Gt expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Gt(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Gt)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_ge_serialization():
        """Test Ge expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Ge(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Ge)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)


class TestBinaryLogicalOperators:
    """Test serialization of binary logical operators."""

    @staticmethod
    def test_and_serialization():
        """Test And expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.BOOL), span)
        y = ir.Var("y", ir.ScalarType(DataType.BOOL), span)
        expr = ir.And(x, y, DataType.BOOL, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.And)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_or_serialization():
        """Test Or expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.BOOL), span)
        y = ir.Var("y", ir.ScalarType(DataType.BOOL), span)
        expr = ir.Or(x, y, DataType.BOOL, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Or)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_xor_serialization():
        """Test Xor expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.BOOL), span)
        y = ir.Var("y", ir.ScalarType(DataType.BOOL), span)
        expr = ir.Xor(x, y, DataType.BOOL, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Xor)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)


class TestBinaryBitwiseOperators:
    """Test serialization of binary bitwise operators."""

    @staticmethod
    def test_bit_and_serialization():
        """Test BitAnd expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        expr = ir.BitAnd(x, y, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.BitAnd)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_bit_or_serialization():
        """Test BitOr expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        expr = ir.BitOr(x, y, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.BitOr)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_bit_xor_serialization():
        """Test BitXor expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        expr = ir.BitXor(x, y, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.BitXor)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_bit_shift_left_serialization():
        """Test BitShiftLeft expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        expr = ir.BitShiftLeft(x, y, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.BitShiftLeft)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_bit_shift_right_serialization():
        """Test BitShiftRight expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        expr = ir.BitShiftRight(x, y, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.BitShiftRight)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)


class TestUnaryOperators:
    """Test serialization of unary operators."""

    @staticmethod
    def test_abs_serialization():
        """Test Abs expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        expr = ir.Abs(x, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Abs)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_neg_serialization():
        """Test Neg expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.FP32), span)
        expr = ir.Neg(x, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Neg)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_not_serialization():
        """Test Not expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.BOOL), span)
        expr = ir.Not(x, DataType.BOOL, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Not)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_bit_not_serialization():
        """Test BitNot expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        expr = ir.BitNot(x, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.BitNot)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_cast_serialization():
        """Test Cast expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        expr = ir.Cast(x, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Cast)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)


class TestTupleOperations:
    """Test serialization of tuple-related expressions."""

    @staticmethod
    def test_make_tuple_serialization():
        """Test MakeTuple expression."""
        span = ir.Span.unknown()
        val1 = ir.ConstInt(10, DataType.INT64, span)
        val2 = ir.ConstBool(True, span)
        make_tuple = ir.MakeTuple([val1, val2], span)

        data = ir.serialize(make_tuple)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.MakeTuple)
        assert len(restored.elements) == 2
        ir.assert_structural_equal(make_tuple, restored)

    @staticmethod
    def test_tuple_get_item_serialization():
        """Test TupleGetItemExpr."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple_var = ir.Var("my_tuple", tuple_type, span)
        get_item = ir.TupleGetItemExpr(tuple_var, 1, span)

        data = ir.serialize(get_item)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.TupleGetItemExpr)
        assert restored.index == 1
        ir.assert_structural_equal(get_item, restored, enable_auto_mapping=True)

    @staticmethod
    def test_empty_tuple_serialization():
        """Test empty MakeTuple expression."""
        span = ir.Span.unknown()
        empty_tuple = ir.MakeTuple([], span)

        data = ir.serialize(empty_tuple)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.MakeTuple)
        assert len(restored.elements) == 0
        ir.assert_structural_equal(empty_tuple, restored)

    @staticmethod
    def test_nested_tuple_serialization():
        """Test nested tuple structures."""
        span = ir.Span.unknown()

        # Inner tuple
        val1 = ir.ConstInt(1, DataType.INT64, span)
        val2 = ir.ConstInt(2, DataType.INT64, span)
        inner_tuple = ir.MakeTuple([val1, val2], span)

        # Outer tuple containing inner tuple
        val3 = ir.ConstInt(3, DataType.INT64, span)
        outer_tuple = ir.MakeTuple([inner_tuple, val3], span)

        data = ir.serialize(outer_tuple)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.MakeTuple)
        assert len(restored.elements) == 2
        assert isinstance(restored.elements[0], ir.MakeTuple)
        assert len(restored.elements[0].elements) == 2
        ir.assert_structural_equal(outer_tuple, restored)


class TestComplexExpressionTrees:
    """Test serialization of complex expression trees."""

    @staticmethod
    def test_deeply_nested_expression():
        """Test serializing deeply nested expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        # Build expression: ((((x + 1) + 2) + 3) + 4) + 5
        expr = x
        for i in range(1, 6):
            const = ir.ConstInt(i, DataType.INT64, span)
            expr = ir.Add(expr, const, DataType.INT64, span)

        # Serialize
        data = ir.serialize(expr)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None

        # Verify depth by counting Add nodes
        def count_adds(node):
            if isinstance(node, ir.Add):
                return 1 + count_adds(node.left) + count_adds(node.right)
            return 0

        assert count_adds(restored) == 5

        # Verify structural equality
        ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

    @staticmethod
    def test_complex_arithmetic_tree():
        """Test serializing complex arithmetic expression tree."""
        span = ir.Span.unknown()
        a = ir.Var("a", ir.ScalarType(DataType.FP32), span)
        b = ir.Var("b", ir.ScalarType(DataType.FP32), span)
        c = ir.Var("c", ir.ScalarType(DataType.FP32), span)
        d = ir.Var("d", ir.ScalarType(DataType.FP32), span)

        add1 = ir.Add(a, b, DataType.FP32, span)
        sub1 = ir.Sub(c, d, DataType.FP32, span)
        mul1 = ir.Mul(add1, sub1, DataType.FP32, span)

        mul2 = ir.Mul(a, c, DataType.FP32, span)
        mul3 = ir.Mul(b, d, DataType.FP32, span)
        add2 = ir.Add(mul2, mul3, DataType.FP32, span)

        div1 = ir.FloatDiv(mul1, add2, DataType.FP32, span)

        # Serialize
        data = ir.serialize(div1)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.FloatDiv)

        # Verify structural equality
        ir.assert_structural_equal(div1, restored, enable_auto_mapping=True)

    @staticmethod
    def test_mixed_operations_with_calls():
        """Test serializing expressions with mixed operations and calls."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.TensorType([], DataType.FP32), span)
        y = ir.Var("y", ir.TensorType([], DataType.FP32), span)

        # relu(x) + sigmoid(y)
        relu_op = ir.Op("relu")
        sigmoid_op = ir.Op("sigmoid")

        relu_call = ir.Call(relu_op, [x], span)
        sigmoid_call = ir.Call(sigmoid_op, [y], span)

        add_expr = ir.Add(relu_call, sigmoid_call, DataType.FP32, span)

        # Serialize
        data = ir.serialize(add_expr)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.Add)
        assert isinstance(restored.left, ir.Call)
        assert isinstance(restored.right, ir.Call)

        # Verify structural equality
        ir.assert_structural_equal(add_expr, restored, enable_auto_mapping=True)

