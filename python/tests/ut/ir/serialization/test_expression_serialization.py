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

    def test_add_serialization(self):
        """Test Add expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Add(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Add)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_sub_serialization(self):
        """Test Sub expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.FP32), span)
        y = ir.Var("y", ir.ScalarType(DataType.FP32), span)
        expr = ir.Sub(x, y, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Sub)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_mul_serialization(self):
        """Test Mul expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        expr = ir.Mul(x, y, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Mul)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_floor_div_serialization(self):
        """Test FloorDiv expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.FloorDiv(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.FloorDiv)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_floor_mod_serialization(self):
        """Test FloorMod expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.FloorMod(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.FloorMod)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_float_div_serialization(self):
        """Test FloatDiv expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.FP32), span)
        y = ir.Var("y", ir.ScalarType(DataType.FP32), span)
        expr = ir.FloatDiv(x, y, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.FloatDiv)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_pow_serialization(self):
        """Test Pow expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.FP32), span)
        y = ir.Var("y", ir.ScalarType(DataType.FP32), span)
        expr = ir.Pow(x, y, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Pow)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)


class TestBinaryComparisonOperators:
    """Test serialization of binary comparison operators."""

    def test_min_serialization(self):
        """Test Min expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Min(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Min)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_max_serialization(self):
        """Test Max expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Max(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Max)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_eq_serialization(self):
        """Test Eq expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Eq(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Eq)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_ne_serialization(self):
        """Test Ne expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Ne(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Ne)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_lt_serialization(self):
        """Test Lt expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.FP32), span)
        y = ir.Var("y", ir.ScalarType(DataType.FP32), span)
        expr = ir.Lt(x, y, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Lt)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_le_serialization(self):
        """Test Le expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Le(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Le)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_gt_serialization(self):
        """Test Gt expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Gt(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Gt)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_ge_serialization(self):
        """Test Ge expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        expr = ir.Ge(x, y, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Ge)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)


class TestBinaryLogicalOperators:
    """Test serialization of binary logical operators."""

    def test_and_serialization(self):
        """Test And expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.BOOL), span)
        y = ir.Var("y", ir.ScalarType(DataType.BOOL), span)
        expr = ir.And(x, y, DataType.BOOL, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.And)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_or_serialization(self):
        """Test Or expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.BOOL), span)
        y = ir.Var("y", ir.ScalarType(DataType.BOOL), span)
        expr = ir.Or(x, y, DataType.BOOL, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Or)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_xor_serialization(self):
        """Test Xor expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.BOOL), span)
        y = ir.Var("y", ir.ScalarType(DataType.BOOL), span)
        expr = ir.Xor(x, y, DataType.BOOL, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Xor)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)


class TestBinaryBitwiseOperators:
    """Test serialization of binary bitwise operators."""

    def test_bit_and_serialization(self):
        """Test BitAnd expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        expr = ir.BitAnd(x, y, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.BitAnd)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_bit_or_serialization(self):
        """Test BitOr expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        expr = ir.BitOr(x, y, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.BitOr)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_bit_xor_serialization(self):
        """Test BitXor expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        expr = ir.BitXor(x, y, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.BitXor)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_bit_shift_left_serialization(self):
        """Test BitShiftLeft expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        expr = ir.BitShiftLeft(x, y, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.BitShiftLeft)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_bit_shift_right_serialization(self):
        """Test BitShiftRight expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        expr = ir.BitShiftRight(x, y, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.BitShiftRight)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)


class TestUnaryOperators:
    """Test serialization of unary operators."""

    def test_abs_serialization(self):
        """Test Abs expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        expr = ir.Abs(x, DataType.INT64, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Abs)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_neg_serialization(self):
        """Test Neg expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.FP32), span)
        expr = ir.Neg(x, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Neg)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_not_serialization(self):
        """Test Not expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.BOOL), span)
        expr = ir.Not(x, DataType.BOOL, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Not)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_bit_not_serialization(self):
        """Test BitNot expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        expr = ir.BitNot(x, DataType.INT32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.BitNot)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_cast_serialization(self):
        """Test Cast expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        expr = ir.Cast(x, DataType.FP32, span)

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Cast)
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)


class TestTupleOperations:
    """Test serialization of tuple-related expressions."""

    def test_make_tuple_serialization(self):
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

    def test_tuple_get_item_serialization(self):
        """Test TupleGetItemExpr."""
        span = ir.Span.unknown()
        tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.FP32)])
        tuple_var = ir.Var("my_tuple", tuple_type, span)
        get_item = ir.TupleGetItemExpr(tuple_var, 1, span)

        data = ir.serialize(get_item)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.TupleGetItemExpr)
        assert restored.index == 1
        ir.assert_structural_equal(get_item, restored, enable_auto_mapping = True)

    def test_empty_tuple_serialization(self):
        """Test empty MakeTuple expression."""
        span = ir.Span.unknown()
        empty_tuple = ir.MakeTuple([], span)

        data = ir.serialize(empty_tuple)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.MakeTuple)
        assert len(restored.elements) == 0
        ir.assert_structural_equal(empty_tuple, restored)

    def test_nested_tuple_serialization(self):
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
