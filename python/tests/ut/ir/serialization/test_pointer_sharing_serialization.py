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
"""Tests for pointer sharing in IR serialization.

This file tests that IR node references are preserved during serialization/deserialization.
"""

import pytest
from pypto import ir
from pypto.ir import DataType


def _make_const_int(value, span=None):
    """Helper to create ConstInt expression."""
    if span is None:
        span = ir.Span.unknown()
    return ir.ConstInt(value, DataType.INT64, span)


def _make_expr_list(values, span=None):
    """Helper to create list of Expr from list of ints."""
    if span is None:
        span = ir.Span.unknown()
    return [_make_const_int(v, span) for v in values]


class TestPointerSharing:
    """Test pointer sharing (id/ref mechanism) in IR serialization.

    When the same IR node is referenced multiple times in a DAG, the serializer
    should store it once with an id and use ref for subsequent references.
    After deserialization, pointer sharing must be preserved.

    This tests the critical invariant: DAG structure is preserved, not converted to a tree.
    """

    def test_var_shared_in_expression(self):
        """Test basic Var pointer sharing: x + x.

        This is the most fundamental pointer sharing case.
        """
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        # x + x: same Var referenced twice
        add_expr = ir.Add(x, x, DataType.INT64, span)

        # Verify sharing before serialization
        assert add_expr.left is add_expr.right

        # Serialize and deserialize
        data = ir.serialize(add_expr)
        restored = ir.deserialize(data)

        # Verify sharing is preserved
        assert isinstance(restored, ir.Add)
        assert restored.left is restored.right, "Var should be shared (same object)"
        assert restored.left.name == "x"

        ir.assert_structural_equal(add_expr, restored, enable_auto_mapping = True)

    def test_constant_shared(self):
        """Test constant pointer sharing: (x + 0) + (y + 0)."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)

        # Shared constant
        zero = ir.ConstInt(0, DataType.INT64, span)

        # Use zero multiple times
        add1 = ir.Add(x, zero, DataType.INT64, span)
        add2 = ir.Add(y, zero, DataType.INT64, span)
        add3 = ir.Add(add1, add2, DataType.INT64, span)

        # Verify zero is shared
        assert add1.right is add2.right

        # Serialize and deserialize
        data = ir.serialize(add3)
        restored = ir.deserialize(data)

        # Verify pointer sharing is preserved
        assert isinstance(restored, ir.Add)
        left_add = restored.left
        right_add = restored.right

        assert isinstance(left_add, ir.Add)
        assert isinstance(right_add, ir.Add)

        # The two references to zero should be the same object
        assert left_add.right is right_add.right, "ConstInt 0 should be shared"
        assert isinstance(left_add.right, ir.ConstInt)
        assert left_add.right.value == 0

        ir.assert_structural_equal(add3, restored, enable_auto_mapping = True)

    def test_complex_dag_with_multiple_shared_nodes(self):
        """Test a DAG with multiple shared nodes at different levels.

        Structure:
            d = (a + b) + (a + c)
        Here 'a' is shared twice.
        """
        span = ir.Span.unknown()
        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        b = ir.Var("b", ir.ScalarType(DataType.INT64), span)
        c = ir.Var("c", ir.ScalarType(DataType.INT64), span)

        # Build (a + b) + (a + c)
        ab = ir.Add(a, b, DataType.INT64, span)
        ac = ir.Add(a, c, DataType.INT64, span)
        d = ir.Add(ab, ac, DataType.INT64, span)

        # Verify 'a' is shared
        assert ab.left is ac.left

        # Serialize and deserialize
        data = ir.serialize(d)
        restored = ir.deserialize(data)

        # Verify pointer sharing is preserved
        assert isinstance(restored, ir.Add)
        restored_ab = restored.left
        restored_ac = restored.right

        assert isinstance(restored_ab, ir.Add)
        assert isinstance(restored_ac, ir.Add)

        # The 'a' Var should be the same object in both sub-expressions
        assert restored_ab.left is restored_ac.left, "'a' should be shared"
        assert restored_ab.left.name == "a"

        ir.assert_structural_equal(d, restored, enable_auto_mapping = True)

    def test_deeply_nested_shared_node(self):
        """Test pointer sharing in a deeply nested expression tree.

        Structure: ((x + x) + (x + x)) + ((x + x) + (x + x))
        'x' is shared many times at different depths.
        """
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        # Build the tree
        xx = ir.Add(x, x, DataType.INT64, span)
        xxxx_left = ir.Add(xx, xx, DataType.INT64, span)
        xxxx_right = ir.Add(xx, xx, DataType.INT64, span)
        result = ir.Add(xxxx_left, xxxx_right, DataType.INT64, span)

        # Verify x is shared
        assert xx.left is xx.right

        # Verify xx is shared
        assert xxxx_left.left is xxxx_left.right
        assert xxxx_right.left is xxxx_right.right
        # And across the top level
        assert xxxx_left.left is xxxx_right.left

        # Serialize and deserialize
        data = ir.serialize(result)
        restored = ir.deserialize(data)

        # Verify structure
        assert isinstance(restored, ir.Add)
        restored_left = restored.left
        restored_right = restored.right

        # Check xx is shared
        assert restored_left.left is restored_left.right, "xx should be shared on left"
        assert restored_right.left is restored_right.right, "xx should be shared on right"

        # Check xx is shared across branches
        assert restored_left.left is restored_right.left, "xx should be shared across top level"

        # Check x is shared within xx
        restored_xx = restored_left.left
        assert restored_xx.left is restored_xx.right, "x should be shared within xx"

        ir.assert_structural_equal(result, restored, enable_auto_mapping = True)

    def test_shared_type(self):
        """Test that Types are properly shared.

        When the same Type object is used in multiple Vars, it should be shared.
        Note: Type sharing may not always be preserved after deserialization,
        but structural equality should still hold.
        """
        span = ir.Span.unknown()

        # Create a shared type
        int64_type = ir.ScalarType(DataType.INT64)

        # Use the same type for multiple vars
        x = ir.Var("x", int64_type, span)
        y = ir.Var("y", int64_type, span)

        # Verify type is shared before serialization
        assert x.type is y.type

        # Create an expression using both
        add_expr = ir.Add(x, y, DataType.INT64, span)

        # Serialize and deserialize
        data = ir.serialize(add_expr)
        restored = ir.deserialize(data)

        # Verify structure is preserved (types may not be shared after deserialization)
        assert isinstance(restored, ir.Add)
        restored_x = restored.left
        restored_y = restored.right

        assert isinstance(restored_x.type, ir.ScalarType)
        assert isinstance(restored_y.type, ir.ScalarType)

        ir.assert_structural_equal(add_expr, restored, enable_auto_mapping = True)

    def test_cyclic_reference_prevention(self):
        """Test that the serializer handles potential cycles correctly.

        Note: IR nodes themselves should not have cycles (it's a DAG),
        but this test ensures the serializer's id/ref mechanism doesn't create issues.
        """
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        # Create a structure where the same node appears multiple times
        # This creates a DAG (not a cycle)
        add1 = ir.Add(x, x, DataType.INT64, span)
        add2 = ir.Add(add1, add1, DataType.INT64, span)

        # Verify sharing
        assert add2.left is add2.right
        assert add1.left is add1.right

        # Serialize and deserialize
        data = ir.serialize(add2)
        restored = ir.deserialize(data)

        # Verify all sharing is preserved
        assert isinstance(restored, ir.Add)
        assert restored.left is restored.right, "add1 should be shared"

        restored_add1 = restored.left
        assert restored_add1.left is restored_add1.right, "x should be shared"

        ir.assert_structural_equal(add2, restored, enable_auto_mapping = True)

    def test_shared_op(self):
        """Test Op node sharing in Call expressions.

        Note: Op sharing may not always be preserved after deserialization,
        but structural equality should still hold.
        """
        span = ir.Span.unknown()

        # Shared Op
        relu_op = ir.Op("relu")

        x = ir.Var("x", ir.TensorType([10], DataType.FP32), span)
        y = ir.Var("y", ir.TensorType([10], DataType.FP32), span)

        # Two calls with the same op
        call1 = ir.Call(relu_op, [x], span)
        call2 = ir.Call(relu_op, [y], span)

        # Verify op is shared
        assert call1.op is call2.op

        # Create a tuple
        tuple_expr = ir.MakeTuple([call1, call2], span)

        # Serialize and deserialize
        data = ir.serialize(tuple_expr)
        restored = ir.deserialize(data)

        # Verify structure is preserved (ops may not be shared after deserialization)
        assert isinstance(restored, ir.MakeTuple)
        restored_call1 = restored.elements[0]
        restored_call2 = restored.elements[1]

        assert isinstance(restored_call1, ir.Call)
        assert isinstance(restored_call2, ir.Call)

        assert restored_call1.op.name == "relu"
        assert restored_call2.op.name == "relu"

        ir.assert_structural_equal(tuple_expr, restored, enable_auto_mapping = True)
