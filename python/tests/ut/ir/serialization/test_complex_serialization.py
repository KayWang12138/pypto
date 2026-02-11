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
"""Tests for complex IR serialization scenarios."""

import math
import os
import tempfile
import pytest
from pypto import ir
from pypto.ir import DataType


class TestComplexExpressionTrees:
    """Test serialization of complex expression trees."""

    def test_deeply_nested_expression(self):
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
        ir.assert_structural_equal(expr, restored, enable_auto_mapping = True)

    def test_complex_arithmetic_tree(self):
        """Test serializing complex arithmetic expression tree."""
        span = ir.Span.unknown()
        a = ir.Var("a", ir.ScalarType(DataType.FP32), span)
        b = ir.Var("b", ir.ScalarType(DataType.FP32), span)
        c = ir.Var("c", ir.ScalarType(DataType.FP32), span)
        d = ir.Var("d", ir.ScalarType(DataType.FP32), span)

        # Build: ((a + b) * (c - d)) / ((a * c) + (b * d))
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
        ir.assert_structural_equal(div1, restored, enable_auto_mapping = True)

    def test_mixed_operations_with_calls(self):
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
        ir.assert_structural_equal(add_expr, restored, enable_auto_mapping = True)


class TestComplexControlFlow:
    """Test serialization of complex control flow structures."""

    def test_complex_control_flow(self):
        """Test serializing complex control flow."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        result = ir.Var("result", ir.ScalarType(DataType.INT64), span)

        # if (x > 0) {
        #   if (y > 0) {
        #     result = x + y
        #   } else {
        #     result = x - y
        #   }
        # } else {
        #   result = 0
        # }

        zero = ir.ConstInt(0, DataType.INT64, span)

        # Inner if-else
        inner_condition = ir.Gt(y, zero, DataType.INT64, span)
        inner_then = ir.AssignStmt(result, ir.Add(x, y, DataType.INT64, span), span)
        inner_else = ir.AssignStmt(result, ir.Sub(x, y, DataType.INT64, span), span)
        inner_if = ir.IfStmt(inner_condition, inner_then, inner_else, return_vars=[], span=span)

        # Outer if-else
        outer_condition = ir.Gt(x, zero, DataType.INT64, span)
        outer_else = ir.AssignStmt(result, zero, span)
        outer_if = ir.IfStmt(outer_condition, inner_if, outer_else, return_vars=[], span=span)

        # Serialize
        data = ir.serialize(outer_if)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.IfStmt)
        assert isinstance(restored.then_body, ir.IfStmt)

        # Verify structural equality
        ir.assert_structural_equal(outer_if, restored, enable_auto_mapping = True)




class TestEdgeCases:
    """Test serialization of edge cases and empty structures."""

    def test_empty_structures(self):
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
