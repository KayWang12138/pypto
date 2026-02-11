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
"""Tests for serialization of IR statements.

NOTE: Most basic serialization tests have been moved to test_IR_node.py.
This file retains additional edge case and variant tests.
"""

import pytest
from pypto import ir
from pypto.ir import DataType


class TestAssignStmtSerialization:
    """Test serialization of assignment statements - additional variants.

    NOTE: Basic AssignStmt test is in test_IR_node.py.
    """

    # NOTE: Basic test moved to test_IR_node.py
    # def test_assign_stmt_with_const(self):

    def test_assign_stmt_with_binary_expr(self):
        """Test serializing AssignStmt with binary expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        add_expr = ir.Add(x, y, DataType.INT64, span)
        result = ir.Var("result", ir.ScalarType(DataType.INT64), span)
        assign = ir.AssignStmt(result, add_expr, span)

        # Serialize
        data = ir.serialize(assign)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.AssignStmt)
        assert isinstance(restored.value, ir.Add)

        # Verify structural equality
        ir.assert_structural_equal(assign, restored, enable_auto_mapping = True)

    def test_assign_stmt_with_call(self):
        """Test serializing AssignStmt with Call expression."""
        span = ir.Span.unknown()
        op = ir.Op("relu")
        x = ir.Var("x", ir.TensorType([], DataType.FP32), span)
        call = ir.Call(op, [x], span)
        result = ir.Var("result", ir.UnknownType(), span)
        assign = ir.AssignStmt(result, call, span)

        # Serialize
        data = ir.serialize(assign)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.AssignStmt)
        assert isinstance(restored.value, ir.Call)
        assert restored.value.op.name == "relu"

        # Verify structural equality
        ir.assert_structural_equal(assign, restored, enable_auto_mapping = True)


class TestSeqStmtsSerialization:
    """Test serialization of sequential statements - additional variants.

    NOTE: Basic SeqStmts test is in test_IR_node.py.
    """

    # NOTE: Basic test moved to test_IR_node.py
    # def test_seq_stmts_with_multiple_stmts(self):

    def test_empty_seq_stmts(self):
        """Test serializing empty SeqStmts."""
        span = ir.Span.unknown()
        seq = ir.SeqStmts([], span)

        # Serialize
        data = ir.serialize(seq)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.SeqStmts)
        assert len(restored.stmts) == 0

        # Verify structural equality
        ir.assert_structural_equal(seq, restored)

    def test_seq_stmts_with_single_stmt(self):
        """Test serializing SeqStmts with single statement - edge case."""
        span = ir.Span.unknown()
        var = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        const = ir.ConstInt(10, DataType.INT64, span)
        assign = ir.AssignStmt(var, const, span)
        seq = ir.SeqStmts([assign], span)

        # Serialize
        data = ir.serialize(seq)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.SeqStmts)
        assert len(restored.stmts) == 1

        # Verify structural equality
        ir.assert_structural_equal(seq, restored, enable_auto_mapping = True)

    # NOTE: Basic test moved to test_IR_node.py
    # def test_seq_stmts_with_multiple_stmts(self):

    def test_nested_seq_stmts(self):
        """Test serializing nested SeqStmts."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        const = ir.ConstInt(5, DataType.INT64, span)
        assign = ir.AssignStmt(x, const, span)

        inner_seq = ir.SeqStmts([assign], span)
        outer_seq = ir.SeqStmts([inner_seq], span)

        # Serialize
        data = ir.serialize(outer_seq)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.SeqStmts)
        assert len(restored.stmts) == 1
        assert isinstance(restored.stmts[0], ir.SeqStmts)

        # Verify structural equality
        ir.assert_structural_equal(outer_seq, restored, enable_auto_mapping = True)


class TestYieldStmtSerialization:
    """Test serialization of YieldStmt - generator return values.

    Source: Moved from test_IR_node_serialization.py (unique test).
    """

    def test_yield_stmt_serialization(self):
        """Test YieldStmt with single value."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        yield_stmt = ir.YieldStmt([x], span)

        data = ir.serialize(yield_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.YieldStmt)
        assert len(restored.value) == 1
        ir.assert_structural_equal(yield_stmt, restored, enable_auto_mapping = True)

    def test_yield_stmt_multiple_values(self):
        """Test YieldStmt with multiple values."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.FP32), span)
        yield_stmt = ir.YieldStmt([x, y], span)

        data = ir.serialize(yield_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.YieldStmt)
        assert len(restored.value) == 2
        ir.assert_structural_equal(yield_stmt, restored, enable_auto_mapping = True)

    def test_yield_stmt_empty(self):
        """Test YieldStmt with no values."""
        span = ir.Span.unknown()
        yield_stmt = ir.YieldStmt([], span)

        data = ir.serialize(yield_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.YieldStmt)
        assert len(restored.value) == 0
        ir.assert_structural_equal(yield_stmt, restored)


class TestReturnStmtSerialization:
    """Test serialization of ReturnStmt - function return values.

    Source: Moved from test_IR_node_serialization.py (unique test).
    """

    def test_return_stmt_serialization(self):
        """Test ReturnStmt with single value."""
        span = ir.Span.unknown()
        const = ir.ConstInt(42, DataType.INT64, span)
        return_stmt = ir.ReturnStmt([const], span)

        data = ir.serialize(return_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ReturnStmt)
        assert len(restored.value) == 1
        ir.assert_structural_equal(return_stmt, restored)

    def test_return_stmt_multiple_values(self):
        """Test ReturnStmt with multiple values."""
        span = ir.Span.unknown()
        val1 = ir.ConstInt(10, DataType.INT64, span)
        val2 = ir.ConstFloat(3.14, DataType.FP32, span)
        return_stmt = ir.ReturnStmt([val1, val2], span)

        data = ir.serialize(return_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ReturnStmt)
        assert len(restored.value) == 2
        ir.assert_structural_equal(return_stmt, restored)

    def test_return_stmt_expression(self):
        """Test ReturnStmt with expression value."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        add_expr = ir.Add(x, y, DataType.INT64, span)
        return_stmt = ir.ReturnStmt([add_expr], span)

        data = ir.serialize(return_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ReturnStmt)
        assert isinstance(restored.value[0], ir.Add)
        ir.assert_structural_equal(return_stmt, restored, enable_auto_mapping = True)


class TestOpStmtsSerialization:
    """Test serialization of OpStmts - operator statements wrapper.

    Source: Moved from test_IR_node_serialization.py (unique test).
    """

    def test_op_stmts_serialization(self):
        """Test OpStmts with single statement."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.TensorType([], DataType.FP32), span)

        op = ir.Op("relu")
        call = ir.Call(op, [x], span)
        result = ir.Var("result", ir.UnknownType(), span)
        assign = ir.AssignStmt(result, call, span)

        op_stmts = ir.OpStmts([assign], span)

        data = ir.serialize(op_stmts)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.OpStmts)
        assert len(restored.stmts) == 1
        ir.assert_structural_equal(op_stmts, restored, enable_auto_mapping = True)

    def test_op_stmts_multiple_statements(self):
        """Test OpStmts with multiple statements."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.TensorType([], DataType.FP32), span)
        y = ir.Var("y", ir.TensorType([], DataType.FP32), span)

        # First operation: relu(x)
        relu_op = ir.Op("relu")
        relu_call = ir.Call(relu_op, [x], span)
        relu_result = ir.Var("relu_result", ir.UnknownType(), span)
        relu_assign = ir.AssignStmt(relu_result, relu_call, span)

        # Second operation: sigmoid(y)
        sigmoid_op = ir.Op("sigmoid")
        sigmoid_call = ir.Call(sigmoid_op, [y], span)
        sigmoid_result = ir.Var("sigmoid_result", ir.UnknownType(), span)
        sigmoid_assign = ir.AssignStmt(sigmoid_result, sigmoid_call, span)

        op_stmts = ir.OpStmts([relu_assign, sigmoid_assign], span)

        data = ir.serialize(op_stmts)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.OpStmts)
        assert len(restored.stmts) == 2
        ir.assert_structural_equal(op_stmts, restored, enable_auto_mapping = True)


class TestEvalStmtSerialization:
    """Test serialization of EvalStmt - expression evaluation statement.

    Source: Moved from test_IR_node_serialization.py (unique test).
    """

    def test_eval_stmt_serialization(self):
        """Test EvalStmt with binary expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        add_expr = ir.Add(x, y, DataType.INT64, span)

        eval_stmt = ir.EvalStmt(add_expr, span)

        data = ir.serialize(eval_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.EvalStmt)
        assert isinstance(restored.expr, ir.Add)
        ir.assert_structural_equal(eval_stmt, restored, enable_auto_mapping = True)

    def test_eval_stmt_with_call(self):
        """Test EvalStmt with Call expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.TensorType([], DataType.FP32), span)
        op = ir.Op("print")
        call = ir.Call(op, [x], span)

        eval_stmt = ir.EvalStmt(call, span)

        data = ir.serialize(eval_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.EvalStmt)
        assert isinstance(restored.expr, ir.Call)
        assert restored.expr.op.name == "print"
        ir.assert_structural_equal(eval_stmt, restored, enable_auto_mapping = True)


class TestIfStmtSerialization:
    """Test serialization of if statements - additional variants.

    NOTE: Basic IfStmt tests are in test_IR_node.py.
    """

    # NOTE: Basic tests moved to test_IR_node.py
    # def test_if_stmt_with_else(self):
    # def test_if_stmt_without_else(self):

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
