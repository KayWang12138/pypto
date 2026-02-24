# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Comprehensive tests for Function serialization with complex bodies."""

import pytest
from pypto import ir
from pypto.ir import DataType


class TestBasicFunction:
    """Test basic Function serialization - moved from test_IR_node.py."""

    @staticmethod
    def test_function_serialization():
        """Test Function serialization."""
        span = ir.Span.unknown()

        # Parameters
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)

        # Body: return x + y
        add_expr = ir.Add(x, y, DataType.INT64, span)
        return_stmt = ir.ReturnStmt([add_expr], span)

        # Create function
        function = ir.Function(
            name="add_function",
            params=[x, y],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=return_stmt,
            span=span
        )

        data = ir.serialize(function)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Function)
        assert restored.name == "add_function"
        assert len(restored.params) == 2
        assert len(restored.return_types) == 1
        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)


class TestFunctionWithIfStmt:
    """Test Function containing IfStmt - critical control flow scenario."""

    @staticmethod
    def test_function_with_simple_if():
        """Test Function with simple if statement."""
        span = ir.Span.unknown()

        # Function: max(a, b) -> if a > b then a else b
        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        b = ir.Var("b", ir.ScalarType(DataType.INT64), span)

        condition = ir.Gt(a, b, DataType.INT64, span)
        then_body = ir.ReturnStmt([a], span)
        else_body = ir.ReturnStmt([b], span)

        if_stmt = ir.IfStmt(condition, then_body, else_body, return_vars=[], span=span)

        function = ir.Function(
            name="max",
            params=[a, b],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=if_stmt,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify
        assert isinstance(restored, ir.Function)
        assert restored.name == "max"
        assert isinstance(restored.body, ir.IfStmt)
        assert isinstance(restored.body.condition, ir.Gt)
        assert isinstance(restored.body.then_body, ir.ReturnStmt)
        assert isinstance(restored.body.else_body, ir.ReturnStmt)

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)

    @staticmethod
    def test_function_with_nested_if():
        """Test Function with nested if-else statements."""
        span = ir.Span.unknown()

        # Function: classify(x) -> returns 0, 1, or 2 based on x value
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        zero = ir.ConstInt(0, DataType.INT64, span)
        one = ir.ConstInt(1, DataType.INT64, span)
        two = ir.ConstInt(2, DataType.INT64, span)
        ten = ir.ConstInt(10, DataType.INT64, span)

        # Outer if: x < 0
        outer_cond = ir.Lt(x, zero, DataType.INT64, span)
        outer_then = ir.ReturnStmt([zero], span)

        # Inner if: x < 10
        inner_cond = ir.Lt(x, ten, DataType.INT64, span)
        inner_then = ir.ReturnStmt([one], span)
        inner_else = ir.ReturnStmt([two], span)
        inner_if = ir.IfStmt(inner_cond, inner_then, inner_else, return_vars=[], span=span)

        outer_if = ir.IfStmt(outer_cond, outer_then, inner_if, return_vars=[], span=span)

        function = ir.Function(
            name="classify",
            params=[x],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=outer_if,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify nested structure
        assert isinstance(restored.body, ir.IfStmt)
        assert isinstance(restored.body.else_body, ir.IfStmt)
        inner_if_restored = restored.body.else_body
        assert isinstance(inner_if_restored.then_body, ir.ReturnStmt)
        assert isinstance(inner_if_restored.else_body, ir.ReturnStmt)

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)

    @staticmethod
    def test_function_with_if_and_assignments():
        """Test Function with if statement containing assignments."""
        span = ir.Span.unknown()

        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        result = ir.Var("result", ir.ScalarType(DataType.INT64), span)
        zero = ir.ConstInt(0, DataType.INT64, span)

        condition = ir.Lt(x, zero, DataType.INT64, span)
        neg_x = ir.Neg(x, DataType.INT64, span)
        then_body = ir.AssignStmt(result, neg_x, span)
        else_body = ir.AssignStmt(result, x, span)

        if_stmt = ir.IfStmt(condition, then_body, else_body, return_vars=[], span=span)
        return_stmt = ir.ReturnStmt([result], span)

        body = ir.SeqStmts([if_stmt, return_stmt], span)

        function = ir.Function(
            name="abs_value",
            params=[x],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=body,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify
        assert isinstance(restored.body, ir.SeqStmts)
        assert len(restored.body.stmts) == 2
        assert isinstance(restored.body.stmts[0], ir.IfStmt)
        assert isinstance(restored.body.stmts[1], ir.ReturnStmt)

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)


class TestFunctionWithForStmt:
    """Test ForStmt within Function context."""

    @staticmethod
    def test_function_with_simple_for_loop():
        """Test Function with simple for loop."""
        span = ir.Span.unknown()

        # Function: sum_to_n(n) -> sum from 0 to n
        n = ir.Var("n", ir.ScalarType(DataType.INT64), span)
        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)
        sum_var = ir.Var("sum", ir.ScalarType(DataType.INT64), span)

        init = ir.AssignStmt(sum_var, ir.ConstInt(0, DataType.INT64, span), span)

        loop_body = ir.AssignStmt(sum_var, ir.Add(sum_var, i, DataType.INT64, span), span)
        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            n,
            ir.ConstInt(1, DataType.INT64, span),
            iter_args=[],
            body=loop_body,
            return_vars=[],
            span=span
        )

        return_stmt = ir.ReturnStmt([sum_var], span)

        body = ir.SeqStmts([init, for_stmt, return_stmt], span)

        function = ir.Function(
            name="sum_to_n",
            params=[n],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=body,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify
        assert isinstance(restored.body, ir.SeqStmts)
        assert len(restored.body.stmts) == 3
        assert isinstance(restored.body.stmts[1], ir.ForStmt)

        for_stmt_restored = restored.body.stmts[1]
        assert for_stmt_restored.loop_var.name == "i"
        assert isinstance(for_stmt_restored.body, ir.AssignStmt)

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)

    @staticmethod
    def test_function_with_nested_for_loops():
        """Test Function with nested for loops."""
        span = ir.Span.unknown()

        # Function: sum_matrix(rows, cols) -> nested loop sum
        rows = ir.Var("rows", ir.ScalarType(DataType.INT64), span)
        cols = ir.Var("cols", ir.ScalarType(DataType.INT64), span)
        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)
        j = ir.Var("j", ir.ScalarType(DataType.INT64), span)
        sum_var = ir.Var("sum", ir.ScalarType(DataType.INT64), span)

        init = ir.AssignStmt(sum_var, ir.ConstInt(0, DataType.INT64, span), span)

        # Inner loop: for j in 0..cols
        inner_body = ir.AssignStmt(sum_var, ir.Add(sum_var, j, DataType.INT64, span), span)
        inner_loop = ir.ForStmt(
            j,
            ir.ConstInt(0, DataType.INT64, span),
            cols,
            ir.ConstInt(1, DataType.INT64, span),
            [],
            inner_body,
            [],
            span
        )

        # Outer loop: for i in 0..rows
        outer_loop = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            rows,
            ir.ConstInt(1, DataType.INT64, span),
            [],
            inner_loop,
            [],
            span
        )

        return_stmt = ir.ReturnStmt([sum_var], span)
        body = ir.SeqStmts([init, outer_loop, return_stmt], span)

        function = ir.Function(
            name="sum_matrix",
            params=[rows, cols],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=body,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify nested structure
        assert isinstance(restored.body, ir.SeqStmts)
        outer_loop_restored = restored.body.stmts[1]
        assert isinstance(outer_loop_restored, ir.ForStmt)
        assert isinstance(outer_loop_restored.body, ir.ForStmt)

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)

    @staticmethod
    def test_function_with_for_loop_using_iter_arg():
        """Test Function with ForStmt using IterArg."""
        span = ir.Span.unknown()

        # Function with IterArg: accumulate(n)
        n = ir.Var("n", ir.ScalarType(DataType.INT64), span)
        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)

        # IterArg: accumulator with initial value 0
        init_value = ir.ConstInt(0, DataType.INT64, span)
        acc = ir.IterArg("acc", ir.ScalarType(DataType.INT64), init_value, span)

        acc_next = ir.Var("acc_next", ir.ScalarType(DataType.INT64), span)
        loop_body = ir.AssignStmt(acc_next, ir.Add(acc, i, DataType.INT64, span), span)

        # For loop with IterArg
        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            n,
            ir.ConstInt(1, DataType.INT64, span),
            iter_args=[acc],
            body=loop_body,
            return_vars=[acc_next],
            span=span
        )

        # Return the final accumulator value
        return_stmt = ir.ReturnStmt([acc], span)
        body = ir.SeqStmts([for_stmt, return_stmt], span)

        function = ir.Function(
            name="accumulate",
            params=[n],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=body,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify IterArg structure
        assert isinstance(restored.body, ir.SeqStmts)
        for_stmt_restored = restored.body.stmts[0]
        assert isinstance(for_stmt_restored, ir.ForStmt)
        assert len(for_stmt_restored.iter_args) == 1
        assert isinstance(for_stmt_restored.iter_args[0], ir.IterArg)
        assert for_stmt_restored.iter_args[0].name == "acc"

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)

    @staticmethod
    def test_function_with_yield_in_loop():
        """Test Function with yield inside loop - classic generator pattern."""
        span = ir.Span.unknown()

        # Generator: range_generator(n) -> yields 0, 1, 2, ..., n-1
        n = ir.Var("n", ir.ScalarType(DataType.INT64), span)
        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)

        # Loop body: yield i
        yield_stmt = ir.YieldStmt([i], span)

        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            n,
            ir.ConstInt(1, DataType.INT64, span),
            [],
            yield_stmt,
            [],
            span
        )

        function = ir.Function(
            name="range_generator",
            params=[n],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=for_stmt,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify
        assert isinstance(restored.body, ir.ForStmt)
        assert isinstance(restored.body.body, ir.YieldStmt)

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)

    @staticmethod
    def test_function_with_iter_arg_and_yield_in_loop():
        """Test Function with IterArg + YieldStmt in loop - stateful generator."""
        span = ir.Span.unknown()

        # Generator: running_sum_generator(n) -> yields running sum
        n = ir.Var("n", ir.ScalarType(DataType.INT64), span)
        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)

        init_value = ir.ConstInt(0, DataType.INT64, span)
        acc = ir.IterArg("acc", ir.ScalarType(DataType.INT64), init_value, span)

        acc_next = ir.Var("acc_next", ir.ScalarType(DataType.INT64), span)
        update_stmt = ir.AssignStmt(acc_next, ir.Add(acc, i, DataType.INT64, span), span)
        yield_stmt = ir.YieldStmt([acc_next], span)
        loop_body = ir.SeqStmts([update_stmt, yield_stmt], span)

        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            n,
            ir.ConstInt(1, DataType.INT64, span),
            iter_args=[acc],
            body=loop_body,
            return_vars=[acc_next],
            span=span
        )

        function = ir.Function(
            name="running_sum_generator",
            params=[n],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=for_stmt,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify
        assert isinstance(restored.body, ir.ForStmt)
        assert len(restored.body.iter_args) == 1
        assert isinstance(restored.body.body, ir.SeqStmts)
        # Find YieldStmt in body
        has_yield = any(isinstance(stmt, ir.YieldStmt) for stmt in restored.body.body.stmts)
        assert has_yield

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)


class TestFunctionWithForStmtAndIfStmt:
    """Test ForStmt combined with other control flow statements."""

    @staticmethod
    def test_function_with_if_and_for_combined():
        """Test Function with both if and for statements."""
        span = ir.Span.unknown()

        # for i in 0..n:
        n = ir.Var("n", ir.ScalarType(DataType.INT64), span)
        threshold = ir.Var("threshold", ir.ScalarType(DataType.INT64), span)
        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)
        sum_var = ir.Var("sum", ir.ScalarType(DataType.INT64), span)

        init = ir.AssignStmt(sum_var, ir.ConstInt(0, DataType.INT64, span), span)

        condition = ir.Gt(i, threshold, DataType.INT64, span)
        then_body = ir.AssignStmt(sum_var, ir.Add(sum_var, i, DataType.INT64, span), span)
        if_stmt = ir.IfStmt(condition, then_body, return_vars=[], span=span)

        # for i in 0..n
        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            n,
            ir.ConstInt(1, DataType.INT64, span),
            [],
            if_stmt,
            [],
            span
        )

        return_stmt = ir.ReturnStmt([sum_var], span)
        body = ir.SeqStmts([init, for_stmt, return_stmt], span)

        function = ir.Function(
            name="conditional_sum",
            params=[n, threshold],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=body,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify nested control flow
        for_stmt_restored = restored.body.stmts[1]
        assert isinstance(for_stmt_restored, ir.ForStmt)
        assert isinstance(for_stmt_restored.body, ir.IfStmt)

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)

    @staticmethod
    def test_function_with_complex_control_flow_graph():
        """Test Function with complex control flow: nested if-for-if pattern."""
        span = ir.Span.unknown()

        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        z = ir.Var("z", ir.ScalarType(DataType.INT64), span)
        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)
        result = ir.Var("result", ir.ScalarType(DataType.INT64), span)

        init = ir.AssignStmt(result, ir.ConstInt(0, DataType.INT64, span), span)

        inner_if_cond = ir.Gt(i, z, DataType.INT64, span)
        inner_if_then = ir.AssignStmt(result, ir.Add(result, i, DataType.INT64, span), span)
        inner_if = ir.IfStmt(inner_if_cond, inner_if_then, return_vars=[], span=span)

        # Middle: for i in 0..y
        for_loop = ir.ForStmt(
            i, ir.ConstInt(0, DataType.INT64, span), 
            y, ir.ConstInt(1, DataType.INT64, span), 
            [], inner_if, [], span)

        # Outer: if x > 0 then execute for loop else result = -1
        outer_if_cond = ir.Gt(x, ir.ConstInt(0, DataType.INT64, span), DataType.INT64, span)
        outer_if_else = ir.AssignStmt(result, ir.ConstInt(-1, DataType.INT64, span), span)
        outer_if = ir.IfStmt(outer_if_cond, for_loop, outer_if_else, return_vars=[], span=span)

        return_stmt = ir.ReturnStmt([result], span)
        body = ir.SeqStmts([init, outer_if, return_stmt], span)

        function = ir.Function(
            name="complex_compute",
            params=[x, y, z],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=body,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify deeply nested structure
        outer_if_restored = restored.body.stmts[1]
        assert isinstance(outer_if_restored, ir.IfStmt)
        assert isinstance(outer_if_restored.then_body, ir.ForStmt)
        assert isinstance(outer_if_restored.then_body.body, ir.IfStmt)

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)



class TestFunctionWithMultipleReturnPaths:
    """Test Function with multiple return statements in different branches."""

    @staticmethod
    def test_function_with_early_return():
        """Test Function with early return pattern."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        b = ir.Var("b", ir.ScalarType(DataType.INT64), span)
        zero = ir.ConstInt(0, DataType.INT64, span)

        # Early return if b == 0
        condition = ir.Eq(b, zero, DataType.INT64, span)
        early_return = ir.ReturnStmt([zero], span)
        if_stmt = ir.IfStmt(condition, early_return, return_vars=[], span=span)

        # Normal return: a / b
        div_expr = ir.FloorDiv(a, b, DataType.INT64, span)
        normal_return = ir.ReturnStmt([div_expr], span)

        body = ir.SeqStmts([if_stmt, normal_return], span)

        function = ir.Function(
            name="safe_divide",
            params=[a, b],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=body,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify both return paths
        assert isinstance(restored.body, ir.SeqStmts)
        assert len(restored.body.stmts) == 2
        assert isinstance(restored.body.stmts[0], ir.IfStmt)
        assert isinstance(restored.body.stmts[0].then_body, ir.ReturnStmt)
        assert isinstance(restored.body.stmts[1], ir.ReturnStmt)

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)

    @staticmethod
    def test_function_with_multiple_branch_returns():
        """Test Function where each branch has its own return."""
        span = ir.Span.unknown()

        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        zero = ir.ConstInt(0, DataType.INT64, span)
        one = ir.ConstInt(1, DataType.INT64, span)
        minus_one = ir.ConstInt(-1, DataType.INT64, span)

        # Outer if: x < 0
        outer_cond = ir.Lt(x, zero, DataType.INT64, span)
        outer_then = ir.ReturnStmt([minus_one], span)

        # Inner if: x > 0
        inner_cond = ir.Gt(x, zero, DataType.INT64, span)
        inner_then = ir.ReturnStmt([one], span)
        inner_else = ir.ReturnStmt([zero], span)
        inner_if = ir.IfStmt(inner_cond, inner_then, inner_else, return_vars=[], span=span)

        outer_if = ir.IfStmt(outer_cond, outer_then, inner_if, return_vars=[], span=span)

        function = ir.Function(
            name="sign",
            params=[x],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=outer_if,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify all three return paths
        assert isinstance(restored.body, ir.IfStmt)
        assert isinstance(restored.body.then_body, ir.ReturnStmt)
        assert isinstance(restored.body.else_body, ir.IfStmt)
        inner_if_restored = restored.body.else_body
        assert isinstance(inner_if_restored.then_body, ir.ReturnStmt)
        assert isinstance(inner_if_restored.else_body, ir.ReturnStmt)

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)


class TestFunctionEdgeCases:
    """Test edge cases for Function serialization."""

    @staticmethod
    def test_function_with_no_params():
        """Test Function with no parameters."""
        span = ir.Span.unknown()

        function = ir.Function(
            name="get_constant",
            params=[],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=ir.ReturnStmt([ir.ConstInt(42, DataType.INT64, span)], span),
            span=span
        )

        data = ir.serialize(function)
        restored = ir.deserialize(data)

        assert len(restored.params) == 0
        ir.assert_structural_equal(function, restored)

    @staticmethod
    def test_function_with_many_params():
        """Test Function with many parameters."""
        span = ir.Span.unknown()

        # Create function with 10 parameters
        params = [ir.Var(f"p{i}", ir.ScalarType(DataType.INT64), span) for i in range(10)]

        # Body: return sum of all params
        sum_expr = params[0]
        for p in params[1:]:
            sum_expr = ir.Add(sum_expr, p, DataType.INT64, span)

        body = ir.ReturnStmt([sum_expr], span)

        function = ir.Function(
            name="sum_many",
            params=params,
            return_types=[ir.ScalarType(DataType.INT64)],
            body=body,
            span=span
        )

        data = ir.serialize(function)
        restored = ir.deserialize(data)

        assert len(restored.params) == 10
        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)

    @staticmethod
    def test_function_with_tuple_return_type():
        """Test Function returning tuple."""
        span = ir.Span.unknown()

        # Function: swap(a, b) -> (b, a)
        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        b = ir.Var("b", ir.ScalarType(DataType.INT64), span)

        tuple_expr = ir.MakeTuple([b, a], span)
        body = ir.ReturnStmt([tuple_expr], span)

        function = ir.Function(
            name="swap",
            params=[a, b],
            return_types=[ir.TupleType([ir.ScalarType(DataType.INT64), ir.ScalarType(DataType.INT64)])],
            body=body,
            span=span
        )

        data = ir.serialize(function)
        restored = ir.deserialize(data)

        assert len(restored.return_types) == 1
        assert isinstance(restored.return_types[0], ir.TupleType)
        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)

    @staticmethod
    def test_function_with_empty_body():
        """Test Function with minimal body."""
        span = ir.Span.unknown()

        # Empty SeqStmts
        function = ir.Function(
            name="noop",
            params=[],
            return_types=[],
            body=ir.SeqStmts([], span),
            span=span
        )

        data = ir.serialize(function)
        restored = ir.deserialize(data)

        assert isinstance(restored.body, ir.SeqStmts)
        assert len(restored.body.stmts) == 0
        ir.assert_structural_equal(function, restored)


class TestFunctionWithTensorOperations:
    """Test Function with tensor-related operations."""

    @staticmethod
    def test_function_with_tensor_params():
        """Test Function accepting tensor parameters."""
        span = ir.Span.unknown()

        dim = ir.ConstInt(100, DataType.INT64, span)
        tensor_type = ir.TensorType([dim], DataType.FP32)
        tensor = ir.Var("tensor", tensor_type, span)
        size = ir.Var("size", ir.ScalarType(DataType.INT64), span)

        # Simple body: return size
        body = ir.ReturnStmt([size], span)

        function = ir.Function(
            name="process_tensor",
            params=[tensor, size],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=body,
            span=span
        )

        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify tensor parameter type
        assert len(restored.params) == 2
        assert isinstance(restored.params[0].type, ir.TensorType)
        assert isinstance(restored.params[1].type, ir.ScalarType)

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)

    @staticmethod
    def test_function_with_call_to_tensor_op():
        """Test Function calling tensor operations."""
        span = ir.Span.unknown()

        tensor_type = ir.TensorType([], DataType.FP32)
        x = ir.Var("x", tensor_type, span)

        # Call external op: relu(x)
        relu_op = ir.Op("relu")
        call = ir.Call(relu_op, [x], span)
        body = ir.ReturnStmt([call], span)

        function = ir.Function(
            name="apply_relu",
            params=[x],
            return_types=[tensor_type],
            body=body,
            span=span
        )

        data = ir.serialize(function)
        restored = ir.deserialize(data)

        # Verify the call is preserved
        assert isinstance(restored.body, ir.ReturnStmt)
        return_expr = restored.body.value[0]
        assert isinstance(return_expr, ir.Call)
        assert return_expr.op.name == "relu"

        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)


class TestFunctionWithComplexBody:
    """Test function with complex body containing multiple statement types - moved from test_IR_node.py."""

    @staticmethod
    def test_function_with_complex_body():
        """Test function with complex body containing multiple statement types."""
        span = ir.Span.unknown()

        # Parameters
        n = ir.Var("n", ir.ScalarType(DataType.INT64), span)

        # Variables
        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)
        sum_var = ir.Var("sum", ir.ScalarType(DataType.INT64), span)

        init_stmt = ir.AssignStmt(sum_var, ir.ConstInt(0, DataType.INT64, span), span)

        loop_body = ir.AssignStmt(
            sum_var,
            ir.Add(sum_var, i, DataType.INT64, span),
            span
        )

        # For loop: for i in 0..n
        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            n,
            ir.ConstInt(1, DataType.INT64, span),
            iter_args=[],
            body=loop_body,
            return_vars=[],
            span=span
        )

        # Return sum
        return_stmt = ir.ReturnStmt([sum_var], span)

        # Function body: sequence of statements
        body = ir.SeqStmts([init_stmt, for_stmt, return_stmt], span)

        # Create function
        function = ir.Function(
            name="sum_to_n",
            params=[n],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=body,
            span=span
        )

        data = ir.serialize(function)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Function)
        assert restored.name == "sum_to_n"
        assert isinstance(restored.body, ir.SeqStmts)
        assert len(restored.body.stmts) == 3
        ir.assert_structural_equal(function, restored, enable_auto_mapping=True)

