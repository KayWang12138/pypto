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
"""Comprehensive tests for ForStmt serialization.

This file consolidates all ForStmt-related tests including:
- Basic ForStmt serialization
- ForStmt with IterArg (SSA loop variables)
- ForStmt with YieldStmt (generator patterns)
- ForStmt with IterArg + YieldStmt (SSA generator pattern) - NEW
- Nested ForStmt
- ForStmt in Functions and Programs
- ForStmt with complex control flow
"""

import pytest
from pypto import ir
from pypto.ir import DataType


class TestBasicForStmtSerialization:
    """Test basic ForStmt serialization patterns."""

    def test_simple_for_stmt(self):
        """Test basic ForStmt serialization."""
        span = ir.Span.unknown()
        loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
        start = ir.ConstInt(0, DataType.INT64, span)
        end = ir.ConstInt(10, DataType.INT64, span)
        step = ir.ConstInt(1, DataType.INT64, span)

        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        add_expr = ir.Add(x, loop_var, DataType.INT64, span)
        body = ir.AssignStmt(x, add_expr, span)

        for_stmt = ir.ForStmt(loop_var, start, end, step, iter_args=[], body=body, return_vars=[], span=span)

        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ForStmt)
        ir.assert_structural_equal(for_stmt, restored, enable_auto_mapping = True)

    def test_nested_for_stmt(self):
        """Test nested ForStmt serialization."""
        span = ir.Span.unknown()
        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)
        j = ir.Var("j", ir.ScalarType(DataType.INT64), span)
        sum_var = ir.Var("sum", ir.ScalarType(DataType.INT64), span)

        # Inner loop body: sum = sum + j
        add_expr = ir.Add(sum_var, j, DataType.INT64, span)
        inner_body = ir.AssignStmt(sum_var, add_expr, span)

        # Inner loop
        inner_loop = ir.ForStmt(
            j,
            ir.ConstInt(0, DataType.INT64, span),
            ir.ConstInt(10, DataType.INT64, span),
            ir.ConstInt(1, DataType.INT64, span),
            iter_args=[],
            body=inner_body,
            return_vars=[],
            span=span
        )

        # Outer loop
        outer_loop = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            ir.ConstInt(5, DataType.INT64, span),
            ir.ConstInt(1, DataType.INT64, span),
            iter_args=[],
            body=inner_loop,
            return_vars=[],
            span=span
        )

        data = ir.serialize(outer_loop)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ForStmt)
        assert isinstance(restored.body, ir.ForStmt)
        ir.assert_structural_equal(outer_loop, restored, enable_auto_mapping = True)


class TestForStmtWithIterArg:
    """Test ForStmt with IterArg - SSA loop-carried variables."""

    def test_for_stmt_with_single_iter_arg(self):
        """Test ForStmt with single IterArg - accumulator pattern."""
        span = ir.Span.unknown()

        # Loop variable
        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)

        # IterArg: accumulator with initial value 0
        init_value = ir.ConstInt(0, DataType.INT64, span)
        acc = ir.IterArg("acc", ir.ScalarType(DataType.INT64), init_value, span)

        # Loop body: acc_next = acc + i
        acc_next = ir.Var("acc_next", ir.ScalarType(DataType.INT64), span)
        loop_body = ir.AssignStmt(acc_next, ir.Add(acc, i, DataType.INT64, span), span)

        # For loop with IterArg
        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            ir.ConstInt(10, DataType.INT64, span),
            ir.ConstInt(1, DataType.INT64, span),
            iter_args=[acc],
            body=loop_body,
            return_vars=[acc_next],
            span=span
        )

        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ForStmt)
        assert len(restored.iter_args) == 1
        assert isinstance(restored.iter_args[0], ir.IterArg)
        assert restored.iter_args[0].name == "acc"
        assert len(restored.return_vars) == 1
        ir.assert_structural_equal(for_stmt, restored, enable_auto_mapping = True)

    def test_for_stmt_with_multiple_iter_args(self):
        """Test ForStmt with multiple IterArgs - multiple accumulators."""
        span = ir.Span.unknown()

        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)

        # Two IterArgs: sum and product
        sum_init = ir.ConstInt(0, DataType.INT64, span)
        prod_init = ir.ConstInt(1, DataType.INT64, span)
        sum_acc = ir.IterArg("sum", ir.ScalarType(DataType.INT64), sum_init, span)
        prod_acc = ir.IterArg("prod", ir.ScalarType(DataType.INT64), prod_init, span)

        # Loop body: sum_next = sum + i; prod_next = prod * i
        sum_next = ir.Var("sum_next", ir.ScalarType(DataType.INT64), span)
        prod_next = ir.Var("prod_next", ir.ScalarType(DataType.INT64), span)

        sum_assign = ir.AssignStmt(sum_next, ir.Add(sum_acc, i, DataType.INT64, span), span)
        prod_assign = ir.AssignStmt(prod_next, ir.Mul(prod_acc, i, DataType.INT64, span), span)
        loop_body = ir.SeqStmts([sum_assign, prod_assign], span)

        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(10, DataType.INT64, span),
            ir.ConstInt(1, DataType.INT64, span),
            iter_args=[sum_acc, prod_acc],
            body=loop_body,
            return_vars=[sum_next, prod_next],
            span=span
        )

        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ForStmt)
        assert len(restored.iter_args) == 2
        assert isinstance(restored.iter_args[0], ir.IterArg)
        assert isinstance(restored.iter_args[1], ir.IterArg)
        assert restored.iter_args[0].name == "sum"
        assert restored.iter_args[1].name == "prod"
        assert len(restored.return_vars) == 2
        ir.assert_structural_equal(for_stmt, restored, enable_auto_mapping = True)


class TestForStmtWithYieldStmt:
    """Test ForStmt with YieldStmt - generator patterns."""

    def test_for_stmt_with_yield(self):
        """Test ForStmt with YieldStmt - simple generator."""
        span = ir.Span.unknown()

        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)

        # Loop body: yield i
        yield_stmt = ir.YieldStmt([i], span)

        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            ir.ConstInt(10, DataType.INT64, span),
            ir.ConstInt(1, DataType.INT64, span),
            [],
            yield_stmt,
            [],
            span
        )

        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ForStmt)
        assert isinstance(restored.body, ir.YieldStmt)
        assert len(restored.iter_args) == 0
        ir.assert_structural_equal(for_stmt, restored, enable_auto_mapping = True)

    def test_for_stmt_with_conditional_yield(self):
        """Test ForStmt with conditional YieldStmt."""
        span = ir.Span.unknown()

        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)

        # Loop body: if i > 5 then yield i
        condition = ir.Gt(i, ir.ConstInt(5, DataType.INT64, span), DataType.INT64, span)
        yield_stmt = ir.YieldStmt([i], span)
        if_stmt = ir.IfStmt(condition, yield_stmt, return_vars=[], span=span)

        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            ir.ConstInt(10, DataType.INT64, span),
            ir.ConstInt(1, DataType.INT64, span),
            [],
            if_stmt,
            [],
            span
        )

        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ForStmt)
        assert isinstance(restored.body, ir.IfStmt)
        assert isinstance(restored.body.then_body, ir.YieldStmt)
        ir.assert_structural_equal(for_stmt, restored, enable_auto_mapping = True)


class TestForStmtWithIterArgAndYieldStmt:
    """Test ForStmt with both IterArg and YieldStmt - SSA generator pattern.

    This is the MISSING TEST CASE that was identified:
    IterArg is the SSA loop-carried variable in ForStmt, and YieldStmt is used
    to produce values in each iteration. This combination represents a stateful
    generator pattern that was not previously covered.
    """

    def test_for_stmt_with_iter_arg_and_yield(self):
        """Test ForStmt with IterArg + YieldStmt - running sum generator.

        This test covers the SSA loop pattern where:
        - IterArg carries state across iterations (accumulator)
        - YieldStmt produces values in each iteration
        - return_vars pass the updated state to next iteration

        Example: Generate running sum: 0, 1, 3, 6, 10, ...
        """
        span = ir.Span.unknown()

        # Loop variable
        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)

        # IterArg: running sum with initial value 0
        init_value = ir.ConstInt(0, DataType.INT64, span)
        sum_acc = ir.IterArg("sum", ir.ScalarType(DataType.INT64), init_value, span)

        # Loop body:
        # 1. sum_next = sum + i
        # 2. yield sum_next
        sum_next = ir.Var("sum_next", ir.ScalarType(DataType.INT64), span)
        update_stmt = ir.AssignStmt(sum_next, ir.Add(sum_acc, i, DataType.INT64, span), span)
        yield_stmt = ir.YieldStmt([sum_next], span)
        loop_body = ir.SeqStmts([update_stmt, yield_stmt], span)

        # ForStmt with both IterArg and YieldStmt
        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            ir.ConstInt(10, DataType.INT64, span),
            ir.ConstInt(1, DataType.INT64, span),
            iter_args=[sum_acc],        # IterArg for state
            body=loop_body,              # Body contains YieldStmt
            return_vars=[sum_next],      # Return updated state
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)

        # Verify structure
        assert isinstance(restored, ir.ForStmt)
        assert len(restored.iter_args) == 1
        assert isinstance(restored.iter_args[0], ir.IterArg)
        assert restored.iter_args[0].name == "sum"
        assert isinstance(restored.body, ir.SeqStmts)
        assert len(restored.body.stmts) == 2
        assert isinstance(restored.body.stmts[0], ir.AssignStmt)
        assert isinstance(restored.body.stmts[1], ir.YieldStmt)
        assert len(restored.return_vars) == 1

        ir.assert_structural_equal(for_stmt, restored, enable_auto_mapping = True)

    def test_for_stmt_with_multiple_iter_args_and_yield(self):
        """Test ForStmt with multiple IterArgs + YieldStmt - complex generator.

        Example: Generate Fibonacci sequence with two accumulators.
        """
        span = ir.Span.unknown()

        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)

        # Two IterArgs: a and b for Fibonacci
        a_init = ir.ConstInt(0, DataType.INT64, span)
        b_init = ir.ConstInt(1, DataType.INT64, span)
        a_acc = ir.IterArg("a", ir.ScalarType(DataType.INT64), a_init, span)
        b_acc = ir.IterArg("b", ir.ScalarType(DataType.INT64), b_init, span)

        # Loop body:
        # 1. yield a
        # 2. a_next = b
        # 3. b_next = a + b
        yield_stmt = ir.YieldStmt([a_acc], span)
        a_next = ir.Var("a_next", ir.ScalarType(DataType.INT64), span)
        b_next = ir.Var("b_next", ir.ScalarType(DataType.INT64), span)

        a_assign = ir.AssignStmt(a_next, b_acc, span)
        b_assign = ir.AssignStmt(b_next, ir.Add(a_acc, b_acc, DataType.INT64, span), span)

        loop_body = ir.SeqStmts([yield_stmt, a_assign, b_assign], span)

        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            ir.ConstInt(10, DataType.INT64, span),
            ir.ConstInt(1, DataType.INT64, span),
            iter_args=[a_acc, b_acc],
            body=loop_body,
            return_vars=[a_next, b_next],
            span=span
        )

        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)

        # Verify structure
        assert isinstance(restored, ir.ForStmt)
        assert len(restored.iter_args) == 2
        assert isinstance(restored.iter_args[0], ir.IterArg)
        assert isinstance(restored.iter_args[1], ir.IterArg)
        assert restored.iter_args[0].name == "a"
        assert restored.iter_args[1].name == "b"
        assert isinstance(restored.body, ir.SeqStmts)
        # Find YieldStmt in body
        has_yield = any(isinstance(stmt, ir.YieldStmt) for stmt in restored.body.stmts)
        assert has_yield
        assert len(restored.return_vars) == 2

        ir.assert_structural_equal(for_stmt, restored, enable_auto_mapping = True)

    def test_for_stmt_with_iter_arg_and_conditional_yield(self):
        """Test ForStmt with IterArg + conditional YieldStmt.

        Example: Accumulate and yield only when condition is met.
        """
        span = ir.Span.unknown()

        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)

        # IterArg: counter
        counter_init = ir.ConstInt(0, DataType.INT64, span)
        counter = ir.IterArg("counter", ir.ScalarType(DataType.INT64), counter_init, span)

        # Loop body:
        # counter_next = counter + 1
        # if i % 2 == 0:
        #     yield counter_next
        counter_next = ir.Var("counter_next", ir.ScalarType(DataType.INT64), span)
        update_stmt = ir.AssignStmt(
            counter_next,
            ir.Add(counter, ir.ConstInt(1, DataType.INT64, span), DataType.INT64, span),
            span
        )

        # Condition: i % 2 == 0 (even numbers)
        mod_expr = ir.FloorMod(i, ir.ConstInt(2, DataType.INT64, span), DataType.INT64, span)
        condition = ir.Eq(mod_expr, ir.ConstInt(0, DataType.INT64, span), DataType.INT64, span)
        yield_stmt = ir.YieldStmt([counter_next], span)
        if_stmt = ir.IfStmt(condition, yield_stmt, return_vars=[], span=span)

        loop_body = ir.SeqStmts([update_stmt, if_stmt], span)

        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, DataType.INT64, span),
            ir.ConstInt(10, DataType.INT64, span),
            ir.ConstInt(1, DataType.INT64, span),
            iter_args=[counter],
            body=loop_body,
            return_vars=[counter_next],
            span=span
        )

        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)

        # Verify structure
        assert isinstance(restored, ir.ForStmt)
        assert len(restored.iter_args) == 1
        assert isinstance(restored.iter_args[0], ir.IterArg)
        assert isinstance(restored.body, ir.SeqStmts)
        # Body should have update + if statement
        assert len(restored.body.stmts) == 2
        assert isinstance(restored.body.stmts[0], ir.AssignStmt)
        assert isinstance(restored.body.stmts[1], ir.IfStmt)
        assert isinstance(restored.body.stmts[1].then_body, ir.YieldStmt)

        ir.assert_structural_equal(for_stmt, restored, enable_auto_mapping = True)
