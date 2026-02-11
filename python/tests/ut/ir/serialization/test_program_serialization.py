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
"""Comprehensive tests for Program serialization - the top-level IR container."""

import pytest
from pypto import ir
from pypto.ir import DataType


class TestProgram:

    def test_program_single_function(self):
        """Test basic Program with a single simple function."""
        span = ir.Span.unknown()

        # Create a simple function that returns a constant
        const = ir.ConstInt(42, DataType.INT64, span)
        return_stmt = ir.ReturnStmt([const], span)

        function = ir.Function(
            name="get_answer",
            params=[],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=return_stmt,
            span=span
        )

        # Create program with the function
        program = ir.Program([function], name="simple_program", span=span)

        # Serialize and deserialize
        data = ir.serialize(program)
        restored = ir.deserialize(data)

        # Verify
        assert isinstance(restored, ir.Program)
        assert restored.name == "simple_program"
        assert len(restored.functions) == 1
        ir.assert_structural_equal(program, restored, enable_auto_mapping = True)

    def test_program_with_two_functions(self):
        """Test Program with two independent functions."""
        span = ir.Span.unknown()

        # Function 1: add(x, y) -> x + y
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y1 = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        add_body = ir.ReturnStmt([ir.Add(x1, y1, DataType.INT64, span)], span)
        add_func = ir.Function(
            name="add",
            params=[x1, y1],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=add_body,
            span=span
        )

        # Function 2: square(x) -> x * x
        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        mul_expr = ir.Mul(x2, x2, DataType.INT64, span)
        square_body = ir.ReturnStmt([mul_expr], span)
        square_func = ir.Function(
            name="square",
            params=[x2],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=square_body,
            span=span
        )

        # Create program with both functions
        program = ir.Program([add_func, square_func], name="two_func_program", span=span)

        # Serialize and deserialize
        data = ir.serialize(program)
        restored = ir.deserialize(data)

        # Verify
        assert isinstance(restored, ir.Program)
        assert restored.name == "two_func_program"
        assert len(restored.functions) == 2

        # Verify function names are preserved
        func_names = {func.name for func in restored.functions.values()}
        assert "add" in func_names
        assert "square" in func_names

        # Verify structural equality
        ir.assert_structural_equal(program, restored, enable_auto_mapping = True)

    def test_program_with_function_calling_another(self):
        """Test Program where one function calls another via GlobalVar."""
        span = ir.Span.unknown()

        # Function 1: helper(x) -> x * 2
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        helper_body = ir.ReturnStmt([ir.Mul(x1, ir.ConstInt(2, DataType.INT64, span), DataType.INT64, span)], span)
        helper_func = ir.Function("helper", [x1], [ir.ScalarType(DataType.INT64)], helper_body, span)

        # Function 2: main(a) -> helper(a) + 1
        # This calls the helper function
        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        helper_gvar = ir.GlobalVar("helper")
        helper_call = ir.Call(helper_gvar, [a], span)
        main_body = ir.ReturnStmt([ir.Add(helper_call, ir.ConstInt(1, DataType.INT64, span), DataType.INT64, span)], span)
        main_func = ir.Function("main", [a], [ir.ScalarType(DataType.INT64)], main_body, span)

        # Create program with both functions
        program = ir.Program([helper_func, main_func], name="call_program", span=span)

        # Serialize and deserialize
        data = ir.serialize(program)
        restored = ir.deserialize(data)

        # Verify structure
        assert isinstance(restored, ir.Program)
        assert len(restored.functions) == 2

        # Verify we can access the main function
        main_restored = restored.get_function("main")
        assert main_restored is not None
        assert main_restored.name == "main"

        # Verify the call to helper is preserved
        assert isinstance(main_restored.body, ir.ReturnStmt)
        return_expr = main_restored.body.value[0]
        assert isinstance(return_expr, ir.Add)
        # The left operand should be the Call to helper
        assert isinstance(return_expr.left, ir.Call)
        assert isinstance(return_expr.left.op, ir.GlobalVar)
        assert return_expr.left.op.name == "helper"

        # Verify structural equality
        ir.assert_structural_equal(program, restored, enable_auto_mapping = True)

    def test_program_with_chain_of_calls(self):
        """Test Program with chain of function calls: f1 -> f2 -> f3."""
        span = ir.Span.unknown()

        # Function 3: base(x) -> x + 1
        x3 = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        base_body = ir.ReturnStmt([ir.Add(x3, ir.ConstInt(1, DataType.INT64, span), DataType.INT64, span)], span)
        base_func = ir.Function("base", [x3], [ir.ScalarType(DataType.INT64)], base_body, span)

        # Function 2: middle(x) -> base(x) * 2
        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        base_call = ir.Call(ir.GlobalVar("base"), [x2], span)
        middle_body = ir.ReturnStmt([ir.Mul(base_call, ir.ConstInt(2, DataType.INT64, span), DataType.INT64, span)], span)
        middle_func = ir.Function("middle", [x2], [ir.ScalarType(DataType.INT64)], middle_body, span)

        # Function 1: top(x) -> middle(x) + 10
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        middle_call = ir.Call(ir.GlobalVar("middle"), [x1], span)
        top_body = ir.ReturnStmt([ir.Add(middle_call, ir.ConstInt(10, DataType.INT64, span), DataType.INT64, span)], span)
        top_func = ir.Function("top", [x1], [ir.ScalarType(DataType.INT64)], top_body, span)

        # Create program
        program = ir.Program([base_func, middle_func, top_func], name="chain_program", span=span)

        # Serialize and deserialize
        data = ir.serialize(program)
        restored = ir.deserialize(data)

        # Verify
        assert len(restored.functions) == 3

        # Verify top function calls middle
        top_restored = restored.get_function("top")
        assert isinstance(top_restored.body, ir.ReturnStmt)
        top_return_expr = top_restored.body.value[0]
        assert isinstance(top_return_expr, ir.Add)
        assert isinstance(top_return_expr.left, ir.Call)
        assert top_return_expr.left.op.name == "middle"

        # Verify middle function calls base
        middle_restored = restored.get_function("middle")
        middle_return_expr = middle_restored.body.value[0]
        assert isinstance(middle_return_expr, ir.Mul)
        assert isinstance(middle_return_expr.left, ir.Call)
        assert middle_return_expr.left.op.name == "base"

        ir.assert_structural_equal(program, restored, enable_auto_mapping = True)

    def test_program_with_recursive_structure(self):
        """Test Program with function that has recursive call structure."""
        span = ir.Span.unknown()

        # Function: factorial(n)
        # if n <= 1:
        #     return 1
        # else:
        #     return n * factorial(n - 1)

        n = ir.Var("n", ir.ScalarType(DataType.INT64), span)
        one = ir.ConstInt(1, DataType.INT64, span)

        # Condition: n <= 1
        condition = ir.Le(n, one, DataType.INT64, span)

        # Then branch: return 1
        then_body = ir.ReturnStmt([one], span)

        # Else branch: return n * factorial(n - 1)
        # Recursive call: factorial(n - 1)
        n_minus_1 = ir.Sub(n, one, DataType.INT64, span)
        factorial_call = ir.Call(ir.GlobalVar("factorial"), [n_minus_1], span)
        n_times_fact = ir.Mul(n, factorial_call, DataType.INT64, span)
        else_body = ir.ReturnStmt([n_times_fact], span)

        # Complete function body
        if_stmt = ir.IfStmt(condition, then_body, else_body, return_vars=[], span=span)

        factorial_func = ir.Function(
            name="factorial",
            params=[n],
            return_types=[ir.ScalarType(DataType.INT64)],
            body=if_stmt,
            span=span
        )

        # Create program with single recursive function
        program = ir.Program([factorial_func], name="recursive_program", span=span)

        # Serialize and deserialize
        data = ir.serialize(program)
        restored = ir.deserialize(data)

        # Verify
        assert len(restored.functions) == 1
        fact_restored = restored.get_function("factorial")
        assert fact_restored is not None

        # Verify the recursive call structure
        assert isinstance(fact_restored.body, ir.IfStmt)
        assert isinstance(fact_restored.body.else_body, ir.ReturnStmt)
        else_return_expr = fact_restored.body.else_body.value[0]
        assert isinstance(else_return_expr, ir.Mul)
        # Right operand should be the recursive call
        assert isinstance(else_return_expr.right, ir.Call)
        assert else_return_expr.right.op.name == "factorial"

        ir.assert_structural_equal(program, restored, enable_auto_mapping = True)

    def test_program_with_complex_functions(self):
        """Test Program with complex functions."""
        span = ir.Span.unknown()

        # Simple function: identity(x) -> x
        x_simple = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        identity_func = ir.Function(
            "identity",
            [x_simple],
            [ir.ScalarType(DataType.INT64)],
            ir.ReturnStmt([x_simple], span),
            span
        )

        # Medium complexity: abs_diff(a, b) -> if a > b then a - b else b - a
        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        b = ir.Var("b", ir.ScalarType(DataType.INT64), span)
        cond = ir.Gt(a, b, DataType.INT64, span)
        then_branch = ir.ReturnStmt([ir.Sub(a, b, DataType.INT64, span)], span)
        else_branch = ir.ReturnStmt([ir.Sub(b, a, DataType.INT64, span)], span)
        abs_diff_func = ir.Function(
            "abs_diff",
            [a, b],
            [ir.ScalarType(DataType.INT64)],
            ir.IfStmt(cond, then_branch, else_branch, return_vars=[], span=span),
            span
        )

        # Complex function: sum_range(start, end)
        # sum = 0
        # for i in start..end:
        #     sum = sum + i
        # return sum
        start = ir.Var("start", ir.ScalarType(DataType.INT64), span)
        end = ir.Var("end", ir.ScalarType(DataType.INT64), span)
        i = ir.Var("i", ir.ScalarType(DataType.INT64), span)
        sum_var = ir.Var("sum", ir.ScalarType(DataType.INT64), span)

        init_stmt = ir.AssignStmt(sum_var, ir.ConstInt(0, DataType.INT64, span), span)
        loop_body = ir.AssignStmt(sum_var, ir.Add(sum_var, i, DataType.INT64, span), span)
        for_stmt = ir.ForStmt(i, start, end, ir.ConstInt(1, DataType.INT64, span), [], loop_body, [], span)
        return_stmt = ir.ReturnStmt([sum_var], span)

        sum_range_func = ir.Function(
            "sum_range",
            [start, end],
            [ir.ScalarType(DataType.INT64)],
            ir.SeqStmts([init_stmt, for_stmt, return_stmt], span),
            span
        )

        # Create program with all three functions
        program = ir.Program(
            [identity_func, abs_diff_func, sum_range_func],
            name="mixed_complexity_program",
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(program)
        restored = ir.deserialize(data)

        # Verify
        assert len(restored.functions) == 3

        # Verify each function is correctly restored
        identity_restored = restored.get_function("identity")
        assert identity_restored is not None
        assert isinstance(identity_restored.body, ir.ReturnStmt)

        abs_diff_restored = restored.get_function("abs_diff")
        assert abs_diff_restored is not None
        assert isinstance(abs_diff_restored.body, ir.IfStmt)

        sum_range_restored = restored.get_function("sum_range")
        assert sum_range_restored is not None
        assert isinstance(sum_range_restored.body, ir.SeqStmts)
        assert len(sum_range_restored.body.stmts) == 3

        ir.assert_structural_equal(program, restored, enable_auto_mapping = True)


class TestProgramEdgeCases:
    """Test edge cases and boundary conditions for Program serialization."""

    def test_empty_program(self):
        """Test Program with no functions (edge case)."""
        span = ir.Span.unknown()
        program = ir.Program([], name="empty_program", span=span)

        data = ir.serialize(program)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.Program)
        assert len(restored.functions) == 0
        ir.assert_structural_equal(program, restored)

    def test_program_with_many_functions(self):
        """Test Program with many functions (stress test)."""
        span = ir.Span.unknown()
        functions = []

        # Create 20 simple functions
        for i in range(20):
            x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
            body = ir.ReturnStmt([ir.Add(x, ir.ConstInt(i, DataType.INT64, span), DataType.INT64, span)], span)
            func = ir.Function(f"func_{i}", [x], [ir.ScalarType(DataType.INT64)], body, span)
            functions.append(func)

        program = ir.Program(functions, name="many_funcs_program", span=span)

        # Serialize and deserialize
        data = ir.serialize(program)
        restored = ir.deserialize(data)

        # Verify
        assert len(restored.functions) == 20
        for i in range(20):
            func = restored.get_function(f"func_{i}")
            assert func is not None
            assert func.name == f"func_{i}"

        ir.assert_structural_equal(program, restored, enable_auto_mapping = True)

    def test_program_with_long_function_name(self):
        """Test Program with function having very long name."""
        span = ir.Span.unknown()
        long_name = "very_long_function_name_" + "x" * 1000

        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        body = ir.ReturnStmt([x], span)
        func = ir.Function(long_name, [x], [ir.ScalarType(DataType.INT64)], body, span)

        program = ir.Program([func], name="long_name_program", span=span)

        data = ir.serialize(program)
        restored = ir.deserialize(data)

        assert len(restored.functions) == 1
        restored_func = restored.get_function(long_name)
        assert restored_func is not None
        assert restored_func.name == long_name
        ir.assert_structural_equal(program, restored, enable_auto_mapping = True)


class TestProgramRealWorldScenarios:
    """Test real-world Program scenarios that would actually be used."""

    def test_program_like_real_compiler_output(self):
        """Test Program structure similar to real compiler output."""
        span = ir.Span.unknown()

        # Helper function: clamp(x, min_val, max_val)
        x_clamp = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        min_val = ir.Var("min_val", ir.ScalarType(DataType.INT64), span)
        max_val = ir.Var("max_val", ir.ScalarType(DataType.INT64), span)

        # if x < min_val: return min_val
        # elif x > max_val: return max_val
        # else: return x
        cond1 = ir.Lt(x_clamp, min_val, DataType.INT64, span)
        then1 = ir.ReturnStmt([min_val], span)

        cond2 = ir.Gt(x_clamp, max_val, DataType.INT64, span)
        then2 = ir.ReturnStmt([max_val], span)
        else2 = ir.ReturnStmt([x_clamp], span)
        inner_if = ir.IfStmt(cond2, then2, else2, return_vars=[], span=span)

        clamp_body = ir.IfStmt(cond1, then1, inner_if, return_vars=[], span=span)
        clamp_func = ir.Function("clamp", [x_clamp, min_val, max_val], [ir.ScalarType(DataType.INT64)], clamp_body, span)

        # Main processing function: process_array(data_tensor, size)
        data_tensor = ir.Var("data", ir.TensorType([], DataType.INT64), span)
        size = ir.Var("size", ir.ScalarType(DataType.INT64), span)
        idx = ir.Var("idx", ir.ScalarType(DataType.INT64), span)
        result = ir.Var("result", ir.ScalarType(DataType.INT64), span)

        # Initialize result
        init = ir.AssignStmt(result, ir.ConstInt(0, DataType.INT64, span), span)

        # Loop through array
        # call clamp on each element
        elem_call = ir.Call(ir.GlobalVar("clamp"), [idx, ir.ConstInt(0, DataType.INT64, span), ir.ConstInt(100, DataType.INT64, span)], span)
        loop_body = ir.AssignStmt(result, ir.Add(result, elem_call, DataType.INT64, span), span)
        for_loop = ir.ForStmt(idx, ir.ConstInt(0, DataType.INT64, span), size, ir.ConstInt(1, DataType.INT64, span), [], loop_body, [], span)

        return_result = ir.ReturnStmt([result], span)

        process_body = ir.SeqStmts([init, for_loop, return_result], span)
        process_func = ir.Function("process_array", [data_tensor, size], [ir.ScalarType(DataType.INT64)], process_body, span)

        # Entry point: main()
        main_call = ir.Call(ir.GlobalVar("process_array"), [ir.ConstInt(0, DataType.INT64, span), ir.ConstInt(10, DataType.INT64, span)], span)
        main_body = ir.ReturnStmt([main_call], span)
        main_func = ir.Function("main", [], [ir.ScalarType(DataType.INT64)], main_body, span)

        # Create complete program
        program = ir.Program([clamp_func, process_func, main_func], name="real_world_program", span=span)

        # Serialize and deserialize
        data = ir.serialize(program)
        restored = ir.deserialize(data)

        # Verify complete structure
        assert len(restored.functions) == 3
        assert restored.get_function("clamp") is not None
        assert restored.get_function("process_array") is not None
        assert restored.get_function("main") is not None

        # Verify main calls process_array
        main_restored = restored.get_function("main")
        assert isinstance(main_restored.body, ir.ReturnStmt)
        main_call_restored = main_restored.body.value[0]
        assert isinstance(main_call_restored, ir.Call)
        assert main_call_restored.op.name == "process_array"

        # Verify process_array calls clamp
        process_restored = restored.get_function("process_array")
        assert isinstance(process_restored.body, ir.SeqStmts)
        # The for loop body contains the call to clamp
        for_stmt_restored = process_restored.body.stmts[1]
        assert isinstance(for_stmt_restored, ir.ForStmt)

        ir.assert_structural_equal(program, restored, enable_auto_mapping = True)
