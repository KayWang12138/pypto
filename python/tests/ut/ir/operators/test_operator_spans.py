# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for automatic span capture in operators."""

import pytest
from pypto import ir
from pypto.ir import DataType


class TestVarOperatorSpans:
    """Test span capture for Var operators."""

    def test_var_binary_operators_capture_span(self):
        """Test that binary operators on Var capture span correctly."""
        ib = ir.IRBuilder()

        with ib.program("test_var_binary_ops_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.INT32))
                y = f.param("y", ir.ScalarType(DataType.INT32))
                f.return_type(ir.ScalarType(DataType.INT32))

                # Test each binary operator
                add_result = x + y
                sub_result = x - y
                mul_result = x * y
                div_result = x / y
                floordiv_result = x // y
                mod_result = x % y
                pow_result = x**y

                base_line = add_result.span.begin_line
                base_column = add_result.span.begin_column
                # All results should have valid spans pointing to this file
                for result, line_offset in [
                    (add_result, 0),
                    (sub_result, 1),
                    (mul_result, 2),
                    (div_result, 3),
                    (floordiv_result, 4),
                    (mod_result, 5),
                    (pow_result, 6),
                ]:
                    assert result.span.filename.endswith("test_operator_spans.py")
                    assert result.span.begin_line == base_line + line_offset
                    assert result.span.begin_column == base_column

                result = ib.let("result", add_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_var_comparison_operators_capture_span(self):
        """Test that comparison operators on Var capture span correctly."""
        ib = ir.IRBuilder()

        with ib.program("test_var_comparison_ops_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.INT32))
                y = f.param("y", ir.ScalarType(DataType.INT32))
                f.return_type(ir.ScalarType(DataType.INT32))

                # Test each comparison operator
                eq_result = x == y
                ne_result = x != y
                lt_result = x < y
                le_result = x <= y
                gt_result = x > y
                ge_result = x >= y

                base_line = eq_result.span.begin_line
                base_column = eq_result.span.begin_column
                for result, line_offset in [
                    (eq_result, 0),
                    (ne_result, 1),
                    (lt_result, 2),
                    (le_result, 3),
                    (gt_result, 4),
                    (ge_result, 5),
                ]:
                    assert result.span.filename.endswith("test_operator_spans.py")
                    assert result.span.begin_line == base_line + line_offset
                    assert result.span.begin_column == base_column

                result = ib.let("result", eq_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_var_unary_operators_capture_span(self):
        """Test that unary operators on Var capture span correctly."""
        ib = ir.IRBuilder()

        with ib.program("test_var_unary_ops_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.INT32))
                f.return_type(ir.ScalarType(DataType.INT32))

                neg_result = -x

                assert neg_result.span.filename.endswith("test_operator_spans.py")
                assert neg_result.span.begin_line > 0

                result = ib.let("result", neg_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_var_reverse_operators_capture_span(self):
        """Test that reverse operators (int on left) capture span correctly."""
        ib = ir.IRBuilder()

        with ib.program("test_var_reverse_ops_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.INT32))
                f.return_type(ir.ScalarType(DataType.INT32))

                # Test reverse operators with Python int on left
                radd_result = 5 + x
                rsub_result = 5 - x
                rmul_result = 5 * x
                rdiv_result = 5 / x
                rfloordiv_result = 5 // x
                rmod_result = 5 % x
                rpow_result = 5**x

                for op_result in [
                    radd_result,
                    rsub_result,
                    rmul_result,
                    rdiv_result,
                    rfloordiv_result,
                    rmod_result,
                    rpow_result,
                ]:
                    assert op_result.span.filename.endswith("test_operator_spans.py")
                    assert op_result.span.begin_line > 0

                result = ib.let("result", radd_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_var_with_tensortype_raises_error(self):
        """Test that operators on Var with TensorType raise appropriate error."""
        ib = ir.IRBuilder()

        with ib.program("test_var_tensortype_error_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                tensor_var = f.param("t", ir.TensorType([128, 256], DataType.FP32))
                scalar_var = f.param("x", ir.ScalarType(DataType.INT32))
                f.return_type(ir.ScalarType(DataType.INT32))

                with pytest.raises(TypeError, match="ScalarType"):
                    _ = tensor_var + scalar_var

                ib.return_stmt(scalar_var)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)


class TestScalarExprOperatorSpans:
    """Test span capture for ScalarExpr operators."""

    def test_constint_binary_operators_capture_span(self):
        """Test that binary operators on ConstInt capture span correctly."""
        ib = ir.IRBuilder()

        with ib.program("test_constint_binary_ops_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                f.return_type(ir.ScalarType(DataType.INT32))

                x = ir.ConstInt(10, DataType.INT32, ir.Span.unknown())
                y = ir.ConstInt(5, DataType.INT32, ir.Span.unknown())

                add_result = x + y
                sub_result = x - y
                mul_result = x * y
                div_result = x / y

                for op_result in [add_result, sub_result, mul_result, div_result]:
                    assert op_result.span.filename.endswith("test_operator_spans.py")
                    assert op_result.span.begin_line > 0

                result = ib.let("result", add_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_constfloat_operators_capture_span(self):
        """Test that operators on ConstFloat capture span correctly."""
        ib = ir.IRBuilder()

        with ib.program("test_constfloat_ops_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                f.return_type(ir.ScalarType(DataType.FP32))

                x = ir.ConstFloat(3.14, DataType.FP32, ir.Span.unknown())
                y = ir.ConstFloat(2.0, DataType.FP32, ir.Span.unknown())

                mul_result = x * y

                assert mul_result.span.filename.endswith("test_operator_spans.py")
                assert mul_result.span.begin_line > 0

                result = ib.let("result", mul_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_scalarexpr_comparison_operators_capture_span(self):
        """Test that comparison operators on ScalarExpr capture span correctly."""
        ib = ir.IRBuilder()

        with ib.program("test_scalarexpr_comparison_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                f.return_type(ir.ScalarType(DataType.INT32))

                x = ir.ConstInt(10, DataType.INT32, ir.Span.unknown())
                y = ir.ConstInt(5, DataType.INT32, ir.Span.unknown())

                eq_result = x == y
                lt_result = x < y
                ge_result = x >= y

                for op_result in [eq_result, lt_result, ge_result]:
                    assert op_result.span.filename.endswith("test_operator_spans.py")
                    assert op_result.span.begin_line > 0

                result = ib.let("result", eq_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_scalarexpr_unary_operator_captures_span(self):
        """Test that unary operator on ScalarExpr captures span correctly."""
        ib = ir.IRBuilder()

        with ib.program("test_scalarexpr_unary_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                f.return_type(ir.ScalarType(DataType.INT32))

                x = ir.ConstInt(42, DataType.INT32, ir.Span.unknown())
                neg_result = -x

                assert neg_result.span.filename.endswith("test_operator_spans.py")
                assert neg_result.span.begin_line > 0

                result = ib.let("result", neg_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)


class TestMixedOperatorSpans:
    """Test span capture for mixed Var and ScalarExpr operators."""

    def test_var_with_constint_captures_span(self):
        """Test operators between Var and ConstInt."""
        ib = ir.IRBuilder()

        with ib.program("test_var_constint_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.INT32))
                f.return_type(ir.ScalarType(DataType.INT32))

                five = ir.ConstInt(5, DataType.INT32, ir.Span.unknown())
                add_result = x + five

                assert add_result.span.filename.endswith("test_operator_spans.py")
                assert add_result.span.begin_line > 0

                result = ib.let("result", add_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_var_with_python_int_captures_span(self):
        """Test operators between Var and Python int (auto-normalized)."""
        ib = ir.IRBuilder()

        with ib.program("test_var_python_int_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.INT32))
                f.return_type(ir.ScalarType(DataType.INT32))

                mul_result = x * 2

                assert mul_result.span.filename.endswith("test_operator_spans.py")
                assert mul_result.span.begin_line > 0

                result = ib.let("result", mul_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_constint_with_python_int_captures_span(self):
        """Test operators between ConstInt and Python int."""
        ib = ir.IRBuilder()

        with ib.program("test_constint_python_int_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                f.return_type(ir.ScalarType(DataType.INT32))

                x = ir.ConstInt(10, DataType.INT32, ir.Span.unknown())
                add_result = x + 5

                assert add_result.span.filename.endswith("test_operator_spans.py")
                assert add_result.span.begin_line > 0

                result = ib.let("result", add_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)


class TestComplexExpressionSpans:
    """Test span capture for complex expressions with multiple operations."""

    def test_nested_operations_capture_different_spans(self):
        """Test that each operation in a complex expression captures its own span."""
        ib = ir.IRBuilder()

        with ib.program("test_nested_ops_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.FP32))
                y = f.param("y", ir.ScalarType(DataType.FP32))
                z = f.param("z", ir.ScalarType(DataType.FP32))
                f.return_type(ir.ScalarType(DataType.FP32))

                # Each operation should capture its own span
                temp1 = x + y
                temp2 = temp1 * z
                final = temp2 - x

                assert temp1.span.filename.endswith("test_operator_spans.py")
                assert temp2.span.filename.endswith("test_operator_spans.py")
                assert final.span.filename.endswith("test_operator_spans.py")

                # Each should have different line numbers
                assert temp1.span.begin_line != temp2.span.begin_line
                assert temp2.span.begin_line != final.span.begin_line

                result = ib.let("result", final)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_complex_arithmetic_expression(self):
        """Test complex arithmetic expressions."""
        ib = ir.IRBuilder()

        with ib.program("test_complex_arith_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.INT32))
                y = f.param("y", ir.ScalarType(DataType.INT32))
                f.return_type(ir.ScalarType(DataType.INT32))

                # Complex expression: (x + y) * 2 - x
                expr_result = (x + y) * 2 - x

                assert expr_result.span.filename.endswith("test_operator_spans.py")
                assert expr_result.span.begin_line > 0

                result = ib.let("result", expr_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_comparison_in_complex_expression(self):
        """Test comparison operators in complex expressions."""
        ib = ir.IRBuilder()

        with ib.program("test_comparison_complex_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.INT32))
                y = f.param("y", ir.ScalarType(DataType.INT32))
                f.return_type(ir.ScalarType(DataType.INT32))

                # Expression with comparison: (x + 5) < (y * 2)
                cmp_result = (x + 5) < (y * 2)

                assert cmp_result.span.filename.endswith("test_operator_spans.py")
                assert cmp_result.span.begin_line > 0

                result = ib.let("result", cmp_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)


class TestSpanWithDifferentDataTypes:
    """Test span capture works correctly with different data types."""

    def test_int_operators_capture_span(self):
        """Test operators on INT32 variables."""
        ib = ir.IRBuilder()

        with ib.program("test_int_ops_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.INT32))
                y = f.param("y", ir.ScalarType(DataType.INT32))
                f.return_type(ir.ScalarType(DataType.INT32))

                add_result = x + y
                assert add_result.span.filename.endswith("test_operator_spans.py")
                assert add_result.span.begin_line > 0

                result = ib.let("result", add_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_float_operators_capture_span(self):
        """Test operators on FP32 variables."""
        ib = ir.IRBuilder()

        with ib.program("test_float_ops_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.FP32))
                y = f.param("y", ir.ScalarType(DataType.FP32))
                f.return_type(ir.ScalarType(DataType.FP32))

                div_result = x / y
                assert div_result.span.filename.endswith("test_operator_spans.py")
                assert div_result.span.begin_line > 0

                result = ib.let("result", div_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_bf16_operators_capture_span(self):
        """Test operators on BF16 variables."""
        ib = ir.IRBuilder()

        with ib.program("test_bf16_ops_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.BF16))
                y = f.param("y", ir.ScalarType(DataType.BF16))
                f.return_type(ir.ScalarType(DataType.BF16))

                mul_result = x * y
                assert mul_result.span.filename.endswith("test_operator_spans.py")
                assert mul_result.span.begin_line > 0

                result = ib.let("result", mul_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)


class TestAllOperators:
    """Comprehensive test covering all operators."""

    def test_all_binary_operators(self):
        """Test all binary operators capture spans."""
        ib = ir.IRBuilder()

        with ib.program("test_all_binary_ops_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.INT32))
                y = f.param("y", ir.ScalarType(DataType.INT32))
                f.return_type(ir.ScalarType(DataType.INT32))

                operators_and_results = [
                    ("+", x + y),
                    ("-", x - y),
                    ("*", x * y),
                    ("/", x / y),
                    ("//", x // y),
                    ("%", x % y),
                    ("**", x**y),
                    ("==", x == y),
                    ("!=", x != y),
                    ("<", x < y),
                    ("<=", x <= y),
                    (">", x > y),
                    (">=", x >= y),
                ]

                for op_name, op_result in operators_and_results:
                    assert op_result.span.filename.endswith("test_operator_spans.py"), f"Operator {op_name} failed"
                    assert op_result.span.begin_line > 0, f"Operator {op_name} has invalid line number"

                result = ib.let("result", operators_and_results[0][1])
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_all_reverse_operators(self):
        """Test all reverse operators capture spans."""
        ib = ir.IRBuilder()

        with ib.program("test_all_reverse_ops_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.ScalarType(DataType.INT32))
                f.return_type(ir.ScalarType(DataType.INT32))

                reverse_operators = [
                    ("radd", 5 + x),
                    ("rsub", 5 - x),
                    ("rmul", 5 * x),
                    ("rtruediv", 5 / x),
                    ("rfloordiv", 5 // x),
                    ("rmod", 5 % x),
                    ("rpow", 5**x),
                ]

                for op_name, op_result in reverse_operators:
                    assert op_result.span.filename.endswith("test_operator_spans.py"), (
                        f"Reverse operator {op_name} failed"
                    )
                    assert op_result.span.begin_line > 0, f"Reverse operator {op_name} has invalid line number"

                result = ib.let("result", reverse_operators[0][1])
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)
