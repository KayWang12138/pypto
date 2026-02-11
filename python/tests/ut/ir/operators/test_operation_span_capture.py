# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for automatic span capture in IR operations."""

import inspect

import pytest
from pypto import ir
from pypto.ir import DataType
from pypto.ir.op import block as block_ops
from pypto.ir.op import tensor as tensor_ops
from pypto.ir.utils import _get_span_or_capture


def get_current_line():
    """Get the current line number in the calling code."""
    frame = inspect.currentframe()
    if frame and frame.f_back:
        return frame.f_back.f_lineno
    return -1


class TestTensorOperationSpanCapture:
    """Test span capture for tensor operations."""

    def test_tensor_add_captures_span(self):
        """Tensor operations should capture caller span automatically."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_add_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([64], DataType.FP32))
                y = f.param("y", ir.TensorType([64], DataType.FP32))
                f.return_type(ir.TensorType([64], DataType.FP32))

                line_before = get_current_line()
                add_result = tensor_ops.add(x, y)

                assert add_result.span.filename.endswith("test_operation_span_capture.py")
                assert add_result.span.is_valid()
                assert add_result.span.begin_line == line_before + 1

                result = ib.let("result", add_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.add" in ir_str
        print(ir_str)

    def test_tensor_mul_captures_span(self):
        """Test mul operation span capture."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_mul_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([64], DataType.FP32))
                f.return_type(ir.TensorType([64], DataType.FP32))

                line_before = get_current_line()
                mul_result = tensor_ops.mul(x, 2.0)

                assert mul_result.span.filename.endswith("test_operation_span_capture.py")
                assert mul_result.span.is_valid()
                assert mul_result.span.begin_line == line_before + 1

                result = ib.let("result", mul_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_tensor_create_captures_span(self):
        """Test create operation span capture."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_create_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                f.return_type(ir.TensorType([64, 32], DataType.FP32))

                line_before = get_current_line()
                create_result = tensor_ops.create([64, 32], DataType.FP32)

                assert create_result.span.filename.endswith("test_operation_span_capture.py")
                assert create_result.span.is_valid()
                assert create_result.span.begin_line == line_before + 1

                result = ib.let("result", create_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.create" in ir_str
        print(ir_str)

    def test_tensor_matmul_captures_span(self):
        """Test matmul operation span capture."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_matmul_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                lhs = f.param("lhs", ir.TensorType([64, 32], DataType.FP32))
                rhs = f.param("rhs", ir.TensorType([32, 16], DataType.FP32))
                f.return_type(ir.TensorType([64, 16], DataType.FP32))

                matmul_result = tensor_ops.matmul(lhs, rhs)

                assert matmul_result.span.filename.endswith("test_operation_span_capture.py")
                assert matmul_result.span.is_valid()

                result = ib.let("result", matmul_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.matmul" in ir_str
        print(ir_str)

    def test_explicit_span_overrides_capture(self):
        """Explicit span should override automatic capture."""
        ib = ir.IRBuilder()

        with ib.program("test_explicit_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([64], DataType.FP32))
                y = f.param("y", ir.TensorType([64], DataType.FP32))
                f.return_type(ir.TensorType([64], DataType.FP32))

                explicit_span = ir.Span("custom.py", 100, 20)
                add_result = tensor_ops.add(x, y, span=explicit_span)

                assert add_result.span.filename == "custom.py"
                assert add_result.span.begin_line == 100
                assert add_result.span.begin_column == 20

                result = ib.let("result", add_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.add" in ir_str
        print(ir_str)

    def test_nested_operations_each_capture_own_span(self):
        """Each nested operation should capture its own span."""
        ib = ir.IRBuilder()

        with ib.program("test_nested_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([64], DataType.FP32))
                y = f.param("y", ir.TensorType([64], DataType.FP32))
                f.return_type(ir.TensorType([64], DataType.FP32))

                line_before_add = get_current_line()
                add_result = tensor_ops.add(x, y)
                line_before_mul = get_current_line()
                mul_result = tensor_ops.mul(add_result, 2.0)

                assert add_result.span.begin_line == line_before_add + 1
                assert mul_result.span.begin_line == line_before_mul + 1
                assert add_result.span.begin_line != mul_result.span.begin_line
                assert add_result.span.is_valid()
                assert mul_result.span.is_valid()

                result = ib.let("result", mul_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_tensor_view_captures_span(self):
        """Test view operation span capture."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_view_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t = f.param("tensor", ir.TensorType([64, 32], DataType.FP32))
                f.return_type(ir.TensorType([32, 16], DataType.FP32))

                view_result = tensor_ops.view(t, [32, 16], [0, 0])

                assert view_result.span.filename.endswith("test_operation_span_capture.py")
                assert view_result.span.is_valid()

                result = ib.let("result", view_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.view" in ir_str
        print(ir_str)

    def test_tensor_cast_captures_span(self):
        """Test cast operation span capture."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_cast_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([64], DataType.FP32))
                f.return_type(ir.TensorType([64], DataType.FP16))

                cast_result = tensor_ops.cast(x, DataType.FP16)

                assert cast_result.span.filename.endswith("test_operation_span_capture.py")
                assert cast_result.span.is_valid()

                result = ib.let("result", cast_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.cast" in ir_str
        print(ir_str)

    def test_tensor_exp_captures_span(self):
        """Test exp operation span capture."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_exp_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([64], DataType.FP32))
                f.return_type(ir.TensorType([64], DataType.FP32))

                exp_result = tensor_ops.exp(x)

                assert exp_result.span.filename.endswith("test_operation_span_capture.py")
                assert exp_result.span.is_valid()

                result = ib.let("result", exp_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.exp" in ir_str
        print(ir_str)

    def test_tensor_row_max_captures_span(self):
        """Test row_max operation span capture."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_row_max_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([64, 32], DataType.FP32))
                f.return_type(ir.TensorType([64, 1], DataType.FP32))

                row_max_result = tensor_ops.row_max(x, axis=-1)

                assert row_max_result.span.filename.endswith("test_operation_span_capture.py")
                assert row_max_result.span.is_valid()

                result = ib.let("result", row_max_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.row_max" in ir_str
        print(ir_str)


class TestBlockOperationSpanCapture:
    """Test span capture for block operations."""

    def test_block_matmul_captures_span(self):
        """Block operations should also capture span."""
        ib = ir.IRBuilder()

        with ib.program("test_block_matmul_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t_in = f.param("t_in", ir.TensorType([16, 16], DataType.FP16))
                f.return_type(ir.TensorType([16, 16], DataType.FP16))

                a = ib.let("a", block_ops.load(t_in, 0, 0, 16, 16))
                b = ib.let("b", block_ops.load(t_in, 0, 0, 16, 16))

                matmul_result = block_ops.matmul(a, b)

                assert matmul_result.span.filename.endswith("test_operation_span_capture.py")
                assert matmul_result.span.is_valid()

                result = ib.let("result", matmul_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "block.matmul" in ir_str
        print(ir_str)

    def test_block_add_captures_span(self):
        """Test block add operation span capture."""
        ib = ir.IRBuilder()

        with ib.program("test_block_add_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t_in = f.param("t_in", ir.TensorType([16, 16], DataType.FP16))
                f.return_type(ir.TensorType([16, 16], DataType.FP16))

                a = ib.let("a", block_ops.load(t_in, 0, 0, 16, 16))
                b = ib.let("b", block_ops.load(t_in, 0, 0, 16, 16))

                add_result = block_ops.add(a, b)

                assert add_result.span.filename.endswith("test_operation_span_capture.py")
                assert add_result.span.is_valid()

                result = ib.let("result", add_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "block.add" in ir_str
        print(ir_str)

    def test_block_load_captures_span(self):
        """Test block load operation span capture."""
        ib = ir.IRBuilder()

        with ib.program("test_block_load_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t = f.param("tensor", ir.TensorType([64, 64], DataType.FP16))
                f.return_type(ir.TensorType([64, 64], DataType.FP16))

                load_result = block_ops.load(t, 0, 0, 16, 16)

                assert load_result.span.filename.endswith("test_operation_span_capture.py")
                assert load_result.span.is_valid()

                result = ib.let("result", load_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "block.load" in ir_str
        print(ir_str)

    def test_block_exp_captures_span(self):
        """Test block exp operation span capture."""
        ib = ir.IRBuilder()

        with ib.program("test_block_exp_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t_in = f.param("t_in", ir.TensorType([16, 16], DataType.FP16))
                f.return_type(ir.TensorType([16, 16], DataType.FP16))

                tile = ib.let("tile", block_ops.load(t_in, 0, 0, 16, 16))
                exp_result = block_ops.exp(tile)

                assert exp_result.span.filename.endswith("test_operation_span_capture.py")
                assert exp_result.span.is_valid()

                result = ib.let("result", exp_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "block.exp" in ir_str
        print(ir_str)

    def test_block_row_max_captures_span(self):
        """Test block row_max operation span capture."""
        ib = ir.IRBuilder()

        with ib.program("test_block_row_max_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t_in = f.param("t_in", ir.TensorType([16, 16], DataType.FP16))
                f.return_type(ir.TensorType([16, 16], DataType.FP16))

                tile = ib.let("tile", block_ops.load(t_in, 0, 0, 16, 16))
                row_max_result = block_ops.row_max(tile)

                assert row_max_result.span.filename.endswith("test_operation_span_capture.py")
                assert row_max_result.span.is_valid()

                result = ib.let("result", row_max_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "block.row_max" in ir_str
        print(ir_str)

    def test_block_explicit_span_override(self):
        """Explicit span should override automatic capture for block ops."""
        ib = ir.IRBuilder()

        with ib.program("test_block_explicit_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t_in = f.param("t_in", ir.TensorType([16, 16], DataType.FP16))
                f.return_type(ir.TensorType([16, 16], DataType.FP16))

                a = ib.let("a", block_ops.load(t_in, 0, 0, 16, 16))
                b = ib.let("b", block_ops.load(t_in, 0, 0, 16, 16))

                explicit_span = ir.Span("block_ops.py", 42, 5)
                add_result = block_ops.add(a, b, span=explicit_span)

                assert add_result.span.filename == "block_ops.py"
                assert add_result.span.begin_line == 42
                assert add_result.span.begin_column == 5

                result = ib.let("result", add_result)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "block.add" in ir_str
        print(ir_str)


class TestUtilityFunction:
    """Test the _get_span_or_capture utility function."""

    def test_get_span_or_capture_returns_explicit_span(self):
        """When span provided, should return it unchanged."""
        ib = ir.IRBuilder()

        with ib.program("test_utility_explicit_span_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                f.return_type(ir.ScalarType(DataType.INT32))

                explicit = ir.Span("test.py", 42, 10)
                result_span = _get_span_or_capture(span=explicit)

                assert result_span.filename == "test.py"
                assert result_span.begin_line == 42
                assert result_span.begin_column == 10

                const = ir.ConstInt(0, DataType.INT32, ir.Span.unknown())
                result = ib.let("result", const)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_get_span_or_capture_auto_captures(self):
        """When span not provided, should capture from caller."""
        ib = ir.IRBuilder()

        with ib.program("test_utility_auto_capture_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                f.return_type(ir.ScalarType(DataType.INT32))

                line_before = get_current_line()
                result_span = _get_span_or_capture(frame_offset=0)

                assert result_span.filename.endswith("test_operation_span_capture.py")
                assert result_span.is_valid()
                assert result_span.begin_line == line_before + 1

                const = ir.ConstInt(0, DataType.INT32, ir.Span.unknown())
                result = ib.let("result", const)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_get_span_or_capture_with_frame_offset(self):
        """Test frame offset parameter works correctly."""
        ib = ir.IRBuilder()

        with ib.program("test_utility_frame_offset_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                f.return_type(ir.ScalarType(DataType.INT32))

                def wrapper():
                    return _get_span_or_capture(frame_offset=1)

                line_before = get_current_line()
                result_span = wrapper()

                assert result_span.filename.endswith("test_operation_span_capture.py")
                assert result_span.is_valid()
                assert result_span.begin_line == line_before + 1

                const = ir.ConstInt(0, DataType.INT32, ir.Span.unknown())
                result = ib.let("result", const)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
