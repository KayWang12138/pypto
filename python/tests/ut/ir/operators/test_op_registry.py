# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comprehensive tests for the operator registration system.

Tests cover:
- TileType construction and validation
- TensorAdd and BlockAdd operations
- Type deduction for various input combinations
- Broadcasting behavior
- Dynamic dimension handling
- Error cases
"""

import pytest
from pypto import ir
from pypto.ir import DataType


class TestDynamicDimension:
    """Test suite for dynamic dimension constant."""

    def test_dynamic_dimension_constant(self):
        """Test dynamic dimension constant."""
        ib = ir.IRBuilder()

        with ib.program("test_dynamic_dim_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                f.return_type(ir.ScalarType(DataType.INT32))
                # Check that DYNAMIC_DIM is -1
                assert ir.DYNAMIC_DIM == -1

                span = ir.Span.unknown()
                dynamic_dim = ir.ConstInt(ir.DYNAMIC_DIM, DataType.INT32, span)
                assert dynamic_dim.value == -1

                result = ib.let("result", dynamic_dim)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)


class TestTensorAddOps:
    """Test suite for tensor add operations with type deduction."""

    def test_tensor_add_same_shape(self):
        """Test TensorAdd with identical shapes."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_add_same_shape_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([4, 8], DataType.FP32))
                b = f.param("b", ir.TensorType([4, 8], DataType.FP32))
                f.return_type(ir.TensorType([4, 8], DataType.FP32))

                result = ib.let("result", ir.create_op_call("tensor.add", [a, b], ir.Span.unknown()))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.add" in ir_str
        print(ir_str)

    def test_tensor_add_broadcasting(self):
        """Test TensorAdd with broadcasting."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_add_broadcast_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([4, 8], DataType.FP32))
                b = f.param("b", ir.TensorType([8], DataType.FP32))
                f.return_type(ir.TensorType([4, 8], DataType.FP32))

                # [4, 8] + [8] should broadcast to [4, 8]
                result = ib.let("result", ir.create_op_call("tensor.add", [a, b], ir.Span.unknown()))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.add" in ir_str
        print(ir_str)

    def test_tensor_add_broadcasting_with_one(self):
        """Test TensorAdd broadcasting with dimension of size 1."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_add_broadcast_one_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([4, 1], DataType.FP32))
                b = f.param("b", ir.TensorType([8], DataType.FP32))
                f.return_type(ir.TensorType([4, 8], DataType.FP32))

                # [4, 1] + [8] should broadcast to [4, 8]
                result = ib.let("result", ir.create_op_call("tensor.add", [a, b], ir.Span.unknown()))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.add" in ir_str
        print(ir_str)

    def test_tensor_add_type_promotion(self):
        """Test TensorAdd with different data types."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_add_type_promo_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([8], DataType.INT32))
                b = f.param("b", ir.TensorType([8], DataType.FP32))
                f.return_type(ir.TensorType([8], DataType.FP32))

                # INT32 + FP32 should promote to FP32
                result = ib.let("result", ir.create_op_call("tensor.add", [a, b], ir.Span.unknown()))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.add" in ir_str
        print(ir_str)

    def test_tensor_add_wrong_arg_count(self):
        """Test TensorAdd with wrong number of arguments."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_add_wrong_args_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([8], DataType.FP32))
                b = f.param("b", ir.TensorType([8], DataType.FP32))
                c = f.param("c", ir.TensorType([8], DataType.FP32))
                f.return_type(ir.TensorType([8], DataType.FP32))

                # Too few arguments
                with pytest.raises(Exception):
                    ir.create_op_call("tensor.add", [a], ir.Span.unknown())

                # Too many arguments
                with pytest.raises(Exception):
                    ir.create_op_call("tensor.add", [a, b, c], ir.Span.unknown())

                # Normal case
                result = ib.let("result", ir.create_op_call("tensor.add", [a, b], ir.Span.unknown()))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.add" in ir_str
        print(ir_str)

    def test_tensor_add_wrong_type(self):
        """Test TensorAdd with non-tensor arguments."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_add_wrong_type_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                s = f.param("s", ir.ScalarType(DataType.FP32))
                t = f.param("t", ir.TensorType([8], DataType.FP32))
                f.return_type(ir.TensorType([8], DataType.FP32))

                # Scalar + Tensor should raise
                with pytest.raises(Exception):
                    ir.create_op_call("tensor.add", [s, t], ir.Span.unknown())

                ib.return_stmt(t)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)


class TestTensorArithOps:
    """Test suite for tensor sub, mul, div operations."""

    def test_tensor_sub_mul_div(self):
        """Test other tensor operations (sub, mul, div)."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_sub_mul_div_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([8], DataType.FP32))
                b = f.param("b", ir.TensorType([8], DataType.FP32))
                f.return_type(ir.TensorType([8], DataType.FP32))

                # Test sub
                call_sub = ib.let("sub_result", ir.create_op_call("tensor.sub", [a, b], ir.Span.unknown()))
                # Test mul
                call_mul = ib.let("mul_result", ir.create_op_call("tensor.mul", [a, b], ir.Span.unknown()))
                # Test div
                call_div = ib.let("div_result", ir.create_op_call("tensor.div", [a, b], ir.Span.unknown()))

                ib.return_stmt(call_div)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.sub" in ir_str
        assert "tensor.mul" in ir_str
        assert "tensor.div" in ir_str
        print(ir_str)

    def test_call_with_explicit_type(self):
        """Test Call constructor with explicit type parameter."""
        ib = ir.IRBuilder()

        with ib.program("test_call_explicit_type_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([8], DataType.FP32))
                b = f.param("b", ir.TensorType([8], DataType.FP32))
                f.return_type(ir.TensorType([8], DataType.FP32))

                span = ir.Span.unknown()
                op = ir.get_op("tensor.add")
                dim8 = ir.ConstInt(8, DataType.INT32, span)
                result_type = ir.TensorType([dim8], DataType.FP32)
                call = ir.Call(op, [a, b], {}, result_type, span)

                # Verify type is set correctly
                assert isinstance(call.type, ir.TensorType)
                assert call.type.dtype == DataType.FP32

                result = ib.let("result", call)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.add" in ir_str
        print(ir_str)


class TestMatMulKwargs:
    """Test suite for tensor.matmul kwargs."""

    def test_matmul_with_valid_kwargs(self):
        """Test tensor.matmul with valid kwargs."""
        ib = ir.IRBuilder()

        with ib.program("test_matmul_valid_kwargs_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([64, 128], DataType.FP16))
                b = f.param("b", ir.TensorType([128, 64], DataType.FP16))
                f.return_type(ir.TensorType([64, 64], DataType.FP32))

                kwargs = {"out_dtype": DataType.FP32, "a_trans": False, "b_trans": False}
                result = ib.let("result", ir.create_op_call("tensor.matmul", [a, b], kwargs, ir.Span.unknown()))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.matmul" in ir_str
        print(ir_str)

    def test_matmul_with_transpose_kwargs(self):
        """Test tensor.matmul with transpose kwargs."""
        ib = ir.IRBuilder()

        with ib.program("test_matmul_transpose_kwargs_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([128, 64], DataType.FP16))  # Will be transposed
                b = f.param("b", ir.TensorType([128, 64], DataType.FP16))
                f.return_type(ir.TensorType([64, 64], DataType.FP16))

                kwargs = {"a_trans": True, "b_trans": False}
                result = ib.let("result", ir.create_op_call("tensor.matmul", [a, b], kwargs, ir.Span.unknown()))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.matmul" in ir_str
        print(ir_str)

    def test_matmul_with_unknown_kwarg(self):
        """Test tensor.matmul with unknown kwarg should raise error."""
        ib = ir.IRBuilder()

        with ib.program("test_matmul_unknown_kwarg_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([64, 64], DataType.FP16))
                b = f.param("b", ir.TensorType([64, 64], DataType.FP16))
                f.return_type(ir.TensorType([64, 64], DataType.FP16))

                # Unknown kwarg should raise ValueError
                kwargs = {"unknown_param": 123, "a_trans": False}
                with pytest.raises(Exception) as exc_info:
                    ir.create_op_call("tensor.matmul", [a, b], kwargs, ir.Span.unknown())

                assert "unknown" in str(exc_info.value).lower() or "Unknown" in str(exc_info.value)

                # Normal case to complete the function
                result = ib.let("result", ir.create_op_call("tensor.matmul", [a, b], {}, ir.Span.unknown()))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.matmul" in ir_str
        print(ir_str)

    def test_matmul_with_wrong_type_kwarg(self):
        """Test tensor.matmul with wrong type kwarg should raise error."""
        ib = ir.IRBuilder()

        with ib.program("test_matmul_wrong_type_kwarg_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([64, 64], DataType.FP16))
                b = f.param("b", ir.TensorType([64, 64], DataType.FP16))
                f.return_type(ir.TensorType([64, 64], DataType.FP16))

                # Wrong type for bool kwarg (passing string instead of bool)
                kwargs = {"a_trans": "true"}  # Should be bool, not string
                with pytest.raises(Exception) as exc_info:
                    ir.create_op_call("tensor.matmul", [a, b], kwargs, ir.Span.unknown())

                error_msg = str(exc_info.value).lower()
                assert "type" in error_msg or "incompatible" in error_msg

                # Normal case
                result = ib.let("result", ir.create_op_call("tensor.matmul", [a, b], {}, ir.Span.unknown()))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.matmul" in ir_str
        print(ir_str)


class TestCastAndReductionKwargs:
    """Test suite for tensor.cast and reduction kwargs."""

    def test_cast_with_datatype_kwarg(self):
        """Test tensor.cast with DataType kwarg."""
        ib = ir.IRBuilder()

        with ib.program("test_cast_datatype_kwarg_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([8], DataType.FP16))
                f.return_type(ir.TensorType([8], DataType.FP32))

                kwargs = {"target_type": DataType.FP32}
                result = ib.let("result", ir.create_op_call("tensor.cast", [a], kwargs, ir.Span.unknown()))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.cast" in ir_str
        print(ir_str)

    def test_reduction_with_kwargs(self):
        """Test tensor reduction operations with kwargs."""
        ib = ir.IRBuilder()

        with ib.program("test_reduction_kwargs_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([4, 8], DataType.FP32))
                f.return_type(ir.TensorType([4, 1], DataType.FP32))

                kwargs = {"axis": -1, "keep_dim": True}
                result = ib.let("result", ir.create_op_call("tensor.row_max", [a], kwargs, ir.Span.unknown()))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.row_max" in ir_str
        print(ir_str)


class TestOperatorRegistration:
    """Test suite for operator registration queries."""

    def test_operator_registration_status(self):
        """Test operator registration queries."""
        assert ir.is_op_registered("tensor.add")
        assert ir.is_op_registered("tensor.sub")
        assert ir.is_op_registered("tensor.mul")
        assert ir.is_op_registered("tensor.div")
        assert not ir.is_op_registered("nonexistent.op")

    def test_get_op(self):
        """Test getting operator instances."""
        tensor_add_op = ir.get_op("tensor.add")
        assert tensor_add_op.name == "tensor.add"

        with pytest.raises(Exception):
            ir.get_op("nonexistent.op")

    def test_test_op_kwarg_schema(self):
        """Test that test.op has kwarg schema defined."""
        test_op = ir.get_op("test.op")
        assert test_op.has_attr("int_attr")
        assert test_op.has_attr("string_attr")
        assert test_op.has_attr("bool_attr")

    def test_test_op_all_kwarg_keys(self):
        """Test all kwarg keys of test.op."""
        test_op = ir.get_op("test.op")
        keys = test_op.get_attr_keys()
        assert "int_attr" in keys
        assert "string_attr" in keys
        assert "bool_attr" in keys
        assert len(keys) == 3

    def test_test_op_nonexistent_kwarg(self):
        """Test checking non-existent kwargs."""
        test_op = ir.get_op("test.op")
        assert not test_op.has_attr("nonexistent")
        assert not test_op.has_attr("device")
        assert not test_op.has_attr("priority")

    def test_test_op_kwarg_isolation(self):
        """Test that test.op kwarg schema is isolated from other operators."""
        test_op = ir.get_op("test.op")
        tensor_add_op = ir.get_op("tensor.add")

        assert test_op.has_attr("int_attr")
        assert test_op.has_attr("string_attr")
        assert test_op.has_attr("bool_attr")

        assert not tensor_add_op.has_attr("int_attr")
        assert not tensor_add_op.has_attr("string_attr")
        assert not tensor_add_op.has_attr("bool_attr")

    def test_matmul_kwarg_schema(self):
        """Test that tensor.matmul has correct kwarg schema."""
        matmul_op = ir.get_op("tensor.matmul")
        assert matmul_op.has_attr("out_dtype")
        assert matmul_op.has_attr("a_trans")
        assert matmul_op.has_attr("b_trans")
        assert matmul_op.has_attr("c_matrix_nz")

        keys = matmul_op.get_attr_keys()
        assert "out_dtype" in keys
        assert "a_trans" in keys
        assert "b_trans" in keys

    def test_cast_kwarg_schema(self):
        """Test that tensor.cast has correct kwarg schema."""
        cast_op = ir.get_op("tensor.cast")
        assert cast_op.has_attr("target_type")
        assert cast_op.has_attr("mode")

    def test_reduction_kwarg_schema(self):
        """Test that tensor reduction ops have correct kwarg schema."""
        row_max_op = ir.get_op("tensor.row_max")
        row_sum_op = ir.get_op("tensor.row_sum")
        assert row_max_op.has_attr("axis")
        assert row_max_op.has_attr("keep_dim")
        assert row_sum_op.has_attr("axis")
        assert row_sum_op.has_attr("keep_dim")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
