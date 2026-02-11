# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comprehensive tests for tensor operations.

Tests cover:
- Memory operations (create, view, assemble)
- Matrix multiplication (matmul)
- Reduction operations (row_max, row_sum)
- Unary operations (exp, cast)
- Binary operations (maximum)
- Python helper functions
"""

import pytest
from pypto import ir
from pypto.ir import DataType
from pypto.ir.op import tensor


class TestTensorMemoryOps:
    """Test suite for tensor memory operations."""

    def test_tensor_create(self):
        """Test tensor.create operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_create_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                f.return_type(ir.TensorType([4, 8], DataType.FP32))
                result = ib.let("result", tensor.create([4, 8], DataType.FP32))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.create" in ir_str
        print(ir_str)

    def test_tensor_view(self):
        """Test tensor.view operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_view_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t = f.param("t", ir.TensorType([16, 32], DataType.FP16))
                f.return_type(ir.TensorType([8, 16], DataType.FP16))
                result = ib.let("result", tensor.view(t, [8, 16], [0, 0]))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.view" in ir_str
        print(ir_str)

    def test_tensor_assemble(self):
        """Test tensor.assemble operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_assemble_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                target = f.param("target", ir.TensorType([64, 128], DataType.FP32))
                source = f.param("source", ir.TensorType([64, 128], DataType.FP32))
                f.return_type(ir.TensorType([64, 128], DataType.FP32))
                result = ib.let("result", tensor.assemble(target, source, [0, 0]))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.assemble" in ir_str
        print(ir_str)


class TestTensorMatMulOps:
    """Test suite for tensor matrix multiplication operations."""

    def test_tensor_matmul(self):
        """Test tensor.matmul operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_matmul_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                lhs = f.param("lhs", ir.TensorType([4, 8], DataType.FP32))
                rhs = f.param("rhs", ir.TensorType([8, 16], DataType.FP32))
                f.return_type(ir.TensorType([4, 16], DataType.FP32))
                result = ib.let("result", tensor.matmul(lhs, rhs, out_dtype=DataType.FP32))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.matmul" in ir_str
        print(ir_str)

    def test_tensor_matmul_with_transpose(self):
        """Test tensor.matmul with transpose flags."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_matmul_transpose_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                lhs = f.param("lhs", ir.TensorType([8, 4], DataType.FP16))
                rhs = f.param("rhs", ir.TensorType([8, 4], DataType.FP16))
                f.return_type(ir.TensorType([4, 4], DataType.FP16))
                # Transpose lhs: [8, 4]^T x [8, 4] -> [4, 4]
                result = ib.let("result", tensor.matmul(lhs, rhs, out_dtype=DataType.FP16, a_trans=True, b_trans=False))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.matmul" in ir_str
        print(ir_str)


class TestTensorReductionOps:
    """Test suite for tensor reduction operations."""

    def test_tensor_row_max(self):
        """Test tensor.row_max reduction."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_row_max_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t = f.param("t", ir.TensorType([64, 128], DataType.FP16))
                f.return_type(ir.TensorType([64, 1], DataType.FP16))
                result = ib.let("result", tensor.row_max(t, axis=-1, keep_dim=1))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.row_max" in ir_str
        print(ir_str)

    def test_tensor_row_sum(self):
        """Test tensor.row_sum reduction."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_row_sum_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t = f.param("t", ir.TensorType([64, 128], DataType.FP16))
                f.return_type(ir.TensorType([64, 1], DataType.FP16))
                result = ib.let("result", tensor.row_sum(t, axis=-1, keep_dim=1))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.row_sum" in ir_str
        print(ir_str)


class TestTensorUnaryOps:
    """Test suite for tensor unary operations."""

    def test_tensor_exp(self):
        """Test tensor.exp operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_exp_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t = f.param("t", ir.TensorType([64, 128], DataType.FP16))
                f.return_type(ir.TensorType([64, 128], DataType.FP16))
                result = ib.let("result", tensor.exp(t))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.exp" in ir_str
        print(ir_str)

    def test_tensor_cast(self):
        """Test tensor.cast operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_cast_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t = f.param("t", ir.TensorType([64, 128], DataType.FP16))
                f.return_type(ir.TensorType([64, 128], DataType.FP32))
                result = ib.let("result", tensor.cast(t, DataType.FP32))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.cast" in ir_str
        print(ir_str)


class TestTensorBinaryOps:
    """Test suite for tensor binary operations."""

    def test_tensor_maximum(self):
        """Test tensor.maximum operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_maximum_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([64, 1], DataType.FP32))
                b = f.param("b", ir.TensorType([64, 1], DataType.FP32))
                f.return_type(ir.TensorType([64, 1], DataType.FP32))
                result = ib.let("result", tensor.maximum(a, b))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.maximum" in ir_str
        print(ir_str)

    def test_tensor_mul(self):
        """Test tensor.mul operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_mul_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t = f.param("t", ir.TensorType([64, 128], DataType.FP16))
                s = f.param("s", ir.TensorType([], DataType.FP32))  # 0-D tensor (scalar)
                f.return_type(ir.TensorType([64, 128], DataType.FP16))
                result = ib.let("result", tensor.mul(t, s))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.mul" in ir_str
        print(ir_str)

    def test_tensor_add(self):
        """Test tensor.add operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_add_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([8], DataType.FP32))
                b = f.param("b", ir.TensorType([8], DataType.FP32))
                f.return_type(ir.TensorType([8], DataType.FP32))
                result = ib.let("result", tensor.add(a, b))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.add" in ir_str
        print(ir_str)

    def test_tensor_sub(self):
        """Test tensor.sub operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_sub_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([8], DataType.FP32))
                b = f.param("b", ir.TensorType([8], DataType.FP32))
                f.return_type(ir.TensorType([8], DataType.FP32))
                result = ib.let("result", tensor.sub(a, b))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.sub" in ir_str
        print(ir_str)

    def test_tensor_div(self):
        """Test tensor.div operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_div_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                a = f.param("a", ir.TensorType([8], DataType.FP32))
                b = f.param("b", ir.TensorType([8], DataType.FP32))
                f.return_type(ir.TensorType([8], DataType.FP32))
                result = ib.let("result", tensor.div(a, b))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.div" in ir_str
        print(ir_str)


class TestTensorTransformOps:
    """Test suite for tensor transform operations."""

    def test_tensor_reshape(self):
        """Test tensor.reshape operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_reshape_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t = f.param("t", ir.TensorType([4, 8], DataType.FP32))
                f.return_type(ir.TensorType([32], DataType.FP32))
                # Reshape to [32] (flatten)
                result = ib.let("result", tensor.reshape(t, [32]))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.reshape" in ir_str
        print(ir_str)
    def test_tensor_reshape_dynamic(self):
        """Test tensor.reshape with dynamic shapes."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_reshape_dynamic_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                span = ir.Span.unknown()
                dim_n = ir.Var("n", ir.ScalarType(DataType.INT64), span)
                dim_m = ir.Var("m", ir.ScalarType(DataType.INT64), span)
                t_type = ir.TensorType([dim_n, dim_m], DataType.FP16)
                t = ib.var("t", t_type)
                dim_k = ir.Var("k", ir.ScalarType(DataType.INT64), span)
                result = ib.let("result", tensor.reshape(t, [dim_k]))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.reshape" in ir_str
        print(ir_str)

    def test_tensor_transpose(self):
        """Test tensor.transpose operation."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_transpose_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t = f.param("t", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([4, 3, 2], DataType.FP32))
                # Transpose by swapping axis 0 and 2: [2, 3, 4] -> [4, 3, 2]
                result = ib.let("result", tensor.transpose(t, 0, 2))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.transpose" in ir_str
        print(ir_str)

    def test_tensor_transpose_negative_axis(self):
        """Test tensor.transpose with negative axis indices."""
        ib = ir.IRBuilder()

        with ib.program("test_tensor_transpose_neg_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                t = f.param("t", ir.TensorType([8, 16], DataType.FP16))
                f.return_type(ir.TensorType([16, 8], DataType.FP16))
                # Transpose using negative indices: [8, 16] -> [16, 8]
                result = ib.let("result", tensor.transpose(t, -2, -1))
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        assert "tensor.transpose" in ir_str
        print(ir_str)


class TestTensorMisc:
    """Test suite for miscellaneous tensor operations and queries."""

    def test_const_float(self):
        """Test ConstFloat expression creation and usage."""
        ib = ir.IRBuilder()

        with ib.program("test_const_float_program") as p:
            p.declare_function("main")

            with ib.function("main", type=ir.FunctionType.InCore) as f:
                f.return_type(ir.ScalarType(DataType.FP32))
                span = ir.Span.unknown()

                # Create and check ConstFloat with FP32
                const_float = ir.ConstFloat(3.14, DataType.FP32, span)
                assert isinstance(const_float, ir.ConstFloat)
                assert const_float.value == 3.14
                assert const_float.dtype == DataType.FP32

                # Create and check ConstFloat with FP16
                const_float_fp16 = ir.ConstFloat(2.718, DataType.FP16, span)
                assert isinstance(const_float_fp16, ir.ConstFloat)
                assert const_float_fp16.value == 2.718
                assert const_float_fp16.dtype == DataType.FP16

                # Test with negative value
                const_float_neg = ir.ConstFloat(-1.5, DataType.FP32, span)
                assert const_float_neg.value == -1.5

                # Test with zero
                const_float_zero = ir.ConstFloat(0.0, DataType.FP32, span)
                assert const_float_zero.value == 0.0

                result = ib.let("result", const_float)
                ib.return_stmt(result)

            p.add_function(f.get_result())

        ir_str = str(p.get_result())
        print(ir_str)

    def test_operator_registration(self):
        """Test that all new operators are registered."""
        assert ir.is_op_registered("tensor.create")
        assert ir.is_op_registered("tensor.view")
        assert ir.is_op_registered("tensor.matmul")
        assert ir.is_op_registered("tensor.row_max")
        assert ir.is_op_registered("tensor.row_sum")
        assert ir.is_op_registered("tensor.exp")
        assert ir.is_op_registered("tensor.cast")
        assert ir.is_op_registered("tensor.assemble")
        assert ir.is_op_registered("tensor.maximum")
        # Check transform operators
        assert ir.is_op_registered("tensor.reshape")
        assert ir.is_op_registered("tensor.transpose")

    def test_get_new_ops(self):
        """Test getting new operator instances."""
        matmul_op = ir.get_op("tensor.matmul")
        assert matmul_op.name == "tensor.matmul"

        exp_op = ir.get_op("tensor.exp")
        assert exp_op.name == "tensor.exp"

        cast_op = ir.get_op("tensor.cast")
        assert cast_op.name == "tensor.cast"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
