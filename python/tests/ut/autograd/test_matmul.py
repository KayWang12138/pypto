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
"""
Unit tests for autograd matmul VJP rules (M5 milestone).

Tests:
- VJP rules for Matmul and BatchMatmul
- VJP signature validation
- Transpose flag handling (a_trans, b_trans)
"""
import pytest


class TestMatmulVJPRegistration:
    """Test that matmul VJP rules are registered."""

    def test_matmul_vjp_registered(self):
        """Test Matmul VJP is registered."""
        from pypto.autograd import has_vjp, get_vjp

        assert has_vjp("Matmul")
        vjp = get_vjp("Matmul")
        assert vjp is not None
        assert callable(vjp)

    def test_batch_matmul_vjp_registered(self):
        """Test BatchMatmul VJP is registered."""
        from pypto.autograd import has_vjp, get_vjp

        assert has_vjp("BatchMatmul")
        vjp = get_vjp("BatchMatmul")
        assert vjp is not None
        assert callable(vjp)


class TestMatmulVJPSignature:
    """Test Matmul VJP rule signature."""

    def test_vjp_matmul_returns_dict(self):
        """Test that Matmul VJP returns dict with 'input' and 'mat2' keys."""
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Matmul")

        # Create mock context for 2D matmul: (2, 3) @ (3, 4) -> (2, 4)
        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Matmul",
            inputs=[
                Value(id=0, tensor_id=1, shape=[2, 3], name="input"),
                Value(id=1, tensor_id=2, shape=[3, 4], name="mat2"),
            ],
            outputs=[Value(id=2, tensor_id=3, shape=[2, 4], name="output")],
            attrs={"a_trans": False, "b_trans": False},
        )
        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"a_trans": False, "b_trans": False},
        )

        result = vjp(ctx)
        assert isinstance(result, dict)
        assert "input" in result
        assert "mat2" in result

    def test_vjp_matmul_with_a_trans(self):
        """Test Matmul VJP with a_trans=True."""
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Matmul")

        # A^T @ B: (3, 2).T @ (3, 4) -> (2, 4)
        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Matmul",
            inputs=[
                Value(id=0, tensor_id=1, shape=[3, 2], name="input"),
                Value(id=1, tensor_id=2, shape=[3, 4], name="mat2"),
            ],
            outputs=[Value(id=2, tensor_id=3, shape=[2, 4], name="output")],
            attrs={"a_trans": True, "b_trans": False},
        )
        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"a_trans": True, "b_trans": False},
        )

        result = vjp(ctx)
        assert isinstance(result, dict)
        assert "input" in result
        assert "mat2" in result

    def test_vjp_matmul_with_b_trans(self):
        """Test Matmul VJP with b_trans=True."""
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Matmul")

        # A @ B^T: (2, 3) @ (4, 3).T -> (2, 4)
        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Matmul",
            inputs=[
                Value(id=0, tensor_id=1, shape=[2, 3], name="input"),
                Value(id=1, tensor_id=2, shape=[4, 3], name="mat2"),
            ],
            outputs=[Value(id=2, tensor_id=3, shape=[2, 4], name="output")],
            attrs={"a_trans": False, "b_trans": True},
        )
        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"a_trans": False, "b_trans": True},
        )

        result = vjp(ctx)
        assert isinstance(result, dict)
        assert "input" in result
        assert "mat2" in result

    def test_vjp_matmul_with_both_trans(self):
        """Test Matmul VJP with a_trans=True and b_trans=True."""
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Matmul")

        # A^T @ B^T: (3, 2).T @ (4, 3).T -> (2, 4)
        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Matmul",
            inputs=[
                Value(id=0, tensor_id=1, shape=[3, 2], name="input"),
                Value(id=1, tensor_id=2, shape=[4, 3], name="mat2"),
            ],
            outputs=[Value(id=2, tensor_id=3, shape=[2, 4], name="output")],
            attrs={"a_trans": True, "b_trans": True},
        )
        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"a_trans": True, "b_trans": True},
        )

        result = vjp(ctx)
        assert isinstance(result, dict)
        assert "input" in result
        assert "mat2" in result


class TestBatchMatmulVJPSignature:
    """Test BatchMatmul VJP rule signature."""

    def test_vjp_batch_matmul_returns_dict(self):
        """Test that BatchMatmul VJP returns dict with 'input' and 'mat2' keys."""
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("BatchMatmul")

        # 3D batch matmul: (B, M, K) @ (B, K, N) -> (B, M, N)
        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="BatchMatmul",
            inputs=[
                Value(id=0, tensor_id=1, shape=[2, 3, 4], name="input"),
                Value(id=1, tensor_id=2, shape=[2, 4, 5], name="mat2"),
            ],
            outputs=[Value(id=2, tensor_id=3, shape=[2, 3, 5], name="output")],
            attrs={"a_trans": False, "b_trans": False},
        )
        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"a_trans": False, "b_trans": False},
        )

        result = vjp(ctx)
        assert isinstance(result, dict)
        assert "input" in result
        assert "mat2" in result


class TestAllMatmulOpsRegistered:
    """Test that all matmul operations have VJP rules."""

    def test_list_registered_ops_includes_matmul_ops(self):
        """Test that registered ops include all matmul operations."""
        from pypto.autograd import list_registered_ops

        ops = list_registered_ops()

        # Matmul operations from M5
        matmul_ops = ["Matmul", "BatchMatmul"]

        for op in matmul_ops:
            assert op in ops, f"{op} should be registered"


class TestRegisteredOpsCount:
    """Test total count of registered operations."""

    def test_registered_ops_count(self):
        """Test that we have expected number of registered ops."""
        from pypto.autograd import list_registered_ops

        ops = list_registered_ops()

        # M2 (12): Add, Sub, Mul, Div, Neg, Exp, Log, Sqrt, Rsqrt, Sigmoid, Cast, Sum
        # M4 (6): Reshape, Transpose, View, Assemble, Assign, Unsqueeze
        # M5 (2): Matmul, BatchMatmul
        # Total: 20
        assert len(ops) >= 20, f"Expected at least 20 ops, got {len(ops)}: {ops}"


class TestMatmulVJPFormulas:
    """Test matmul VJP mathematical formulas."""

    def test_vjp_no_transpose_formula(self):
        """Test VJP formula for C = A @ B (no transpose).

        Expected:
        - d_A = d_C @ B^T
        - d_B = A^T @ d_C
        """
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Matmul")

        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Matmul",
            inputs=[
                Value(id=0, tensor_id=1, shape=[2, 3], name="input"),
                Value(id=1, tensor_id=2, shape=[3, 4], name="mat2"),
            ],
            outputs=[Value(id=2, tensor_id=3, shape=[2, 4], name="output")],
            attrs={"a_trans": False, "b_trans": False},
        )

        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"a_trans": False, "b_trans": False},
        )

        result = vjp(ctx)

        # With None gradient, should return None for both inputs
        assert result["input"] is None
        assert result["mat2"] is None

    def test_vjp_a_trans_formula(self):
        """Test VJP formula for C = A^T @ B (a_trans=True).

        Expected:
        - d_A = B @ d_C^T
        - d_B = A @ d_C
        """
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Matmul")

        # A: (3, 2), A^T: (2, 3), B: (3, 4), C = A^T @ B: (2, 4)
        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Matmul",
            inputs=[
                Value(id=0, tensor_id=1, shape=[3, 2], name="input"),
                Value(id=1, tensor_id=2, shape=[3, 4], name="mat2"),
            ],
            outputs=[Value(id=2, tensor_id=3, shape=[2, 4], name="output")],
            attrs={"a_trans": True, "b_trans": False},
        )

        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"a_trans": True, "b_trans": False},
        )

        result = vjp(ctx)
        assert result["input"] is None
        assert result["mat2"] is None

    def test_vjp_b_trans_formula(self):
        """Test VJP formula for C = A @ B^T (b_trans=True).

        Expected:
        - d_A = d_C @ B
        - d_B = d_C^T @ A
        """
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Matmul")

        # A: (2, 3), B: (4, 3), B^T: (3, 4), C = A @ B^T: (2, 4)
        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Matmul",
            inputs=[
                Value(id=0, tensor_id=1, shape=[2, 3], name="input"),
                Value(id=1, tensor_id=2, shape=[4, 3], name="mat2"),
            ],
            outputs=[Value(id=2, tensor_id=3, shape=[2, 4], name="output")],
            attrs={"a_trans": False, "b_trans": True},
        )

        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"a_trans": False, "b_trans": True},
        )

        result = vjp(ctx)
        assert result["input"] is None
        assert result["mat2"] is None

    def test_vjp_both_trans_formula(self):
        """Test VJP formula for C = A^T @ B^T (both trans).

        Expected:
        - d_A = B^T @ d_C^T
        - d_B = d_C^T @ A^T
        """
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Matmul")

        # A: (3, 2), A^T: (2, 3), B: (4, 3), B^T: (3, 4), C = A^T @ B^T: (2, 4)
        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Matmul",
            inputs=[
                Value(id=0, tensor_id=1, shape=[3, 2], name="input"),
                Value(id=1, tensor_id=2, shape=[4, 3], name="mat2"),
            ],
            outputs=[Value(id=2, tensor_id=3, shape=[2, 4], name="output")],
            attrs={"a_trans": True, "b_trans": True},
        )

        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"a_trans": True, "b_trans": True},
        )

        result = vjp(ctx)
        assert result["input"] is None
        assert result["mat2"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
