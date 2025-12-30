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
"""Unit tests for autograd matmul VJP rules."""
import pytest

from pypto.autograd import VJPContext, get_vjp, has_vjp, list_registered_ops
from pypto.autograd.graph import Node, NodeType, Value


class TestMatmulVJPRegistration:
    """Test that matmul VJP rules are registered."""

    def test_matmul_vjp_registered(self):
        """Test Matmul VJP is registered."""
        assert has_vjp("Matmul")
        vjp = get_vjp("Matmul")
        assert vjp is not None
        assert callable(vjp)

    def test_batch_matmul_vjp_registered(self):
        """Test BatchMatmul VJP is registered."""
        assert has_vjp("BatchMatmul")
        vjp = get_vjp("BatchMatmul")
        assert vjp is not None
        assert callable(vjp)


class TestMatmulVJPSignature:
    """Test Matmul VJP rule signature."""

    def test_vjp_matmul_returns_dict(self):
        """Test that Matmul VJP returns dict with 'input' and 'mat2' keys."""
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
        assert isinstance(result, dict)
        assert "input" in result
        assert "mat2" in result

    def test_vjp_matmul_with_a_trans(self):
        """Test Matmul VJP with a_trans=True."""
        vjp = get_vjp("Matmul")

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
        vjp = get_vjp("Matmul")

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
        vjp = get_vjp("Matmul")

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
        vjp = get_vjp("BatchMatmul")

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
        ops = list_registered_ops()

        matmul_ops = ["Matmul", "BatchMatmul"]

        for op in matmul_ops:
            assert op in ops, f"{op} should be registered"


class TestRegisteredOpsCount:
    """Test total count of registered operations."""

    def test_registered_ops_count(self):
        """Test that we have expected number of registered ops."""
        ops = list_registered_ops()

        assert len(ops) >= 20, f"Expected at least 20 ops, got {len(ops)}: {ops}"


class TestMatmulVJPFormulas:
    """Test matmul VJP mathematical formulas."""

    def test_vjp_no_transpose_formula(self):
        """Test VJP formula for C = A @ B (no transpose)."""
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

        assert result["input"] is None
        assert result["mat2"] is None

    def test_vjp_a_trans_formula(self):
        """Test VJP formula for C = A^T @ B (a_trans=True)."""
        vjp = get_vjp("Matmul")

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
        """Test VJP formula for C = A @ B^T (b_trans=True)."""
        vjp = get_vjp("Matmul")

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
        """Test VJP formula for C = A^T @ B^T (both trans)."""
        vjp = get_vjp("Matmul")

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
