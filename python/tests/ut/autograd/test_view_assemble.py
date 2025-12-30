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
Unit tests for autograd view/assemble VJP rules (M4 milestone).

Tests:
- VJP rules for reshape, transpose, view, assemble, clone, unsqueeze
- VJP signature validation
- View + assemble patterns common in kernels
"""
import pytest


class TestViewVJPRegistration:
    """Test that view VJP rules are registered."""

    def test_reshape_vjp_registered(self):
        """Test Reshape VJP is registered."""
        from pypto.autograd import has_vjp, get_vjp

        assert has_vjp("Reshape")
        vjp = get_vjp("Reshape")
        assert vjp is not None
        assert callable(vjp)

    def test_transpose_vjp_registered(self):
        """Test Transpose VJP is registered."""
        from pypto.autograd import has_vjp, get_vjp

        assert has_vjp("Transpose")
        vjp = get_vjp("Transpose")
        assert vjp is not None
        assert callable(vjp)

    def test_view_vjp_registered(self):
        """Test View VJP is registered."""
        from pypto.autograd import has_vjp, get_vjp

        assert has_vjp("View")
        vjp = get_vjp("View")
        assert vjp is not None
        assert callable(vjp)

    def test_assemble_vjp_registered(self):
        """Test Assemble VJP is registered."""
        from pypto.autograd import has_vjp, get_vjp

        assert has_vjp("Assemble")
        vjp = get_vjp("Assemble")
        assert vjp is not None
        assert callable(vjp)

    def test_clone_vjp_registered(self):
        """Test clone (Assign) VJP is registered."""
        from pypto.autograd import has_vjp, get_vjp

        assert has_vjp("Assign")
        vjp = get_vjp("Assign")
        assert vjp is not None
        assert callable(vjp)

    def test_unsqueeze_vjp_registered(self):
        """Test Unsqueeze VJP is registered."""
        from pypto.autograd import has_vjp, get_vjp

        assert has_vjp("Unsqueeze")
        vjp = get_vjp("Unsqueeze")
        assert vjp is not None
        assert callable(vjp)


class TestReshapeVJPSignature:
    """Test Reshape VJP rule signature."""

    def test_vjp_reshape_returns_dict(self):
        """Test that Reshape VJP returns dict with 'input' key."""
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Reshape")

        # Create mock context
        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Reshape",
            inputs=[Value(id=0, tensor_id=1, shape=[2, 3], name="input")],
            outputs=[Value(id=1, tensor_id=2, shape=[6], name="output")],
        )
        ctx = VJPContext(
            node=node,
            out_grads={"output": None},  # No gradient
            attrs={},
        )

        result = vjp(ctx)
        assert isinstance(result, dict)
        assert "input" in result


class TestTransposeVJPSignature:
    """Test Transpose VJP rule signature."""

    def test_vjp_transpose_returns_dict(self):
        """Test that Transpose VJP returns dict with 'input' key."""
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Transpose")

        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Transpose",
            inputs=[Value(id=0, tensor_id=1, shape=[2, 3], name="input")],
            outputs=[Value(id=1, tensor_id=2, shape=[3, 2], name="output")],
            attrs={"dim0": 0, "dim1": 1},
        )
        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"dim0": 0, "dim1": 1},
        )

        result = vjp(ctx)
        assert isinstance(result, dict)
        assert "input" in result


class TestViewVJPSignature:
    """Test View VJP rule signature."""

    def test_vjp_view_returns_dict(self):
        """Test that View VJP returns dict with 'input' key."""
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("View")

        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="View",
            inputs=[Value(id=0, tensor_id=1, shape=[4, 8], name="input")],
            outputs=[Value(id=1, tensor_id=2, shape=[2, 4], name="output")],
            attrs={"offsets": [0, 0], "shape": [2, 4]},
        )
        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"offsets": [0, 0], "shape": [2, 4]},
        )

        result = vjp(ctx)
        assert isinstance(result, dict)
        assert "input" in result


class TestAssembleVJPSignature:
    """Test Assemble VJP rule signature."""

    def test_vjp_assemble_returns_dict(self):
        """Test that Assemble VJP returns dict with 'input' and 'out' keys."""
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Assemble")

        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Assemble",
            inputs=[
                Value(id=0, tensor_id=1, shape=[2, 4], name="input"),
                Value(id=1, tensor_id=2, shape=[4, 8], name="out"),
            ],
            outputs=[Value(id=2, tensor_id=2, shape=[4, 8], name="output")],
            attrs={"offsets": [0, 0]},
        )
        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"offsets": [0, 0]},
        )

        result = vjp(ctx)
        assert isinstance(result, dict)
        assert "input" in result
        assert "out" in result


class TestCloneVJPSignature:
    """Test Clone (Assign) VJP rule signature."""

    def test_vjp_clone_returns_dict(self):
        """Test that Clone VJP returns dict with 'input' key."""
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Assign")

        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Assign",
            inputs=[Value(id=0, tensor_id=1, shape=[2, 3], name="input")],
            outputs=[Value(id=1, tensor_id=2, shape=[2, 3], name="output")],
        )
        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={},
        )

        result = vjp(ctx)
        assert isinstance(result, dict)
        assert "input" in result


class TestUnsqueezeVJPSignature:
    """Test Unsqueeze VJP rule signature."""

    def test_vjp_unsqueeze_returns_dict(self):
        """Test that Unsqueeze VJP returns dict with 'input' key."""
        from pypto.autograd import get_vjp, VJPContext
        from pypto.autograd.graph import Node, NodeType, Value

        vjp = get_vjp("Unsqueeze")

        node = Node(
            id=0,
            node_type=NodeType.OP,
            op_name="Unsqueeze",
            inputs=[Value(id=0, tensor_id=1, shape=[2, 3], name="input")],
            outputs=[Value(id=1, tensor_id=2, shape=[1, 2, 3], name="output")],
            attrs={"dim": 0},
        )
        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"dim": 0},
        )

        result = vjp(ctx)
        assert isinstance(result, dict)
        assert "input" in result


class TestAllViewOpsRegistered:
    """Test that all view operations have VJP rules."""

    def test_list_registered_ops_includes_view_ops(self):
        """Test that registered ops include all view operations."""
        from pypto.autograd import list_registered_ops

        ops = list_registered_ops()

        # View operations from M4
        view_ops = ["Reshape", "Transpose", "View", "Assemble", "Assign", "Unsqueeze"]

        for op in view_ops:
            assert op in ops, f"{op} should be registered"


class TestRegisteredOpsCount:
    """Test total count of registered operations."""

    def test_registered_ops_count(self):
        """Test that we have expected number of registered ops."""
        from pypto.autograd import list_registered_ops

        ops = list_registered_ops()

        # M2 (12): Add, Sub, Mul, Div, Neg, Exp, Log, Sqrt, Rsqrt, Sigmoid, Cast, Sum
        # M4 (6): Reshape, Transpose, View, Assemble, Assign, Unsqueeze
        # Total: 18
        assert len(ops) >= 18, f"Expected at least 18 ops, got {len(ops)}: {ops}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
