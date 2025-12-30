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
Unit tests for autograd VJP rules (M2 milestone).

Tests:
- VJP registry functionality
- Math VJP rules: add, sub, mul, div, neg, exp, log, sqrt, rsqrt, sigmoid
- Reduction VJP rules: sum
- Unbroadcast utility
"""
import pytest


class TestVJPRegistry:
    """Test VJP registry functionality."""

    def test_list_registered_ops(self):
        """Test that VJP rules are registered."""
        from pypto.autograd import list_registered_ops

        ops = list_registered_ops()
        assert "Add" in ops
        assert "Sub" in ops
        assert "Mul" in ops
        assert "Div" in ops
        assert "Neg" in ops
        assert "Exp" in ops
        assert "Log" in ops
        assert "Sqrt" in ops
        assert "Rsqrt" in ops
        assert "Sigmoid" in ops
        assert "Sum" in ops

    def test_get_vjp(self):
        """Test getting VJP rule by name."""
        from pypto.autograd import get_vjp, has_vjp

        assert has_vjp("Add")
        vjp_add = get_vjp("Add")
        assert vjp_add is not None
        assert callable(vjp_add)

    def test_has_vjp_false_for_unregistered(self):
        """Test has_vjp returns False for unregistered ops."""
        from pypto.autograd import has_vjp

        assert not has_vjp("NonExistentOp")

    def test_register_vjp_decorator(self):
        """Test registering a custom VJP rule."""
        from pypto.autograd import register_vjp, get_vjp, VJP_REGISTRY

        @register_vjp("CustomTestOp")
        def vjp_custom_test(ctx):
            return {"input": None}

        assert "CustomTestOp" in VJP_REGISTRY
        assert get_vjp("CustomTestOp") is vjp_custom_test

        # Cleanup
        del VJP_REGISTRY["CustomTestOp"]


class TestVJPContext:
    """Test VJPContext functionality."""

    def test_vjp_context_creation(self):
        """Test VJPContext creation and methods."""
        from pypto.autograd import VJPContext
        from pypto.autograd.graph import Graph, NodeType

        graph = Graph("test")
        v_in = graph.create_value(tensor_id=1, shape=[2, 3], dtype="fp32", name="input")
        v_out = graph.create_value(tensor_id=2, shape=[2, 3], dtype="fp32", name="output")
        node = graph.create_node(
            node_type=NodeType.OP,
            op_name="Add",
            inputs=[v_in],
            outputs=[v_out],
        )

        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            saved_tensors={},
            attrs={"alpha": 1.0},
        )

        assert ctx.get_attr("alpha") == 1.0
        assert ctx.get_attr("beta", 0.0) == 0.0
        assert ctx.get_input_shape("input") == [2, 3]
        assert ctx.get_output_grad("output") is None


class TestUnbroadcast:
    """Test unbroadcast utility function."""

    def test_compute_broadcast_axes_same_shape(self):
        """Test broadcast axes for same shapes."""
        from pypto.autograd.utils import _compute_broadcast_axes

        axes = _compute_broadcast_axes([2, 3], [2, 3])
        assert axes == []

    def test_compute_broadcast_axes_scalar(self):
        """Test broadcast axes for scalar broadcast."""
        from pypto.autograd.utils import _compute_broadcast_axes

        # Broadcasting [1] to [2, 3] means axes 0 and 1 (where dim=1) were broadcast
        axes = _compute_broadcast_axes([2, 3], [1])
        # Axis 0 is added (target has fewer dims), and axis 1 has 1->3
        assert 0 in axes

    def test_compute_broadcast_axes_dim_1(self):
        """Test broadcast axes when target has dim=1."""
        from pypto.autograd.utils import _compute_broadcast_axes

        # Broadcasting [1, 3] to [2, 3] means axis 0 was broadcast
        axes = _compute_broadcast_axes([2, 3], [1, 3])
        assert axes == [0]

    def test_broadcast_shapes(self):
        """Test broadcast shape computation."""
        from pypto.autograd.utils import broadcast_shapes

        assert broadcast_shapes([2, 3], [2, 3]) == [2, 3]
        assert broadcast_shapes([1, 3], [2, 1]) == [2, 3]
        assert broadcast_shapes([3], [2, 3]) == [2, 3]


class TestDifferentiableDtype:
    """Test dtype differentiation check."""

    def test_float_dtypes_are_differentiable(self):
        """Test that float dtypes are differentiable."""
        from pypto.autograd import is_differentiable_dtype

        assert is_differentiable_dtype("DT_FP32")
        assert is_differentiable_dtype("DT_FP16")
        assert is_differentiable_dtype("DT_BF16")

    def test_int_dtypes_not_differentiable(self):
        """Test that int dtypes are not differentiable."""
        from pypto.autograd import is_differentiable_dtype

        assert not is_differentiable_dtype("DT_INT32")
        assert not is_differentiable_dtype("DT_INT8")
        assert not is_differentiable_dtype("DT_BOOL")


class TestMathVJPFormulas:
    """Test math VJP formulas are correctly defined.

    These tests verify the VJP function signatures and basic behavior,
    not the actual gradient computation (which requires engine execution).
    """

    def test_vjp_add_signature(self):
        """Test vjp_add returns correct keys."""
        from pypto.autograd.rules.math import vjp_add
        from pypto.autograd import VJPContext
        from pypto.autograd.graph import Graph, NodeType

        graph = Graph("test")
        v_in1 = graph.create_value(tensor_id=1, shape=[2, 3], name="input")
        v_in2 = graph.create_value(tensor_id=2, shape=[2, 3], name="other")
        v_out = graph.create_value(tensor_id=3, shape=[2, 3], name="output")
        node = graph.create_node(
            node_type=NodeType.OP,
            op_name="Add",
            inputs=[v_in1, v_in2],
            outputs=[v_out],
        )

        ctx = VJPContext(node=node, out_grads={"output": None})
        result = vjp_add(ctx)

        assert "input" in result
        assert "other" in result

    def test_vjp_sub_signature(self):
        """Test vjp_sub returns correct keys."""
        from pypto.autograd.rules.math import vjp_sub
        from pypto.autograd import VJPContext
        from pypto.autograd.graph import Graph, NodeType

        graph = Graph("test")
        v_in1 = graph.create_value(tensor_id=1, shape=[2, 3], name="input")
        v_in2 = graph.create_value(tensor_id=2, shape=[2, 3], name="other")
        v_out = graph.create_value(tensor_id=3, shape=[2, 3], name="output")
        node = graph.create_node(
            node_type=NodeType.OP,
            op_name="Sub",
            inputs=[v_in1, v_in2],
            outputs=[v_out],
        )

        ctx = VJPContext(node=node, out_grads={"output": None})
        result = vjp_sub(ctx)

        assert "input" in result
        assert "other" in result

    def test_vjp_neg_signature(self):
        """Test vjp_neg returns correct keys."""
        from pypto.autograd.rules.math import vjp_neg
        from pypto.autograd import VJPContext
        from pypto.autograd.graph import Graph, NodeType

        graph = Graph("test")
        v_in = graph.create_value(tensor_id=1, shape=[2, 3], name="input")
        v_out = graph.create_value(tensor_id=2, shape=[2, 3], name="output")
        node = graph.create_node(
            node_type=NodeType.OP,
            op_name="Neg",
            inputs=[v_in],
            outputs=[v_out],
        )

        ctx = VJPContext(node=node, out_grads={"output": None})
        result = vjp_neg(ctx)

        assert "input" in result

    def test_vjp_sum_signature(self):
        """Test vjp_sum returns correct keys."""
        from pypto.autograd.rules.reduction import vjp_sum
        from pypto.autograd import VJPContext
        from pypto.autograd.graph import Graph, NodeType

        graph = Graph("test")
        v_in = graph.create_value(tensor_id=1, shape=[2, 3], name="input")
        v_out = graph.create_value(tensor_id=2, shape=[2, 1], name="output")
        node = graph.create_node(
            node_type=NodeType.OP,
            op_name="Sum",
            inputs=[v_in],
            outputs=[v_out],
            attrs={"dim": 1, "keepdim": True},
        )

        ctx = VJPContext(
            node=node,
            out_grads={"output": None},
            attrs={"dim": 1, "keepdim": True},
        )
        result = vjp_sum(ctx)

        assert "input" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
