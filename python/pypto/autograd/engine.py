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
"""Backward engine for autograd.

Implements reverse-mode automatic differentiation (VJP/backprop).
The engine traverses the autograd graph in reverse topological order,
calling VJP rules for each operation and accumulating gradients.
"""
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .graph import Graph, Node, NodeType, Value
from .registry import VJPContext, get_vjp, has_vjp
from .utils import is_differentiable_dtype

if TYPE_CHECKING:
    from ..tensor import Tensor


class BackwardEngine:
    """Backward engine for computing gradients via VJP.

    The engine performs reverse-mode automatic differentiation:
    1. Initialize gradients for loss with seed value (default 1.0)
    2. Traverse nodes in reverse topological order
    3. For each node, call its VJP rule to compute input gradients
    4. Accumulate gradients for values that are used multiple times
    5. Cast gradients to match target dtypes

    Args:
        graph: The autograd graph built during forward pass
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self._value_grads: Dict[int, Optional["Tensor"]] = {}

    def backward(
        self,
        loss_value: Value,
        wrt_values: List[Value],
        seed: float = 1.0,
    ) -> Dict[str, Optional["Tensor"]]:
        """Compute gradients of loss with respect to specified values.

        Args:
            loss_value: The Value representing the loss tensor
            wrt_values: List of Values to compute gradients for
            seed: Initial gradient value for loss (default 1.0)

        Returns:
            Dict mapping value names to gradient tensors
        """
        # Initialize gradient for loss
        self._initialize_loss_grad(loss_value, seed)

        # Traverse in reverse topological order
        for node in self.graph.reverse_topological_order():
            self._process_node(node)

        # Collect gradients for requested values
        result = {}
        for value in wrt_values:
            grad = self._value_grads.get(value.id)
            # Cast gradient to match target dtype if needed
            if grad is not None and value.dtype is not None:
                grad = self._cast_to_dtype(grad, value.dtype)
            result[value.name] = grad

        return result

    def _initialize_loss_grad(self, loss_value: Value, seed: float) -> None:
        """Initialize the gradient for the loss value.

        Creates a scalar tensor filled with the seed value.
        """
        import pypto

        if loss_value.shape is None:
            self._value_grads[loss_value.id] = None
            return

        dtype = loss_value.dtype or pypto.DT_FP32

        # Seed is typically 1.0 for scalar loss.
        try:
            grad = pypto.ones(loss_value.shape, dtype=dtype)
            if seed != 1.0:
                grad = pypto.mul(grad, float(seed))
        except Exception:
            # In some environments, PyPTO ops cannot run in EAGER mode.
            # Keep autograd engine importable/testable without requiring a
            # full pypto.function/jit recording context.
            grad = None

        self._value_grads[loss_value.id] = grad

    def _process_node(self, node: Node) -> None:
        """Process a single node during backward pass.

        Looks up the VJP rule for the node's operation and computes
        gradients for its inputs.
        """
        if node.node_type == NodeType.INPUT:
            # Input nodes don't contribute gradients
            return

        if node.node_type == NodeType.MOVE:
            # Move nodes pass gradient through unchanged
            self._process_move_node(node)
            return

        if node.node_type == NodeType.OP:
            self._process_op_node(node)
            return

    def _process_move_node(self, node: Node) -> None:
        """Process a MOVE node (gradient identity)."""
        if len(node.outputs) != 1 or len(node.inputs) != 1:
            return

        out_value = node.outputs[0]
        in_value = node.inputs[0]

        # Pass gradient through unchanged
        out_grad = self._value_grads.get(out_value.id)
        self._accumulate_grad(in_value, out_grad)

    def _process_op_node(self, node: Node) -> None:
        """Process an OP node using its VJP rule."""
        op_name = node.op_name

        # Check if VJP rule exists
        if not has_vjp(op_name):
            # No VJP rule - only error if this op contributes to gradients.
            if any(self._value_grads.get(out.id) is not None for out in node.outputs):
                source = node.source_location
                source_msg = f"{source[0]}:{source[1]}" if source else "<unknown>"
                raise NotImplementedError(
                    f"Autograd VJP for '{op_name}' is not supported. "
                    f"Source: {source_msg}"
                )
            return

        # Collect output gradients (by logical output names)
        out_grads: Dict[str, Optional["Tensor"]] = {}
        if getattr(node, "output_names", None):
            for out_name, out_value in zip(node.output_names, node.outputs):
                out_grads[out_name] = self._value_grads.get(out_value.id)
        else:
            for idx, out_value in enumerate(node.outputs):
                out_name = "output" if idx == 0 else f"output{idx}"
                out_grads[out_name] = self._value_grads.get(out_value.id)

        # Create VJP context
        ctx = VJPContext(
            node=node,
            out_grads=out_grads,
            saved_tensors=getattr(node, "saved_tensors", {}) or {},
            attrs=node.attrs,
        )

        # Call VJP rule
        vjp_rule = get_vjp(op_name)
        try:
            in_grads = vjp_rule(ctx)
        except Exception as e:
            raise RuntimeError(
                f"VJP rule for '{op_name}' failed: {e}\n"
                f"Source: {node.source_location}"
            ) from e

        # Accumulate gradients for inputs (by logical input names)
        if getattr(node, "input_names", None):
            for in_name, in_value in zip(node.input_names, node.inputs):
                self._accumulate_grad(in_value, in_grads.get(in_name))
        else:
            # Fallback: try matching by Value.name
            for in_value in node.inputs:
                self._accumulate_grad(in_value, in_grads.get(in_value.name))

    def _accumulate_grad(
        self,
        value: Value,
        grad: Optional["Tensor"],
    ) -> None:
        """Accumulate gradient for a value.

        Handles the case where a value is used by multiple operations.
        """
        if grad is None:
            return

        if value.dtype is not None and not is_differentiable_dtype(value.dtype):
            return

        target_dtype = value.dtype
        existing = self._value_grads.get(value.id)
        if target_dtype is None and existing is not None and hasattr(existing, "dtype"):
            target_dtype = existing.dtype

        if target_dtype is not None:
            grad = self._cast_to_dtype(grad, target_dtype)
            if existing is not None:
                existing = self._cast_to_dtype(existing, target_dtype)

        if existing is None:
            self._value_grads[value.id] = grad
        else:
            # Accumulate gradients
            import pypto
            self._value_grads[value.id] = pypto.add(existing, grad)

    def _cast_to_dtype(
        self,
        grad: "Tensor",
        target_dtype: Any,
    ) -> "Tensor":
        """Cast gradient tensor to target dtype if needed."""
        import pypto

        if grad is None:
            return None

        # Get current dtype
        current_dtype = grad.dtype if hasattr(grad, "dtype") else None
        if current_dtype is None or current_dtype == target_dtype:
            return grad

        # Cast to target dtype
        return pypto.cast(grad, target_dtype)


def compute_gradients(
    graph: Graph,
    loss_value: Value,
    wrt_values: List[Value],
    seed: float = 1.0,
) -> Dict[str, Optional["Tensor"]]:
    """Convenience function to compute gradients.

    Args:
        graph: The autograd graph
        loss_value: The Value representing loss
        wrt_values: Values to compute gradients for
        seed: Initial gradient value for loss

    Returns:
        Dict mapping value names to gradient tensors
    """
    engine = BackwardEngine(graph)
    return engine.backward(loss_value, wrt_values, seed)
