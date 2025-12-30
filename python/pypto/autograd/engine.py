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
"""Backward engine for autograd."""
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pypto
from .graph import Graph, Node, NodeType, Value
from .registry import VJPContext, get_vjp, has_vjp
from .utils import is_differentiable_dtype

if TYPE_CHECKING:
    from ..tensor import Tensor


class BackwardEngine:
    """Backward engine for computing gradients via VJP."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self._value_grads: Dict[int, Optional["Tensor"]] = {}

    def backward(
        self,
        loss_value: Value,
        wrt_values: List[Value],
        seed: float = 1.0,
    ) -> Dict[str, Optional["Tensor"]]:
        """Compute gradients of loss with respect to specified values."""
        self._initialize_loss_grad(loss_value, seed)

        for node in self.graph.reverse_topological_order():
            self._process_node(node)

        result = {}
        for value in wrt_values:
            grad = self._value_grads.get(value.id)
            if grad is not None and value.dtype is not None:
                grad = self._cast_to_dtype(grad, value.dtype)
            result[value.name] = grad

        return result

    def _initialize_loss_grad(self, loss_value: Value, seed: float) -> None:
        if loss_value.shape is None:
            self._value_grads[loss_value.id] = None
            return

        dtype = loss_value.dtype or pypto.DT_FP32
        try:
            grad = pypto.ones(loss_value.shape, dtype=dtype)
            if seed != 1.0:
                grad = pypto.mul(grad, float(seed))
        except Exception:
            # PyPTO ops require JIT context; allow testing without it
            grad = None
        self._value_grads[loss_value.id] = grad

    def _process_node(self, node: Node) -> None:
        if node.node_type == NodeType.INPUT:
            return
        if node.node_type == NodeType.MOVE:
            self._process_move_node(node)
            return
        if node.node_type == NodeType.OP:
            self._process_op_node(node)

    def _process_move_node(self, node: Node) -> None:
        if len(node.outputs) != 1 or len(node.inputs) != 1:
            return
        out_grad = self._value_grads.get(node.outputs[0].id)
        self._accumulate_grad(node.inputs[0], out_grad)

    def _process_op_node(self, node: Node) -> None:
        op_name = node.op_name

        if not has_vjp(op_name):
            if any(self._value_grads.get(out.id) is not None for out in node.outputs):
                source = node.source_location
                source_msg = f"{source[0]}:{source[1]}" if source else "<unknown>"
                raise NotImplementedError(
                    f"Autograd VJP for '{op_name}' is not supported. Source: {source_msg}"
                )
            return

        out_grads: Dict[str, Optional["Tensor"]] = {}
        if getattr(node, "output_names", None):
            for out_name, out_value in zip(node.output_names, node.outputs):
                out_grads[out_name] = self._value_grads.get(out_value.id)
        else:
            for idx, out_value in enumerate(node.outputs):
                out_name = "output" if idx == 0 else f"output{idx}"
                out_grads[out_name] = self._value_grads.get(out_value.id)

        ctx = VJPContext(
            node=node,
            out_grads=out_grads,
            saved_tensors=getattr(node, "saved_tensors", {}) or {},
            attrs=node.attrs,
        )

        vjp_rule = get_vjp(op_name)
        in_grads = vjp_rule(ctx)

        if getattr(node, "input_names", None):
            for in_name, in_value in zip(node.input_names, node.inputs):
                self._accumulate_grad(in_value, in_grads.get(in_name))
        else:
            for in_value in node.inputs:
                self._accumulate_grad(in_value, in_grads.get(in_value.name))

    def _accumulate_grad(self, value: Value, grad: Optional["Tensor"]) -> None:
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
            self._value_grads[value.id] = pypto.add(existing, grad)

    def _cast_to_dtype(self, grad: "Tensor", target_dtype: Any) -> "Tensor":
        if grad is None:
            return None
        current_dtype = grad.dtype if hasattr(grad, "dtype") else None
        if current_dtype is None or current_dtype == target_dtype:
            return grad
        return pypto.cast(grad, target_dtype)


def compute_gradients(
    graph: Graph,
    loss_value: Value,
    wrt_values: List[Value],
    seed: float = 1.0,
) -> Dict[str, Optional["Tensor"]]:
    """Convenience function to compute gradients."""
    engine = BackwardEngine(graph)
    return engine.backward(loss_value, wrt_values, seed)
