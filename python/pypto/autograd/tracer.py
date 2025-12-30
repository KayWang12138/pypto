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
"""Autograd Tracer."""
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .graph import Graph, Node, NodeType, Value
from ._core import get_current_tracer, is_tracing

__all__ = [
    "Tracer",
]


# Attributes that should be captured for VJP rules
_IMPORTANT_ATTRS = frozenset([
    "dim",
    "keepdim",
    "dim0",
    "dim1",
    "axis",
    "a_trans",
    "b_trans",
    "offsets",
    "perm",
    "shape",
    "valid_shape",
    "dtype",
    "out_dtype",
    "alpha",
    "beta",
    "mode",
    # Derived attrs (not necessarily part of op signature)
    "input_dtype",
])


class Tracer:
    """Autograd tracer that builds a computation graph from forward operations."""

    def __init__(self, name: str = "autograd"):
        self.graph = Graph(name=name)
        self._tensor_id_to_value: Dict[int, Value] = {}

    def _get_tensor_id(self, tensor: Any) -> int:
        if hasattr(tensor, "id"):
            return tensor.id
        return id(tensor)

    def _get_tensor_shape(self, tensor: Any) -> Optional[List[Any]]:
        if hasattr(tensor, "shape"):
            return list(tensor.shape)
        return None

    def _get_tensor_dtype(self, tensor: Any) -> Optional[Any]:
        if hasattr(tensor, "dtype"):
            return tensor.dtype
        return None

    def _get_or_create_value(
        self,
        tensor: Any,
        name: str = "",
        increment_version: bool = False,
    ) -> Value:
        tensor_id = self._get_tensor_id(tensor)

        existing = self.graph.get_value(tensor_id)

        if not increment_version and existing is not None:
            # Update debug name if we learn it later (e.g., record_input after use)
            if name and not existing.name:
                existing.name = name
            return existing

        if increment_version and existing is not None and not name and existing.name:
            # Preserve name across SSA versions (useful for loss/out buffers)
            name = existing.name

        value = self.graph.create_value(
            tensor_id=tensor_id,
            shape=self._get_tensor_shape(tensor),
            dtype=self._get_tensor_dtype(tensor),
            name=name,
            increment_version=increment_version,
        )
        return value

    def _extract_attrs(self, bound_args: Dict[str, Any]) -> Dict[str, Any]:
        attrs = {}
        for key in _IMPORTANT_ATTRS:
            if key in bound_args and not self._is_tensor(bound_args[key]):
                attrs[key] = bound_args[key]
        return attrs

    def _get_source_location(self, level: int = 3) -> Optional[Tuple[str, int]]:
        try:
            frame = inspect.currentframe()
            for _ in range(level):
                if frame is not None:
                    frame = frame.f_back
            if frame is not None:
                return (frame.f_code.co_filename, frame.f_lineno)
        except Exception:
            pass
        return None

    def record_op(
        self,
        op_callable: Callable,
        op_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        outputs: Any,
        source_location: Optional[Tuple[str, int]] = None,
    ) -> None:
        """Record an operation in the autograd graph."""
        bound_args: Dict[str, Any] = {}
        try:
            sig = inspect.signature(op_callable)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            bound_args = dict(bound.arguments)
        except Exception:
            # Fallback: best-effort (no names)
            bound_args = {f"arg{i}": a for i, a in enumerate(args)}
            bound_args.update(kwargs)

        input_values: List[Value] = []
        input_names: List[str] = []
        saved_tensors: Dict[str, Any] = {}

        for name, value_obj in bound_args.items():
            if self._is_tensor(value_obj):
                value = self._get_or_create_value(value_obj)
                input_values.append(value)
                input_names.append(name)
                saved_tensors[name] = value_obj

        output_values: List[Value] = []
        output_names: List[str] = []

        if outputs is not None:
            if isinstance(outputs, (list, tuple)):
                for idx, out in enumerate(outputs):
                    if self._is_tensor(out):
                        value = self._get_or_create_value(out)
                        output_values.append(value)
                        output_names.append("output" if idx == 0 else f"output{idx}")
                        saved_tensors[output_names[-1]] = out
            elif self._is_tensor(outputs):
                value = self._get_or_create_value(outputs)
                output_values.append(value)
                output_names.append("output")
                saved_tensors["output"] = outputs

        # In-place ops that return None
        if outputs is None and op_name == "Assemble":
            out_tensor = bound_args.get("out")
            if self._is_tensor(out_tensor):
                out_value_new = self._get_or_create_value(out_tensor, increment_version=True)
                output_values = [out_value_new]
                output_names = ["output"]
                saved_tensors["output"] = out_tensor

        attrs = self._extract_attrs(bound_args)

        # Derived attrs
        if op_name == "Cast":
            input_tensor = saved_tensors.get("input")
            if input_tensor is not None and hasattr(input_tensor, "dtype"):
                attrs["input_dtype"] = input_tensor.dtype

        self.graph.create_node(
            node_type=NodeType.OP,
            op_name=op_name,
            op_callable=op_callable,
            inputs=input_values,
            outputs=output_values,
            input_names=input_names,
            output_names=output_names,
            saved_tensors=saved_tensors,
            attrs=attrs,
            source_location=source_location or self._get_source_location(),
        )

    def record_move(
        self,
        dst_tensor: Any,
        src_tensor: Any,
        source_location: Optional[Tuple[str, int]] = None,
    ) -> None:
        """Record a Tensor.move() operation (alias/SSA mapping)."""
        src_value = self._get_or_create_value(src_tensor)

        # Create new version for destination (SSA semantics)
        dst_value = self._get_or_create_value(
            dst_tensor,
            increment_version=True,
        )

        self.graph.create_node(
            node_type=NodeType.MOVE,
            op_name="move",
            inputs=[src_value],
            outputs=[dst_value],
            input_names=["src"],
            output_names=["dst"],
            saved_tensors={"src": src_tensor, "dst": dst_tensor},
            source_location=source_location or self._get_source_location(),
        )

    def record_input(
        self,
        tensor: Any,
        name: str = "",
    ) -> Value:
        """Record a graph input (function parameter)."""
        value = self._get_or_create_value(tensor, name=name)

        self.graph.create_node(
            node_type=NodeType.INPUT,
            op_name="input",
            outputs=[value],
            attrs={"name": name},
        )

        return value

    def _is_tensor(self, obj: Any) -> bool:
        if hasattr(obj, "_base") and hasattr(obj, "id"):
            return True
        type_name = type(obj).__name__
        return type_name == "Tensor"

    def get_graph(self) -> Graph:
        return self.graph

    def clear(self) -> None:
        self.graph = Graph(name=self.graph.name)
        self._tensor_id_to_value.clear()


def _autograd_hook_op(
    op_callable: Callable,
    op_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    outputs: Any,
) -> None:
    """Global hook function called by op_wrapper."""
    if not is_tracing():
        return

    tracer = get_current_tracer()
    if tracer is not None:
        tracer.record_op(
            op_callable=op_callable,
            op_name=op_name,
            args=args,
            kwargs=kwargs,
            outputs=outputs,
        )


def _autograd_hook_move(dst_tensor: Any, src_tensor: Any) -> None:
    """Global hook function called by Tensor.move()."""
    if not is_tracing():
        return

    tracer = get_current_tracer()
    if tracer is not None:
        tracer.record_move(dst_tensor, src_tensor)
