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
"""Training graph builder for autograd.

This module provides functionality to build combined forward+backward graphs
that can be compiled and executed via pypto.jit.

The key design principle is to generate the backward pass within the same
graph recording context as the forward pass, avoiding the need for separate
runtime support for saving/loading activations.
"""
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .tracer import Tracer
from ._core import trace
from .engine import BackwardEngine
from .utils import is_differentiable_dtype

if TYPE_CHECKING:
    from ..tensor import Tensor


class TrainGraphBuilder:
    """Builder for creating combined forward+backward training graphs.

    The builder:
    1. Traces the forward function to build an autograd graph
    2. Generates backward operations using VJP rules
    3. Produces a combined training graph for jit compilation

    Args:
        forward_fn: The forward function to differentiate
        wrt: Parameter names or indices to compute gradients for
        loss: Name or index of the loss output parameter
    """

    def __init__(
        self,
        forward_fn: Callable,
        wrt: Tuple[Any, ...],
        loss: Any,
    ):
        self.forward_fn = forward_fn
        self.wrt = wrt
        self.loss = loss
        self._signature = inspect.signature(forward_fn)
        self._param_names = list(self._signature.parameters.keys())

    def _resolve_param_index(self, name_or_index: Any) -> int:
        """Resolve parameter name or index to index."""
        if isinstance(name_or_index, int):
            if name_or_index < 0 or name_or_index >= len(self._param_names):
                raise ValueError(
                    f"Parameter index {name_or_index} out of range "
                    f"(function has {len(self._param_names)} parameters)"
                )
            return name_or_index

        if isinstance(name_or_index, str):
            try:
                return self._param_names.index(name_or_index)
            except ValueError:
                raise ValueError(
                    f"Parameter '{name_or_index}' not found in function signature. "
                    f"Available: {self._param_names}"
                )

        raise TypeError(
            f"Parameter identifier must be str or int, got {type(name_or_index)}"
        )

    def _resolve_param_name(self, name_or_index: Any) -> str:
        """Resolve parameter name or index to name."""
        idx = self._resolve_param_index(name_or_index)
        return self._param_names[idx]

    def validate_wrt(self, bound_args: Dict[str, Any]) -> List[str]:
        """Validate wrt parameters are valid and differentiable.

        Args:
            bound_args: Bound arguments from function call

        Returns:
            List of resolved parameter names

        Raises:
            ValueError: If wrt parameter is invalid
            TypeError: If wrt parameter is non-differentiable dtype
        """
        resolved_names = []

        for w in self.wrt:
            name = self._resolve_param_name(w)
            resolved_names.append(name)

            # Check that parameter exists in bound args
            if name not in bound_args:
                raise ValueError(f"wrt parameter '{name}' not provided in call")

            # Check dtype is differentiable
            tensor = bound_args[name]
            if hasattr(tensor, "dtype"):
                if not is_differentiable_dtype(tensor.dtype):
                    raise TypeError(
                        f"Cannot differentiate with respect to '{name}': "
                        f"dtype {tensor.dtype} is not differentiable"
                    )

        return resolved_names

    def validate_loss(self, bound_args: Dict[str, Any]) -> str:
        """Validate loss parameter.

        Args:
            bound_args: Bound arguments from function call

        Returns:
            Resolved loss parameter name

        Raises:
            ValueError: If loss parameter is invalid
            NotImplementedError: If loss shape is dynamic
        """
        loss_name = self._resolve_param_name(self.loss)

        if loss_name not in bound_args:
            raise ValueError(f"Loss parameter '{loss_name}' not provided in call")

        tensor = bound_args[loss_name]

        # Check loss is scalar (1-element tensor)
        if hasattr(tensor, "shape"):
            shape = tensor.shape
            # Check for dynamic shape (SymbolicScalar dimensions)
            has_symbolic = any(
                hasattr(dim, "__class__") and "Symbolic" in type(dim).__name__
                for dim in shape
            )
            if has_symbolic:
                raise NotImplementedError(
                    f"Loss parameter '{loss_name}' has dynamic shape. "
                    "MVP requires static scalar loss shape [1]"
                )

            # Check shape is scalar-like
            total_elements = 1
            for dim in shape:
                if isinstance(dim, int):
                    total_elements *= dim
            if total_elements != 1:
                raise ValueError(
                    f"Loss parameter '{loss_name}' must be scalar (1 element), "
                    f"got shape {shape}"
                )

        return loss_name

    def build_training_function(
        self,
        *,
        wrt_names: List[str],
        loss_name: str,
        jit_options: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Build a jit-compiled training function (forward + backward).

        The returned callable is a `pypto.jit`-wrapped function that:
        1) Records forward ops while building an autograd graph (Tracer hook)
        2) Generates backward ops (VJP rules) in the same pypto.function recording
        3) Writes gradients into the provided grad buffers

        Args:
            wrt_names: Resolved parameter names to differentiate w.r.t.
            loss_name: Resolved loss parameter name (must be scalar)
            jit_options: Options forwarded to `pypto.jit(...)`

        Returns:
            A `pypto.jit`-wrapped training function. It takes positional args:
                (*forward_args, *grad_buffers)
            where `grad_buffers` are in the same order as `wrt_names`.
        """
        import pypto

        # Resolve indices once (used by the training kernel)
        wrt_indices = [self._resolve_param_index(w) for w in self.wrt]
        loss_index = self._resolve_param_index(self.loss)

        num_fwd_params = len(self._param_names)
        num_grads = len(wrt_names)

        def training_impl(*all_args):
            if len(all_args) < num_fwd_params + num_grads:
                raise ValueError(
                    f"Training function expects at least {num_fwd_params + num_grads} "
                    f"positional args (forward args + grad buffers), got {len(all_args)}"
                )

            fwd_args = all_args[:num_fwd_params]
            grad_buffers = all_args[num_fwd_params:num_fwd_params + num_grads]

            # Build autograd graph from forward pass (only forward is traced)
            tracer = Tracer(name="train")
            with trace(tracer):
                for name, value in zip(self._param_names, fwd_args):
                    if self._is_tensor(value):
                        tracer.record_input(value, name=name)

                # Forward ops (recorded by both pypto.function and tracer hooks)
                self.forward_fn(*fwd_args)

            graph = tracer.get_graph()

            # Resolve loss and wrt Values by tensor identity (SSA-aware)
            loss_tensor = fwd_args[loss_index]
            if not self._is_tensor(loss_tensor):
                raise TypeError(f"loss parameter '{loss_name}' must be a Tensor")

            loss_value = graph.get_value(loss_tensor.id)
            if loss_value is None:
                raise RuntimeError(f"Could not resolve loss Value for '{loss_name}'")

            wrt_values = []
            for idx in wrt_indices:
                t = fwd_args[idx]
                if not self._is_tensor(t):
                    raise TypeError(
                        f"wrt parameter '{self._param_names[idx]}' must be a Tensor"
                    )
                v = graph.get_value(t.id)
                if v is None:
                    raise RuntimeError(
                        f"Could not resolve wrt Value for '{self._param_names[idx]}'"
                    )
                wrt_values.append(v)

            # Generate backward ops (still inside pypto.function recording via pypto.jit)
            engine = BackwardEngine(graph)
            computed = engine.backward(loss_value, wrt_values, seed=1.0)

            # Write gradients into buffers (one buffer per wrt, in wrt_names order)
            for name, buf in zip(wrt_names, grad_buffers):
                grad = computed.get(name)
                if grad is None:
                    continue
                if self._is_tensor(buf) and hasattr(buf, "dtype") and hasattr(grad, "dtype"):
                    if buf.dtype != grad.dtype:
                        grad = pypto.cast(grad, buf.dtype)
                buf[:] = grad

        training_impl.__name__ = f"{self.forward_fn.__name__}__train_step"

        jit_options = jit_options or {}
        return pypto.jit(training_impl, **jit_options)

    def _is_tensor(self, obj: Any) -> bool:
        """Check if object is a tensor."""
        if hasattr(obj, "_base") and hasattr(obj, "id"):
            return True
        type_name = type(obj).__name__
        return type_name == "Tensor"
