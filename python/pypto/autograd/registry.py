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
"""VJP (Vector-Jacobian Product) rule registry for autograd.

This module provides:
- VJP_REGISTRY: mapping from op_name to VJP function
- register_vjp: decorator/function to register VJP rules
- get_vjp: lookup VJP rule by op_name

VJP function signature:
    def vjp_rule(ctx: VJPContext) -> Dict[str, Optional[Tensor]]:
        '''
        Args:
            ctx: VJPContext containing:
                - node: the Node from forward graph
                - out_grads: dict mapping output name to gradient tensor
                - saved_tensors: tensors saved during forward for backward
                - attrs: op attributes (dim, keepdim, etc.)

        Returns:
            Dict mapping input name to gradient tensor (or None if no gradient).
        '''
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .graph import Node
    from ..tensor import Tensor


def _snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part)


def _normalize_op_name(op: Union[str, Callable]) -> str:
    """Normalize an op identifier to the canonical VJP registry key."""
    if isinstance(op, str):
        return op

    name = getattr(op, "__name__", "")
    # Keep consistent with python/pypto/_op_wrapper.py canonicalization.
    if name == "clone":
        return "Assign"
    if name == "matmul":
        # Callable registration maps to the 2D op by default.
        return "Matmul"
    return _snake_to_pascal(name)


@dataclass
class VJPContext:
    """Context passed to VJP rules containing all information needed for backward."""

    node: "Node"
    out_grads: Dict[str, Optional["Tensor"]]
    saved_tensors: Dict[str, "Tensor"] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def get_saved(self, name: str) -> Optional["Tensor"]:
        """Get a saved tensor by name."""
        return self.saved_tensors.get(name)

    def get_attr(self, name: str, default: Any = None) -> Any:
        """Get an attribute by name."""
        return self.attrs.get(name, default)

    def get_input_shape(self, name: str) -> Optional[List[Any]]:
        """Get the shape of an input tensor."""
        if hasattr(self.node, "input_names") and self.node.input_names:
            try:
                idx = self.node.input_names.index(name)
                if idx < len(self.node.inputs):
                    return self.node.inputs[idx].shape
            except ValueError:
                pass

        # Fallback: older graphs may encode names on Value.name
        for value in self.node.inputs:
            if getattr(value, "name", "") == name:
                return value.shape
        return None

    def get_output_grad(self, name: str = "output") -> Optional["Tensor"]:
        """Get the gradient of an output tensor."""
        if name in self.out_grads:
            return self.out_grads.get(name)

        # Common case: single-output op, but output name not recorded.
        if len(self.out_grads) == 1:
            return next(iter(self.out_grads.values()))
        return None


# VJP rule type: takes context, returns dict of input gradients
VJPRule = Callable[[VJPContext], Dict[str, Optional["Tensor"]]]

# Global registry mapping op_name -> VJP rule
VJP_REGISTRY: Dict[str, VJPRule] = {}


def register_vjp(op_name: Union[str, Callable], rule: Optional[VJPRule] = None):
    """Register a VJP rule for an operation.

    Can be used as a decorator or function:

    As decorator:
        @register_vjp("Add")
        def vjp_add(ctx):
            ...

    As function:
        register_vjp("Add", vjp_add)

    Or using an op callable (recommended, matches docs/tutorials plan):
        @register_vjp(pypto.add)
        def vjp_add(ctx):
            ...

    Args:
        op_name: The operation name (str) or the op callable
        rule: The VJP rule function (optional if used as decorator)

    Returns:
        The rule function (for decorator usage)
    """
    def decorator(fn: VJPRule) -> VJPRule:
        key = _normalize_op_name(op_name)
        if key in VJP_REGISTRY:
            import warnings
            warnings.warn(f"VJP rule for '{key}' is being overwritten")
        VJP_REGISTRY[key] = fn
        return fn

    if rule is not None:
        return decorator(rule)
    return decorator


def get_vjp(op_name: str) -> Optional[VJPRule]:
    """Get the VJP rule for an operation.

    Args:
        op_name: The operation name

    Returns:
        The VJP rule function, or None if not registered
    """
    # Trigger lazy rule registration
    from ._core import _ensure_rules_registered
    _ensure_rules_registered()
    return VJP_REGISTRY.get(op_name)


def has_vjp(op_name: str) -> bool:
    """Check if a VJP rule exists for an operation.

    Args:
        op_name: The operation name

    Returns:
        True if a VJP rule is registered
    """
    # Trigger lazy rule registration
    from ._core import _ensure_rules_registered
    _ensure_rules_registered()
    return op_name in VJP_REGISTRY


def list_registered_ops() -> List[str]:
    """List all operations with registered VJP rules.

    Returns:
        List of operation names
    """
    # Trigger lazy rule registration
    from ._core import _ensure_rules_registered
    _ensure_rules_registered()
    return list(VJP_REGISTRY.keys())
