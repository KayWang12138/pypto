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
"""VJP rules for reduction operations.

Implements VJP (Vector-Jacobian Product) rules for:
- sum: reduce sum along dimension

VJP formulas:
- sum(input, dim, keepdim):
    d_input = broadcast(d_out, input_shape)

    If keepdim=False, need to unsqueeze d_out at dim first.
    Then broadcast/expand to match input shape.
"""
from typing import Dict, Optional, List, Any, TYPE_CHECKING

from ..registry import register_vjp, VJPContext
from ..utils import get_shape

if TYPE_CHECKING:
    from ...tensor import Tensor


def _expand_to_shape(tensor: "Tensor", target_shape: List[Any], dim: int) -> "Tensor":
    """Expand a tensor to target shape by broadcasting along specified dimension.

    Args:
        tensor: The input tensor (reduced along dim)
        target_shape: The target shape to broadcast to
        dim: The dimension that was reduced

    Returns:
        Tensor broadcast to target_shape
    """
    import pypto

    tensor_shape = get_shape(tensor)

    # If shapes already match, return as-is
    if tensor_shape == target_shape:
        return tensor

    # Use expand_clone to broadcast the tensor
    # This replicates the reduced dimension
    return pypto.expand_clone(tensor, target_shape)


@register_vjp("Sum")
def vjp_sum(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for sum: out = sum(input, dim, keepdim)

    d_input = broadcast(d_out, input_shape)

    If keepdim=False:
        1. unsqueeze d_out at dim to restore the reduced dimension
        2. broadcast/expand to input_shape
    If keepdim=True:
        1. broadcast/expand d_out to input_shape
    """
    import pypto

    d_out = ctx.get_output_grad("output")
    if d_out is None:
        return {"input": None}

    # Get attributes
    dim = ctx.get_attr("dim")
    keepdim = ctx.get_attr("keepdim", False)

    # Get original input shape
    input_shape = ctx.get_input_shape("input")
    if input_shape is None:
        return {"input": None}

    # Handle negative dim
    if dim is not None and dim < 0:
        dim = len(input_shape) + dim

    # If keepdim=False, need to unsqueeze first
    if not keepdim and dim is not None:
        d_out = pypto.unsqueeze(d_out, dim)

    # Expand to input shape
    d_input = _expand_to_shape(d_out, input_shape, dim)

    return {"input": d_input}


# Note: amax/amin VJP rules are more complex because they are not differentiable
# everywhere (gradient is non-zero only at the max/min positions).
# These will be implemented in Phase D (discrete/sparse operators).
