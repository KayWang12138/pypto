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
"""VJP rules for view/shape manipulation operations.

Implements VJP (Vector-Jacobian Product) rules for:
- reshape: reshapes tensor to new shape
- transpose: swaps two dimensions
- view: extracts a slice/sub-tensor at given offsets
- assemble: writes a tensor into a larger tensor at offsets (in-place)
- clone: creates a copy of a tensor
- unsqueeze: adds a dimension of size 1

VJP formulas:
- reshape(input, shape): d_input = reshape(d_out, input_shape)
- transpose(input, dim0, dim1): d_input = transpose(d_out, dim0, dim1)
- view(input, shape, offsets): d_input = zeros(input_shape); assemble(d_out, offsets, d_input)
- assemble(input, offsets, out):
    - d_input = view(d_out_{n+1}, input_shape, offsets)
    - d_out_n = clone(d_out_{n+1}); assemble(zeros(input_shape), offsets, d_out_n)
- clone(input): d_input = d_out
- unsqueeze(input, dim): d_input = squeeze(d_out, dim) or reshape(d_out, input_shape)
"""
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from ..registry import register_vjp, VJPContext
from ..utils import zeros_like, get_shape

if TYPE_CHECKING:
    from ...tensor import Tensor


def _get_out_grad(ctx: VJPContext) -> Optional["Tensor"]:
    """Get the output gradient from context."""
    return ctx.get_output_grad("output")


def _get_saved_input(ctx: VJPContext, name: str = "input") -> Optional["Tensor"]:
    """Get a saved input tensor from context."""
    return ctx.get_saved(name)


# =============================================================================
# Reshape
# =============================================================================

@register_vjp("Reshape")
def vjp_reshape(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for reshape: out = reshape(input, shape)

    d_input = reshape(d_out, input_shape)

    The gradient just needs to be reshaped back to the original input shape.
    This is a simple identity operation in terms of data - only the shape changes.
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    # Get original input shape
    input_shape = ctx.get_input_shape("input")
    if input_shape is None:
        # Try to get from saved tensor
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            input_shape = get_shape(input_tensor)

    if input_shape is None:
        return {"input": None}

    # Reshape gradient back to input shape
    # Convert shape to list of ints if needed
    shape_list = [int(s) if isinstance(s, int) else s for s in input_shape]
    d_input = pypto.reshape(d_out, shape_list)

    return {"input": d_input}


# =============================================================================
# Transpose
# =============================================================================

@register_vjp("Transpose")
def vjp_transpose(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for transpose: out = transpose(input, dim0, dim1)

    d_input = transpose(d_out, dim0, dim1)

    Transposing is its own inverse - applying the same transpose operation
    on the gradient reverses the dimension swap.
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    # Get the dimensions that were transposed
    # The attrs should contain the axis/dims info
    # Check for various attribute names
    dim0 = ctx.get_attr("dim0")
    dim1 = ctx.get_attr("dim1")

    # If individual dims not found, try axis list
    if dim0 is None or dim1 is None:
        axis = ctx.get_attr("axis")
        if axis is not None and len(axis) == 2:
            dim0, dim1 = axis

    if dim0 is None or dim1 is None:
        # Default to 0,1 if not specified (common case)
        dim0, dim1 = 0, 1

    # Apply the same transpose to gradient
    d_input = pypto.transpose(d_out, dim0, dim1)

    return {"input": d_input}


# =============================================================================
# View (Slice/Extract)
# =============================================================================

@register_vjp("View")
def vjp_view(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for view: out = view(input, shape, offsets)

    View extracts a sub-tensor from input at given offsets.
    The gradient needs to be scattered back to the original positions.

    d_input = zeros(input_shape)
    assemble(d_out, offsets, d_input)

    Note: This creates a "contribution" to d_input. If the same input
    is used by multiple views, the engine will accumulate the gradients.
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    # Get original input shape
    input_shape = ctx.get_input_shape("input")
    if input_shape is None:
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            input_shape = get_shape(input_tensor)

    if input_shape is None:
        return {"input": None}

    # Get offsets from attributes
    offsets = ctx.get_attr("offsets")
    if offsets is None:
        # Can't compute gradient without knowing offsets
        return {"input": None}

    # Get dtype from d_out
    dtype = d_out.dtype if hasattr(d_out, "dtype") else pypto.DT_FP32

    # Create zeros tensor with input shape
    shape_list = [int(s) if isinstance(s, int) else s for s in input_shape]
    d_input = pypto.zeros(shape_list, dtype=dtype)

    # Scatter d_out to the positions where the view was taken
    pypto.assemble(d_out, offsets, d_input)

    return {"input": d_input}


# =============================================================================
# Assemble (In-place write)
# =============================================================================

@register_vjp("Assemble")
def vjp_assemble(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for assemble: assemble(input, offsets, out)

    Assemble writes 'input' into 'out' at 'offsets' (in-place operation).
    In SSA semantics: out_{n+1} = assemble(input, offsets, out_n)

    Gradients:
    - d_input = view(d_out_{n+1}, input_shape, offsets)
        Extract the gradient at the positions where input was written.

    - d_out_n = clone(d_out_{n+1}); assemble(zeros_like(input), offsets, d_out_n)
        The old values at the assembled positions don't affect the result,
        so their gradient should be zero. We copy the gradient and zero out
        the assembled region.

    Note: assemble returns None, so "output" here refers to the modified out tensor.
    The tracer needs to track this as an in-place modification that creates
    a new version of the out tensor.
    """
    import pypto

    # Get gradient of the modified output tensor
    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None, "out": None}

    # Get input shape
    input_shape = ctx.get_input_shape("input")
    if input_shape is None:
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            input_shape = get_shape(input_tensor)

    # Get offsets from attributes
    offsets = ctx.get_attr("offsets")
    if offsets is None:
        return {"input": None, "out": None}

    # d_input: extract gradient at the assembled positions
    d_input = None
    if input_shape is not None:
        shape_list = [int(s) if isinstance(s, int) else s for s in input_shape]
        d_input = pypto.view(d_out, shape_list, offsets)

    # d_out_old: gradient for the old version of out tensor
    # We need to zero out the assembled region to cut the gradient path
    # for the old values at those positions
    dtype = d_out.dtype if hasattr(d_out, "dtype") else pypto.DT_FP32
    d_out_old = pypto.clone(d_out)

    # Create zeros with input shape to zero out the assembled region
    if input_shape is not None:
        shape_list = [int(s) if isinstance(s, int) else s for s in input_shape]
        zeros_input = pypto.zeros(shape_list, dtype=dtype)
        pypto.assemble(zeros_input, offsets, d_out_old)

    return {"input": d_input, "out": d_out_old}


# =============================================================================
# Clone (Copy)
# =============================================================================

@register_vjp("Assign")  # clone uses Assign internally
def vjp_clone(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for clone: out = clone(input)

    Clone creates a copy of the tensor. The gradient passes through unchanged.

    d_input = d_out
    """
    d_out = _get_out_grad(ctx)
    return {"input": d_out}


# =============================================================================
# Unsqueeze
# =============================================================================

@register_vjp("Unsqueeze")
def vjp_unsqueeze(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for unsqueeze: out = unsqueeze(input, dim)

    Unsqueeze adds a dimension of size 1 at the specified position.
    The gradient needs to remove that dimension (squeeze).

    d_input = reshape(d_out, input_shape) or squeeze(d_out, dim)

    We use reshape for simplicity since squeeze may not be available.
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    # Get original input shape
    input_shape = ctx.get_input_shape("input")
    if input_shape is None:
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            input_shape = get_shape(input_tensor)

    if input_shape is None:
        return {"input": None}

    # Reshape gradient back to input shape (removing the added dimension)
    shape_list = [int(s) if isinstance(s, int) else s for s in input_shape]
    d_input = pypto.reshape(d_out, shape_list)

    return {"input": d_input}
