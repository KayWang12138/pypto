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
"""VJP rules for mathematical operations.

Implements VJP (Vector-Jacobian Product) rules for:
- Binary: add, sub, mul, div
- Unary: neg, exp, log, sqrt, rsqrt, sigmoid
- Cast (dtype conversion)

VJP formulas:
- add(a, b): d_a = d_out, d_b = d_out
- sub(a, b): d_a = d_out, d_b = -d_out
- mul(a, b): d_a = d_out * b, d_b = d_out * a
- div(a, b): d_a = d_out / b, d_b = -d_out * a / (b * b)
- neg(a): d_a = -d_out
- exp(a): d_a = d_out * exp(a) = d_out * out
- log(a): d_a = d_out / a
- sqrt(a): d_a = d_out / (2 * sqrt(a)) = d_out / (2 * out)
- rsqrt(a): d_a = -0.5 * d_out * rsqrt(a)^3 = -0.5 * d_out * out^3
- sigmoid(a): d_a = d_out * sigmoid(a) * (1 - sigmoid(a)) = d_out * out * (1 - out)
"""
from typing import Dict, Optional, TYPE_CHECKING

from ..registry import register_vjp, VJPContext
from ..utils import unbroadcast, is_differentiable_dtype, get_shape

if TYPE_CHECKING:
    from ...tensor import Tensor


def _get_out_grad(ctx: VJPContext) -> Optional["Tensor"]:
    """Get the output gradient from context."""
    return ctx.get_output_grad("output")


def _get_saved_input(ctx: VJPContext, name: str = "input") -> Optional["Tensor"]:
    """Get a saved input tensor from context."""
    return ctx.get_saved(name)


# =============================================================================
# Binary Operations
# =============================================================================

@register_vjp("Add")
def vjp_add(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for add: out = input + other

    d_input = d_out (unbroadcast to input shape)
    d_other = d_out (unbroadcast to other shape)
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None, "other": None}

    alpha = ctx.get_attr("alpha", 1)

    input_shape = ctx.get_input_shape("input")
    other_shape = ctx.get_input_shape("other")

    d_input = d_out
    d_other = d_out
    if alpha != 1 and alpha != 1.0:
        d_other = pypto.mul(d_other, alpha)

    # Unbroadcast if shapes differ
    if input_shape is not None:
        d_input = unbroadcast(d_out, input_shape)
    if other_shape is not None:
        d_other = unbroadcast(d_other, other_shape)

    return {"input": d_input, "other": d_other}


@register_vjp("Sub")
def vjp_sub(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for sub: out = input - other

    d_input = d_out (unbroadcast to input shape)
    d_other = -d_out (unbroadcast to other shape)
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None, "other": None}

    alpha = ctx.get_attr("alpha", 1)

    input_shape = ctx.get_input_shape("input")
    other_shape = ctx.get_input_shape("other")

    d_input = d_out
    d_other = pypto.neg(d_out)
    if alpha != 1 and alpha != 1.0:
        d_other = pypto.mul(d_other, alpha)

    if input_shape is not None:
        d_input = unbroadcast(d_out, input_shape)
    if other_shape is not None:
        d_other = unbroadcast(d_other, other_shape)

    return {"input": d_input, "other": d_other}


@register_vjp("Mul")
def vjp_mul(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for mul: out = input * other

    d_input = d_out * other (unbroadcast to input shape)
    d_other = d_out * input (unbroadcast to other shape)
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None, "other": None}

    # Get saved inputs (saved during forward pass)
    input_tensor = _get_saved_input(ctx, "input")
    other_tensor = _get_saved_input(ctx, "other")

    input_shape = ctx.get_input_shape("input")
    other_shape = ctx.get_input_shape("other")

    d_input = None
    d_other = None

    if other_tensor is not None:
        d_input = pypto.mul(d_out, other_tensor)
        if input_shape is not None:
            d_input = unbroadcast(d_input, input_shape)

    if input_tensor is not None:
        d_other = pypto.mul(d_out, input_tensor)
        if other_shape is not None:
            d_other = unbroadcast(d_other, other_shape)

    return {"input": d_input, "other": d_other}


@register_vjp("Div")
def vjp_div(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for div: out = input / other

    d_input = d_out / other (unbroadcast to input shape)
    d_other = -d_out * input / (other * other) (unbroadcast to other shape)
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None, "other": None}

    input_tensor = _get_saved_input(ctx, "input")
    other_tensor = _get_saved_input(ctx, "other")

    input_shape = ctx.get_input_shape("input")
    other_shape = ctx.get_input_shape("other")

    d_input = None
    d_other = None

    if other_tensor is not None:
        # d_input = d_out / other
        d_input = pypto.div(d_out, other_tensor)
        if input_shape is not None:
            d_input = unbroadcast(d_input, input_shape)

    if input_tensor is not None and other_tensor is not None:
        # d_other = -d_out * input / (other * other)
        neg_d_out = pypto.neg(d_out)
        numerator = pypto.mul(neg_d_out, input_tensor)
        denominator = pypto.mul(other_tensor, other_tensor)
        d_other = pypto.div(numerator, denominator)
        if other_shape is not None:
            d_other = unbroadcast(d_other, other_shape)

    return {"input": d_input, "other": d_other}


# =============================================================================
# Unary Operations
# =============================================================================

@register_vjp("Neg")
def vjp_neg(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for neg: out = -input

    d_input = -d_out
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    d_input = pypto.neg(d_out)

    grads = {}
    input_names = getattr(ctx.node, "input_names", None)
    if input_names:
        grads[input_names[0]] = d_input
    elif ctx.node.inputs:
        name = getattr(ctx.node.inputs[0], "name", "")
        if name:
            grads[name] = d_input
    grads.setdefault("input", d_input)
    return grads


@register_vjp("Exp")
def vjp_exp(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for exp: out = exp(input)

    d_input = d_out * exp(input) = d_out * out
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    # Use saved output (exp(input)) instead of recomputing
    output = _get_saved_input(ctx, "output")
    if output is not None:
        d_input = pypto.mul(d_out, output)
    else:
        # Fallback: recompute exp
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            exp_input = pypto.exp(input_tensor)
            d_input = pypto.mul(d_out, exp_input)
        else:
            d_input = None

    return {"input": d_input}


@register_vjp("Log")
def vjp_log(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for log: out = log(input)

    d_input = d_out / input
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    input_tensor = _get_saved_input(ctx, "input")
    if input_tensor is not None:
        d_input = pypto.div(d_out, input_tensor)
    else:
        d_input = None

    return {"input": d_input}


@register_vjp("Sqrt")
def vjp_sqrt(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for sqrt: out = sqrt(input)

    d_input = d_out / (2 * sqrt(input)) = d_out / (2 * out)
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    # Use saved output (sqrt(input))
    output = _get_saved_input(ctx, "output")
    if output is not None:
        # d_input = d_out / (2 * out)
        two_out = pypto.mul(output, 2.0)
        d_input = pypto.div(d_out, two_out)
    else:
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            sqrt_input = pypto.sqrt(input_tensor)
            two_sqrt = pypto.mul(sqrt_input, 2.0)
            d_input = pypto.div(d_out, two_sqrt)
        else:
            d_input = None

    return {"input": d_input}


@register_vjp("Rsqrt")
def vjp_rsqrt(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for rsqrt: out = 1 / sqrt(input) = rsqrt(input)

    d_input = -0.5 * d_out * rsqrt(input)^3 = -0.5 * d_out * out^3
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    output = _get_saved_input(ctx, "output")
    if output is not None:
        # d_input = -0.5 * d_out * out^3
        out_cubed = pypto.mul(pypto.mul(output, output), output)
        d_input = pypto.mul(d_out, out_cubed)
        d_input = pypto.mul(d_input, -0.5)
    else:
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            rsqrt_input = pypto.rsqrt(input_tensor)
            out_cubed = pypto.mul(pypto.mul(rsqrt_input, rsqrt_input), rsqrt_input)
            d_input = pypto.mul(d_out, out_cubed)
            d_input = pypto.mul(d_input, -0.5)
        else:
            d_input = None

    return {"input": d_input}


@register_vjp("Sigmoid")
def vjp_sigmoid(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for sigmoid: out = sigmoid(input) = 1 / (1 + exp(-input))

    d_input = d_out * sigmoid(input) * (1 - sigmoid(input))
           = d_out * out * (1 - out)
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    output = _get_saved_input(ctx, "output")
    if output is not None:
        # d_input = d_out * out * (1 - out)
        one_minus_out = pypto.sub(1.0, output)
        d_input = pypto.mul(d_out, output)
        d_input = pypto.mul(d_input, one_minus_out)
    else:
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            sig = pypto.sigmoid(input_tensor)
            one_minus_sig = pypto.sub(1.0, sig)
            d_input = pypto.mul(d_out, sig)
            d_input = pypto.mul(d_input, one_minus_sig)
        else:
            d_input = None

    return {"input": d_input}


# =============================================================================
# Type Conversion
# =============================================================================

@register_vjp("Cast")
def vjp_cast(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for cast: out = cast(input, target_dtype)

    For floating point dtypes:
        d_input = cast(d_out, input.dtype)

    For non-floating point dtypes:
        Raise error (non-differentiable)
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    # Get the original input dtype
    input_dtype = ctx.get_attr("input_dtype")

    if input_dtype is None:
        # Try to get from saved tensor
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            input_dtype = input_tensor.dtype

    if input_dtype is not None:
        if not is_differentiable_dtype(input_dtype):
            raise RuntimeError(
                f"Cannot compute gradient for cast to non-differentiable dtype: {input_dtype}"
            )
        d_input = pypto.cast(d_out, input_dtype)
    else:
        d_input = d_out

    return {"input": d_input}
