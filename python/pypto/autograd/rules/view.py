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
"""VJP rules for view/shape manipulation operations."""
from typing import Dict, List, Optional, Any, TYPE_CHECKING

import pypto
from ..registry import register_vjp, VJPContext
from ..utils import get_shape

if TYPE_CHECKING:
    from ...tensor import Tensor


def _get_out_grad(ctx: VJPContext) -> Optional["Tensor"]:
    return ctx.get_output_grad("output")


def _get_saved_input(ctx: VJPContext, name: str = "input") -> Optional["Tensor"]:
    return ctx.get_saved(name)


@register_vjp("Reshape")
def vjp_reshape(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """d_input = reshape(d_out, input_shape)"""
    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    input_shape = ctx.get_input_shape("input")
    if input_shape is None:
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            input_shape = get_shape(input_tensor)

    if input_shape is None:
        return {"input": None}

    shape_list = [int(s) if isinstance(s, int) else s for s in input_shape]
    d_input = pypto.reshape(d_out, shape_list)

    return {"input": d_input}


@register_vjp("Transpose")
def vjp_transpose(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """d_input = transpose(d_out, dim0, dim1) - transpose is its own inverse"""
    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    dim0 = ctx.get_attr("dim0")
    dim1 = ctx.get_attr("dim1")

    if dim0 is None or dim1 is None:
        axis = ctx.get_attr("axis")
        if axis is not None and len(axis) == 2:
            dim0, dim1 = axis

    if dim0 is None or dim1 is None:
        dim0, dim1 = 0, 1

    d_input = pypto.transpose(d_out, dim0, dim1)

    return {"input": d_input}


@register_vjp("View")
def vjp_view(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """d_input = zeros(input_shape); assemble(d_out, offsets, d_input)"""
    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    input_shape = ctx.get_input_shape("input")
    if input_shape is None:
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            input_shape = get_shape(input_tensor)

    if input_shape is None:
        return {"input": None}

    offsets = ctx.get_attr("offsets")
    if offsets is None:
        return {"input": None}

    dtype = d_out.dtype if hasattr(d_out, "dtype") else pypto.DT_FP32
    shape_list = [int(s) if isinstance(s, int) else s for s in input_shape]
    d_input = pypto.zeros(shape_list, dtype=dtype)
    pypto.assemble(d_out, offsets, d_input)

    return {"input": d_input}


@register_vjp("Assemble")
def vjp_assemble(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """d_input = view(d_out, input_shape, offsets); d_out_old = clone with zeros at assembled region"""
    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None, "out": None}

    input_shape = ctx.get_input_shape("input")
    if input_shape is None:
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            input_shape = get_shape(input_tensor)

    offsets = ctx.get_attr("offsets")
    if offsets is None:
        return {"input": None, "out": None}

    d_input = None
    if input_shape is not None:
        shape_list = [int(s) if isinstance(s, int) else s for s in input_shape]
        d_input = pypto.view(d_out, shape_list, offsets)

    dtype = d_out.dtype if hasattr(d_out, "dtype") else pypto.DT_FP32
    d_out_old = pypto.clone(d_out)

    if input_shape is not None:
        shape_list = [int(s) if isinstance(s, int) else s for s in input_shape]
        zeros_input = pypto.zeros(shape_list, dtype=dtype)
        pypto.assemble(zeros_input, offsets, d_out_old)

    return {"input": d_input, "out": d_out_old}


@register_vjp("Assign")
def vjp_clone(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """d_input = d_out (gradient passes through unchanged)"""
    d_out = _get_out_grad(ctx)
    return {"input": d_out}


@register_vjp("Unsqueeze")
def vjp_unsqueeze(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """d_input = reshape(d_out, input_shape) - removes the added dimension"""
    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None}

    input_shape = ctx.get_input_shape("input")
    if input_shape is None:
        input_tensor = _get_saved_input(ctx, "input")
        if input_tensor is not None:
            input_shape = get_shape(input_tensor)

    if input_shape is None:
        return {"input": None}

    shape_list = [int(s) if isinstance(s, int) else s for s in input_shape]
    d_input = pypto.reshape(d_out, shape_list)

    return {"input": d_input}
