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
"""VJP rules for matrix multiplication operations."""
from typing import Dict, Optional, TYPE_CHECKING

import pypto
from ..registry import register_vjp, VJPContext
from ..utils import unbroadcast

if TYPE_CHECKING:
    from ...tensor import Tensor


def _get_out_grad(ctx: VJPContext) -> Optional["Tensor"]:
    return ctx.get_output_grad("output")


def _get_saved_input(ctx: VJPContext, name: str) -> Optional["Tensor"]:
    return ctx.get_saved(name)


@register_vjp("Matmul")
def vjp_matmul(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for matmul: d_A = d_C @ B^T, d_B = A^T @ d_C (with transpose handling)."""
    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None, "mat2": None}

    input_tensor = _get_saved_input(ctx, "input")
    mat2_tensor = _get_saved_input(ctx, "mat2")
    if input_tensor is None or mat2_tensor is None:
        return {"input": None, "mat2": None}

    a_trans = ctx.get_attr("a_trans") or False
    b_trans = ctx.get_attr("b_trans") or False

    input_dtype = input_tensor.dtype if hasattr(input_tensor, 'dtype') else d_out.dtype
    mat2_dtype = mat2_tensor.dtype if hasattr(mat2_tensor, 'dtype') else d_out.dtype

    if not a_trans and not b_trans:
        d_input = pypto.matmul(d_out, mat2_tensor, input_dtype, a_trans=False, b_trans=True)
        d_mat2 = pypto.matmul(input_tensor, d_out, mat2_dtype, a_trans=True, b_trans=False)
    elif a_trans and not b_trans:
        d_input = pypto.matmul(mat2_tensor, d_out, input_dtype, a_trans=False, b_trans=True)
        d_mat2 = pypto.matmul(input_tensor, d_out, mat2_dtype, a_trans=False, b_trans=False)
    elif not a_trans and b_trans:
        d_input = pypto.matmul(d_out, mat2_tensor, input_dtype, a_trans=False, b_trans=False)
        d_mat2 = pypto.matmul(d_out, input_tensor, mat2_dtype, a_trans=True, b_trans=False)
    else:
        d_input = pypto.matmul(mat2_tensor, d_out, input_dtype, a_trans=True, b_trans=True)
        d_mat2 = pypto.matmul(d_out, input_tensor, mat2_dtype, a_trans=True, b_trans=True)

    return {"input": d_input, "mat2": d_mat2}


@register_vjp("BatchMatmul")
def vjp_batch_matmul(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for batch_matmul (3D/4D). Same formulas as regular matmul."""
    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None, "mat2": None}

    input_tensor = _get_saved_input(ctx, "input")
    mat2_tensor = _get_saved_input(ctx, "mat2")
    if input_tensor is None or mat2_tensor is None:
        return {"input": None, "mat2": None}

    a_trans = ctx.get_attr("a_trans") or False
    b_trans = ctx.get_attr("b_trans") or False

    input_dtype = input_tensor.dtype if hasattr(input_tensor, 'dtype') else d_out.dtype
    mat2_dtype = mat2_tensor.dtype if hasattr(mat2_tensor, 'dtype') else d_out.dtype

    input_shape = ctx.get_input_shape("input")
    mat2_shape = ctx.get_input_shape("mat2")

    if not a_trans and not b_trans:
        d_input = pypto.matmul(d_out, mat2_tensor, input_dtype, a_trans=False, b_trans=True)
        d_mat2 = pypto.matmul(input_tensor, d_out, mat2_dtype, a_trans=True, b_trans=False)
    elif a_trans and not b_trans:
        d_input = pypto.matmul(mat2_tensor, d_out, input_dtype, a_trans=False, b_trans=True)
        d_mat2 = pypto.matmul(input_tensor, d_out, mat2_dtype, a_trans=False, b_trans=False)
    elif not a_trans and b_trans:
        d_input = pypto.matmul(d_out, mat2_tensor, input_dtype, a_trans=False, b_trans=False)
        d_mat2 = pypto.matmul(d_out, input_tensor, mat2_dtype, a_trans=True, b_trans=False)
    else:
        d_input = pypto.matmul(mat2_tensor, d_out, input_dtype, a_trans=True, b_trans=True)
        d_mat2 = pypto.matmul(d_out, input_tensor, mat2_dtype, a_trans=True, b_trans=True)

    if d_input is not None and input_shape is not None:
        d_input = unbroadcast(d_input, input_shape)
    if d_mat2 is not None and mat2_shape is not None:
        d_mat2 = unbroadcast(d_mat2, mat2_shape)

    return {"input": d_input, "mat2": d_mat2}
