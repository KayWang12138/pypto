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
"""VJP rules for matrix multiplication operations.

Implements VJP (Vector-Jacobian Product) rules for:
- matmul: matrix multiplication with optional transpose

VJP formulas for C = matmul(A, B, a_trans=at, b_trans=bt):
Equivalent to: C = T(A,at) @ T(B,bt) where T(X,True)=X^T, T(X,False)=X

Case 1: at=False, bt=False: C = A @ B
    d_A = matmul(d_C, B, a_trans=False, b_trans=True)   # d_C @ B^T
    d_B = matmul(A, d_C, a_trans=True, b_trans=False)   # A^T @ d_C

Case 2: at=True, bt=False: C = A^T @ B
    d_A = matmul(B, d_C, a_trans=False, b_trans=True)   # B @ d_C^T
    d_B = matmul(A, d_C, a_trans=False, b_trans=False)  # A @ d_C

Case 3: at=False, bt=True: C = A @ B^T
    d_A = matmul(d_C, B, a_trans=False, b_trans=False)  # d_C @ B
    d_B = matmul(d_C, A, a_trans=True, b_trans=False)   # d_C^T @ A

Case 4: at=True, bt=True: C = A^T @ B^T
    d_A = matmul(B, d_C, a_trans=True, b_trans=True)    # B^T @ d_C^T
    d_B = matmul(d_C, A, a_trans=True, b_trans=True)    # d_C^T @ A^T
"""
from typing import Dict, Optional, TYPE_CHECKING

from ..registry import register_vjp, VJPContext
from ..utils import unbroadcast

if TYPE_CHECKING:
    from ...tensor import Tensor


def _get_out_grad(ctx: VJPContext) -> Optional["Tensor"]:
    """Get the output gradient from context."""
    return ctx.get_output_grad("output")


def _get_saved_input(ctx: VJPContext, name: str) -> Optional["Tensor"]:
    """Get a saved input tensor from context."""
    return ctx.get_saved(name)


# =============================================================================
# Matmul (2D Matrix Multiplication)
# =============================================================================

@register_vjp("Matmul")
def vjp_matmul(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for matmul: out = matmul(input, mat2, out_dtype, a_trans, b_trans)

    The gradients depend on the transpose flags used in the forward pass.
    We need to use the original input tensors and apply appropriate transposes
    in the backward matmul operations.
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None, "mat2": None}

    # Get saved inputs
    input_tensor = _get_saved_input(ctx, "input")
    mat2_tensor = _get_saved_input(ctx, "mat2")

    if input_tensor is None or mat2_tensor is None:
        return {"input": None, "mat2": None}

    # Get transpose flags from attributes
    a_trans = ctx.get_attr("a_trans")
    b_trans = ctx.get_attr("b_trans")

    # Default to False if not specified
    if a_trans is None:
        a_trans = False
    if b_trans is None:
        b_trans = False

    # Get input dtypes for gradient computation
    input_dtype = input_tensor.dtype if hasattr(input_tensor, 'dtype') else None
    mat2_dtype = mat2_tensor.dtype if hasattr(mat2_tensor, 'dtype') else None

    # Use input dtypes for gradients (gradient should match input dtype)
    # If not available, use d_out's dtype
    d_input_dtype = input_dtype if input_dtype is not None else d_out.dtype
    d_mat2_dtype = mat2_dtype if mat2_dtype is not None else d_out.dtype

    # Compute gradients based on transpose flags
    d_input = None
    d_mat2 = None

    if not a_trans and not b_trans:
        # Case 1: C = A @ B
        # d_A = d_C @ B^T
        # d_B = A^T @ d_C
        d_input = pypto.matmul(d_out, mat2_tensor, d_input_dtype,
                               a_trans=False, b_trans=True)
        d_mat2 = pypto.matmul(input_tensor, d_out, d_mat2_dtype,
                              a_trans=True, b_trans=False)

    elif a_trans and not b_trans:
        # Case 2: C = A^T @ B
        # d_A = B @ d_C^T
        # d_B = A @ d_C
        d_input = pypto.matmul(mat2_tensor, d_out, d_input_dtype,
                               a_trans=False, b_trans=True)
        d_mat2 = pypto.matmul(input_tensor, d_out, d_mat2_dtype,
                              a_trans=False, b_trans=False)

    elif not a_trans and b_trans:
        # Case 3: C = A @ B^T
        # d_A = d_C @ B
        # d_B = d_C^T @ A
        d_input = pypto.matmul(d_out, mat2_tensor, d_input_dtype,
                               a_trans=False, b_trans=False)
        d_mat2 = pypto.matmul(d_out, input_tensor, d_mat2_dtype,
                              a_trans=True, b_trans=False)

    else:  # a_trans and b_trans
        # Case 4: C = A^T @ B^T
        # d_A = B^T @ d_C^T
        # d_B = d_C^T @ A^T
        d_input = pypto.matmul(mat2_tensor, d_out, d_input_dtype,
                               a_trans=True, b_trans=True)
        d_mat2 = pypto.matmul(d_out, input_tensor, d_mat2_dtype,
                              a_trans=True, b_trans=True)

    return {"input": d_input, "mat2": d_mat2}


# =============================================================================
# BatchMatmul (3D/4D Batch Matrix Multiplication)
# =============================================================================

@register_vjp("BatchMatmul")
def vjp_batch_matmul(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """VJP for batch_matmul: out = batch_matmul(input, mat2, out_dtype, a_trans, b_trans)

    Same VJP formulas as regular matmul, but for batched operations.
    The batch dimensions are preserved through the backward pass.

    Note: MVP focuses on 2D matmul; batch matmul support is provided for completeness
    but may have limitations with broadcast handling in the batch dimensions.
    """
    import pypto

    d_out = _get_out_grad(ctx)
    if d_out is None:
        return {"input": None, "mat2": None}

    # Get saved inputs
    input_tensor = _get_saved_input(ctx, "input")
    mat2_tensor = _get_saved_input(ctx, "mat2")

    if input_tensor is None or mat2_tensor is None:
        return {"input": None, "mat2": None}

    # Get transpose flags from attributes
    a_trans = ctx.get_attr("a_trans")
    b_trans = ctx.get_attr("b_trans")

    if a_trans is None:
        a_trans = False
    if b_trans is None:
        b_trans = False

    # Get input dtypes for gradient computation
    input_dtype = input_tensor.dtype if hasattr(input_tensor, 'dtype') else None
    mat2_dtype = mat2_tensor.dtype if hasattr(mat2_tensor, 'dtype') else None

    d_input_dtype = input_dtype if input_dtype is not None else d_out.dtype
    d_mat2_dtype = mat2_dtype if mat2_dtype is not None else d_out.dtype

    input_shape = ctx.get_input_shape("input")
    mat2_shape = ctx.get_input_shape("mat2")

    d_input = None
    d_mat2 = None

    # Same VJP formulas as regular matmul
    if not a_trans and not b_trans:
        d_input = pypto.matmul(d_out, mat2_tensor, d_input_dtype,
                               a_trans=False, b_trans=True)
        d_mat2 = pypto.matmul(input_tensor, d_out, d_mat2_dtype,
                              a_trans=True, b_trans=False)

    elif a_trans and not b_trans:
        d_input = pypto.matmul(mat2_tensor, d_out, d_input_dtype,
                               a_trans=False, b_trans=True)
        d_mat2 = pypto.matmul(input_tensor, d_out, d_mat2_dtype,
                              a_trans=False, b_trans=False)

    elif not a_trans and b_trans:
        d_input = pypto.matmul(d_out, mat2_tensor, d_input_dtype,
                               a_trans=False, b_trans=False)
        d_mat2 = pypto.matmul(d_out, input_tensor, d_mat2_dtype,
                              a_trans=True, b_trans=False)

    else:  # a_trans and b_trans
        d_input = pypto.matmul(mat2_tensor, d_out, d_input_dtype,
                               a_trans=True, b_trans=True)
        d_mat2 = pypto.matmul(d_out, input_tensor, d_mat2_dtype,
                              a_trans=True, b_trans=True)

    if d_input is not None and input_shape is not None:
        d_input = unbroadcast(d_input, input_shape)
    if d_mat2 is not None and mat2_shape is not None:
        d_mat2 = unbroadcast(d_mat2, mat2_shape)

    return {"input": d_input, "mat2": d_mat2}
