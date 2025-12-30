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
"""Utility functions for autograd."""
from typing import Any, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..tensor import Tensor


_DIFFERENTIABLE_DTYPES = {
    "DT_FP32",
    "DT_FP16",
    "DT_BF16",
    "DT_FLOAT",
    "DT_HALF",
    "FP32",
    "FP16",
    "BF16",
    "FLOAT32",
    "FLOAT16",
    "BFLOAT16",
    "F32",
    "F16",
    "FLOAT",
    "HALF",
}


def is_differentiable_dtype(dtype: Any) -> bool:
    """Check if a dtype is differentiable (floating point)."""
    dtype_name = str(dtype) if not isinstance(dtype, str) else dtype
    if "." in dtype_name:
        dtype_name = dtype_name.split(".")[-1]
    dtype_name = dtype_name.upper()
    return dtype_name in _DIFFERENTIABLE_DTYPES


def get_shape(tensor: "Tensor") -> List[Any]:
    """Get the shape of a tensor."""
    return tensor.shape


def get_dtype(tensor: "Tensor") -> Any:
    """Get the dtype of a tensor."""
    return tensor.dtype


def zeros_like(tensor: "Tensor") -> "Tensor":
    """Create a tensor of zeros with the same shape and dtype."""
    import pypto
    shape = get_shape(tensor)
    dtype = get_dtype(tensor)
    return pypto.zeros(shape, dtype)


def ones_like(tensor: "Tensor") -> "Tensor":
    """Create a tensor of ones with the same shape and dtype."""
    import pypto
    shape = get_shape(tensor)
    dtype = get_dtype(tensor)
    return pypto.ones(shape, dtype)


def full_like(tensor: "Tensor", fill_value: Union[int, float]) -> "Tensor":
    """Create a tensor filled with a value, with same shape and dtype."""
    import pypto
    shape = get_shape(tensor)
    dtype = get_dtype(tensor)
    return pypto.full(shape, fill_value, dtype)


def _compute_broadcast_axes(
    grad_shape: List[int],
    target_shape: List[int]
) -> List[int]:
    """Compute which axes were broadcast from target to grad shape."""
    grad_ndim = len(grad_shape)
    target_ndim = len(target_shape)

    broadcast_axes = []

    for i in range(grad_ndim - target_ndim):
        broadcast_axes.append(i)

    for i in range(target_ndim):
        grad_axis = i + (grad_ndim - target_ndim)
        target_dim = target_shape[i]
        grad_dim = grad_shape[grad_axis]

        if target_dim == 1 and grad_dim != 1:
            broadcast_axes.append(grad_axis)

    return broadcast_axes


def unbroadcast(
    grad: "Tensor",
    target_shape: List[Any],
    keepdim: bool = False
) -> "Tensor":
    """Reduce gradient to match the original input shape before broadcasting."""
    import pypto
    grad_shape = get_shape(grad)

    grad_shape_concrete = [int(s) if hasattr(s, '__int__') else s for s in grad_shape]
    target_shape_concrete = [int(s) if hasattr(s, '__int__') else s for s in target_shape]

    if grad_shape_concrete == target_shape_concrete:
        return grad

    broadcast_axes = _compute_broadcast_axes(grad_shape_concrete, target_shape_concrete)

    if not broadcast_axes:
        return grad

    result = grad
    for axis in sorted(broadcast_axes, reverse=True):
        result = pypto.sum(result, dim=axis, keepdim=True)

    result_shape = get_shape(result)
    result_shape_concrete = [int(s) if hasattr(s, '__int__') else s for s in result_shape]

    if result_shape_concrete != target_shape_concrete:
        if len(result_shape_concrete) > len(target_shape_concrete):
            result = pypto.reshape(result, target_shape_concrete)
        elif len(result_shape_concrete) == len(target_shape_concrete):
            result = pypto.reshape(result, target_shape_concrete)

    return result


def broadcast_shapes(shape1: List[int], shape2: List[int]) -> List[int]:
    """Compute the broadcast shape of two shapes."""
    ndim = max(len(shape1), len(shape2))
    shape1 = [1] * (ndim - len(shape1)) + list(shape1)
    shape2 = [1] * (ndim - len(shape2)) + list(shape2)

    result = []
    for d1, d2 in zip(shape1, shape2):
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")

    return result


def maybe_cast(tensor: "Tensor", target_dtype: Any) -> "Tensor":
    """Cast tensor to target dtype if different."""
    import pypto
    if get_dtype(tensor) == target_dtype:
        return tensor
    return pypto.cast(tensor, target_dtype)
