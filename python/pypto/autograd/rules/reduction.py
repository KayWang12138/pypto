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
"""VJP rules for reduction operations."""
from typing import Dict, Optional, List, Any, TYPE_CHECKING

import pypto
from ..registry import register_vjp, VJPContext
from ..utils import get_shape

if TYPE_CHECKING:
    from ...tensor import Tensor


def _expand_to_shape(tensor: "Tensor", target_shape: List[Any], dim: int) -> "Tensor":
    """Expand a tensor to target shape by broadcasting along specified dimension."""
    tensor_shape = get_shape(tensor)
    if tensor_shape == target_shape:
        return tensor
    return pypto.expand_clone(tensor, target_shape)


@register_vjp("Sum")
def vjp_sum(ctx: VJPContext) -> Dict[str, Optional["Tensor"]]:
    """d_input = broadcast(d_out, input_shape) - with unsqueeze if keepdim=False"""
    d_out = ctx.get_output_grad("output")
    if d_out is None:
        return {"input": None}

    dim = ctx.get_attr("dim")
    keepdim = ctx.get_attr("keepdim", False)

    input_shape = ctx.get_input_shape("input")
    if input_shape is None:
        return {"input": None}

    if dim is not None and dim < 0:
        dim = len(input_shape) + dim

    if not keepdim and dim is not None:
        d_out = pypto.unsqueeze(d_out, dim)

    d_input = _expand_to_shape(d_out, input_shape, dim)

    return {"input": d_input}
