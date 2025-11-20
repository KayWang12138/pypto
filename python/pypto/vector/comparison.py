# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO"""
from typing import Optional, Tuple

from .. import pto_impl

from ..enum import OpType, OutType
from ..op_wrapper import op_wrapper
from ..tensor import Tensor


@op_wrapper
def greater(input: Tensor, other: Tensor) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is greater than other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is not a Tensor.


    Examples
    --------
    a = pto.tensor([3], pto.DT_FP32)
    b = pto.tensor([3], pto.DT_FP32)
    out = pto.greater(a, b)

    Input a:    [1 2 3]
    Input b:    [2 2 2]
    Output out: [False False True]

    """
    return pto_impl.compare(input, other, OpType.GT, OutType.BOOL)


@op_wrapper
def topk(
    input: Tensor, k: int, dim: Optional[int] = None, largest: bool = True
) -> Tuple[Tensor, Tensor]:
    """Returns the k largest elements of the given input tensor along a given dimension.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    k : int
        The k in "top-k".
    dim : int, optional
        The dimension to sort along, if dim is not given, the last dimension of the input is chosen.
    largest : bool
        Controling whether to return the elements in sorted order, if largest is False then the k smallest
        elements are returned.

    Returns
    -------
    tuple
        A tuple of (values, indices) is returned with the values and indices of the largest k elements of
        each row of the input tensor in the given dimension dim.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.topk(x, 2, -1, True)

    Input x:     [[1 2 3],
                  [1 2 3]]
    Output y[0]: [[3 2],
                  [3 2]]
    Output y[1]: [[2 1],
                  [2 1]]
    """

    return pto_impl.topk(input, k, (-1 if dim is None else dim), largest)
