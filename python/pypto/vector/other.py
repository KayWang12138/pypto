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
from typing import Union

from .. import pto_impl

from ..op_wrapper import op_wrapper
from ..tensor import Tensor


@op_wrapper
def where(
    condition: Tensor, input: Union[Tensor, float], other: Union[Tensor, float]
) -> Tensor:
    """
    Return a tensor of elements selected from either `input` or `other`, depending on `condition`.

    This function implements element-wise selection:
    'out[i] = input[i] if condition[i] else other[i]'.
    It supports broadcasting among `condition`, `input`, and `other`.

    Parameters
    ----------
    condition : Tensor of bool
        A boolean tensor indicating which elements to select from `input` (True) or `other` (False).
    input : Tensor or Number
        A tensor or scalar value to be selected where `condition` is True.
    other : Tensor or Number
        A tensor or scalar value to be selected where `condition` is False.

    Returns
    -------
    Tensor
        A tensor with the same shape as the broadcasted `condition`, containing elements
        from `input` where `condition` is True, and from `other` otherwise.
        The data type is determined by type promotion rules between `input` and `other`.

    Raises
    ------
    RuntimeError
        If `condition`, `input`, and `other` cannot be broadcasted to a common shape.
    TypeError
        If `condition` is not a boolean tensor.

    See Also
    --------
    logical_not : Computes element-wise logical NOT.
    add : Element-wise addition with optional scaling.

    Examples
    --------
    cond = pto.tensor([4], pto.DT_BOOL)
    x = pto.tensor([4], pto.DT_FP32)
    y = pto.tensor([4], pto.DT_FP32)
    out1 = pto.where(cond, x, y)

    Input cond:  [True False True False]
    Input x:     [1 2 3 4]
    Input y:     [10 20 30 40]
    Output out1: [1 20 3 40]

    # Using scalar inputs
    out2 = pto.where(cond, 1, 0)

    Output out2: [1 0 1 0]

    # Broadcasting example
    cond = pto.tensor([2, 2], pto.DT_BOOL)
    x = pto.tensor([1, 2], pto.DT_FP32)  # Will be broadcasted
    y = 0
    out3 = pto.where(cond, x, y)

    Input cond:  [[True False], [False True]]
    Input x:     [1 2]
    Input y:     0

    Output out3: [[1 0],
                  [0 2]])
    """
    if isinstance(input, pto_impl.Tensor):
        input_base = input
    else:
        input_base = pto_impl.Element(pto_impl.DT_FP32, input)

    if isinstance(other, pto_impl.Tensor):
        other_base = other
    else:
        other_base = pto_impl.Element(pto_impl.DT_FP32, other)
    return pto_impl.where(condition, input_base, other_base)


@op_wrapper
def one_hot(input: Tensor, num_classes: int) -> Tensor:
    """
    Converts a tensor of indices to one-hot encoded tensor.

    Parameters
    ----------
    input : Tensor
        LongTensor containing class indices of any shape (*)
    num_classes : int
        Total number of classes.

    Returns
    -------
    Tensor
        One-hot encoded tensor(LongTensor) of shape (*, num_classes) where:
        - 1 is placed at the index specified by input value
        - 0 is placed everywhere else

    Examples
    --------
    a = pto.tensor([3], pto.DT_INT32)
    out = pto.one_hot(a, 1)

    Input a:    [0 2 4]
    Input num_classes:  5
    Output out: [[1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1]]

    """
    if not isinstance(input, pto_impl.Tensor):
        raise TypeError("input must be a `Tensor`")
    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be an `int`")
    if num_classes == -1:
        raise RuntimeError("num_classes must be specified")
    if num_classes <= 0:
        raise RuntimeError("num_classes must be a positive integer")
    return pto_impl.one_hot(input, num_classes)
