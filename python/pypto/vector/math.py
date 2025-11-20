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
from typing import Optional, Union

from .. import pto_impl

from ..element import Element
from ..op_wrapper import op_wrapper
from ..tensor import Tensor


@op_wrapper
def add(
    input: Tensor, other: Union[Tensor, int, float], *, alpha: Union[int, float] = 1
) -> Tensor:
    """Computes the element-wise addition of `input` and `other`.

    This function calculates the formula: `out = input + alpha * other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to be added.
    alpha : float, optional, keyword-only
        A scaling factor for the `other` input. Default is 1.0.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise sum.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    See Also
    --------
    sub : The inverse operation, element-wise subtraction.
    mul : Element-wise multiplication.

    Examples
    --------
    a = pto.tensor([1, 3], pto.DT_FP32)
    b = pto.tensor([1, 3], pto.DT_FP32)
    out = pto.add(a, b)

    Input a:    [1 2 3]
    Input b:    [2 3 4]
    Output out: [3 5 7]
    """
    if isinstance(other, pto_impl.Tensor):
        if alpha == 1 or alpha == 1.0:
            return pto_impl.add(input, other)
        else:
            return pto_impl.add(
                input, pto_impl.mul_s(other, pto_impl.Element(input.dtype, alpha))
            )
    else:
        if alpha == 1 or alpha == 1.0:
            return pto_impl.add_s(input, pto_impl.Element(input.dtype, other))
        else:
            if not isinstance(other, (int, float)):
                raise TypeError(f"alpha must be int or float, but got {type(other)}.")
            return pto_impl.add_s(input, pto_impl.Element(input.dtype, other * alpha))


@op_wrapper
def sub(
    input: Tensor, other: Union[Tensor, int, float], *, alpha: Union[int, float] = 1
) -> Tensor:
    """Computes the element-wise subtraction of `input` and `other`.

    This function calculates the formula: `out = input - alpha * other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to be subtracted.
    alpha : float, optional, keyword-only
        A scaling factor for the `other` input. Default is 1.0.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise subtraction.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.tensor([2, 3], pto.DT_FP32)
    out1 = pto.sub(a, b)

    Input x:      [[9 9 9],
                   [9 9 9]]
    Input y:      [[1 2 3],
                   [1 2 3]]
    Output out1 : [[8 7 6],
                   [8 7 6]]

    # Using a scalar and alpha
    c = pto.sub(x, 2, alpha=3) # Computes x - 2 * 3

    Output c:[[3 3 3],
              [3 3 3]]
    """
    if isinstance(other, pto_impl.Tensor):
        if alpha == 1 or alpha == 1.0:
            return pto_impl.sub(input, other)
        else:
            return pto_impl.sub(
                input, pto_impl.mul_s(other, pto_impl.Element(input.dtype, alpha))
            )
    else:
        if alpha == 1 or alpha == 1.0:
            return pto_impl.sub(input, pto_impl.Element(input.dtype, other))
        else:
            if not isinstance(other, (int, float)):
                raise TypeError(f"alpha must be int or float, but got {type(other)}.")
            return pto_impl.sub(input, pto_impl.Element(input.dtype, other * alpha))


@op_wrapper
def mul(input: Tensor, other: Union[Tensor, int, float]) -> Tensor:
    """Computes the element-wise multiplication of `input` and `other`.

    This function calculates the formula: `out = input * other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to be multiplied.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise multiplication.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.tensor([2, 3], pto.DT_FP32)
    z = pto.mul(a, b)

    Input x:[[1 2 3],
             [1 2 3]]
    Input y:[[1 2 3],
             [1 2 3]]
    Output z:[[1 4 9],
              [1 4 9]]
    """
    if isinstance(other, pto_impl.Tensor):
        return pto_impl.mul(input, other)
    else:
        return pto_impl.mul_s(input, pto_impl.Element(input.dtype, other))


@op_wrapper
def div(input: Tensor, other: Union[Tensor, int, float]) -> Tensor:
    """Computes the element-wise division of `input` and `other`.

    This function calculates the formula: `out = input / other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to divide.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise division.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    See Also
    --------
    sub : The inverse operation, element-wise subtraction.
    mul : Element-wise multiplication.

    Examples
    --------
    a = pto.tensor([3], pto.DT_FP32)
    b = pto.tensor([3], pto.DT_FP32)
    out = pto.div(a, b)

    Input a:    [2 4 6]
    Input b:    [2 2 2]
    Output out: [1 2 3]
    """
    if isinstance(other, pto_impl.Tensor):
        return pto_impl.div(input, other)
    else:
        return pto_impl.div_s(input, pto_impl.Element(input.dtype, other))


@op_wrapper
def pow(input: Tensor, other: Union[int, float]) -> Tensor:
    """Computes the element-wise power of `input` raised to `other`.

    This function calculates the formula: `out = input ** other`.

    Parameters
    ----------
    input : Tensor
        The base input tensor.
    other : Number
        The exponent to which each element in `input` will be raised.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise power operation results.

    Examples
    --------
    x = pto.tensor([2, 2], pto.DT_FP32)
    a = 2
    y = pto.pow(x, a)

    Input x:[[1 2],
             [3 4]]
    Output y:[[1  4],
              [9 16]]
    """
    if not isinstance(other, (int, float)):
        raise TypeError(f"other must be int or float, but got {type(other)}.")
    return pto_impl.pow(input, pto_impl.Element(input.dtype, other))


@op_wrapper
def exp(input: Tensor) -> Tensor:
    """Computes the element-wise exponential of `input`.

    This function calculates the formula: `out = e ** input`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise exponential.

    See Also
    -------
    sqrt : Element-wise square-root

    Examples
    --------
    x = pto.tensor([3], pto.DT_FP32)
    y = pto.exp(x)

    Input x: [0 1 2]
    Output y:[1.0000 2.7183 7.3891]
    """
    return pto_impl.exp(input)


@op_wrapper
def abs(a) -> Tensor:
    return pto_impl.abs(a)


@op_wrapper
def reciprocal(a) -> Tensor:
    return pto_impl.reciprocal(a)


@op_wrapper
def logical_not(input: Tensor) -> Tensor:
    """
    Computes the element-wise logical NOT of 'input'

    This funtion calculates the formula: 'out = input == 0? True : False'.

    Parameters
    ----------
    input : Tensor
        The input tensor

    Returns
    -------
    Tensor
        A tensor of bool with the same shape as input

    Examples
    --------
    a = pto.tensor([5], pto.DT_INT32)
    out = pto.logical_not(a)

    Input a:    [0 1 2 3 4]
    Output out: [True False False False False False]

    """
    return pto_impl.logical_not(input)


@op_wrapper
def logical_and(input: Tensor, other: Tensor) -> Tensor:
    """Computes the element-wise logical AND of `input` and `other`.

    This function calculates the formula: `out = input && other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor. Should be broadcastable to the shape of `input`.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise logical AND operation results.

    Examples
    --------
    x = pto.tensor([True, False], pto.DT_BOOL)
    y = pto.tensor([True, True], pto.DT_BOOL)
    z = pto.logical_and(x, y)

    Input x: [True, False]
    Input y: [True, True]
    Output z: [True, False]

    # 支持广播
    x = pto.tensor([[True, False], [False, True]], pto.DT_BOOL)
    y = pto.tensor([True, False], pto.DT_BOOL)
    z = pto.logical_and(x, y)

    Input x: [[True, False], [False, True]]
    Input y: [True, False]
    Output z: [[True, False], [False, False]]
    """
    return pto_impl.logical_and(input, other)


@op_wrapper
def rsqrt(input: Tensor) -> Tensor:
    """Computes the element-wise reciprocal of the square-root of `input`

    This function calculates the formula: `out = 1 / sqrt(input)`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor with the reciprocal of the square-root of each of the element of input.

    Raises
    ------
    TODO

    See Also
    --------
    sqrt : square-root of each of the element

    Examples
    --------
    x = pto.tensor([2, 2], pto.DT_FP32)
    y = pto.rsqrt(x)

    Input x: [[1  4],
             [16 9]]
    Output y:[[1  0.5],
              [0.25 0.33333]]
    """
    return pto_impl.rsqrt(input)


@op_wrapper
def sqrt(input: Tensor) -> Tensor:
    """Computes the element-wise squareroot of `input`.

    This function calculates the formula: `out = √input`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise squareroot.

    See Also
    --------
    exp : Element-wise exponential function.

    Examples
    --------
    x = pto.tensor([5], pto.DT_FP32)
    y = pto.sqrt(x)

    Input x:  [1.0 4.0 9.0 16.0 25.0]
    Output y: [1.0 2.0 3.0 4.0 5.0]
    """
    return pto_impl.sqrt(input)


@op_wrapper
def neg(a) -> Tensor:
    return pto_impl.neg(a)


@op_wrapper
def log(input: Tensor) -> Tensor:
    """Computes the element-wise log of `input`.

    This function calculates the formula: `out = log(input)`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise log.

    See Also
    -------
    sqrt : Element-wise square-root

    Examples
    --------
    x = pto.tensor([3], pto.DT_FP32)
    y = pto.log(x)

    Input x:[1 2 3]
    Output y:[0.0000 0.6931 1.0986]
    """

    return pto_impl.log(input, pto_impl.LogBaseType.LOG_E)


@op_wrapper
def clip(
    input: Tensor,
    min_: Optional[Union[Tensor, Element, float, int]] = None,
    max_: Optional[Union[Tensor, Element, float, int]] = None
):
    """
    Make the values in `input` greater than `min_` and less than `max_`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    min_ : Tensor or Element
        The minimum value.
    max_: Tensor or Element
        The maximum value

    Returns
    -------
    Tensor
        A new tensor containing the element-wise minimum.

    Examples
    --------
    a = pto.tensor([3], pto.DT_INT32)
    b = pto.tensor([3], pto.DT_INT32)
    out = pto.minimum(a, b)

    Input a:    [0 2 4]
    Input b:    [3 1 3]
    Output out: [0 1 3]
    """
    if min_ is None and max_ is None:
        return input

    element_types = (pto_impl.Element, int, float)
    is_element_mode = isinstance(min_, element_types) or isinstance(max_, element_types)
    default = (
        pto_impl.Tensor()
        if not is_element_mode
        else pto_impl.Element(pto_impl.DataType.DT_BOTTOM, 0)
    )
    min_ = min_ or default
    max_ = max_ or default
    return pto_impl.clip(input, min_, max_)
