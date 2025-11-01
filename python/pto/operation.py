#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
# pyright: reportReturnType=false
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
"""
"""
import typing
from typing import Optional, Union, Tuple, List

import pto
from pto import pto_impl

from .element import Element
from .pto_utils import to_syms
from .symbolic_scalar import SymbolicScalar
from .tensor import Tensor
from .enum import DataType, OpType, OutType


def _to_base(arg):
    if isinstance(arg, (Tensor, Element, SymbolicScalar)):
        return arg.base()
    elif isinstance(arg, (list, tuple)):
        return [_to_base(a) for a in arg]
    elif isinstance(arg, dict):
        return {k: _to_base(v) for k, v in arg.items()}
    else:
        return arg


def op_wrapper(func):
    def wrapper(*args, **kwargs):
        args = _to_base(args)
        assert isinstance(args, (list, tuple))
        out = func(*args, **kwargs)
        if out is None:
            return None
        elif isinstance(out, pto_impl.Tensor):
            return Tensor.from_base(out)
        else:
            return out
    return wrapper


@op_wrapper
def add(
    input: Tensor,
    other: Union[Tensor, int, float],
    *,
    alpha: Union[int, float] = 1
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
    >>> a = pto.tensor([1, 2, 3])
    >>> b = pto.tensor([4, 5, 6])
    >>> pto.add(a, b)
    tensor([5, 7, 9])

    >>> # Using a scalar and alpha
    >>> pto.add(a, 5, alpha=2) # Computes a + 2 * 5
    tensor([11, 12, 13])
    """
    if isinstance(other, pto_impl.Tensor):
        if alpha == 1 or alpha == 1.0:
            return pto_impl.add(input, other)
        else:
            return pto_impl.add(input, pto_impl.mul_s(other, pto_impl.Element(input.dtype, alpha)))
    else:
        if alpha == 1 or alpha == 1.0:
            return pto_impl.add_s(input, pto_impl.Element(input.dtype, other))
        else:
            assert isinstance(other, (int, float)), "alpha must be a number"
            return pto_impl.add_s(input, pto_impl.Element(input.dtype, other * alpha))


@op_wrapper
def sub(
    input: Tensor,
    other: Union[Tensor, int, float],
    *,
    alpha: Union[int, float] = 1
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
    >>> a = pto.tensor([4, 5, 6])
    >>> b = pto.tensor([1, 2, 3])
    >>> pto.sub(a, b)
    tensor([3, 3, 3])

    >>> # Using a scalar and alpha
    >>> pto.sub(a, 2, alpha=2) # Computes a + 2 * 5
    tensor([0, 1, 2])
    """
    if isinstance(other, pto_impl.Tensor):
        if alpha == 1 or alpha == 1.0:
            return pto_impl.sub(input, other)
        else:
            return pto_impl.sub(input, pto_impl.mul_s(other, pto_impl.Element(input.dtype, alpha)))
    else:
        if alpha == 1 or alpha == 1.0:
            return pto_impl.sub(input, pto_impl.Element(input.dtype, other))
        else:
            assert isinstance(other, (int, float)), "alpha must be a number"
            return pto_impl.sub(input, pto_impl.Element(input.dtype, other * alpha))


@op_wrapper
def mul(
    input: Tensor,
    other: Union[Tensor, int, float],
    *,
    alpha: Union[int, float] = 1
) -> Tensor:
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
    >>> a = pto.tensor([1, 2, 3])
    >>> b = pto.tensor([4, 5, 6])
    >>> pto.mul(a, b)
    tensor([4, 10, 18])

    >>> # Using a scalar
    >>> pto.mul(a, 5) # Computes a * 5
    tensor([5, 10, 15])
    """
    if isinstance(other, pto_impl.Tensor):
        if alpha == 1 or alpha == 1.0:
            return pto_impl.mul(input, other)
        else:
            return pto_impl.mul(input, pto_impl.mul_s(other, pto_impl.Element(input.dtype, alpha)))
    else:
        if alpha == 1 or alpha == 1.0:
            return pto_impl.mul_s(input, pto_impl.Element(input.dtype, other))
        else:
            assert isinstance(other, (int, float)), "alpha must be a number"
            return pto_impl.mul_s(input, pto_impl.Element(input.dtype, other * alpha))


@op_wrapper
def div(
    input: Tensor,
    other: Union[Tensor, int, float],
    *,
    alpha: Union[int, float] = 1
) -> Tensor:
    """Computes the element-wise division of `input` and `other`.

    This function calculates the formula: `out = input / alpha * other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to divide.
    alpha : float, optional, keyword-only
        A scaling factor for the `other` input. Default is 1.0.

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
    >>> a = pto.tensor([2, 4, 6])
    >>> b = pto.tensor([2, 2, 2])
    >>> pto.div(a, b)
    tensor([1, 2, 3])

    >>> # Using a scalar and alpha
    >>> pto.div(a, 2, alpha=3) # Computes a / 2 * 3
    tensor([3, 6, 9])
    """
    if isinstance(other, pto_impl.Tensor):
        if alpha == 1 or alpha == 1.0:
            return pto_impl.div(input, other)
        else:
            return pto_impl.div(input, pto_impl.mul_s(other, pto_impl.Element(input.dtype, alpha)))
    else:
        if alpha == 1 or alpha == 1.0:
            return pto_impl.div_s(input, pto_impl.Element(input.dtype, other))
        else:
            assert isinstance(other, (int, float)), "alpha must be a number"
            return pto_impl.div_s(input, pto_impl.Element(input.dtype, other * alpha))


@op_wrapper
def assemble(input: Tensor, offsets: List[Union[int, SymbolicScalar]], out: Tensor) -> None:
    """
    Assembles a small Tensor into a larger Tensor based on specified offsets.

    Parameters
    ---------
    input: Tensor
        The small input tensor to be assembled into the larger tensor
        
    offsets : List[int] or List[SymbolicScalar]
        List of offset values indicating where the input tensor should be placed in the output tensor. 
        It is required that the offsets is smaller than the shape of out.
        
    out: Tensor
        The larger output tensor that will contain the assembled input tensor
    Examples
    ---------
    >>> x = pto.tensor([2, 2], pto.data_type.DT_FP32)  # 2x2 tensor with all 1s
    >>> out = pto.tensor([4, 4], pto.data_type.DT_FP32)  # 4x4 tensor with all 0s
    >>> pto.assemble(x, [0, 0], out) 
    >>> print(out)
    [[1 1 0 0]
    [1 1 0 0]
    [0 0 0 0]
    [0 0 0 0]]
    """
    pto_impl.assemble(input, to_syms(offsets), out)


def min(a: 'SymbolicScalar | int', b: 'SymbolicScalar | int') -> 'SymbolicScalar':
    if isinstance(a, int):
        a = SymbolicScalar(a)
    return a.min(b)


def max(a: 'SymbolicScalar | int', b: 'SymbolicScalar | int') -> 'SymbolicScalar':
    if isinstance(a, int):
        a = SymbolicScalar(a)
    return a.max(b)


@op_wrapper
def exp(
    input: Tensor,
) -> Tensor:
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
    >>> a = pto.tensor([0, 1, 2])
    >>> pto.exp(a, b)
    tensor([1.0000, 2.7183, 7.3891])
    """
    return pto_impl.exp(input)


@op_wrapper
def transpose(
    input: Tensor,
    dim0: int,
    dim1: int
) -> Tensor:
    """Returns a tensor that is a transposed version of `input`. The given dimensions `dim0` and `dim1` are swapped.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim0 : int
        The first dimension to be transposed.
    dim1 : int
        The second dimension to be transposed.

    Returns
    -------
    Tensor
        A new tensor that is a transposed version of `input`.

    Raises
    ------
    RuntimeError
        If dim0 or dim1 is greater than or equal to the input dimension.

    Examples
    --------
    >>> x = pto.tensor(2, 3)
    >>> x
    tensor([[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]])
    >>> pto.transpose(x, 0, 1)
    tensor([[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [ 0.5809,  0.4942]])
    """
    return pto_impl.transpose(input, {dim0, dim1})


@op_wrapper
def abs(a) -> Tensor:
    return pto_impl.abs(a)


@op_wrapper
def reciprocal(a) -> Tensor:
    return pto_impl.reciprocal(a)


@op_wrapper
def logical_not(
    input: Tensor,
) -> Tensor:
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
    >>> input = pto.tensor([0, 1, 2, 3, 4])
    >>> pto.logical_not(input)
    tensor([True, False, False, False, False, False,])

    """
    return pto_impl.logical_not(input)


@op_wrapper
def rsqrt(
    input: Tensor,
) -> Tensor:
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
    >>> a = pto.tensor([1, 4, 16])
    >>> pto.rsqrt(a)
    tensor([1, 0.5, 0.25])
    """
    return pto_impl.rsqrt(input)


@op_wrapper
def sqrt(
    input: Tensor
) -> Tensor:
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
    >>> a = pto.tensor([1, 4, 9])
    >>> pto.sqrt(a)
    tensor([1, 2, 3])
    """
    return pto_impl.sqrt(input)


@op_wrapper
def neg(a) -> Tensor:
    return pto_impl.neg(a)


@op_wrapper
def topk(
    input: Tensor,
    k: int,
    dim: Optional[int] = None,
    largest: Optional[bool] = True
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
    largest : bool, optional
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
    >>> in = pto.tensor([[4, 5, 6],
                          [1, 2, 3]] )
    >>> out = pto.topk(in, 2, -1, True)
    >>> out[0]
    tensor([[6, 5],
            [3, 2]])
    >>> out1
    tensor([[2, 1],
            [2, 1]])
    """

    return pto_impl.topk(input, k, (-1 if dim is None else dim), largest)


@op_wrapper
def gather(
    input: Tensor,
    dim: int,
    index: Tensor
) -> Tensor:
    """
    Gather elements from `input` along `dim` according to `index`.

    This function specified output for a 3-D tensor:
    output[i][j][k] = input[index[i][j][k]][j][k] # if dim == 0
    output[i][j][k] = input[i][index[i][j][k]][k] # if dim == 1
    output[i][j][k] = input[i][j][index[i][j][k]] # if dim == 2

    Parameters
    ----------
    input : Tensor
        Source tensor from which to gather values.
    index : Tensor
        Integer tensor containing the subscripts to pick along `dim`. It must have
        the same numbers of dimensions as `input`. And it is also required that
        index.shape[d] <= input.shape[d] for all dimensions d != dim.
    dim : int
        Dimension in `input` along which to gather. Negative indexing is supported.

    Returns
    -------
    Tensor
        A new tensor, with the same dtype as `input` and the same shape as `index`.

    Raises
    ------
    IndexError
        If any value in `index` is outside the inclusive range
        [0, input.shape[dim]-1].
    RuntimeError
        If the broadcast shape of `index` against `input` is incompatible.

    Examples
    --------
    >>> a = pto.tensor([[0, 1, 2, 3, 4],
    ...                   [5, 6, 7, 8, 9],
    ...                   [10, 11, 12, 13, 14]])        # shape (3, 5)

    >>> index = pto.tensor([[0, 1, 2, 0],
    ...                       [1, 2, 0, 1],
    ...                       [2, 2, 1, 0])             # shape (3, 4)
    >>> pto.gather(a, 0, index)
    tensor([[0, 6, 12, 3],
            [5, 11, 2, 8],
            [10, 11, 7, 3]])                    # shape (3, 4)

    """

    return pto_impl.gather_element(input, index, dim)


@op_wrapper
def scatter(
    input: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor
) -> Tensor:
    """Write all values from the tensor 'src' into 'input' at the indices specified in the 'index' tensor.

    This function calculates the formula:
    For dim2,
    input[indexp[i][j]][:] = src[i][:]
    For dim4,
    input[indexp[i][j]][indexp[i][j]][0][:] = src[i][j][0][:]

    Parameters
    ----------
    input : Tensor
        The input tensor to be chenged.
    dim : int
        The axis along which to index.
    index : Tensor
        The indices of elements to scatter.
    src : Tensor
        The source elements to scatter.

    Returns
    -------
    Tensor
        A new tensor containing the elements of input after scatter.

    Raises
    ------
    RuntimeError
        If the dimension of 'index' is not equal 2.
        If the dimension of 'input' and 'src' is not equal 2 or 4.
        If the value of 'index' is not less than the blockSize of 'input'.

    See Also
    --------
    gather : The inverse operation, gather values along an axis specified by dim.

    Examples
    --------
    >>> a = pto.tensor([[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0]])
    >>> b = pto.tensor([[1, 2],[4, 5]])
    >>> c = pto.tensor([[1, 2, 3],[4, 5, 6],[7, 8, 9],[10, 11, 12]])
    >>> pto.scatter(a, -2, b, c)
    tensor([[0, 0, 0],[1, 2, 3],[4, 5, 6],[0, 0, 0],[7, 8, 9],[10, 11, 12],[0, 0, 0],[0, 0, 0]])

    >>> a = pto.tensor([[[[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0]]],
        [[[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0]]]])
    >>> b = pto.tensor([[1, 2],[0, 2]])
    >>> c = pto.tensor([[[[1, 2, 3],[4, 5, 6]]],[[[7, 8, 9],[10, 11, 12]]]])
    >>> pto.scatter(a, -2, b, c)
    tensor([[[[0, 0, 0],[1, 2, 3],[4, 5, 6]]],[[[7, 8, 9],[0, 0, 0],[10, 11, 12]]]])
    """
    dims = len(input.Dim())
    if dims == 4:
        chunk_size = input.GetShapeAt(1)
    elif dims == 2:
        chunk_size = 1
    else:
        raise ValueError("dim must be 2 or 4")

    return pto_impl.scatter_update(input, index, src, -2, "PA_BSND", chunk_size)


@op_wrapper
def where(
    condition: Tensor,
    input: Union[Tensor, float],
    other: Union[Tensor, float],
    *,
    dtype: DataType = None
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
    >>> cond = pto.tensor([True, False, True, False])
    >>> x = pto.tensor([1, 2, 3, 4])
    >>> y = pto.tensor([10, 20, 30, 40])
    >>> pto.where(cond, x, y)
    tensor([ 1, 20,  3, 40])

    >>> # Using scalar inputs
    >>> pto.where(cond, 1, 0)
    tensor([1, 0, 1, 0])

    >>> # Broadcasting example
    >>> cond = pto.tensor([[True, False], [False, True]])
    >>> a = pto.tensor([1, 2])  # Will be broadcasted
    >>> b = 0
    >>> pto.where(cond, a, b)
    tensor([[1, 0],
            [0, 2]])
    """
    if isinstance(input, pto_impl.Tensor):
        input_base = input
    elif dtype is None:
        input_base = pto_impl.Element(pto_impl.DT_FP32, input)
    else:
        input_base = pto_impl.Element(dtype, input)
    if isinstance(other, pto_impl.Tensor):
        other_base = other
    elif dtype is None:
        other_base = pto_impl.Element(pto_impl.DT_FP32, other)
    else:
        other_base = pto_impl.Element(dtype, other)
    return pto_impl.where(condition, input_base, other_base)


def convert_to_element(value) -> pto_impl.Element:
    if isinstance(value, (int)):
        if value >= -2**31 and value <= 2**31 - 1:
            return pto_impl.Element(pto_impl.DT_INT32, value)
        else:
            return pto_impl.Element(pto_impl.DT_INT64, value)
    else:
        return pto_impl.Element(pto_impl.DT_FP32, value)


@op_wrapper
def arange(
    *args: Union[int, float]
) -> Tensor:
    """Creates a 1-dimensional tensor containing a sequence of values in the range [start, end) with a given step.

    This function generates values from 'start' to 'end' (exclusive) in increments of 'step'.
    If only one argument is provided, it is treated as 'end', and 'start' defaults to 0 (INT32) and 'step'
    defaults to 1 (INT32).
    If two arguments are provided, they are treated as 'start' and 'end', and 'step' defailts to 1 (INT32).

    Parameters
    ----------
    start : Number
        The starting value of the sequence (inclusive), default is 0.
    end : Number
        The ending value of the sequence (exclusive).
    step : Number
        The step size between consecutive values, default is 1.

    Returns
    -------
    Tensor
        A 1-dimensional tensor containing the sequence of values.

    Raises
    ------
    ValueError
        If 'step' is zero or directionally incorrect (e.g., step > 0 when start > end).

    Examples
    --------
    >>> pto.arange(1.0, 4.0, 0.5)
    tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    >>> pto.arange(1.0, 4.0)
    tensor([1.0, 2.0, 3.0])
    >>> pto.arange(4)
    tensor([0, 1, 2, 3])

    """
    if len(args) == 1:
        end = args[0]
        return pto_impl.range(pto_impl.Element(pto_impl.DataType.DT_INT32, 0),
                              convert_to_element(end),
                              pto_impl.Element(pto_impl.DataType.DT_INT32, 1))
    elif len(args) == 2:
        start, end = args
        return pto_impl.range(convert_to_element(start),
                              convert_to_element(end),
                              pto_impl.Element(pto_impl.DataType.DT_INT32, 1))
    elif len(args) == 3:
        start, end, step = args
        return pto_impl.range(convert_to_element(start),
                              convert_to_element(end),
                              convert_to_element(step))

@op_wrapper
def log(
    input: Tensor,
) -> Tensor:
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
    >>> a = pto.tensor([1, 2, 3])
    >>> pto.log(a)
    tensor([0.0000, 0.6931, 1.0986])
    """

    return pto_impl.log(input, pto_impl.LogBaseType.LOG_e)


@op_wrapper
def cast(
    input: Tensor,
    dtype: DataType
) -> Tensor:
    """Casting the operand to the specified type.

    Parameters
    ----------
    operand : Tensor
        The input tensor.
    dtype : DataType
        The desired type.

    Returns
    -------
    Tensor
        Return a tensor after type cast, if this is already of the correct type, no copy is performed and the
        original object is returned

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    """
    if dtype == input.dtype:
        return input
    else:
        return pto_impl.cast(input, dtype, pto_impl.CastMode.CAST_NONE)


@op_wrapper
def amax(
    input: Tensor,
    dim: Optional[int] = -1
) -> Tensor:
    """Returns the maximum value of each slice of the input tensor in the given dimension dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    keepdim : bool, optional
        whether the output tensor has dim retained or not. Default: False.

    Returns
    -------
    Tensor
        If keepdim is True, the return tensor is of the same size as input except in the dimension dim where it
        is of size 1.
        Otherwise, dim is squeezed, resulting in the return tensor having 1 fewer dimension.

    Examples
    --------
    >>> in = pto.tensor([[4, 5, 6],
                          [1, 2, 3]] )
    >>> pto.amax(in, -1, true)
    tensor([[6],
            [3]])

    """
    return pto_impl.row_max_single(input, dim)


@op_wrapper
def sum(
    input: Tensor,
    dim: Optional[int] = -1
) -> Tensor:
    """Returns the sum value of each slice of the input tensor in the given dimension dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    keepdim : bool, optional
        whether the output tensor has dim retained or not. Default: False.

    Returns
    -------
    Tensor
        If keepdim is True, the return tensor is of the same size as input except in the dimension dim where it
        is of size 1.
        Otherwise, dim is squeezed, resulting in the return tensor having 1 fewer dimension.

    Examples
    --------
    >>> a = pto.tensor([[4, 5, 6],
                          [1, 2, 3]] )
    >>> pto.sum(a, -1, true)
    tensor([[15],
            [6]])

    """
    return pto_impl.row_sum_single(input, dim)


@op_wrapper
def full(size: List[int],
         fill_value: Union[int, float, SymbolicScalar, Element],
         dtype: DataType,
         *,
         valid_shape: Optional[Union[List[int], List[SymbolicScalar]]] = None
         ) -> Tensor:
    """
    Creates a tensor of the specified shape whose every entry equals the scalar elem.

    Parameters
    ----------
    size : List[int] 
        target shape; must be non-negative integers
    fill_value : int | float | SymbolicScalar | pto.element
        scalar value to replicate
    dtype : pto.DataType 
        desired data type; only int/float are supported (DT_FP32, DT_INT32).  
        If elem is a SymbolicScalar, dtype must be int32.
    valid_shape : List[int] | List[SymbolicScalar]]
        runtime actual shape

    Returns
    -------
    Tensor of shape shape filled with elem.

    Examples
    --------
    >>> # Valid shapes use keyword argument
    >>> a = 1.0 # must be 1.0; implicit conversion is not support
    >>> pto.full([2,2], a, pto.data_type.DT_FP32, valid_shape=[pto.symbolic_scalar(2), pto.symbolic_scalar(2)])
    tensor([[1.0,1.0],
            [1.0,1.0]])

    >>> b = pto.symbolic_scalar(1)
    >>> pto.full([2,2], b, pto.data_type.DT_INT32, valid_shape=[pto.symbolic_scalar(2), pto.symbolic_scalar(2)])
    tensor([[1,1],
            [1,1]])

    >>> c = pto.element(1)
    >>> pto.full([2,2], c, pto.data_type.DT_INT32, valid_shape=[pto.symbolic_scalar(2), pto.symbolic_scalar(2)])
    tensor([[1,1],
            [1,1]])

    >>> #  In static graphs, validshape can be ignored
    >>> d = pto.element(1)
    >>> pto.full([2,2], d, pto.data_type.DT_INT32)
    tensor([[1,1],
            [1,1]])
    """

    if valid_shape is None:
        valid_shape = []
    if isinstance(fill_value, pto_impl.SymbolicScalar):
        return pto_impl.vector_duplicate(fill_value, dtype, size, valid_shape)
    elif isinstance(fill_value, pto_impl.Element):
        return pto_impl.vector_duplicate(fill_value, dtype, size, valid_shape)
    else:
        return pto_impl.vector_duplicate(pto_impl.Element(dtype, fill_value), dtype, size, valid_shape)


@op_wrapper
def amin(
    input: Tensor,
    dim: Optional[int] = -1
) -> Tensor:
    """Returns the minimum value of each slice of the input tensor in the given dimension dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    keepdim : bool, optional
        whether the output tensor has dim retained or not. Default: False.

    Returns
    -------
    Tensor
        If keepdim is True, the return tensor is of the same size as input except in the dimension dim
        where it is of size 1.
        Otherwise, dim is squeezed, resulting in the return tensor having 1 fewer dimension.

    Examples
    --------
    >>> in = pto.tensor([[4, 5, 6],
                          [1, 2, 3]] )
    >>> pto.amin(in, -1, true)
    tensor([[4],
            [1]])

    """
    return pto_impl.row_min_single(input, dim)


@op_wrapper
def greater(
    input: Tensor,
    other: Union[Tensor, int, float]
) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    This function supports both BIT-packed and BOOLEAN output modes.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or scalar for comparison.

    Returns
    -------
    Tensor
        A new tensor containing the comparison results.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is not a Tensor or Element.


    Examples
    --------
    >>> a = pto.tensor([1, 2, 3])
    >>> b = pto.tensor([2, 2, 2])
    
    >>> pto.greater(a, b)
    tensor([False, False, True])

    """
    if isinstance(other, pto_impl.Tensor):
        return pto_impl.compare(input, other, pto_impl.OpType.GT, OutType.BOOL)
    else:
        return pto_impl.compare(input, pto_impl.Element(input.dtype, other), pto_impl.OpType.GT, OutType.BOOL)



@op_wrapper
def concat(
    tensors: List[Tensor],
    dim: Optional[int] = 0
) -> Tensor:
    """
    Concatenate multiple tensors according to the specified dimension.

    Parameters
    ---------
    tensors: Tensors
        tensor to be spliced.

    dim : int
        specified dimensions.

    out: Tensor
        The concatenated tensor
    Examples
    ---------
    >>> x = pto.tensor([2, 2], pto.data_type.DT_FP32)  # 2x2 tensor with all 1s
    >>> y = pto.tensor([2, 2], pto.data_type.DT_FP32)  # 2x2 tensor with all 0s
    >>> dim = 0
    >>> out = pto.Concat({x, y}, dim)
    >>> print(out)
    [[1 1]
    [1 1]
    [0 0]
    [0 0]]
    """
    return pto_impl.concat(tensors, dim)


@op_wrapper
def matmul(input, mat2, out_dtype, *, a_trans=False, b_trans=False, c_matrix_nz=False) -> Tensor:
    """
    Supports two forms of matrix multiplication compution:
    (1) Performs a matrix multiplication of the matrices `input` and `mat2`
    (2) Performs a batch matrix-matrix multiplication of the matrices `input` and `mat2`

    `input` and `mat2` support 2-D or 3-D or 4-D tensors each containing the same number of matrices.
    If `input` is a (n x k) tensor, `mat2` is a (k x m) tensor, output will be a (m x n) tensor.
    If `input` is a (b x n x k) tensor, `mat2` is a (b x k x m) tensor, output will be a (b x m x n) tensor.

    NOTES:
    If `input` of `mat2` is 3-D or 4-D, this function support broadcast.
    For example, if `input` is a (1 x n x k)tensor and `mat2` is a (b x k x m) tensor, the batch dimensions are (1)
    and (b), and the matrix dimensions are (n x k) and (k x m). output will be a (b x m x n) tensor.

    Parameters
    --------
    input : Tensor
        the left matrix to be matrix multiplied.
    mat2 : Tensor
        the right matrix to be matrix multiplied.
    out_dtye : dtype
        the dtype of the output tensor.

    Keyword Arguments
    --------
    a_trans : bool
        whether to transpose the left matrix. Default is False.
    b_trans : bool
        whether to transpose the right matrix. Default is False.
    c_matrix_nz : bool
        whether output matrix is in NZ format. Default is False.

    Returns
    --------
    Tensor
        A new Tensor containing the matrix multiplied result.

    Raise
    --------
    RuntimeError
        If the dimensions of matrix `input` and `mat2` are not equal and greater than 4-D or less than 2-D.

    Examples
    --------
    >>> # matrix x matrix
    >>> a = pto.tensor((16, 32), pto.data_type.DT_BF16, "tensor_a")
    >>> b = pto.tensor((32, 64), pto.data_type.DT_BF16, "tensor_b")
    >>> pto.matmul(a, b, pto.data_type.DT_BF16)
    tensors([16, 64])

    >>> # batched matrix x batched matrix
    >>> a = pto.tensor((2, 16, 32), pto.data_type.DT_FP16, "tensor_a")
    >>> b = pto.tensor((2, 32, 16), pto.data_type.DT_FP16, "tensor_b")
    >>> pto.matmul(a, b, pto.data_type.DT_FP16)
    tensors([2, 16, 16])

    >>> # batched matrix x batched matrix with broadcasted
    >>> a = pto.tensor((1, 32, 64), pto.data_type.DT_FP32, "tensor_a")
    >>> b = pto.tensor((3, 64, 16), pto.data_type.DT_FP32, "tensor_b")
    >>> pto.matmul(a, b, pto.data_type.DT_DT_FP32)
    tensors([3, 32, 16])

    """
    input_dim = input.Dim()
    mat2_dim = mat2.Dim()
    if input_dim == mat2_dim == 2:
        return pto_impl.matmul(out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz)
    elif (input_dim == mat2_dim == 3) or (input_dim == mat2_dim == 4):
        return pto_impl.batch_matmul(out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz)
    else:
        raise RuntimeError("input dim and mat dim must equals, which only support 2-D/3-D/4-D currently")


@op_wrapper
def reshape(input: Tensor, shape: List[int], valid_shape: Union[List[int], List[SymbolicScalar]] = None) -> Tensor:
    """
    Reshape the input Tensor into a new tensor with the specific shape.

    Parameters
    ---------
    input: pto.Tensor
        The input tensor to be reshaped.
    
    shape : List[int]
        The new shape of the tensor. The total number of elements must match the input tensor.
        
    valid_shape : List[int], optional
        An optional parameter specifying the valid shape for partial reshapeing or padding.
        If provided, it may be used to define the effective part of the new shape.

    Return
    ------ 
    pto.Tensor
        A new tensor with the specific shape.

    Examples
    ---------
    >>> x = pto.tensor([2, 2], pto.data_type.DT_FP32)  # 2x2 tensor
    x = [[1,2], [3,4]]
    >>> y = pto.reshape(x, [4, 1]) 
    >>> print(y.shape)
    [4, 1]
    >>> print(y)
    y = [1, 2, 3, 4]
    """
    if valid_shape is None:
        out = pto_impl.reshape(input, shape)
    else:
        out = pto_impl.reshape(input, shape, valid_shape)
    return out


@op_wrapper
def unsqueeze(input: Tensor, dim: int) -> Tensor:
    """Add a new dimension of size 1 to a tensor at a specified position.
    
    This operation increases the tensor's dimensionality while preserving 
    the total number of elements (since the new dimension has size 1).

    Parameters
    ----------
    input : Tensor
        The input tensor to which a new dimension will be added.
        Supported data types are: DT_FP32, DT_FP16, DT_BF16.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).

    dim : int 
        The position(index) where the new dimension is inserted. 
        It must be within the range of [-input.dim - 1, input.dim]

    Returns
    -------
    Tensor
        A new tensor with the same data as the input tensor, but with an additional dimension 
        of size 1 inserted at the specified dim position.

    Examples
    --------
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.unsqueeze(x, 0)

    Input x:[[1, 2, 3],
             [4, 5, 6]]
    Output y:[[[1, 2, 3], 
               [4, 5, 6]]]

    """
    return pto_impl.unsqueeze(input, dim)


@op_wrapper
def view(input: Tensor, shape: List[int], offsets: Union[List[int], List[SymbolicScalar]],
         *, valid_shape: Union[List[int], List[SymbolicScalar]] = None) -> Tensor:
    """Extract a partial view from the input tensor for subsequent computations.
       WARNING: view has a very different behavior from torch.view, it is more like slice.

    Parameters
    ----------
    input: Tensor
        The input tensor to extract a partial view.
        The supported data types are: DT_FP32, DT_FP16, DT_BF16.Empty Tensors are not supported, 
        and the Shape Size must not exceed 2147483647 (i.e., INT32_MAX).
    shape: List[int]
        Get the shape of the view.
        The Shape Size must not exceed 2147483647 (i.e., INT32_MAX).
    offsets: List[int]
        Get the offset of each dimension relative to the input when obtaining the view.
        It is required that thr offsets are smaller than the shape of the input.
    valid_shape: List[int] = None
        Optional parameter to retrieve the effective data size of the schematic block.
        It is required that thr offsets are smaller than the shape of the input.
    
    Returns
    -------
    Tensor
        A partial view from the input tensor with the size of shape.

    Examples
    --------
    x = pto.tensor([4, 8], pto.DT_FP32)
    shape = [4, 4]
    offsets = [0, 4]
    y = pto.view(x, shape, offsets)

    Input x:[[1 1 2 2 3 3 4 4],
             [1 1 2 2 3 3 4 4],
             [1 1 2 2 3 3 4 4],
             [1 1 2 2 3 3 4 4]]
    Output y:[[3 3 4 4],
              [3 3 4 4],
              [3 3 4 4],
              [3 3 4 4]]

    # add valid_shape
    x = pto.tensor([4, 8], pto.DT_FP32)
    shape = [4, 4]
    offsets = [2, 4]
    valid_shape = [2, 4]   
    y = pto.view(x, shape, offsets, valid_shape=valid_shape) 

    Input x:[[1 1 2 2 3 3 4 4],
             [1 1 2 2 3 3 4 4],
             [1 1 2 2 5 5 6 6],
             [1 1 2 2 5 5 6 6]]
    Output y:[[5 5 6 6],
              [5 5 6 6],
              [0 0 0 0],
              [0 0 0 0]]  
    """
    if valid_shape is None:
        return pto_impl.view(input, shape, offsets)
    else:
        return pto_impl.view(input, shape, to_syms(valid_shape), to_syms(offsets))


@op_wrapper
def maximum(input: Tensor, other: Tensor) -> Tensor:
    """
    Computes the element-wise maximum of input and other.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor 
        The second input tensor.
    
    Returns
    -------
    Tensor
        A new tensor containing the element-wise exponential.

    Examples
    --------
    >>> a = pto.tensor([0, 2, 4])
    >>> a = pto.tensor([3, 1, 3])

    >>> pto.maximum(a, b)
    tensor([3, 2, 4])
    """
    return pto_impl.maximum(input, other)
