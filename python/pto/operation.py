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
import functools
from typing import Optional, Union, Tuple, List, overload

from . import pto_impl

from .element import Element
from .pto_utils import *  # noqa
from .symbolic_scalar import SymbolicScalar
from .tensor import Tensor
from .enum import * # noqa


def _to_base(arg):
    if isinstance(arg, (Tensor, Element, SymbolicScalar)):
        return arg.base()
    elif isinstance(arg, (list, tuple)):
        return [_to_base(a) for a in arg]
    elif isinstance(arg, dict):
        return {k: _to_base(v) for k, v in arg.items()}
    else:
        return arg


def _from_base(out):
    if isinstance(out, pto_impl.Tensor):
        return Tensor.from_base(out)
    elif isinstance(out, (list, tuple)):
        return [_from_base(a) for a in out]
    elif isinstance(out, dict):
        return {k: _from_base(v) for k, v in out.items()}
    else:
        return out


def op_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = _to_base(args)
        kwargs = _to_base(kwargs)
        assert isinstance(args, (list, tuple))
        set_source_location()
        out = func(*args, **kwargs)
        clear_source_location()
        if out is None:
            return None
        else:
            return _from_base(out)
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
    other: Union[Tensor, int, float]
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
def div(
    input: Tensor,
    other: Union[Tensor, int, float]
) -> Tensor:
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
def pow(
    input: Tensor,
    other: Union[int, float]
) -> Tensor:
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
    assert isinstance(other, (int, float)), "other must be a number"
    return pto_impl.pow(input, pto_impl.Element(input.dtype, other))


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
    x = pto.tensor([2, 2], pto.data_type.DT_FP32)
    out = pto.tensor([4, 4], pto.data_type.DT_FP32)
    offsets = [0, 0]
    pto.assemble(x, offsets, out)

    Input x:[[1 1],
            [1,1]]
          out:[[0 0 0 0],
               [0 0 0 0],
               [0 0 0 0],
               [0 0 0 0]]

    Output out:[[1 1 0 0]
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
    input: Tensor
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
    x = pto.tensor([3], pto.DT_FP32)
    y = pto.exp(x)

    Input x: [0 1 2]
    Output y:[1.0000 2.7183 7.3891]
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
    x = pto.tensor([2, 3], pto.DT_FP32)
    out = pto.transpose(x, 0, 1)

    Input x:    [[ 1.0028 -0.9893 0.5809],
                 [-0.1669 0.7299 0.4942]])
    Output out: [[ 1.0028 -0.1669],
                 [-0.9893 0.7299],
                 [ 0.5809 0.4942]])
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
    input: Tensor
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
    a = pto.tensor([5], pto.DT_INT32)
    out = pto.logical_not(a)

    Input a:    [0 1 2 3 4]
    Output out: [True False False False False False]

    """
    return pto_impl.logical_not(input)


@op_wrapper
def rsqrt(
    input: Tensor
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
    x = pto.tensor([2, 2], pto.DT_FP32)
    y = pto.rsqrt(x)

    Input x: [[1  4],
             [16 9]]
    Output y:[[1  0.5],
              [0.25 0.33333]]
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
def topk(
    input: Tensor,
    k: int,
    dim: Optional[int] = None,
    largest: bool = True
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
    x = pto.tensor([3, 5], pto.DT_FP32)        # shape (3, 5)

    index = pto.tensor([3, 4], pto.DT_INT32)   # shape (3, 4)
    dim = 0
    y = pto.gather(x, dim, index)

    Input x:  [[0 1 2 3 4],
               [5 6 7 8 9],
               [10 11 12 13 14]]
      index:  [[0 1 2 0],
               [1 2 0 1],
               [2 2 1 0]]

    Output y: [[0 6 12 3],
               [5 11 2 8],
               [10 11 7 3]]               # shape (3, 4)

    """

    return pto_impl.gather_element(input, index, dim)


@op_wrapper
def scatter_update(
    input: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor
) -> Tensor:
    """Write all values from the tensor 'src' into 'input' at the indices specified in the 'index' tensor.

    This function calculates the formula:
    For dim2,
    input[index[i][j]][:] = src[i][:]
    For dim4,
    input[index[i][j]][index[i][j]][0][:] = src[i][j][0][:]

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
    # dim2
    x = pto.tensor([8, 3], pto.DT_FP32)
    y = pto.tensor([2, 2], pto.DT_INT64)
    z = pto.tensor([4, 3], pto.DT_FP32)
    o = pto.scatter_update(x, -2, y, z)

    Input x:[[0 0 0],
             [0 0 0],
             [0 0 0],
             [0 0 0],
             [0 0 0],
             [0 0 0],
             [0 0 0],
             [0 0 0]]
    Input y:[[1 2],
             [4 5]]
    Input z:[[1 2 3],
             [4 5 6],
             [7 8 9],
             [10 11 12]]
    Output o:[[0 0 0],
              [1 2 3],
              [4 5 6],
              [0 0 0],
              [7 8 9],
              [10 11 12],
              [0 0 0],
              [0 0 0]])

    #dim4
    x = pto.tensor([2, 6, 1, 3], pto.DT_FP32)
    y = pto.tensor([2, 2], pto.DT_INT64)
    z = pto.tensor([2, 2, 1, 3], pto.DT_FP32)
    o = pto.scatter_update(x, -2, y, z)

    Input x:[[
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
             ],
             [
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
             ]]
    Input y:[[1 8],
             [4 10]]
    Input z:[[
                [[1 2 3]],
                [[4 5 6]],
             ],
             [
                [[7 8 9]],
                [[10 11 12]],
             ]]
    Output o:[[
                [[0 0 0]],
                [[1 2 3]],
                [[0 0 0]],
                [[0 0 0]],
                [[7 8 9]],
                [[0 0 0]],
             ],
             [
                [[0 0 0]],
                [[0 0 0]],
                [[4 5 6]],
                [[0 0 0]],
                [[10 11 12]],
                [[0 0 0]],
             ]]
    """
    if dim != -2:
        raise ValueError(
            "scatter currection only support the case where dim = -2.")
    dims = input.Dim()
    if dims == 4:
        chunk_size = input.GetShape()[1]
    elif dims == 2:
        chunk_size = 1
    else:
        raise ValueError("dim must be 2 or 4")

    return pto_impl.scatter_update(input, index, src, -2, "PA_BSND", chunk_size)


@op_wrapper
def scatter_(
    input: Tensor,
    dim: int,
    index: Tensor,
    src: float
) -> Tensor:
    """Write all values from the value 'src' into 'input' at the indices specified in the 'index' tensor.

    This function calculates the formula:
    For a 3-D tensor, 'input' is update as:
    self[index[i][j][k]][j][k] = src  # if dim == 0
    self[i][index[i][j][k]][k] = src  # if dim == 1
    self[i][j][index[i][j][k]] = src  # if dim == 2

    Parameters
    ----------
    input : Tensor
        The input tensor to be chenged.
    dim : int
        The axis along which to index.
    index : Tensor
        The indices of elements to scatter.
    src : float
        The source scalar value to scatter.

    Returns
    -------
    Tensor
        A new tensor containing the elements of input after scatter.

    Raises
    ------
    RuntimeError
        If the dimension of 'index' is not equal to the dimension of 'input'.
        If the shape size of any dimension of 'index' is bigger than the shape size of same dimension of 'input'.
        If the value of 'input[i][j][k]' is bigger than the shape size of the dimension of 'input'.

    See Also
    --------
    gather : The inverse operation, gather values along an axis specified by dim.

    Examples
    --------
    # dim2
    x = pto.tensor([3, 5], pto.DT_FP32)
    y = pto.tensor([2, 2], pto.DT_INT64)
    o = pto.scatter_(x, 0, y, 2.0)

    Input x:[[0 0 0 0 0],
             [0 0 0 0 0],
             [0 0 0 0 0]]
    Input y:[[1 2],
             [0 1]]
    Output o:[[2.0 0   0 0 0],
              [2.0 2.0 0 0 0],
              [0   2.0 0 0 0]]
    """

    return pto_impl.scatter_(input, index, pto_impl.Element(input.dtype, src), dim)


@op_wrapper
def scatter(
    input: Tensor,
    dim: int,
    index: Tensor,
    src: float
) -> Tensor:
    """Out-of-place version of 'scatter_'."""

    return pto_impl.scatter(input, index, pto_impl.Element(input.dtype, src), dim)


@op_wrapper
def where(
    condition: Tensor,
    input: Union[Tensor, float],
    other: Union[Tensor, float]
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


def convert_to_element(value) -> pto_impl.Element:
    if isinstance(value, (int)):
        if value >= -2**31 and value <= 2**31 - 1:
            return pto_impl.Element(pto_impl.DT_INT32, value)
        else:
            return pto_impl.Element(pto_impl.DT_INT64, value)
    else:
        return pto_impl.Element(pto_impl.DT_FP32, value)


@overload
def arange(end: Union[int, float]) -> Tensor:
    ...


@overload
def arange(start: Union[int, float], end: Union[int, float]) -> Tensor:
    ...


@overload
def arange(start: Union[int, float],
           end: Union[int, float], step: Union[int, float]) -> Tensor:
    ...


@op_wrapper
def arange(*args: Union[int, float]) -> Tensor:
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
    a = pto.arange(1.0, 4.0, 0.5)
    b = pto.arange(1.0, 4.0)
    c = pto.arange(4)

    Output a: [1.0 1.5 2.0 2.5 3.0 3.5]
    Output b: [1.0 2.0 3.0]
    Output c: [0 1 2 3]
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
    input: Tensor
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
    x = pto.tensor([3], pto.DT_FP32)
    y = pto.log(x)

    Input x:[1 2 3]
    Output y:[0.0000 0.6931 1.0986]
    """

    return pto_impl.log(input, pto_impl.LogBaseType.LOG_E)


@op_wrapper
def cast(
    input: Tensor,
    dtype: DataType,
    mode: CastMode = CastMode.CAST_NONE
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
    x = pto.tensor([2], pto.DT_FP32)
    y = pto.cast(x, pto.DT_FP16)

    Input  x: [2.0, 3.0] x.dtype: pto.DT_FP32
    Output y: [2.0, 3.0] y.dtype: pto.DT_FP16
    """
    if dtype == input.dtype and mode == CastMode.CAST_NONE:
        return input
    else:
        return pto_impl.cast(input, dtype, mode)


@op_wrapper
def amax(
    input: Tensor,
    dim: int = -1,
    keepdim: bool = False
) -> Tensor:
    """Returns the maximum value of each slice of the input tensor in the given dimension dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    keepdim : bool
        whether the output tensor has dim retained or not. Default: False.

    Returns
    -------
    Tensor
        If keepdim is True, the return tensor is of the same size as input except in the dimension dim where it
        is of size 1.
        Otherwise, dim is squeezed, resulting in the return tensor having 1 fewer dimension.

    Examples
    --------
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.amax(x, -1, True)

    Input x:[[1 2 3],
             [1 2 3]]
    Output y:[[3],
              [3]]

    """
    return pto_impl.amax(input, dim, keepdim)


@op_wrapper
def sum(
    input: Tensor,
    dim: int = -1,
    keepdim: bool = False
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
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.sum(x, -1, True)

    Input x:[[1 2 3],
             [1 2 3]]
    Output y:[[6],
              [6]]

    """
    return pto_impl.sum(input, dim, keepdim)


@op_wrapper
def full(size: List[int],
         fill_value: Union[int, float, SymbolicScalar, Element],
         dtype: DataType,
         *,
         valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None
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
    # Valid shapes use keyword argument
    x1 = 1.0
    y1 = pto.full([2,2], x1, pto.DT_FP32, valid_shape=[2, 2])

    x2 = pto.symbolic_scalar(1)
    y2 = pto.full([2,2], x2, pto.DT_INT32, valid_shape=[2, 2])

    #  In static graphs, validshape can be ignored
    x3 = 1
    y3 = pto.full([2,2], x3, pto.DT_INT32)

    Output y1: [[1.0 1.0], [1.0 1.0]]
    Output y2: [[1 1], [1 1]]
    Output y3: [[1 1], [1 1]]
    """

    if valid_shape is None:
        valid_shape = []
    if isinstance(fill_value, pto_impl.SymbolicScalar):
        return pto_impl.full(fill_value, dtype, size, to_syms(valid_shape))
    elif isinstance(fill_value, pto_impl.Element):
        return pto_impl.full(fill_value, dtype, size, to_syms(valid_shape))
    else:
        return pto_impl.full(pto_impl.Element(dtype, fill_value), dtype,
                                         size, to_syms(valid_shape))


@op_wrapper
def amin(
    input: Tensor,
    dim: int = -1,
    keepdim: bool = False
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
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.amin(x, -1, True)

    Input x:[[1 2 3],
             [1 2 3]]
    Output y:[[1],
              [1]]

    """
    return pto_impl.amin(input, dim, keepdim)


@op_wrapper
def greater(
    input: Tensor,
    other: Tensor
) -> Tensor:
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
def concat(
    tensors: List[Tensor],
    dim: int = 0
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
    x = pto.tensor([2, 2], pto.data_type.DT_FP32)  # 2x2 tensor with all 1s
    y = pto.tensor([2, 2], pto.data_type.DT_FP32)  # 2x2 tensor with all 0s
    dim = 0
    out = pto.concat([x, y], dim)

    Input  x : [[1 1],
                [1 1]]
           y : [[0 0],
                [0 0]]
    Output out:[[1 1],
                [1 1],
                [0 0],
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
        raise RuntimeError(
            "input dim and mat dim must equals, which only support 2-D/3-D/4-D currently")


@op_wrapper
def reshape(input: Tensor, shape: List[int], *,
            valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None, inplace: bool = False) -> Tensor:
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

    inplace : bool, optional
        An optional parameter determines memory sharing behavior between input and out tensors.
        If True, performs the reshape operation in-place, sharing the same storage with the input tensor.
        If False, creates a new tensor with the reshaped shape.

    Return
    ------
    pto.Tensor
        A new tensor with the specific shape.

    Examples
    ---------
    x = pto.tensor([2, 2], pto.DT_FP32)
    y = pto.reshape(x, [4, 1], [2, 1])
    z = pto.add(y, 1.0)

    input x: [[1, 2],
              [3, 4]]
    output y: [[1],
               [2],
               [3],
               [4]
           z: [[2],
               [3],
               [3],
               [4]]

    # inplace
    x = pto.tensor([2, 2], pto.DT_FP32)
    y = pto.reshape(x, [1, 4], inplace=True)

    input x: [[1, 2],
              [3, 4]]
    output y: [1, 2, 3, 4]
    """
    if inplace:
        out = pto_impl.reshape(input, to_syms(shape), inplace)
    else:
        if valid_shape is None:
            out = pto_impl.reshape(input, shape)
        else:
            out = pto_impl.reshape(input, shape, valid_shape)
    return out


@op_wrapper
def clone(input: Tensor) -> Tensor:
    """
    Clone the input Tensor into a new tensor with the same shape.

    Parameters
    ---------
    input: pto.Tensor
        The input tensor to be cloned.

    Return
    ------
    pto.Tensor
        A new tensor with the same shape.

    Examples
    ---------
    x = pto.tensor([2, 2], pto.DT_FP32)
    y = pto.clone(y)

    input x: [[1, 2],
              [3, 4]]
    output y: [[1, 2],
              [3, 4]]
    """
    return pto_impl.clone(input)

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
def view(input: Tensor, shape: List[int], offsets: List[Union[int, SymbolicScalar]],
         *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None) -> Tensor:
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
def maximum(input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]) -> Tensor:
    """
    Computes the element-wise maximum of input and other.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Element
        The second input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise maximum.

    Examples
    --------
    a = pto.tensor([3], pto.DT_INT32)
    b = pto.tensor([3], pto.DT_INT32)
    out = pto.maximum(a, b)

    Input a:    [0 2 4]
    Input b:    [3 1 3]
    Output out: [3 2 4]
    """
    if not isinstance(input, pto_impl.Tensor) and not isinstance(other, pto_impl.Tensor):
        raise TypeError("one of `input` and `other` should be `Tensor`")

    if not isinstance(input, pto_impl.Tensor) and isinstance(other, pto_impl.Tensor):
        input, other = other, input
    if isinstance(other, (int, float)):
        other = pto_impl.Element(input.dtype, other)
    return pto_impl.maximum(input, other)


@op_wrapper
def minimum(input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]) -> Tensor:
    """
    Computes the element-wise minimum of input and other.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Element
        The second input tensor.

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
    if not isinstance(input, pto_impl.Tensor) and not isinstance(other, pto_impl.Tensor):
        raise TypeError("one of `input` and `other` should be `Tensor`")

    if not isinstance(input, pto_impl.Tensor) and isinstance(other, pto_impl.Tensor):
        input, other = other, input
    if isinstance(other, (int, float)):
        other = pto_impl.Element(input.dtype, other)
    return pto_impl.minimum(input, other)


@op_wrapper
def expand_clone(input: Tensor, shape: List[int], *,
           valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None) -> Tensor:
    if valid_shape is None:
        valid_shape = []
    return pto_impl.expand(input, shape, valid_shape)


@op_wrapper
def logical_and(
    input: Tensor,
    other: Tensor
) -> Tensor:
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
def minimum(input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]) -> Tensor:
    """
    Computes the element-wise minimum of input and other.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Element
        The second input tensor.

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
    if not isinstance(input, pto_impl.Tensor) and not isinstance(other, pto_impl.Tensor):
        raise TypeError("one of `input` and `other` should be `Tensor`")

    if not isinstance(input, pto_impl.Tensor) and isinstance(other, pto_impl.Tensor):
        input, other = other, input
    if isinstance(other, (int, float)):
        other = pto_impl.Element(input.dtype, other)
    return pto_impl.minimum(input, other)


@op_wrapper
def expand_clone(input: Tensor, shape: List[int], *,
           valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None) -> Tensor:
    if valid_shape is None:
        valid_shape = []
    return pto_impl.expand(input, shape, valid_shape)


@op_wrapper
def logical_and(
    input: Tensor,
    other: Tensor
) -> Tensor:
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


@op_wrapper
def minimum(input: Tensor, other: Union[Tensor, Element]) -> Tensor:
    """
    Computes the element-wise minimum of input and other.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Element
        The second input tensor.

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
    return pto_impl.minimum(input, other)


@op_wrapper
def clip(
    input: Tensor,
    min_: Optional[Union[Tensor, Element, float, int]] = None,
    max_: Optional[Union[Tensor, Element, float, int]] = None,
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
    default = pto_impl.Tensor() if not is_element_mode else pto_impl.Element(pto_impl.DataType.DT_BOTTOM, 0)
    min_ = min_ or default
    max_ = max_ or default
    return pto_impl.clip(input, min_, max_)
