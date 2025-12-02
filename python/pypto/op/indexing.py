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
"""PyPTO"""
from .. import pypto_impl

from ..op_wrapper import op_wrapper
from ..tensor import Tensor


@op_wrapper
def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
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

    return pypto_impl.GatherElements(input, index, dim)


@op_wrapper
def scatter_update(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
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
        raise ValueError("scatter currection only support the case where dim = -2.")
    dims = input.Dim()
    if dims == 4:
        chunk_size = input.GetShape()[1]
    elif dims == 2:
        chunk_size = 1
    else:
        raise ValueError("dim must be 2 or 4")

    return pypto_impl.ScatterUpdate(input, index, src, -2, "PA_BSND", chunk_size)


@op_wrapper
def scatter_(input: Tensor, dim: int, index: Tensor, src: float) -> Tensor:
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

    return pypto_impl.Scatter_(input, index, pypto_impl.Element(input.dtype, src), dim)


@op_wrapper
def scatter(input: Tensor, dim: int, index: Tensor, src: float) -> Tensor:
    """Out-of-place version of 'scatter_'."""

    return pypto_impl.Scatter(input, index, pypto_impl.Element(input.dtype, src), dim)
