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
import pto
from pto import pto_impl
from .tensor import Tensor
from .operation import op_wrapper


@op_wrapper
def sin(input: Tensor) -> Tensor:
    """Return a tensor containing the element-wise sine values of input.

    Parameters
    ----------
    input: Tensor
        The input tensor to compute.
        The supported data type is DT_FP32.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).
    
    Returns
    -------
    Tensor
        A tensor with the same shape and data type as the input, whose elements 
        are the sine values of the corresponding elements in the input tensor.


    Examples
    --------
    x = pto.tensor([4], pto.DT_FP32)
    y = pto.sin(x)

    Input x:[-0.5461,  0.1347, -2.7266, -0.2746]
    Output y:[-0.5194,  0.1343, -0.4032, -0.2711]
    """
    @op_wrapper
    def _cast(_input, _dtype, _cast_mode = pto.CastMode.CAST_NONE):
        return pto_impl.cast(_input, _dtype, _cast_mode)

    dtype = input.dtype
    if dtype != pto.DT_FP32:
        input = _cast(input, pto.DT_FP32)

    number2048 = 2048.0
    one_over_n = 1.0 / 2048.0
    inv_half_pi = 0.63661975
    pi0 = 1.5708008
    pi1 = -0.0000044535846
    pi2 = -8.706138e-10
    f_025 = 0.25
    f_05 = 0.5
    f_4 = 4.0
    f_1 = 1.0
    f_nega_1 = -1.0
    f_nega_2 = -2.0

    x_scaled = pto.mul(input, one_over_n)
    x_over_pi = pto.mul(x_scaled, inv_half_pi)
    n = _cast(x_over_pi, pto.DT_FP32, pto.CastMode.CAST_ROUND)
    n0 = pto.mul(x_over_pi, one_over_n)
    n0 = _cast(n0, pto.DT_FP32, pto.CastMode.CAST_ROUND)
    n0 = pto.mul(n0, number2048)

    n1 = pto.sub(n, n0)

    fix = pto.mul(n0, pi0)
    x_fix = pto.sub(x_scaled, fix)
    fix = pto.mul(n1, pi0)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi1)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi1)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi2)
    x_fix = pto.sub(x_fix, fix)

    pi_02 = 1.5703125
    pi_12 = 0.0004837513

    remain_x = pto.mul(x_fix, number2048)
    temp = pto.mul(remain_x, inv_half_pi)
    n2 = _cast(temp, pto.DT_FP32, pto.CastMode.CAST_ROUND)

    n0 = pto.mul(n0, number2048)
    n1 = pto.mul(n1, number2048)
    fix = pto.mul(n0, pi_02)
    x_fix = pto.sub(input, fix)
    fix = pto.mul(n1, pi_02)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi_12)
    x_fix = pto.sub(x_fix, fix)

    pi_22 = 0.000000075495336
    fix = pto.mul(n2, pi_02)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi_12)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi_22)
    x_fix = pto.sub(x_fix, fix)

    pi_32 = 2.5579538e-12
    fix = pto.mul(n2, pi_12)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi_22)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi_32)
    x_fix = pto.sub(x_fix, fix)

    pi_42 = 5.389786e-15
    fix = pto.mul(n2, pi_22)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi_32)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi_42)
    x_fix = pto.sub(x_fix, fix)

    pi_52 = 5.166901e-19
    fix = pto.mul(n2, pi_32)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi_42)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi_52)
    x_fix = pto.sub(x_fix, fix)

    pi_62 = 3.281839e-22
    fix = pto.mul(n2, pi_42)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi_52)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi_62)
    x_fix = pto.sub(x_fix, fix)

    fix = pto.mul(n2, pi_52)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi_62)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n2, pi_62)
    x_fix = pto.sub(x_fix, fix)

    half_n2 = pto.mul(n2, f_05)
    half4_n2 = pto.mul(n2, f_025)
    n_half2 = _cast(half_n2, pto.DT_FP32, pto.CastMode.CAST_FLOOR)
    n_half4 = _cast(half4_n2, pto.DT_FP32, pto.CastMode.CAST_FLOOR)

    k1 = pto.mul(n_half2, f_nega_2)
    k2 = pto.mul(n_half4, f_4)
    sign = pto.add(k1, k2)
    sign = pto.add(sign, f_1)

    ifcos = pto.add(n2, k1)
    ifsin = pto.mul(ifcos, f_nega_1)
    ifsin = pto.add(ifsin, f_1)

    scoef4 = 0.0000027183114939898219064
    scoef3 = -0.000198393348360966317347
    scoef2 = 0.0083333293858894631756
    scoef1 = -0.166666666416265235595
    x_pow = pto.mul(x_fix, x_fix)
    sin_poly = pto.mul(x_pow, scoef4)
    sin_poly = pto.add(sin_poly, scoef3)
    sin_poly = pto.mul(x_pow, sin_poly)
    sin_poly = pto.add(sin_poly, scoef2)
    sin_poly = pto.mul(x_pow, sin_poly)
    sin_poly = pto.add(sin_poly, scoef1)
    sin_poly = pto.mul(x_pow, sin_poly)
    sin_poly = pto.add(sin_poly, f_1)
    sin_poly = pto.mul(x_fix, sin_poly)

    ccoef4 = 0.0000243904487962774090654
    ccoef3 = -0.00138867637746099294692
    ccoef2 = 0.0416666233237390631894
    ccoef1 = -0.499999997251031003120
    cos_poly = pto.mul(x_pow, ccoef4)
    cos_poly = pto.add(cos_poly, ccoef3)
    cos_poly = pto.mul(x_pow, cos_poly)
    cos_poly = pto.add(cos_poly, ccoef2)
    cos_poly = pto.mul(x_pow, cos_poly)
    cos_poly = pto.add(cos_poly, ccoef1)
    cos_poly = pto.mul(x_pow, cos_poly)
    cos_poly = pto.add(cos_poly, f_1)

    temp1 = pto.mul(sin_poly, ifsin)
    cos_poly = pto.mul(cos_poly, ifcos)
    res = pto.add(temp1, cos_poly)
    res = pto.mul(res, sign)
    if dtype != res.dtype: 
        res = _cast(res, dtype)

    return res    


@op_wrapper
def cos(input: Tensor) -> Tensor:
    """Return a tensor containing the element-wise cosine values of input.

    Parameters
    ----------
    input: Tensor
        The input tensor to compute.
        The supported data type is DT_FP32.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).
    
    Returns
    -------
    Tensor
        A tensor with the same shape and data type as the input, whose elements are 
        the cosine values of the corresponding elements in the input tensor.


    Examples
    --------
    x = pto.tensor([4], pto.DT_FP32)
    y = pto.cos(x)

    Input x:[0.0000, 0.7854, 1.5708, 2.3562]
    Output y:[1.0000, 0.7071, 0.0000, -0.7071]
    """
    @op_wrapper
    def _cast(_input, _dtype, _cast_mode = pto.CastMode.CAST_NONE):
        return pto_impl.cast(_input, _dtype, _cast_mode)
    
    dtype = input.dtype
    if dtype != pto.DT_FP32:
        input = _cast(input, pto.DT_FP32)

    number2048 = 2048.0
    one_over_n = 1.0 / 2048.0
    inv_half_pi = 0.63661975

    pi0 = 1.5708008
    pi1 = -0.0000044535846
    pi2 = -8.706138e-10
    f_025 = 0.25
    f_05 = 0.5
    f_4 = 4.0
    f_1 = 1.0
    f_nega_1 = -1.0
    f_nega_2 = -2.0

    x_scaled = pto.mul(input, one_over_n)
    x_over_pi = pto.mul(x_scaled, inv_half_pi)
    n = _cast(x_over_pi, pto.DT_FP32, pto.CastMode.CAST_ROUND)
    n0 = pto.mul(x_over_pi, one_over_n)
    n0 = _cast(n0, pto.DT_FP32, pto.CastMode.CAST_ROUND)
    n0 = pto.mul(n0, number2048)
    n1 = pto.sub(n, n0)

    fix = pto.mul(n0, pi0)
    x_fix = pto.sub(x_scaled, fix)
    fix = pto.mul(n1, pi0)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi1)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi1)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi2)
    x_fix = pto.sub(x_fix, fix)

    pi_02 = 1.5703125
    pi_12 = 0.0004837513

    remain_x = pto.mul(x_fix, number2048)
    temp = pto.mul(remain_x, inv_half_pi)
    n2 = _cast(temp, pto.DT_FP32, pto.CastMode.CAST_ROUND)
    n0 = pto.mul(n0, number2048)
    n1 = pto.mul(n1, number2048)
    fix = pto.mul(n0, pi_02)
    x_fix = pto.sub(input, fix)
    fix = pto.mul(n1, pi_02)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi_12)
    x_fix = pto.sub(x_fix, fix)

    pi_22 = 0.000000075495336
    fix = pto.mul(n2, pi_02)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi_12)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi_22)
    x_fix = pto.sub(x_fix, fix)

    pi_32 = 2.5579538e-12
    fix = pto.mul(n2, pi_12)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi_22)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi_32)
    x_fix = pto.sub(x_fix, fix)

    pi_42 = 5.389786e-15
    fix = pto.mul(n2, pi_22)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi_32)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi_42)
    x_fix = pto.sub(x_fix, fix)

    pi_52 = 5.166901e-19
    fix = pto.mul(n2, pi_32)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi_42)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi_52)
    x_fix = pto.sub(x_fix, fix)

    pi_62 = 3.281839e-22
    fix = pto.mul(n2, pi_42)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi_52)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n0, pi_62)
    x_fix = pto.sub(x_fix, fix)

    fix = pto.mul(n2, pi_52)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n1, pi_62)
    x_fix = pto.sub(x_fix, fix)
    fix = pto.mul(n2, pi_62)
    x_fix = pto.sub(x_fix, fix)

    n2 = pto.add(n2, f_1)
    half_n2 = pto.mul(n2, f_05)
    half4_n2 = pto.mul(n2, f_025)
    n_half2 = _cast(half_n2, pto.DT_FP32, pto.CastMode.CAST_FLOOR)
    n_half4 = _cast(half4_n2, pto.DT_FP32, pto.CastMode.CAST_FLOOR)

    k1 = pto.mul(n_half2, f_nega_2)
    k2 = pto.mul(n_half4, f_4)
    sign = pto.add(k1, k2)
    sign = pto.add(sign, f_1)

    ifcos = pto.add(n2, k1)
    ifsin = pto.mul(ifcos, f_nega_1)
    ifsin = pto.add(ifsin, f_1)

    scoef4 = 0.0000027183114939898219064
    scoef3 = -0.000198393348360966317347
    scoef2 = 0.0083333293858894631756
    scoef1 = -0.166666666416265235595
    x_pow = pto.mul(x_fix, x_fix)
    sin_poly = pto.mul(x_pow, scoef4)
    sin_poly = pto.add(sin_poly, scoef3)
    sin_poly = pto.mul(x_pow, sin_poly)
    sin_poly = pto.add(sin_poly, scoef2)
    sin_poly = pto.mul(x_pow, sin_poly)
    sin_poly = pto.add(sin_poly, scoef1)
    sin_poly = pto.mul(x_pow, sin_poly)
    sin_poly = pto.add(sin_poly, f_1)
    sin_poly = pto.mul(x_fix, sin_poly)

    ccoef4 = 0.0000243904487962774090654
    ccoef3 = -0.00138867637746099294692
    ccoef2 = 0.0416666233237390631894
    ccoef1 = -0.499999997251031003120
    cos_poly = pto.mul(x_pow, ccoef4)
    cos_poly = pto.add(cos_poly, ccoef3)
    cos_poly = pto.mul(x_pow, cos_poly)
    cos_poly = pto.add(cos_poly, ccoef2)
    cos_poly = pto.mul(x_pow, cos_poly)
    cos_poly = pto.add(cos_poly, ccoef1)
    cos_poly = pto.mul(x_pow, cos_poly)
    cos_poly = pto.add(cos_poly, f_1)

    temp1 = pto.mul(sin_poly, ifsin)
    cos_poly = pto.mul(cos_poly, ifcos)
    res = pto.add(temp1, cos_poly)
    res = pto.mul(res, sign)
    if res.dtype != dtype:
        res = _cast(res, dtype)

    return res


@op_wrapper
def sigmoid(input: Tensor) -> Tensor:
    """ Return a tensor containing the element-wise sigmoid values of input.
        The sigmoid function is a common activation function in machine learning, 
        defined mathematically as: sigmoid(x) = 1 / (1 + exp(-x))

    Parameters
    ----------
    input: Tensor
        The input tensor to compute.
        The supported data type is DT_FP32.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).
    
    Returns
    -------
    Tensor
        A tensor with the same shape and data type as the input, whose elements are 
        the results of the input elements mapped to the interval (0, 1) via the sigmoid function.

    Examples
    --------
    x = pto.tensor([4], pto.DT_FP32)
    y = pto.sigmoid(x) 

    Input x:[-3.0, 0.0, 2.0, 5.0]
    Output y:[0.0474, 0.5000, 0.8808, 0.9933]
    """
    @op_wrapper
    def _cast(_input, _dtype, _cast_mode = pto.CastMode.CAST_NONE):
        return pto_impl.cast(_input, _dtype, _cast_mode)

    @op_wrapper
    def _vector_duplicate(_elem, _dtype, _shape, _valid_shape):            
        if isinstance(_elem, pto_impl.SymbolicScalar):
            return pto_impl.vector_duplicate(_elem, _dtype, _shape, _valid_shape)
        else:
            return pto_impl.vector_duplicate(pto_impl.Element(_dtype, _elem), _dtype, _shape, _valid_shape)
        
    dtype = input.dtype
    if dtype != pto.DT_FP32:
        input = _cast(input, pto.DT_FP32)

    f_1 = 1.0
    f_nega_1 = -1.0
    
    exp_res = pto.exp(pto.mul(input, f_nega_1))
    res = pto.add(exp_res, f_1)
    src = 1.0
    ones = _vector_duplicate(src, pto.DT_FP32, res.shape, 
                            [pto.symbolic_scalar(res.shape[0]), pto.symbolic_scalar(res.shape[1])])
    res = pto.div(ones, res)
    if dtype != pto.DT_FP32:
        res = _cast(res, dtype)
    
    return res


@op_wrapper
def softmax(input: Tensor, dim: int) -> Tensor:
    """ Return a tensor obtained by applying the softmax activation function to the input.
        Mathematically, for an input tensor x along a specified dimension dim, 
        the softmax of element x_i is computed as:
        softmax(x_i) = exp(x_i) / sum(exp(x_j) for j in dimension dim

    Parameters
    ----------
    input: Tensor
        The input tensor to compute.
        The supported data type is DT_FP32.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).
    dim: int
        Specify the dimension for normalization.
        Negative indices are supported (e.g., -1 indicates the last dimension).
        It must be within the range of [-input.dim, input.dim - 1].
    
    Returns
    -------
    Tensor
        A tensor with the same shape as the input, where the sum of elements along the 
        specified dimension is 1, and the data type is determined by dtype or the input type.

    Examples
    --------    
    x = pto.tensor([2, 3], pto.DT_FP32)
    y = pto.softmax(x, -1)

    Input x:[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    Output y:[[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]]
    """    
    @op_wrapper
    def _cast(_input, _dtype, _cast_mode = pto.CastMode.CAST_NONE):
        return pto_impl.cast(_input, _dtype, _cast_mode)    

    dtype = input.dtype
    if dtype != pto.DT_FP32:
        input = _cast(input, pto.DT_FP32)

    rowmax = pto.amax(input, dim)
    sub_res = pto.sub(input, rowmax)
    exp_res = pto.exp(sub_res)
    esum = pto.sum(exp_res, dim)
    output = pto.div(exp_res, esum)

    return output
