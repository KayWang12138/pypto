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
""" """
import struct

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..tensor import Tensor


@op_wrapper
def matmul(
    input,
    mat2,
    out_dtype,
    *,
    a_trans=False,
    b_trans=False,
    c_matrix_nz=False,
    extend_params=None
) -> Tensor:
    """
    Supports two forms of matrix multiplication computation:
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
    extend_params: dict
        the extend features of matrix multiplication computation:
        (1) bias: adds a learnable bias to the output.
        keyword arguments: 'bias_tensor': Tensor
        (2) Dequantization: C = DEQF16(ReLu(A @ B))
        keyword arguments: 'scale_tensor': Tensor, 'scale': float, 'relu_type': ReLuType

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
    # matrix x matrix
    a = pto.tensor((16, 32), pto.DT_BF16, "tensor_a")
    b = pto.tensor((32, 64), pto.DT_BF16, "tensor_b")
    pto.matmul(a, b, pto.DT_BF16)


    # batched matrix multiplication
    a = pto.tensor((2, 16, 32), pto.DT_FP16, "tensor_a")
    b = pto.tensor((2, 32, 16), pto.DT_FP16, "tensor_b")
    pto.matmul(a, b, pto.DT_FP16)


    # batched matrix multiplication with broadcast
    a = pto.tensor((1, 32, 64), pto.DT_FP32, "tensor_a")
    b = pto.tensor((3, 64, 16), pto.DT_FP32, "tensor_b")
    pto.matmul(a, b, pto.DT_DT_FP32)

    # matrix multiplication with bias
    a = pto.tensor((16, 32), ptoDT_FP16, "tensor_a")
    b = pto.tensor((32, 64), pto.DT_FP16, "tensor_b")
    bias = pto.tensor((1, 64), pto.DT_FP16, "tensor_bias")
    extend_params = {'bias_tensor': bias}
    pto.matmul(a, b, pto.DT_BF16, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)

    # matrix multiplication with dequantization
    a = pto.tensor((16, 32), pto.DT_INT8, "tensor_a")
    b = pto.tensor((32, 64), pto.DT_INT8, "tensor_b")
    extend_params = {'scale': 0.2}
    pto.matmul(a, b, pto.DT_BF16, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)

    """
    input_dim = input.Dim()
    mat2_dim = mat2.Dim()
    if input_dim == mat2_dim == 2:
        if (extend_params is None) or (not extend_params):
            return pypto_impl.Matmul(
                out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz
            )
        else:
            extend_params = pypto_impl.MatmulExtendParam(
                **convert_matmul_extend_params(extend_params)
            )
            return pypto_impl.Matmul(
                out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz, extend_params
            )
    elif (input_dim == mat2_dim == 3) or (input_dim == mat2_dim == 4):
        return pypto_impl.BatchMatmul(
            out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz
        )
    else:
        raise RuntimeError(
            "input dim and mat dim must equals, which only support 2-D/3-D/4-D currently"
        )


def convert_matmul_extend_params(extend_params) -> dict:
    extend_params.setdefault('bias_tensor', pypto_impl.Tensor())
    extend_params.setdefault('scale_tensor', pypto_impl.Tensor())
    extend_params.setdefault('relu_type', pypto_impl.ReLuType.NoReLu)
    extend_params.setdefault('scale', 0.0)
    return extend_params
