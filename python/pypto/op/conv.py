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
import struct
from typing import Any, Type

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..enum import DataType
from ..symbolic_scalar import SymbolicScalar
from ..tensor import Tensor


@op_wrapper
def conv(
    input,
    weight,
    out_dtype,
    bias,
    strides,
    paddings,
    dilations,
    groups = 1,
    transposed = False,
    output_paddings = [],
    extra_params = None
) -> Tensor:
    """
    input: 支持1d/2d/3d 的大小的tensor规格输入，新增format NCL，NCHW，NCDHW（tf 框架对标的format NLC，NHWC，NDHWC）
    weight: 支持1d/2d/3d 的卷积核的tensor规格输入，新增format NCL，NCHW，NCDHW（tf 框架对标的format LCN，HWCN，DHWCN）
    out_dytpe：normal 场景与输入一致，使能fixpipe随路量化、反量化、重量化能力时配置使用
    bais：可选tensor，bias输入，shape大小[cout]
    strides: strides参数，1d/2d/3d 分别在w，hw，dhw方向可进行配置
    paddings: paddings参数，1d/2d/3d 分别在w，hw，dhw方向可进行配置
    dilations: dilations参数，1d/2d/3d 分别在w，hw，dhw方向可进行配置
    groups：分组卷积参数
    transposed：反卷积flag
    output_paddings：transposed = true时被使用，反卷积输出的paddings配置参数
    extra_params: dict，额外参数配置
        -- scale：per_tensor场景下，dequant，quant，requant的scale参数值float
        -- scale_tensor：per_channel场景下，dequant，requant的scale tensor输入
        -- relu_type：fixpipe随路relu（normal relu, leakyrelu，prelu）
    """
    # __validate_inputs(input, weight, out_dtype, bias, strides, paddings, dilations, groups)
    return pypto_impl.Conv(
        out_dtype, input, weight, bias, strides, paddings, dilations, groups
    )


def __validate_type(value: Any, expect_type: Type, arg_name: str = "input") -> None:
    if value is None:
        return
    if not isinstance(value, expect_type):
        raise TypeError(
            f"Argument '{arg_name}' must be of type {expect_type.__name__}, but got {type(value).__name__}."
        )


def __get_valid_shape(tensor):
    return [SymbolicScalar.from_base(n) for n in tensor.GetValidShape()]


def __validate_shape(input: Tensor, weight: Tensor, bias: Tensor, transposed: bool) -> None:
    input_dim = input.Dim()
    weight_dim = weight.Dim()
    bias_dim = bias.Dim() if bias is not None else None
    if input_dim != weight_dim or input_dim not in {3, 4 ,5}:
        raise RuntimeError(
            "Tensor dimension mismatch. Expect input_dim == weight_dim and both in [3, 4, 5], "
            f"got input_dim: {input_dim}, weight_dim: {weight_dim}."
        )
    


def __validate_inputs(input, weight, out_dtype, bias, strides, paddings, dilations, groups) -> None:
    __validate_type(out_dtype, DataType, "out_dtype")
    __validate_type(strides, bool, "strides")
    __validate_type(paddings, bool, "paddings")
    __validate_type(dilations, bool, "dilations")
    __validate_shape(input, weight, bias, False)

    if is_out_nz:
        raise ValueError("Output tensor do not support NZ currently.")
    input1_valid = input_tensor1.GetDataType() == pypto_impl.DataType.DT_FP32 \
        and input_tensor1.Format() == pypto_impl.TileOpFormat.TILEOP_NZ
    input2_valid = input_tensor2.GetDataType() == pypto_impl.DataType.DT_FP32 \
        and input_tensor2.Format() == pypto_impl.TileOpFormat.TILEOP_NZ
    if input1_valid or input2_valid:
        raise ValueError("Input tensor with DT_FP32 must use ND format, NZ format is not support currently.")
    if input_tensor1.GetDataType() != input_tensor2.GetDataType():
        raise ValueError("All input tensors must have the same data type")
    if input_tensor1.Dim() != 2 and extend_params is not None:
        raise RuntimeError(
            "extend_params is not supported for batched matrix multiplication."
        )