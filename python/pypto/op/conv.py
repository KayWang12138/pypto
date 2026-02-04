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
    strides,
    paddings,
    dilations,
    *,
    groups = 1,
    transposed = False,
    output_paddings = [],
    extend_params = None
) -> Tensor:
    """
    input: 支持1d/2d/3d 的大小的tensor规格输入，新增format NCL，NCHW，NCDHW（tf 框架对标的format NLC，NHWC，NDHWC）
    weight: 支持1d/2d/3d 的卷积核的tensor规格输入，新增format NCL，NCHW，NCDHW（tf 框架对标的format LCN，HWCN，DHWCN）
    out_dytpe：normal 场景与输入一致，使能fixpipe随路量化、反量化、重量化能力时配置使用
    strides: strides参数，1d/2d/3d 分别在w，hw，dhw方向可进行配置
    paddings: paddings参数，1d/2d/3d 分别在w，hw，dhw方向可进行配置
    dilations: dilations参数，1d/2d/3d 分别在w，hw，dhw方向可进行配置
    groups：分组卷积参数
    transposed：反卷积flag
    output_paddings：transposed = true时被使用，反卷积输出的paddings配置参数
    extend_params: dict，额外参数配置
        -- bais：可选tensor，bias输入，shape大小[cout]
        -- scale：per_tensor场景下，dequant，quant，requant的scale参数值float
        -- scale_tensor：per_channel场景下，dequant，requant的scale tensor输入
        -- relu_type：fixpipe随路relu（normal relu, leakyrelu，prelu）
    """
    __validate_inputs(
        out_dtype, input, weight, strides, paddings, dilations, extend_params, groups, transposed, output_paddings
    )
    
    if extend_params is not None:
            extend_params = pypto_impl.ConvExtendParam(
                **__convert_conv_extend_params(extend_params)
            )

    if not transposed:
        return pypto_impl.Conv(
            out_dtype, input, weight, strides, paddings, dilations, extend_params, groups
        )
    else:
        raise RuntimeError(
            "Conv transpose true is not supported yet."
        )


def __validate_type(value: Any, expect_type: Type, arg_name: str = "input") -> None:
    if value is None:
        return
    if not isinstance(value, expect_type):
        raise TypeError(
            f"Argument '{arg_name}' must be of type {expect_type.__name__}, but got {type(value).__name__}."
        )


def __validate_shape(input: Tensor, weight: Tensor, transposed: bool) -> None:
    input_dim = input.Dim()
    weight_dim = weight.Dim()
    if input_dim != weight_dim or input_dim not in {3, 4 ,5}:
        raise RuntimeError(
            "Tensor dimension mismatch. Expect input_dim == weight_dim and both in [3, 4, 5], "
            f"got input_dim: {input_dim}, weight_dim: {weight_dim}."
        )


def __validate_inputs(out_dtype, input, weight, strides, paddings, dilations, extend_params, groups, transposed, output_paddings) -> None:
    __validate_type(input, pypto_impl.Tensor, "input")
    __validate_type(weight, pypto_impl.Tensor, "weight")
    __validate_type(out_dtype, DataType, "out_dtype")
    __validate_type(strides, list, "strides")
    __validate_type(paddings, list, "paddings")
    __validate_type(dilations, list, "dilations")
    __validate_type(groups, int, "groups")
    __validate_type(transposed, bool, "transposed")
    __validate_type(output_paddings, list, "output_paddings")
    __validate_type(extend_params, dict, "extend_params")
    __validate_shape(input, weight, False)

    if input.GetDataType() not in (pypto_impl.DataType.DT_BF16, pypto_impl.DataType.DT_FP16, pypto_impl.DataType.DT_FP32):
        raise ValueError(
            "Input tensor data type must in [bf16, fp16, fp32],"
            f"but Input tensor got {input.GetDataType()}"
        )
    
    if weight.GetDataType() not in (pypto_impl.DataType.DT_BF16, pypto_impl.DataType.DT_FP16, pypto_impl.DataType.DT_FP32):
        raise ValueError(
            "Weight tensor data type must in [bf16, fp16, fp32],"
            f"but Weight tensor got {input.GetDataType()}"
        )


def __convert_conv_extend_params(extend_params) -> dict:
    extend_params.setdefault('bias_tensor', pypto_impl.Tensor())
    extend_params.setdefault('scale_tensor', pypto_impl.Tensor())
    extend_params.setdefault('relu_type', pypto_impl.ConvReLuType.NO_RELU)
    extend_params.setdefault('scale', 0.0)
    return extend_params
