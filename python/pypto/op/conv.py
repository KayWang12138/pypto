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
    groups
) -> Tensor:
    """
    接口说明 + 用例调用示例
    """
    # __validate_inputs(input, weight, out_dtype, bias, strides, paddings, dilations, groups)
    return pypto_impl.Conv(
        out_dtype, input, weight, bias, strides, paddings, dilations, groups
    )