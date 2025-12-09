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

from . import pypto_impl

from .op_wrapper import op_wrapper


@op_wrapper
def load(a: pypto_impl.Tensor, offsets: pypto_impl.Tensor) -> pypto_impl.Tensor:
    return pypto_impl.Load(a, offsets)


@op_wrapper
def gather_in_ub(param: pypto_impl.Tensor, indices: pypto_impl.Tensor, axis: int):
    """gather_in_ub."""

    return pypto_impl.gather_in_ub(param, indices, axis)


@op_wrapper
def gather_in_l1(src: pypto_impl.Tensor, offsets: pypto_impl.Tensor, size: int, is_b_matrix: bool, is_trans: bool):
    """gather_in_l1."""

    return pypto_impl.gather_in_l1(src, offsets, size, is_b_matrix, is_trans)