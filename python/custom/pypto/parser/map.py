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
"""
"""
from .util import get_var_str, get_curlybrace_list
from ..utils import CodeHelper, Instruction, Tensor, Shape, TensorMap


def new_map(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.dst, TensorMap):
        raise Exception()
    h(f'std::map<std::vector<int>, Tensor> mp{inst.dst.idx};')


def map_set(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Shape, list)):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.dst, TensorMap):
        raise Exception()

    shape = inst.src[0]
    if isinstance(shape, list):
        shape_str = get_curlybrace_list(shape)
    else:
        shape_str = get_var_str(shape)

    h(f'mp{inst.dst.idx}[{shape_str}] = tsr{inst.src[1].idx};')


def map_ret(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Shape, list)):
        raise Exception()
    if not isinstance(inst.src[1], TensorMap):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    shape = inst.src[0]
    if isinstance(shape, list):
        shape_str = get_curlybrace_list(shape)
    else:
        shape_str = get_var_str(shape)

    h(f'auto tsr{inst.dst.idx} = mp{inst.src[1].idx}[{shape_str}];')
