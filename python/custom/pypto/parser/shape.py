#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
from typing import Sequence
from .util import get_var_str
from ..utils import CodeHelper, Instruction, Tensor, Var, Shape


def get_shape(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Shape):
        raise Exception()
    h(f'auto shape{inst.dst.idx} = tsr{inst.src[0].idx}->shape;')


def get_shape_dim(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Shape):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()
    dim_str = get_var_str(inst.src[1])
    h(f'[[maybe_unused]] auto v{inst.dst.idx} = shape{inst.src[0].idx}[{dim_str}];')


def get_shape_size(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Shape):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()
    h(f'auto v{inst.dst.idx} = (int) shape{inst.src[0].idx}.size();')


def set_shape(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Shape):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int)):
        raise Exception()
    if not isinstance(inst.src[2], (Var, int)):
        raise Exception()
    var_str1 = get_var_str(inst.src[1])
    var_str2 = get_var_str(inst.src[2])
    h(f'shape{inst.src[0].idx}[{var_str1}] = {var_str2};')


def new_shape(h: CodeHelper, inst: Instruction):
    def new_shape_sub(h: CodeHelper, inst: Instruction, var_str0: str):
        if not isinstance(inst.src[1], (Var, int, Sequence)):
            raise Exception()
        if isinstance(inst.src[1], Sequence):
            dims = []
            for dim in inst.src[1]:
                if not isinstance(dim, (Var, int)):
                    raise Exception()
                dims.append(get_var_str(dim))
            var_str = "{" + ", ".join(dims) + "}"
            h(f"std::vector<int> shape{inst.dst.idx} = {var_str};")
        else:
            var_str1 = get_var_str(inst.src[1])
            h(f'std::vector<int> shape{inst.dst.idx}({var_str0}, {var_str1});')

    if not isinstance(inst.src[0], (Var, int)):
        raise Exception()
    if not isinstance(inst.dst, Shape):
        raise Exception()

    var_str0 = get_var_str(inst.src[0])

    if len(inst.src) == 1:
        h(f'std::vector<int> shape{inst.dst.idx}({var_str0});')
    else:
        new_shape_sub(h, inst, var_str0)


def shape_emplace_back(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Shape):
        raise Exception()
    if not isinstance(inst.src[1], (int, Var)):
        raise Exception()

    s1 = get_var_str(inst.src[1])
    h(f'shape{inst.src[0].idx}.emplace_back({s1});')
