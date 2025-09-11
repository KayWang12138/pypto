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

from .util import get_var_str, get_curlybrace_list
from ..utils import CodeHelper, Instruction, Tensor, Vector, Var, Shape


def vec_declare(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], list):
        raise Exception()
    if not isinstance(inst.dst, Vector):
        raise Exception()
    dtype_str = f'std::vector<{inst.src[0][-1]}>'
    for _ in range(-2, -len(inst.src[0]) - 1, -1):
        dtype_str = f'std::vector<{dtype_str}>'
    h(f'{dtype_str} vec{inst.dst.idx};')


def retrive_dtype_str(inst: Instruction, src1_cls: tuple):
    if not isinstance(inst.src[0], list):
        raise Exception()
    if not isinstance(inst.src[1], src1_cls):
        raise Exception()
    if not isinstance(inst.dst, Vector):
        raise Exception()

    dtype_str = f'std::vector<{inst.src[0][-1]}>'
    for _ in range(-2, -len(inst.src[0]) - 1, -1):
        dtype_str = f'std::vector<{dtype_str}>'

    return dtype_str


def vec_declare_len(h: CodeHelper, inst: Instruction):
    dtype_str = retrive_dtype_str(inst, (Var, int))

    s1 = get_var_str(inst.src[1])
    if inst.src[2] is None:
        h(f'{dtype_str} vec{inst.dst.idx}({s1});')
    else:
        if not isinstance(inst.src[2], (Var, int, Tensor)):
            raise Exception()
        s2 = get_var_str(inst.src[2])
        h(f'{dtype_str} vec{inst.dst.idx}({s1}, {s2});')


def vec_declare_list(h: CodeHelper, inst: Instruction):
    dtype_str = retrive_dtype_str(inst, (Shape, list))

    shape_1 = inst.src[1]
    if isinstance(shape_1, list):
        shape_1_str = get_curlybrace_list(shape_1)
    else:
        shape_1_str = get_var_str(shape_1)
    h(f'{dtype_str} vec{inst.dst.idx} = {shape_1_str};')


def make_mod_vec(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], list):
        raise Exception()
    if not isinstance(inst.dst, Vector):
        raise Exception()
    dtype_str = f'std::vector<{inst.src[0][-1]}>'
    for _ in range(-2, -len(inst.src[0]) - 1, -1):
        dtype_str = f'std::vector<{dtype_str}>'
    h(f'vec{inst.dst.idx} = {dtype_str}();')


def make_mod_vec_len(h: CodeHelper, inst: Instruction):
    dtype_str = retrive_dtype_str(inst, (Var, int))

    s1 = get_var_str(inst.src[1])
    if inst.src[2] is None:
        h(f'vec{inst.dst.idx} = {dtype_str}({s1});')
    else:
        if not isinstance(inst.src[2], (Var, int)):
            raise Exception()
        s2 = get_var_str(inst.src[2])
        h(f'vec{inst.dst.idx} = {dtype_str}({s1}, {s2});')


def make_mod_vec_list(h: CodeHelper, inst: Instruction):
    dtype_str = retrive_dtype_str(inst, (Shape, list))

    shape_1 = inst.src[1]
    if isinstance(shape_1, list):
        shape_1_str = get_curlybrace_list(shape_1)
    else:
        shape_1_str = get_var_str(shape_1)
    h(f'vec{inst.dst.idx} = {shape_1_str};')


def get_vec_size(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Vector):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()
    h(f'auto v{inst.dst.idx} = (int) vec{inst.src[0].idx}.size();')


def get_vec_index(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Vector):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int)):
        raise Exception()

    s1 = get_var_str(inst.src[1])

    if inst.src[0].dtype[0] == 'int':
        if not isinstance(inst.dst, Var):
            raise Exception()
        h(f'auto v{inst.dst.idx} = vec{inst.src[0].idx}[{s1}];')
    elif inst.src[0].dtype[0] == 'vec':
        if not isinstance(inst.dst, Vector):
            raise Exception()
        h(f'auto vec{inst.dst.idx} = vec{inst.src[0].idx}[{s1}];')
    else:
        if not isinstance(inst.dst, Tensor):
            raise Exception()
        h(f'auto tsr{inst.dst.idx} = vec{inst.src[0].idx}[{s1}];')


def set_vec_index(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Vector):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int)):
        raise Exception()

    s1 = get_var_str(inst.src[1])

    if inst.src[0].dtype[0] == 'int':
        if not isinstance(inst.src[2], (Var, int)):
            raise Exception()
        if isinstance(inst.src[2], Var):
            s2 = f'v{inst.src[2].idx}'
        else:
            s2 = str(inst.src[2])
        h(f'vec{inst.src[0].idx}[{s1}] = {s2};')
    else:
        if not isinstance(inst.src[2], Tensor):
            raise Exception()
        h(f'vec{inst.src[0].idx}[{s1}] = tsr{inst.src[2].idx};')


def vec_emplace_back(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Vector):
        raise Exception()
    h(f'vec{inst.src[0].idx}.emplace_back({get_curlybrace_list(inst.src[1])});')
