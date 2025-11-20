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
from typing import Sequence

from .util import get_var_str, get_vartype_str
from ..utils import CodeHelper, Instruction, CustStruct, Tensor, Vector, Var, ConfigMap


def get_attr(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], CustStruct):
        raise Exception()
    if not isinstance(inst.src[1], str):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()
    h(f'auto v{inst.dst.idx} = stru{inst.src[0].idx}.{inst.src[1]};')


def get_attr_tensor(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], CustStruct):
        raise Exception()
    if not isinstance(inst.src[1], str):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = stru{inst.src[0].idx}.{inst.src[1]};')


def get_attr_vector(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], CustStruct):
        raise Exception()
    if not isinstance(inst.src[1], str):
        raise Exception()
    if not isinstance(inst.dst, Vector):
        raise Exception()
    h(f'auto vec{inst.dst.idx} = stru{inst.src[0].idx}.{inst.src[1]};')


def get_attr_config(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], ConfigMap):
        raise Exception()
    if not isinstance(inst.src[1], str):
        raise Exception()
    if not inst.src[2] in [int, float, bool, str]:
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()
    if not isinstance(inst.src[2], type):
        raise Exception()

    type_str = get_vartype_str(inst.src[2])
    config_var_name = get_var_str(inst.src[0])
    h(f'auto v{inst.dst.idx} = std::get<{type_str}>({config_var_name}["{inst.src[1]}"]);')


def call_func(h: CodeHelper, inst: Instruction):
    funcname = inst.src[0]
    params = inst.src[1:]
    if not isinstance(funcname, str):
        raise Exception()

    params_str = []
    for p in params:
        if not isinstance(p, (Tensor, CustStruct, ConfigMap, Var, Vector)):
            raise Exception()
        params_str.append(get_var_str(p))

    params_str = ', '.join(params_str)

    if isinstance(inst.dst, Tensor):
        left = f'auto tsr{inst.dst.idx} = '
        h(f'{left}{funcname}({params_str});')
    elif isinstance(inst.dst, Var):
        left = f'{inst.dst.dtype} v{inst.dst.idx} = '
        h(f'{left}{funcname}({params_str});')
    elif isinstance(inst.dst, Sequence):
        left = f'auto tsrbundle{inst.dst[0].idx} = '
        h(f'{left}{funcname}({params_str});')
        for ii, _ in enumerate(inst.dst):
            h(f'auto tsr{inst.dst[ii].idx} = tsrbundle{inst.dst[0].idx}[{ii}];')
    elif isinstance(inst.dst, Vector):
        left = f'{inst.dst.dtype_str} vec{inst.dst.idx} = '
        h(f'{left}{funcname}({params_str});')
    else:
        h(f'{funcname}({params_str});')


def retrieve_clsname_and_params_str(inst: Instruction):
    classname = inst.src[0]
    params = inst.src[1:]
    if not isinstance(classname, str):
        raise Exception()

    from pypto.module import AscppModule
    if not isinstance(inst.dst, AscppModule):
        raise Exception()

    params_str = []
    for p in params:
        if not (isinstance(p, (Tensor, CustStruct, ConfigMap, Var, Vector))):
            raise Exception()
        params_str.append(get_var_str(p))
    params_str = ', '.join(params_str)

    return classname, params_str


def make_mod_class(h: CodeHelper, inst: Instruction):
    classname, params_str = retrieve_clsname_and_params_str(inst)

    h(f'cls{inst.dst.idx} = {classname}({params_str});')


def new_class(h: CodeHelper, inst: Instruction):
    classname, params_str = retrieve_clsname_and_params_str(inst)

    h(f'auto cls{inst.dst.idx} = {classname}({params_str});')


def add_comment(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], str):
        raise Exception()
    h(f'// {inst.src[0]}')


def call_module_function(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], str):
        raise Exception()
    if not isinstance(inst.src[1], (type(None), str)):
        raise Exception()
    if not isinstance(inst.src[2], Sequence):
        raise Exception()

    func_name = inst.src[0]
    return_type = inst.src[1]
    args = inst.src[2]
    for arg in args:
        if not isinstance(arg, (Tensor, Var, Vector, float, int, str)):
            raise Exception()
    arg_str = ', '.join([get_var_str(arg) for arg in args])

    if inst.src[1] is None:  # no return
        h(f'{func_name}({arg_str});')
    else:
        if not isinstance(inst.dst, (Var, Tensor)):
            raise Exception()
        h(f'auto {get_var_str(inst.dst)} = {func_name}({arg_str});')
