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

from typing import Union
from .util import get_var_str, get_curlybrace_list
from ..utils import CodeHelper, Instruction, Tensor, Vector, Var, Shape


def new_tensor(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Shape, list)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, str)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    if isinstance(inst.src[0], list):
        shape_str = get_curlybrace_list(inst.src[0])
    else:
        shape_str = get_var_str(inst.src[0])
    dtype_str = get_var_str(inst.src[1])

    h(f'Tensor tsr{inst.dst.idx}({dtype_str}, {shape_str});')


def tsr_declare(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'Tensor tsr{inst.dst.idx};')


def tsreq(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    h(f'tsr{inst.src[0].idx} = tsr{inst.src[1].idx};')


def retrieve_shape_str(inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (Shape, list)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    shape = inst.src[1]
    if isinstance(shape, list):
        shape_str = get_curlybrace_list(shape)
    else:
        shape_str = get_var_str(shape)

    return shape_str


def retrive_pair_shape_str(shape_1: Union[Shape, Vector, list], shape_2: Union[Shape, Vector, list]):
    if isinstance(shape_1, list):
        shape_1_str = get_curlybrace_list(shape_1)
    else:
        shape_1_str = get_var_str(shape_1)
    if isinstance(shape_2, list):
        shape_2_str = get_curlybrace_list(shape_2)
    else:
        shape_2_str = get_var_str(shape_2)
    return shape_1_str, shape_2_str


def view(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (Shape, Vector, list)):
        raise Exception()
    if not isinstance(inst.src[2], (Shape, Vector, list)):
        raise Exception()
    shape_1 = inst.src[1]
    shape_2 = inst.src[2]
    shape_1_str, shape_2_str = retrive_pair_shape_str(shape_1, shape_2)
    h(f'auto tsr{inst.dst.idx} = View(tsr{inst.src[0].idx}, {shape_1_str}, {shape_2_str});')


def dview(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (Shape, list)):
        raise Exception()
    if not isinstance(inst.src[2], (Shape, list)):
        raise Exception()
    shape_1 = inst.src[1]
    shape_2 = inst.src[2]
    shape_1_str, shape_2_str = retrive_pair_shape_str(shape_1, shape_2)
    h(f'auto tsr{inst.dst.idx} = DView(tsr{inst.src[0].idx}, {shape_1_str}, {shape_2_str});')


def dviewpad(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (Shape, list)):
        raise Exception()
    if not isinstance(inst.src[2], (Shape, list)):
        raise Exception()
    if not isinstance(inst.src[3], (Shape, list)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    shape_1 = inst.src[1]
    shape_2 = inst.src[2]
    shape_3 = inst.src[3]
    shape_1_str, shape_2_str = retrive_pair_shape_str(shape_1, shape_2)
    if isinstance(shape_3, list):
        shape_3_str = get_curlybrace_list(shape_3)
    else:
        shape_3_str = get_var_str(shape_3)
    h(f'auto tsr{inst.dst.idx} = DViewPad(tsr{inst.src[0].idx}, {shape_1_str}, {shape_2_str}, {shape_3_str});')


def dassemble(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (Shape, list, Vector)):
        raise Exception()
    if not isinstance(inst.src[2], Tensor):
        raise Exception()

    if isinstance(inst.src[1], list):
        var_str1 = get_curlybrace_list(inst.src[1])
    else:
        var_str1 = get_var_str(inst.src[1])

    h(f'DAssemble(tsr{inst.src[0].idx}, {var_str1}, tsr{inst.src[2].idx});')


def reshape(h: CodeHelper, inst: Instruction):
    shape_str = retrieve_shape_str(inst)

    h(f'auto tsr{inst.dst.idx} = Reshape(tsr{inst.src[0].idx}, {shape_str});')


def pad(h: CodeHelper, inst: Instruction):
    shape_str = retrieve_shape_str(inst)

    h(f'auto tsr{inst.dst.idx} = Pad(tsr{inst.src[0].idx}, {shape_str});')


def unsqueeze(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (int, Var)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    s = get_var_str(inst.src[1])
    h(f'auto tsr{inst.dst.idx} = Unsqueeze(tsr{inst.src[0].idx}, {s});')


def expand(h: CodeHelper, inst: Instruction):
    shape_str = retrieve_shape_str(inst)

    h(f'auto tsr{inst.dst.idx} = Expand(tsr{inst.src[0].idx}, {shape_str});')


def get_datatype(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    h(f'auto v{inst.dst.idx} = tsr{inst.src[0].idx}->Datatype();')


def transpose(h: CodeHelper, inst: Instruction):
    shape_str = retrieve_shape_str(inst)

    h(f'auto tsr{inst.dst.idx} = Transpose(tsr{inst.src[0].idx}, {shape_str});')


def concat(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (list, Vector)):
        raise Exception()
    if not isinstance(inst.src[1], (int, Var)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    if isinstance(inst.src[0], list):
        tensor_str = get_curlybrace_list(inst.src[0])
    else:
        tensor_str = get_var_str(inst.src[0])

    axis_str = get_var_str(inst.src[1])
    h(f'auto tsr{inst.dst.idx} = Concat({tensor_str}, {axis_str});')


def less_than0_inv(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, int)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int)):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()

    str0 = get_var_str(inst.src[0])
    str1 = get_var_str(inst.src[1])
    dest_str = f'auto {get_var_str(inst.dst)}'

    h(f'{dest_str} = {str0} >= 0 ? {str0} : ({str0} + {str1});')
