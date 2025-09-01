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

from .util import get_var_str
from ..utils import CodeHelper, Instruction, Var


def var_eq(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], str):
        raise Exception()
    if not isinstance(inst.src[1], Var):
        raise Exception()
    if not isinstance(inst.src[2], (str, int, float, Var)):
        raise Exception()
    dest_str = get_var_str(inst.src[2])
    h(f'v{inst.src[1].idx} = ({inst.src[0]}) {dest_str};')


def var_declare(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], str):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()
    h(f'{inst.src[0]} v{inst.dst.idx};')
