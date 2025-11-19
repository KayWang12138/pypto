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
from typing import Union, Sequence

from .instruction import Instruction
from .tensor import Tensor
from .var import Var
from .. import context


class TensorMap():
    idx: int = -1
    _previdx: list[int]

    def __init__(self):
        self._previdx = []
        if context.active_module is not None:
            self.idx = context.active_module.args_counter
            context.active_module.args_counter += 1
            context.active_module.add_inst(Instruction('new_map', [], self))

    def __str__(self):
        return f'Map[{self.idx}]'

    def __repr__(self):
        return str(self)

    def set_idx(self, idx: int):
        self.idx = idx

    def push(self):
        self._previdx.append(self.idx)

    def pop(self):
        self.idx = self._previdx.pop(-1)

    def set(self, key: Union[Sequence[Union[Var, int]], 'Shape'], val: Tensor):
        if context.active_module is None:
            raise Exception()
        context.active_module.add_inst(Instruction('map_set', [key, val], self))

    def retrieve(self, key: Union[Sequence[Union[Var, int]], 'Shape']):
        if context.active_module is None:
            raise Exception()
        result = context.active_module.create_tensor()
        context.active_module.add_inst(Instruction('map_ret', [key, self], result))
        return result
