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

from typing import Optional, Union

from .instruction import Instruction
from .var import Var
from .. import context


class Shape():
    idx: int = -1

    def __init__(self, size: Optional[Union[int, Var]] = None,
                 value: Optional[Union[list[Union[int, Var]], int, Var]] = None):
        if size is not None:
            if context.active_module is not None:
                self.idx = context.active_module.args_counter
                context.active_module.args_counter += 1
                if value is not None:
                    context.active_module.add_inst(Instruction('new_shape', [size, value], self))
                else:
                    context.active_module.add_inst(Instruction('new_shape', [size], self))

    def __getitem__(self, idx: Union[int, Var, slice]):
        if context.active_module is None:
            raise Exception()
        if isinstance(idx, int) and idx < 0:
            idx = idx + self.size()
        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            if start is None:
                start = 0
            if step is None:
                step = 1
            lst = []
            for i in range(start, stop, step):
                new_var = context.active_module.create_var('int')
                context.active_module.add_inst(Instruction('get_shape_dim', [self, i], new_var))
                lst.append(new_var)
            return lst
        else:
            new_var = context.active_module.create_var('int')
            context.active_module.add_inst(Instruction('get_shape_dim', [self, idx], new_var))
            return new_var

    def __setitem__(self, idx: Union[int, Var], value: Union[int, Var]):
        if context.active_module is None:
            raise Exception()
        if isinstance(idx, int) and idx < 0:
            idx = idx + self.size()
        context.active_module.add_inst(Instruction('set_shape', [self, idx, value], None))

    def __str__(self):
        return f'Shape[{self.idx}]'

    def __repr__(self):
        return str(self)
    
    def set_idx(self, idx: int):
        self.idx = idx

    def size(self):
        from ..stub_fun import shapesize
        res = shapesize(self)
        return res

    def emplace_back(self, val: Union[int, Var]):
        from ..stub_fun import shape_emplace_back
        shape_emplace_back(self, val)
