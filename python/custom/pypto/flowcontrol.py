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
from typing import Union, Optional

from . import context
from .utils import DATATYPE as DT
from .utils import Var, Instruction


class Loop():
    def __init__(self, start: Union[Var, int], end: Union[Var, int], step: Union[Var, int] = 1) -> None:
        self.start = start
        self.end = end
        self.step = step
        self.tmp_var = Var(DT.int, is_declare=False)

    def __enter__(self):
        if context.active_module is not None:
            self.tmp_var = context.active_module.create_var(DT.int)
            context.active_module.add_inst(Instruction('start_loop', \
                                                       [self.tmp_var, self.start, self.end, self.step], None))
            return self.tmp_var
        else:
            raise Exception('Loop must be called in ActiveModule')

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            return False
        g_active_module = context.active_module
        if g_active_module is not None:
            g_active_module.add_inst(Instruction('end_loop', [self.tmp_var, self.end], None))
        else:
            raise Exception('Loop must be called in ActiveModule')
        return True


class If():
    start_inst: Optional[Instruction]

    def __init__(self, cond: Union[Var, bool]):
        self.cond = cond
        self.start_inst = None

    def __enter__(self):
        if context.active_module is None:
            raise Exception()
        self.start_inst = context.active_module.start_if(self.cond)

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            return False
        if context.active_module is None:
            raise Exception()
        if self.start_inst is None:
            raise Exception()
        context.active_module.end_if(self.start_inst)
        return True


class Else():
    def __init__(self):
        pass

    @classmethod
    def __enter__(cls):
        if context.active_module is None:
            raise Exception()
        context.active_module.start_else()

    @classmethod
    def __exit__(cls, exec_type, exec_val, exec_traceback):
        if exec_type:
            return False
        if context.active_module is None:
            raise Exception()
        context.active_module.end_else()
        return True


class Elif():
    start_inst: Optional[Instruction]

    def __init__(self, cond: Union[Var, bool]):
        self.cond = cond
        self.start_inst = None

    def __enter__(self):
        if context.active_module is None:
            raise Exception()
        self.start_inst = context.active_module.start_elseif(self.cond)

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            return False
        if context.active_module is None:
            raise Exception()
        if self.start_inst is None:
            raise Exception()
        context.active_module.end_if(self.start_inst)
        return True