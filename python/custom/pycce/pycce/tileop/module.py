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
from collections import defaultdict
from typing import Any, Union, TypeVar
from types import FunctionType
from ..utils import Var, GMTensor, Tensor, DBuff, Instruction, DT, Position, SEvent, DEvent, CodeHelper, DTYPE, PIPE, Pos, sizeof, PipeInst
from .. import context
from .. import autosync
from . import parser

# from rich.style import Style
# from rich.text import Text
# from rich.console import Console
# from functools import partial
# print = partial(Console(width=150).print, justify='center')


BUFFER_T = TypeVar('BUFFER_T', bound=Union[DBuff, Tensor])


class TileOpModule():
    _input_vars: list[Var]
    _input_tensors: list[Tensor]
    _inst_list: list[Instruction]
    _name: str
    _tmp_idx: int
    _registered: bool
    _curr_phase: str

    def __init__(self):
        self._name = ''
        self._input_vars = []
        self._input_tensors = []
        self._inst_list = []
        self._tmp_idx = 0
        self._registered = False
        self._curr_phase = ''

    def __call__(self, *args: Union[Tensor, int, float, Var]):
        for a in args:
            assert isinstance(a, (Tensor, int, float, Var)), 'Forward inputs must be GMTensors'

        if not self._registered and context.active_vec is None:
            self.register_args(*args)
            context.active_vec = self   # register current active module and record instructions

        if hasattr(self, 'forward'):
            self.forward(*args)  # type: ignore

        if not self._registered and context.active_vec==self:
            context.active_vec = None
            self._registered = True

    def fetch_tmp_idx(self):
        res = self._tmp_idx
        self._tmp_idx += 1
        return res

    def get_name(self):
        return self._name

    def set_name(self, name: str):
        self._name = name

    def add_buffer(self, *args):
        raise TypeError('Cannot add any buffers inside tileop function')

    def get_input_tensors(self):
        return self._input_tensors

    def register_args(self, *args: Union[Var, int, float, Tensor]):
        for a in args:
            if isinstance(a, Tensor):
                self._input_tensors.append(a)
            elif isinstance(a, Var):
                self._input_vars.append(a)
            elif isinstance(a, (float, int)):
                ...
            else:
                raise TypeError()

    def append(self, inst: Instruction):
        self._inst_list.append(inst)

    # codegen related
    def gen_code(self, filename: str):
        print()
        h = CodeHelper()
        # generate template args
        tmplt_args = []
        for i in range(len(self._input_tensors)):
            tmplt_args.append('typename T%d'%i)
        for v in self._input_vars:
            tmplt_args.append(f'{v.dtype.ctype} {v.name}')
        tmplt_args_str = f'template <{", ".join(tmplt_args)}>'
        h(tmplt_args_str)

        # generate input args
        input_args = []
        for i,t in enumerate(self._input_tensors):
            input_args.append(f'__ubuf__ T{i} *{t.name}')
        input_args_str = ', '.join(input_args)
        h(f'TILEOP void {self._name}_pytileopfn({input_args_str}){{')
        h.ir()

        parser.parse_all(self._inst_list, h)

        h.il()
        h('}')

        return h.result
        # print(h.result)

        # fout = open(filename, 'w')
        # fout.write(h.result)
        # fout.close()
