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
from collections import defaultdict
from typing import Any, Union, TypeVar
from types import FunctionType
from ..utils import Var, GMTensor, Tensor, DBuff, Instruction, DT, Position, SEvent, DEvent, CodeHelper, DTYPE, PIPE, Pos, sizeof, PipeInst
from .. import context
from .. import autosync
from . import parser

from rich.style import Style
from rich.text import Text
from rich.console import Console
from functools import partial
print = partial(Console(width=150).print, justify='center')


BUFFER_T = TypeVar('BUFFER_T', bound=Union[DBuff, Tensor])


class CubeModule():
    _curr_phase: str
    _input_vars: list[Var]
    _input_tensors: list[GMTensor]
    _init_inst_list: list[Instruction]
    _compute_inst_list: list[Instruction]
    _buffer_list: list[Union[Tensor, DBuff]]
    _event_list: list[Union[SEvent, DEvent]]
    _name: str
    _tmp_idx: int
    _var_mapping: dict[Var, Var]
    _flag_id_dict: dict[tuple[PipeInst, PipeInst], int]

    def __init__(self, *args: Var):
        self._name = ''
        self._curr_phase = 'init'
        self._input_vars = []
        self._input_tensors = []
        self._init_inst_list = []
        self._compute_inst_list = []
        self._event_list = []
        self._buffer_list = []
        self._tmp_idx = 0
        self._var_mapping = {}
        self._flag_id_dict = defaultdict(int)

        for a in args:
            assert isinstance(a, Var), 'All input args should be Var'
            self._input_vars.append(a)

        context.active_cube = self
        self.l1_empty = DEvent(PIPE.MTE1, PIPE.MTE2)
        self.l1_ready = DEvent(PIPE.MTE2, PIPE.MTE1)
        self.l0_empty = DEvent(PIPE.M, PIPE.MTE1)
        self.l0_ready = DEvent(PIPE.MTE1, PIPE.M)
        self.out_empty = DEvent(PIPE.FIX, PIPE.M)
        self.out_ready = DEvent(PIPE.M, PIPE.FIX)
        if hasattr(self, 'initialize'):
            self.initialize(*args)  # type: ignore
        context.active_cube = None

    @property
    def var_mapping(self):
        return self._var_mapping

    def fetch_tmp_idx(self):
        res = self._tmp_idx
        self._tmp_idx += 1
        return res

    def assign_vars_funcmode(self, *vs: Var):
        for value in vs:
            new_var = value.copy()
            new_var.name = f'shape.{value.name}'
            new_var.varstr = f'shape.{value.name}'
            self._var_mapping[value] = new_var
            self.append(Instruction('ASSIGNVAR', v=new_var, src=value))

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            super().__setattr__(name, value)
            return
        assert self._curr_phase=='init', 'Can only assign class variables in initialize'
        if isinstance(value, float):
            new_var = Var(f'shape.{name}', DT.float, value)
            self.append(Instruction('ASSIGNVAR', v=new_var, src=value))
        elif isinstance(value, int):
            new_var = Var(f'shape.{name}', DT.int, value)
            self.append(Instruction('ASSIGNVAR', v=new_var, src=value))
        elif isinstance(value, Var):
            new_var = value.copy()
            new_var.name = f'shape.{name}'
            new_var.varstr = f'shape.{name}'
            self._var_mapping[value] = new_var
            self.append(Instruction('ASSIGNVAR', v=new_var, src=value))
        elif isinstance(value, (Tensor, DBuff)):
            new_var = value
            new_var.name = name
            if new_var.pos is None:
                new_var.pos = Position.L1
            self._buffer_list.append(new_var)
        elif isinstance(value, (SEvent, DEvent)):
            new_var = value
            new_var.set_name(name)
            new_var.set_module(self)
            self._event_list.append(new_var)
        elif isinstance(value, FunctionType):
            new_var = value
        else:
            raise TypeError(f'Not supported type {type(value)}')
        super().__setattr__(name, new_var)

    def add_buffer(self, buf: Union[Tensor, DBuff]):
        buf.name = '_local_buffer_%d'%self._tmp_idx
        self._tmp_idx += 1
        self._buffer_list.append(buf)

    def set_name(self, name: str):
        self._name = name

    def __call__(self, *args: GMTensor):
        for a in args:
            assert isinstance(a, GMTensor), 'Forward inputs must be GMTensors'
        self.inner_forward(*args)

    def inner_forward(self, *args: Union[Var, GMTensor]):
        for a in args:
            if isinstance(a, GMTensor):
                self._input_tensors.append(a)
        assert context.active_kernel is not None
        context.active_kernel.run_module(self)
        self._curr_phase = 'computing'

        context.active_cube = self
        self.l1_empty.setall()
        self.l0_empty.setall()
        self.out_empty.setall()
        if hasattr(self, 'forward'):
            self.forward(*args)  # type: ignore
        self.l1_empty.release()
        self.l0_empty.release()
        self.out_empty.release()
        context.active_cube = None
        self._curr_phase = 'finished'
        self._compute_inst_list = autosync.parse_autosync(self._compute_inst_list, 'cube')

    def append(self, inst: Instruction):
        if self._curr_phase=='init':
            self._init_inst_list.append(inst)
        elif self._curr_phase=='computing':
            self._compute_inst_list.append(inst)
        else:
            raise Exception(f'Can only call in initialize or forward. Not {self._curr_phase}')

    def create_l0a(self, t: type[BUFFER_T], dtype: DTYPE, length: Union[int, Var]) -> BUFFER_T:
        new_obj = t(dtype, length)
        new_obj.pos = Position.L0A
        return new_obj

    def create_l0b(self, t: type[BUFFER_T], dtype: DTYPE, length: Union[int, Var]) -> BUFFER_T:
        new_obj = t(dtype, length)
        new_obj.pos = Position.L0B
        return new_obj

    def create_l0c(self, t: type[BUFFER_T], dtype: DTYPE, length: Union[int, Var]) -> BUFFER_T:
        new_obj = t(dtype, length)
        new_obj.pos = Position.L0C
        return new_obj

    def create_ub(self, t: type[BUFFER_T], dtype: DTYPE, length: Union[int, Var]) -> BUFFER_T:
        new_obj = t(dtype, length)
        new_obj.pos = Position.UB
        return new_obj

    def wait_l1_empty(self):
        self.l1_empty.wait()

    def set_l1_empty(self):
        self.l1_empty.set()

    def wait_l1_ready(self):
        self.l1_ready.wait()

    def set_l1_ready(self):
        self.l1_ready.set()

    def wait_l0_empty(self):
        self.l0_empty.wait()

    def set_l0_empty(self):
        self.l0_empty.set()

    def wait_l0_ready(self):
        self.l0_ready.wait()

    def set_l0_ready(self):
        self.l0_ready.set()

    def wait_out_empty(self):
        self.out_empty.wait()

    def set_out_empty(self):
        self.out_empty.set()

    def wait_out_ready(self):
        self.out_ready.wait()

    def set_out_ready(self):
        self.out_ready.set()

    # codegen related
    def gen_header(self, h: CodeHelper):
        # some includes and defines
        h('#pragma once')
        h('#include "tensorutils.h"')
        h()
        h()

    def gen_tiling(self, h: CodeHelper):
        h(f'struct {self._name}ShapeInfo{{')
        h.ir()
        var_list = []
        for i in self._init_inst_list:
            if i._inst=='ASSIGNVAR':
                v: Var = i.v
                if v.name in var_list:
                    continue
                if 'shape.' in v.name:
                    var_list.append(v.name)
                    h(f'{v.dtype} {v.name.replace("shape.", "")};')
        h.il()
        h('};')
        h()
        h()

        arg_list = [f'{v.dtype} {v.name}' for v in self._input_vars]
        arg_list.append(f'{self._name}ShapeInfo &shape')
        arg_str = ', '.join(arg_list)
        h(f'__aicore__ inline void tilingShape{self._name}({arg_str}){{')
        h.ir()
        parser.parse_all(self._init_inst_list, h)
        h.il()
        h('}')
        h()
        h()

    def gen_init(self, h: CodeHelper, name: str):
        arg_list = ''
        for i in self._input_tensors:
            arg_list += f'GM_ADDR {i.name}_, '
        arg_list += f'{name}ShapeInfo shape_'
        h(f'__aicore__ inline void Init({arg_list}){{')
        h.ir()
        h('shape = shape_;')
        h('set_nd_para(0x100010001);')
        h('// Global Tensors')
        for i in self._input_tensors:
            h(f'{i.name} = Tensor<{i.dtype}, PGM>({i.name}_);')

        h('// L1 Buffers')
        h('int offset = 0;')
        for i in self._buffer_list:
            if i.pos==Position.L1:
                if isinstance(i, DBuff):
                    h(f'{i.name} = DBuff<{i.dtype}, PL1>(0, {i.length}, offset);')
                else:
                    h(f'{i.name} = Tensor<{i.dtype}, PL1>(0, {i.length}, offset);')

        h('// L0A Buffers')
        h('offset = 0;')
        for i in self._buffer_list:
            if i.pos==Position.L0A:
                if isinstance(i, DBuff):
                    h(f'{i.name} = DBuff<{i.dtype}, PL0A>(0, {i.length}, offset);')
                else:
                    h(f'{i.name} = Tensor<{i.dtype}, PL0A>(0, {i.length}, offset);')

        h('// L0B Buffers')
        h('offset = 0;')
        for i in self._buffer_list:
            if i.pos==Position.L0B:
                if isinstance(i, DBuff):
                    h(f'{i.name} = DBuff<{i.dtype}, PL0B>(0, {i.length}, offset);')
                else:
                    h(f'{i.name} = Tensor<{i.dtype}, PL0B>(0, {i.length}, offset);')

        h('// L0C Buffers')
        h('offset = 0;')
        for i in self._buffer_list:
            if i.pos==Position.L0C:
                if isinstance(i, DBuff):
                    h(f'{i.name} = DBuff<{i.dtype}, PL0C>(0, {i.length}, offset);')
                else:
                    h(f'{i.name} = Tensor<{i.dtype}, PL0C>(0, {i.length}, offset);')

        h('// UB Buffers')
        h('offset = 0;')
        for i in self._buffer_list:
            if i.pos==Position.UB:
                if isinstance(i, DBuff):
                    h(f'{i.name} = DBuff<{i.dtype}, PUB>(0, {i.length}, offset);')
                else:
                    h(f'{i.name} = Tensor<{i.dtype}, PUB>(0, {i.length}, offset);')

        h.il()
        h('}')
        h()

    def gen_compute(self, h: CodeHelper):
        h('__aicore__ inline void Compute(){')
        h.ir()
        parser.parse_all(self._compute_inst_list, h)
        h.il()
        h('}')
        h()

    def gen_members(self, h: CodeHelper, name: str):
        h(f'{name}ShapeInfo shape;')
        h('// Global Tensors')
        for i in self._input_tensors:
            h(f'Tensor<{i.dtype}, PGM> {i.name};')
        h('// Local buffers')
        for i in self._buffer_list:
            if i.pos==Position.L1:
                if isinstance(i, DBuff):
                    h(f'DBuff<{i.dtype}, PL1> {i.name};')
                else:
                    h(f'Tensor<{i.dtype}, PL1> {i.name};')
        for i in self._buffer_list:
            if i.pos==Position.L0A:
                if isinstance(i, DBuff):
                    h(f'DBuff<{i.dtype}, PL0A> {i.name};')
                else:
                    h(f'Tensor<{i.dtype}, PL0A> {i.name};')
        for i in self._buffer_list:
            if i.pos==Position.L0B:
                if isinstance(i, DBuff):
                    h(f'DBuff<{i.dtype}, PL0B> {i.name};')
                else:
                    h(f'Tensor<{i.dtype}, PL0B> {i.name};')
        for i in self._buffer_list:
            if i.pos==Position.L0C:
                if isinstance(i, DBuff):
                    h(f'DBuff<{i.dtype}, PL0C> {i.name};')
                else:
                    h(f'Tensor<{i.dtype}, PL0C> {i.name};')
        for i in self._buffer_list:
            if i.pos==Position.UB:
                if isinstance(i, DBuff):
                    h(f'DBuff<{i.dtype}, PUB> {i.name};')
                else:
                    h(f'Tensor<{i.dtype}, PUB> {i.name};')
        h()
        h('// Events')
        for e in self._event_list:
            if isinstance(e, SEvent):
                next_id = self._flag_id_dict[(e.src, e.dst)]
                h(f'SEvent<PIPE_{e.src}, PIPE_{e.dst}> {e.name}{{{next_id}}};' )
                self._flag_id_dict[(e.src, e.dst)] += 1
            if isinstance(e, DEvent):
                next_id = self._flag_id_dict[(e.src, e.dst)]
                h(f'DEvent<PIPE_{e.src}, PIPE_{e.dst}> {e.name}{{{next_id}, {next_id+1}}};' )
                self._flag_id_dict[(e.src, e.dst)] += 2

    def check_usage_pos(self, pos: Pos):
        print(f'┌───────────────── {str(pos).center(5)} Usage ─────────────────┐')
        total = 0
        for i in self._buffer_list:
            multiplier = 1
            if isinstance(i, DBuff):
                multiplier = 2
            if i.pos==pos:
                if isinstance(i.length, Var):
                    if i.length.value is None:
                        print(f'Cannot determine size of buffer {i.name}')
                    else:
                        s = i.length.value * sizeof(i.dtype) * multiplier
                        total += s
                        print(f'[{i.__class__.__name__}] {i.name} : {s/1024} KB')
                elif isinstance(i.length, int):
                    s = i.length * sizeof(i.dtype) * multiplier
                    total += s
                    print(f'[{i.__class__.__name__}] {i.name} : {s/1024} KB')
        print('-------------------------------------------')
        print(f'Total {pos} usage: {total/1024} / {context.get_max_size(pos)/1024} KB')
        if total>context.get_max_size(pos):
            print(Text(f'WARNING: Exceed max capacity!!', style=Style(color='deep_pink4', bold=True, reverse=True)))
        print('└───────────────────────────────────────────────┘')

    def gen_code(self, filename: str):
        print()
        outstr = Text()
        outstr.append('CUBE: Generating [')
        outstr.append(Text(self._name, style=Style(color='red', bold=True, underline=True)))
        outstr.append(']')
        Console(width=150).rule(outstr)
        if self._curr_phase!='finished':
            print(Text('Warning', style='yellow').append(Text(': [', style='white')).append(Text(f'{self._name}', style='red')).append(Text('] is not forwarded. Skip generation.', style='white')))
            return

        h = CodeHelper()
        self.gen_header(h)
        self.gen_tiling(h)

        h(f'class {self._name}{{')
        h('public:')
        h.ir()
        h(f'__aicore__ inline {self._name}(){{}}')
        self.gen_init(h, self._name)
        self.gen_compute(h)
        h.il()
        h('private:')
        h.ir()
        self.gen_members(h, self._name)
        h.il()
        h('};')

        for p in [Position.L1, Position.L0A, Position.L0B, Position.L0C, Position.UB]:
            self.check_usage_pos(p)

        fout = open(filename, 'w')
        fout.write(h.result)
        fout.close()

