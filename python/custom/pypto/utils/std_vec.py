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

from typing import Union, Optional, Sequence

from .instruction import Instruction
from .shape import Shape
from .tensor import Tensor
from .utilfuncs import get_vartype_str
from .var import Var
from .. import context


class Vector():
    _previdx: list[int]
    idx: int = -1
    dtype: list[str]

    def __init__(self, dtype: Union[list[str], str, 'Vector', type],
                 init_vec: Optional[Union[Shape, Sequence[Union[Var, int, Tensor]]]] = None,
                 init_len: Optional[Union[Var, int]] = None, init_val: Optional[Union[Var, int, Tensor]] = None,
                 is_declare: bool = True):

        def check_dtype_instance(dtype: Union[list, str, 'Vector', type]):
            if isinstance(dtype, Vector):
                self.dtype = ['vec'] + dtype.dtype
                if dtype.idx != -1:
                    self.idx = dtype.idx
                else:
                    if is_declare and context.active_module is not None:
                        self.idx = context.active_module.args_counter
                        context.active_module.args_counter += 1
                dtype.remove_declaration()
            elif isinstance(dtype, str):
                self.dtype = [dtype]
            elif isinstance(dtype, list):
                self.dtype = dtype
            elif isinstance(dtype, type):
                self.dtype = [get_vartype_str(dtype)]
            else:
                raise NotImplementedError()

        def check_context_stage(dtype: Union[list[str], str, 'Vector', type],
                                init_vec: Union[Shape, Sequence[Union[Var, int, Tensor]]],
                                init_len: Union[Var, int],
                                init_val: Union[Var, int, Tensor]):
            if context.active_module is None:
                return
            if not isinstance(dtype, Vector):
                self.idx = context.active_module.args_counter
                context.active_module.args_counter += 1
            if context.active_module.get_stage() == 'init':
                if init_vec is not None:
                    context.active_module.add_inst(Instruction('make_mod_vec_list', [self.dtype, init_vec], self))
                elif init_len is not None:
                    context.active_module.add_inst(
                        Instruction('make_mod_vec_len', [self.dtype, init_len, init_val], self))
                else:
                    context.active_module.add_inst(Instruction('make_mod_vec', [self.dtype], self))
            elif context.active_module.get_stage() == 'call':
                if init_vec is not None:
                    context.active_module.add_inst(Instruction('vec_declare_list', [self.dtype, init_vec], self))
                elif init_len is not None:
                    context.active_module.add_inst(
                        Instruction('vec_declare_len', [self.dtype, init_len, init_val], self))
                else:
                    context.active_module.add_inst(Instruction('vec_declare', [self.dtype], self))

        self._previdx = []
        check_dtype_instance(dtype)
        if is_declare:
            check_context_stage(dtype, init_vec, init_len, init_val)

    def __str__(self):
        return f'Vector[{self.idx}]'

    def __repr__(self):
        return str(self)

    def __getitem__(self, idx: Union[int, Var, Sequence[Union[int, Var]], slice]) -> Union[
        Var, Tensor, 'Vector', Sequence]:
        if context.active_module is None:
            raise Exception()
        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            if start is None:
                start = 0
            if step is None:
                step = 1
            lst = []
            for i in range(start, stop, step):
                new_var = context.active_module.create_var('int')
                context.active_module.add_inst(Instruction('get_vec_index', [self, i], new_var))
                lst.append(new_var)
            return lst

        if isinstance(idx, int) and idx < 0:
            idx = idx + self.size()
        if self.dtype[0] == 'int':
            res = context.active_module.create_var('int')
        elif self.dtype[0] == 'vec':
            res = context.active_module.create_vector(self.dtype[1:])
        else:
            res = context.active_module.create_tensor()
        context.active_module.add_inst(Instruction('get_vec_index', [self, idx], res))
        return res

    def __setitem__(self, idx: Union[int, Var], value: Union[int, Var, Tensor]):
        if context.active_module is None:
            raise Exception()
        if isinstance(idx, int) and idx < 0:
            idx = idx + self.size()
        context.active_module.add_inst(Instruction('set_vec_index', [self, idx, value], None))

    @classmethod
    def __class_getitem__(cls, type_):
        return Vector(dtype=type_)

    @property
    def dtype_str(self):
        res = f'std::vector<{self.dtype[-1]}>'
        for _ in range(-2, -len(self.dtype) - 1, -1):
            res = f'std::vector<{res}>'
        return res

    def remove_declaration(self):
        if context.active_module is None:
            return
        if context.active_module is None:
            raise Exception()
        candidate = None
        for inst in context.active_module.inst_list:
            if 'vec_declare' in inst.inst and inst.dst == self:
                candidate = inst
        if candidate is not None:
            context.active_module.inst_list.remove(candidate)

    def set_idx(self, idx: int):
        self.idx = idx

    def push(self):
        self._previdx.append(self.idx)

    def pop(self):
        self.idx = self._previdx.pop(-1)

    def size(self):
        from ..stub_fun import vec_size
        res = vec_size(self)
        return res

    def emplace_back(self, value: Union[Var, int, Tensor]):
        from ..stub_fun import vec_emplace_back
        vec_emplace_back(self, value)
