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

from .instruction import Instruction
from .utilfuncs import get_obj_dtype
from .. import context


class Var():
    idx: int = -1
    _previdx: list[int]
    dtype: str

    def __init__(self, dtype: str, is_declare: bool = True):
        self._previdx = []
        if not isinstance(dtype, str):
            raise Exception()
        self.dtype = dtype
        if is_declare:
            if context.active_module is None:
                raise Exception()
            self.idx = context.active_module.args_counter
            context.active_module.args_counter += 1
            context.active_module.add_inst(Instruction('var_declare', [self.dtype], self))

    def __str__(self):
        return f'Var[{self.idx}]'

    def __repr__(self):
        return str(self)

    def __mul__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype=get_obj_dtype(self))
        context.active_module.add_inst(Instruction('mul', [self, other], new_var))
        return new_var

    def __rmul__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype=get_obj_dtype(other))
        context.active_module.add_inst(Instruction('mul', [other, self], new_var))
        return new_var

    def __imul__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        context.active_module.add_inst(Instruction('imul', [self, other], self))
        return self

    def __eq__(self, other: Union['Var', int, float, str]):
        if not isinstance(other, (int, float, Var, str)):
            return False
        if context.active_module is None:
            raise Exception()
        if get_obj_dtype(self) == 'float' or get_obj_dtype(other) == 'float':
            if isinstance(other, str):
                raise Exception()
            eps = 1e-9
            return (other - eps < self) & (self < other + eps)
        new_var = context.active_module.create_var(dtype='bool')
        context.active_module.add_inst(Instruction('compare_eq', [self, other], new_var))
        return new_var

    def __ne__(self, other: Union['Var', int, float, str]):
        if context.active_module is None:
            raise Exception()
        if get_obj_dtype(self) == 'float' or get_obj_dtype(other) == 'float':
            if isinstance(other, str):
                raise Exception()
            eps = 1e-9
            return (self <= other - eps) | (other + eps <= self)
        new_var = context.active_module.create_var(dtype='bool')
        context.active_module.add_inst(Instruction('compare_neq', [self, other], new_var))
        return new_var

    def __gt__(self, other: Union['Var', int, float]):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype='bool')
        context.active_module.add_inst(Instruction('compare_gt', [self, other], new_var))
        return new_var

    def __ge__(self, other: Union['Var', int, float]):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype='bool')
        context.active_module.add_inst(Instruction('compare_ge', [self, other], new_var))
        return new_var

    def __lt__(self, other: Union['Var', int, float]):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype='bool')
        context.active_module.add_inst(Instruction('compare_gt', [other, self], new_var))
        return new_var

    def __le__(self, other: Union['Var', int, float]):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype='bool')
        context.active_module.add_inst(Instruction('compare_ge', [other, self], new_var))
        return new_var

    def __add__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype=get_obj_dtype(self))
        context.active_module.add_inst(Instruction('add', [self, other], new_var))
        return new_var

    def __radd__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype=get_obj_dtype(other))
        context.active_module.add_inst(Instruction('add', [self, other], new_var))
        return new_var

    def __iadd__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        context.active_module.add_inst(Instruction('iadd', [self, other], self))
        return self

    def __sub__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype=get_obj_dtype(self))
        context.active_module.add_inst(Instruction('sub', [self, other], new_var))
        return new_var

    def __rsub__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype=get_obj_dtype(other))
        context.active_module.add_inst(Instruction('sub', [other, self], new_var))
        return new_var

    def __isub__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        context.active_module.add_inst(Instruction('isub', [self, other], self))
        return self

    def __truediv__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype=get_obj_dtype(self))
        context.active_module.add_inst(Instruction('truediv', [self, other], new_var))
        return new_var

    def __rtruediv__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype=get_obj_dtype(other))
        context.active_module.add_inst(Instruction('truediv', [other, self], new_var))
        return new_var

    def __itruediv__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        context.active_module.add_inst(Instruction('itruediv', [self, other], self))
        return self

    def __floordiv__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype='int')
        context.active_module.add_inst(Instruction('floordiv', [self, other], new_var))
        return new_var

    def __rfloordiv__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype='int')
        context.active_module.add_inst(Instruction('floordiv', [other, self], new_var))
        return new_var

    def __ifloordiv__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        context.active_module.add_inst(Instruction('ifloordiv', [self, other], self))
        return self

    def __mod__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype='int')
        context.active_module.add_inst(Instruction('mod', [self, other], new_var))
        return new_var

    def __imod__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        context.active_module.add_inst(Instruction('imod', [self, other], self))
        return self

    def __pow__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype='float')
        context.active_module.add_inst(Instruction('pow', [self, other], new_var))
        return new_var

    def __rpow__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype='float')
        context.active_module.add_inst(Instruction('pow', [other, self], new_var))
        return new_var

    def __ipow__(self, other: Union[int, float, 'Var']):
        if context.active_module is None:
            raise Exception()
        context.active_module.add_inst(Instruction('ipow', [self, other], self))
        return self

    def __and__(self, other: 'Var'):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype=get_obj_dtype(self))
        context.active_module.add_inst(Instruction('var_and', [other, self], new_var))
        return new_var

    def __or__(self, other: 'Var'):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype=get_obj_dtype(self))
        context.active_module.add_inst(Instruction('var_or', [self, other], new_var))
        return new_var

    def __invert__(self):
        if context.active_module is None:
            raise Exception()
        new_var = context.active_module.create_var(dtype=get_obj_dtype(self))
        context.active_module.add_inst(Instruction('var_inv', [self], new_var))
        return new_var

    def set_idx(self, idx: int):
        self.idx = idx

    def push(self):
        self._previdx.append(self.idx)

    def pop(self):
        self.idx = self._previdx.pop(-1)

    def eq(self, expr: Union[int, float, str, 'Var']):
        if context.active_module is None:
            raise Exception()
        context.active_module.add_inst(Instruction('var_eq', [self.dtype, self, expr], None))
        return expr