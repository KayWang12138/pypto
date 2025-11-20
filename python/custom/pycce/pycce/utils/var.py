# -----------------------------------------------------------------------------------------------------------
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
from typing import Union, Optional, TYPE_CHECKING
from .instruction import Instruction
from .datatype import DTYPE
from .datatype import DATATYPE as DT
from .. import context

if TYPE_CHECKING:
    from ..cube import CubeModule
    from ..vec import VecModule
    from ..kernel import KernelBase
    from .tensor import Tensor, GMTensor


class Var():
    _module: Optional[Union['CubeModule', 'VecModule', 'KernelBase']]
    name: str
    dtype: DTYPE
    varstr: str
    value: Optional[Union[int, float]]

    def __init__(self, name_or_value: Union[str, int, float, 'Var'], dtype: Optional[DTYPE]=None, value: Optional[Union[int, float]]=None, auto_declare: bool=True):
        active_mod = context.active_cube or context.active_vec

        if isinstance(name_or_value, str):
            if dtype is None:
                raise Exception('Must specify datatype')
            self.name = name_or_value
            self.varstr = name_or_value
            self.dtype = dtype
            self.value = value
        else:
            if active_mod is None:
                raise Exception('Must be in cube or vector function')
            self.name = '_local_var_%d'%active_mod._tmp_idx
            active_mod._tmp_idx += 1
            self.varstr = self.name
            if isinstance(name_or_value, Var):
                self.dtype = name_or_value.dtype
                self.value = name_or_value.value
            elif isinstance(name_or_value, float):
                self.dtype = DT.float
                self.value = name_or_value
            elif isinstance(name_or_value, int):
                self.dtype = DT.int
                self.value = name_or_value
            else:
                raise TypeError(f'Got not supported type {type(name_or_value)}')

        self._module = None
        if active_mod is not None and active_mod._curr_phase=='computing':
            self._module = active_mod
            if auto_declare:
                if isinstance(name_or_value, str):
                    active_mod.append(Instruction('CREATEVAR', v=self, value=value))
                else:
                    active_mod.append(Instruction('CREATEVAR', v=self, value=name_or_value))

    def __hash__(self) -> int:
        return hash(self.name+self.varstr)

    def copy(self):
        new_const = Var(self.name, self.dtype)
        new_const.varstr = self.varstr
        new_const._module = self._module
        new_const.value = self.value
        return new_const

    def assign_module(self, m: Union['CubeModule', 'VecModule', 'KernelBase']):
        self._module = m

    def inc(self):
        self.eq(self+1)
        return self

    def __ilshift__(self, other: Union[int, float, 'Var']):
        self.eq(other)
        return self

    def __iadd__(self, other: Union[int, float, 'Var']):
        self.eq(self + other)
        return self

    def __isub__(self, other: Union[int, float, 'Var']):
        self.eq(self - other)
        return self

    def __imul__(self, other: Union[int, float, 'Var']):
        self.eq(self * other)
        return self

    def eq(self, expr: Union[int, float, 'Var']):
        assert self._module is not None
        if self.dtype in [DT.half, DT.float]:
            expr2 = Var('', self.dtype)
            expr2.varstr = f'(({self.dtype}){str(expr)})'
            expr2.name = f'(({self.dtype}){str(expr)})'
            expr = expr2
        self._module.append(Instruction('ASSIGNVAR', v=self, src=expr))

    def GetValue(self, v: Union['Tensor', 'GMTensor']):
        from .tensor import Tensor, GMTensor
        from .position import Position
        assert isinstance(v, (Tensor, GMTensor)), 'Can only get value from Tensor or GMTensor'
        if isinstance(v, Tensor):
            assert v.pos==Position.UB, 'Tensor must be on UB'

        active_mod = context.active_cube or context.active_vec
        assert active_mod is not None, 'GetValue must be in either CubeModule forward or VecModule forward'
        active_mod.append(Instruction('GETVAL', dst=self, src=v))

    def SetValue(self, v: 'Tensor'):
        from .tensor import Tensor
        from .position import Position
        assert isinstance(v, Tensor), 'Only UB Tensor is support for SetValue'
        assert v.pos==Position.UB, 'Tensor must be on UB'
        active_mod = context.active_vec
        assert active_mod is not None, 'SetValue must be in VecModule forward'
        active_mod.append(Instruction('SETVAL', dst=self, src=v))

    def __str__(self) -> str:
        return self.varstr

    def __repr__(self) -> str:
        return f'[Var] name={self.name} dtype={self.dtype} varstr={self.varstr} value={self.value}'

    def __mul__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        if self.dtype==DT.float:
            new_const.varstr = '(' + self.varstr + ' * (float)' + str(other) + ')'
        else:
            new_const.varstr = '(' + self.varstr + ' * ' + str(other) + ')'
        if self.value is not None:
            if isinstance(other, Var):
                if other.value is not None:
                    new_const.value = self.value * other.value
            elif isinstance(other, (float, int)):
                new_const.value = self.value * other
            else:
                raise Exception(f'Var computation only supports [Var, float, int], but got {type(other)}')
        return new_const

    def __rmul__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        if self.dtype==DT.float:
            new_const.varstr = '((float)' + str(other) + ' * ' + self.varstr + ')'
        else:
            new_const.varstr = '(' + str(other) + ' * ' + self.varstr + ')'
        if self.value is not None:
            if isinstance(other, Var):
                if other.value is not None:
                    new_const.value = self.value * other.value
            elif isinstance(other, (float, int)):
                new_const.value = self.value * other
            else:
                raise Exception(f'Var computation only supports [Var, float, int], but got {type(other)}')
        return new_const

    def __add__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + self.varstr + ' + ' + str(other) + ')'
        if self.value is not None:
            if isinstance(other, Var):
                if other.value is not None:
                    new_const.value = self.value + other.value
            elif isinstance(other, (float, int)):
                new_const.value = self.value + other
            else:
                raise Exception(f'Var computation only supports [Var, float, int], but got {type(other)}')
        return new_const

    def __radd__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + str(other) + ' + ' + self.varstr + ')'
        if self.value is not None:
            if isinstance(other, Var):
                if other.value is not None:
                    new_const.value = self.value + other.value
            elif isinstance(other, (float, int)):
                new_const.value = self.value + other
            else:
                raise Exception(f'Var computation only supports [Var, float, int], but got {type(other)}')
        return new_const

    def __sub__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + self.varstr + ' - ' + str(other) + ')'
        if self.value is not None:
            if isinstance(other, Var):
                if other.value is not None:
                    new_const.value = self.value - other.value
            elif isinstance(other, (float, int)):
                new_const.value = self.value - other
            else:
                raise Exception(f'Var computation only supports [Var, float, int], but got {type(other)}')
        return new_const

    def __rsub__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + str(other) + ' - ' + self.varstr + ')'
        if self.value is not None:
            if isinstance(other, Var):
                if other.value is not None:
                    new_const.value = other.value - self.value
            elif isinstance(other, (float, int)):
                new_const.value = other - self.value
            else:
                raise Exception(f'Var computation only supports [Var, float, int], but got {type(other)}')
        return new_const

    def __truediv__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', DT.float, auto_declare=False)
        new_const.varstr = '((float)' + self.varstr + ' / ' + '(float)' + str(other) + ')'
        if self.value is not None:
            if isinstance(other, Var):
                if other.value is not None:
                    new_const.value = self.value / other.value
            elif isinstance(other, (float, int)):
                new_const.value = self.value / other
            else:
                raise Exception(f'Var computation only supports [Var, float, int], but got {type(other)}')
        return new_const

    def __rtruediv__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', DT.float, auto_declare=False)
        new_const.varstr = f'((float)' + str(other) + ' / (float)' + self.varstr + ')'
        if self.value is not None:
            if isinstance(other, Var):
                if other.value is not None:
                    new_const.value = other.value / self.value
            elif isinstance(other, (float, int)):
                new_const.value = other / self.value
            else:
                raise Exception(f'Var computation only supports [Var, float, int], but got {type(other)}')
        return new_const

    def __floordiv__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', DT.int, auto_declare=False)
        new_const.varstr = '((int)' + self.varstr + ' / (int)' + str(other) + ')'
        if self.value is not None:
            if isinstance(other, Var):
                if other.value is not None:
                    new_const.value = self.value // other.value
            elif isinstance(other, (float, int)):
                new_const.value = self.value // other
            else:
                raise Exception(f'Var computation only supports [Var, float, int], but got {type(other)}')
        return new_const

    def __rfloordiv__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', DT.int, auto_declare=False)
        new_const.varstr = '((int)' + str(other) + ' / (int)' + self.varstr + ')'
        if self.value is not None:
            if isinstance(other, Var):
                if other.value is not None:
                    new_const.value = other.value // self.value
            elif isinstance(other, (float, int)):
                new_const.value = other // self.value
            else:
                raise Exception(f'Var computation only supports [Var, float, int], but got {type(other)}')
        return new_const

    def __mod__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + self.varstr + ' % ' + str(other) + ')'
        if self.value is not None:
            if isinstance(other, Var):
                if other.value is not None:
                    new_const.value = self.value % other.value
            elif isinstance(other, (float, int)):
                new_const.value = self.value % other
            else:
                raise Exception(f'Var computation only supports [Var, float, int], but got {type(other)}')
        return new_const

    def __rmod__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + str(other) + ' % ' + self.varstr + ')'
        if self.value is not None:
            if isinstance(other, Var):
                if other.value is not None:
                    new_const.value = other.value % self.value
            elif isinstance(other, (float, int)):
                new_const.value = other % self.value
            else:
                raise Exception(f'Var computation only supports [Var, float, int], but got {type(other)}')
        return new_const

    def __eq__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + self.varstr + ' == ' + str(other) + ')'
        return new_const

    def __lt__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + self.varstr + ' < ' + str(other) + ')'
        return new_const

    def __gt__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + self.varstr + ' > ' + str(other) + ')'
        return new_const

    def __le__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + self.varstr + ' <= ' + str(other) + ')'
        return new_const

    def __ge__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + self.varstr + ' >= ' + str(other) + ')'
        return new_const

    def __ne__(self, other: Union['Var', int, float]):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + self.varstr + ' != ' + str(other) + ')'
        return new_const

    def __and__(self, other: 'Var'):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + self.varstr + ' && ' + str(other) + ')'
        return new_const

    def __or__(self, other: 'Var'):
        new_const = Var('tmp_const', self.dtype, auto_declare=False)
        new_const.varstr = '(' + self.varstr + ' || ' + str(other) + ')'
        return new_const

    def __bool__(self):
        return True
