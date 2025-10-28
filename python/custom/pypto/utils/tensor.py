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

from types import EllipsisType
from typing import Union, Optional, Sequence, TYPE_CHECKING, Tuple, Any
import numpy as np
from .enums import DATATYPE
from .instruction import Instruction
from .shape import Shape
from .var import Var
from .. import context


if TYPE_CHECKING:
    pass


class Tensor():
    idx: int = -1
    is_transpose: bool = False
    _previdx: list[int]
    _prev_insts: list[Optional['Instruction']]
    _shape: Optional[Union[list[Union[Var, int]], 'Shape']]
    dtype: Optional[Union[Var, str]]
    is_output: bool = False
    data: Optional[np.ndarray]

    def __init__(self, shape: Optional[Union[list[Union[Var, int]], 'Shape']] = None,
                 dtype: Optional[Union[Var, str]] = None, is_output: bool = False, force_declare: bool = False):
        self.data = None
        self._previdx = []
        self._prev_insts = [None, ]
        self._shape, self.dtype = shape, dtype
        self.is_output = is_output
        if shape is not None and dtype is not None:
            if context.active_module is not None:
                self.idx = context.active_module.args_counter
                context.active_module.args_counter += 1
                src = [shape, dtype]
                context.active_module.add_inst(Instruction('new_tensor', src, self))
        elif force_declare:
            if context.active_module is not None:
                self.idx = context.active_module.args_counter
                context.active_module.args_counter += 1
                context.active_module.add_inst(Instruction('declare', [], self))

    def __str__(self):
        return f'Tensor[{self.idx}]'

    def __repr__(self):
        return str(self)

    def __add__(self, other: Union['Tensor', Var, int, float]):
        from ..stub_fun import add, adds
        if isinstance(other, Tensor):
            res = add(self, other)
            return res
        elif isinstance(other, (Var, int, float)):
            res = adds(self, other)
            return res
        else:
            raise TypeError(f'Not supported type for tensor add {type(other)}')

    def __radd__(self, other: Union['Tensor', Var, int, float]):
        return self + other

    def __iadd__(self, other: Union['Tensor', Var, int, float]):
        from ..stub_fun import inplace_add, inplace_adds
        if isinstance(other, Tensor):
            res = inplace_add(self, other)
            return res
        elif isinstance(other, (Var, int, float)):
            res = inplace_adds(self, other)
            return res
        else:
            raise TypeError(f'Not supported type for tensor add {type(other)}')

    def __sub__(self, other: Union['Tensor', Var, int, float]):
        from ..stub_fun import sub
        if isinstance(other, Tensor):
            res = sub(self, other)
            return res
        elif isinstance(other, (Var, int, float)):
            res = sub(self, other)
            return res
        else:
            raise TypeError(f'Not supported type for tensor sub {type(other)}')

    def __rsub__(self, other: Union['Tensor', Var, int, float]):
        return (self - other) * (-1)

    def __isub__(self, other: Union['Tensor', Var, int, float]):
        from ..stub_fun import inplace_sub, inplace_subs
        if isinstance(other, Tensor):
            res = inplace_sub(self, other)
            return res
        elif isinstance(other, (Var, int, float)):
            res = inplace_subs(self, other)
            return res
        else:
            raise TypeError(f'Not supported type for tensor sub {type(other)}')

    def __mul__(self, other: Union['Tensor', Var, int, float]):
        from ..stub_fun import mul, muls
        if isinstance(other, Tensor):
            res = mul(self, other)
            return res
        elif isinstance(other, (Var, int, float)):
            res = muls(self, other)
            return res
        else:
            raise TypeError(f'Not supported type for tensor mul {type(other)}')

    def __rmul__(self, other: Union['Tensor', Var, int, float]):
        return self * other

    def __imul__(self, other: Union['Tensor', Var, int, float]):
        from ..stub_fun import inplace_mul, inplace_muls
        if isinstance(other, Tensor):
            res = inplace_mul(self, other)
            return res
        elif isinstance(other, (Var, int, float)):
            res = inplace_muls(self, other)
            return res
        else:
            raise TypeError(f'Not supported type for tensor mul {type(other)}')

    def __truediv__(self, other: Union['Tensor', Var, int, float]):
        from ..stub_fun import div, divs
        if isinstance(other, Tensor):
            res = div(self, other)
            return res
        elif isinstance(other, (Var, int, float)):
            res = divs(self, other)
            return res
        else:
            raise TypeError(f'Not supported type for tensor div {type(other)}')

    def __itruediv__(self, other: Union['Tensor', Var, int, float]):
        from ..stub_fun import inplace_div, inplace_divs
        if isinstance(other, Tensor):
            res = inplace_div(self, other)
            return res
        elif isinstance(other, (Var, int, float)):
            res = inplace_divs(self, other)
            return res
        else:
            raise TypeError(f'Not supported type for tensor div {type(other)}')

    def __lt__(self, other: Union['Tensor', Var, int, float]):
        from ..stub_fun import lt
        if isinstance(other, (Tensor, Var, int, float)):
            return lt(self, other)
        else:
            raise TypeError(f'Not supported type for tensor less than {type(other)}')

    def __le__(self, other: Union['Tensor', Var, int, float]):
        from ..stub_fun import le
        if isinstance(other, (Tensor, Var, int, float)):
            return le(self, other)
        else:
            raise TypeError(f'Not supported type for tensor less than {type(other)}')

    def __gt__(self, other: Union['Tensor', Var, int, float]):
        from ..stub_fun import gt
        if isinstance(other, (Tensor, Var, int, float)):
            return gt(self, other)
        else:
            raise TypeError(f'Not supported type for tensor less than {type(other)}')

    def __ge__(self, other: Union['Tensor', Var, int, float]):
        from ..stub_fun import ge
        if isinstance(other, (Tensor, Var, int, float)):
            return ge(self, other)
        else:
            raise TypeError(f'Not supported type for tensor less than {type(other)}')

    def __getitem__(self, indexes: Sequence[Optional[Union[Var, int, slice, EllipsisType]]]):
        from ..flowcontrol import Loop
        from ..stub_fun import add_comment, reshape, less_than_zero_inv

        inst_name = f"{self} getitem with index {indexes}"

        indexes = list(indexes)
        if indexes.count(Ellipsis) == 0:
            indexes.append(Ellipsis)
        if indexes.count(Ellipsis) != 1:
            raise Exception("Too many ellipsis, Not possible to decipher!")

        ellipsis_ind = indexes.index(Ellipsis)

        def update_front_back(indexes: Sequence[Optional[Union[Var, int, slice, EllipsisType]]]) \
                -> Optional[Tuple[Any, Union[int, Any], int, int]]:
            front_num, front_dim, back_num, back_dim = 0, 0, 0, 0
            for idx in indexes[:ellipsis_ind]:
                if idx is None:
                    front_num += 1
                elif isinstance(idx, (int, Var)):
                    front_dim += 1
                elif isinstance(idx, slice):
                    front_dim += 1
                    front_num += 1
                else:
                    raise Exception("Type not identified!")

            for idx in indexes[ellipsis_ind + 1: len(indexes)]:
                if idx is None:
                    back_num += 1
                elif isinstance(idx, (int, Var)):
                    back_dim += 1
                elif isinstance(idx, slice):
                    back_dim += 1
                    back_num += 1
                else:
                    raise Exception("Type not identified!")

            return front_num, front_dim, back_num, back_dim

        # front/back refers to before/after Ellipsis
        front_num, front_dim, back_num, back_dim = update_front_back(indexes)

        def update_shape(idx: Optional[Union[Var, int, slice, EllipsisType]], \
                         offset: Shape, shape1: Shape, shape2: Shape, front_dim2: int):
            if idx is None:
                shape2.emplace_back(1)
            elif isinstance(idx, (int, Var)):
                shape1.emplace_back(1)
                offset.emplace_back(less_than_zero_inv(idx, self.shape[front_dim2]))  # type: ignore
                front_dim2 += 1
            elif isinstance(idx, slice):
                start, stop, step = idx.start, idx.stop, idx.step
                if idx.step is not None:
                    raise Exception()
                if start is None:
                    start = 0
                if stop is None:
                    stop = self.shape[front_dim2]
                start = less_than_zero_inv(start, self.shape[front_dim2])  # type: ignore
                stop = less_than_zero_inv(stop, self.shape[front_dim2])  # type: ignore
                length = stop - start
                shape1.emplace_back(length)
                offset.emplace_back(start)
                shape2.emplace_back(length)
                front_dim2 += 1
            return front_dim2

        offset = Shape(size=0)
        shape1 = Shape(size=0)
        shape2 = Shape(size=0)
        front_dim2 = 0  # corresponding index in original tensor dimension

        for idx in indexes[:ellipsis_ind]:
            front_dim2 = update_shape(idx, offset, shape1, shape2, front_dim2)

        if not front_dim == front_dim2:
            raise Exception()
        with Loop(front_dim, self.shape.size() - back_dim) as i:
            x = self.shape[i]
            if not isinstance(x, Var):
                raise Exception()
            shape1.emplace_back(x)
            offset.emplace_back(0)
            shape2.emplace_back(x)
        if ellipsis_ind + 1 < len(indexes):
            front_dim2 = (front_dim2 - back_dim - front_dim) + self.shape.size()
            for idx in indexes[ellipsis_ind + 1: len(indexes)]:
                front_dim2 = update_shape(idx, offset, shape1, shape2, front_dim2)

        result = reshape(self.view(shape1, offset), shape2)
        add_comment("End of", inst_name)
        return result

    def __matmul__(self, other: 'Tensor'):
        from ..stub_fun import matmul
        if isinstance(other, Tensor):
            res = matmul(self, other, DATATYPE.fp32)
            return res
        else:
            raise TypeError('Only support matmul for tensor @ tensor')

    def __rmatmul__(self, other: 'Tensor'):
        from ..stub_fun import matmul
        if isinstance(other, Tensor):
            res = matmul(other, self, DATATYPE.fp32)
            return res
        else:
            raise TypeError('Only support matmul for tensor @ tensor')
        
    @property
    def shape(self):
        if context.active_module is None:
            raise Exception
        if self._shape is not None:
            new_shape = Shape(len(self._shape), self._shape)
        else:
            new_shape = Shape()
        new_shape.set_idx(context.active_module.args_counter)
        context.active_module.args_counter += 1
        context.active_module.add_inst(Instruction('get_shape', [self], new_shape))
        return new_shape
    
    def get_shape(self):
        return self._shape
        
    def load(self, filename: str):
        self.data = np.load(filename)
        return self.data
    
    def from_np(self, arr: Optional[np.ndarray]):
        self.data = arr
        return self

    def set_idx(self, idx: int):
        self.idx = idx

    def t(self):
        new_tensor = Tensor()
        new_tensor.idx = self.idx
        new_tensor.is_transpose = not self.is_transpose
        return new_tensor

    def push(self):
        self._previdx.append(self.idx)
        self._prev_insts.append(None)

    def pop(self):
        self.idx = self._previdx.pop(-1)
        self._prev_insts.pop(-1)

    def set_vec_tile_shapes(self, a: Union[Sequence[Union[Var, int]], 'Vector', Shape]):
        if context.active_module is None:
            raise Exception()
        if self._prev_insts[-1] is None:
            raise Exception()
        new_inst = Instruction('set_vec_tile_shapes', [a], None)
        context.active_module.insert_inst(new_inst, self._prev_insts[-1])
        return self

    def exp(self):
        from ..stub_fun import exp
        res = exp(self)
        return res

    def view(self, new_shape: Union[Shape, list[Union[int, Var]]], \
             new_offset: Union[Shape, list[Union[int, Var]]]):
        from ..stub_fun import view
        res = view(self, new_shape, new_offset)
        return res

    def dview(self, new_shape: Union[Shape, list[Union[int, Var]]], \
              new_offset: Union[Shape, list[Union[int, Var]]]):
        from ..stub_fun import dview
        res = dview(self, new_shape, new_offset)
        return res

    def dview_pad(self, new_shape: Union[Shape, list[Union[int, Var]]], \
                 new_offset: Union[Shape, list[Union[int, Var]]], block_size: Union[Shape, list[Union[int, Var]]]):
        from ..stub_fun import dview_pad
        res = dview_pad(self, new_shape, new_offset, block_size)
        return res

    def data_type(self):
        from ..stub_fun import data_type
        res = data_type(self)
        return res

    def astype(self, dtype: str):
        from ..stub_fun import cast
        res = cast(self, dtype)
        return res

    def assign(self, expr: 'Tensor'):
        if context.active_module is None:
            raise Exception()
        context.active_module.add_inst(Instruction('assign', [self, expr], None))
        return expr

    def row_max_single(self, axis=None):
        from ..stub_fun import row_max_single
        res = row_max_single(self, axis)
        return res

    def row_sum_single(self, axis=None):
        from ..stub_fun import row_sum_single
        res = row_sum_single(self, axis)
        return res

    def reciprocal(self):
        from ..stub_fun import reciprocal
        res = reciprocal(self)
        return res

    def transpose(self, dim0, dim1):
        from ..stub_fun import transpose
        res = transpose(self, [dim0, dim1])
        return res

    def reshape(self, *shape):
        from ..stub_fun import reshape
        if isinstance(shape[0], tuple):
            res = reshape(self, list(shape[0]))
        else:
            res = reshape(self, list(shape))
        self.eq(res)
        return res

    def masked_fill_(self, mask: 'Tensor', value: Union['Var', int, float]):
        if not isinstance(mask, Tensor):
            raise Exception()
        if not isinstance(value, (Var, int, float)):
            raise Exception()
        self -= mask * self
        self += mask * value