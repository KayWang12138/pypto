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
from .datatype import DTYPE
from .position import Pos, Position
from .utilfuncs import sizeof
from .instruction import Instruction
from .. import context
from typing import Optional, Union, Sequence
from .var import Var


def ComputeOffset(idx: list[Union[Var, int]], shape: list[Union[Var, int]]) -> Var:
    assert len(idx)==len(shape)
    offset = None
    strides = []
    for i in range(1, len(shape)):
        tmp = None
        for j in range(i, len(shape)):
            if tmp is None:
                tmp = shape[j]
            else:
                tmp = tmp * shape[j]
        strides.append(tmp)
    strides.append(1)

    for i,s in zip(idx, strides):
        if isinstance(i, int) and i==0:
            continue
        if offset is None:
            offset = i*s
        else:
            offset = offset + i*s

    assert offset is not None
    return offset


class GMTensor():
    name: str
    dtype: DTYPE
    length: Optional[Union[int, Var]]
    shape: Optional[Sequence[Union[Var, int]]]
    is_output: bool

    def __init__(self, name: str, dtype: DTYPE, shape: Optional[Sequence[Union[Var, int]]]=None, length: Optional[Union[int, Var]]=None, is_output: bool=False):
        assert not (shape and length), 'Can only set either one of shape or length'
        self.name = name
        self.dtype = dtype
        if shape is not None:
            assert isinstance(shape, Sequence)
            self.length = shape[0]
            for i in range(1, len(shape)):
                self.length = self.length * shape[i]
        else:
            self.length = length
        self.shape = shape
        self.is_output = is_output

    def __getitem__(self, idx: Union[int, Var, Sequence[Union[int, Var]]]):
        from ..tileop import TileOpModule
        if isinstance(idx, (int, Var)) and self.shape is None:
            new_tensor = GMTensor(f'{self.name}[{str(idx)}]', self.dtype)
            return new_tensor
        elif isinstance(self.shape, Sequence):
            if isinstance(idx, (int, Var)):
                idx_list = [idx,]
            else:
                idx_list = idx
            shape = []
            index = []
            active_mod = context.active_cube or context.active_vec
            if active_mod is None or isinstance(active_mod, TileOpModule):
                raise TypeError()
            for i in range(len(self.shape)):
                if i==0:
                    shape.append(-1)
                else:
                    s = self.shape[i]
                    if isinstance(s, Var):
                        if s not in active_mod.var_mapping:
                            raise KeyError('GMTensor shape must be assigned to class variables (i.e. self.xxx)')
                        shape.append(active_mod.var_mapping[s])
                    elif isinstance(s, int):
                        shape.append(s)
                    else:
                        raise TypeError()
                # process index
                if i<len(idx_list):
                    index.append(idx_list[i])
                else:
                    index.append(0)
            offset = ComputeOffset(index, shape)
            new_tensor = GMTensor(f'{self.name}[{str(offset)}]', self.dtype)
            return new_tensor
        else:
            raise TypeError()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'[GMTensor] name={self.name} dtype={self.dtype}'

    def load_bin(self, file: str):
        ...


class Tensor():
    dtype: DTYPE
    length: Union[int, Var]
    name: str
    pos: Pos
    is_output: bool

    def __init__(self, dtype: DTYPE, length: Union[int, Var], position: Optional[Pos] = None,
                 auto_declare: bool=True, is_output: bool = False):
        self.dtype = dtype
        self.length = length
        self.name = ''
        self.is_output = is_output
        # get position
        if position is not None:
            self.pos = position
        else:
            if context.active_cube is not None:
                self.pos = Position.L1
            elif context.active_vec is not None:
                self.pos = Position.UB
            else:
                # >>> Tensor created outside will be regarded as UB
                self.pos = Position.UB
        # assign to module
        active_mod = context.active_cube or context.active_vec
        if active_mod is not None and auto_declare:
            if active_mod._curr_phase=='computing':
                active_mod.add_buffer(self)
        if active_mod is None:
            self.name = 'tsr%d'%context.tsr_idx
            context.tsr_idx += 1

    def reinterpret(self, dtype: DTYPE):
        g_vec = context.active_vec
        g_cube = context.active_cube
        curr_mod = g_vec or g_cube
        assert curr_mod is not None
        new_tsr = Tensor(dtype, self.length*sizeof(self.dtype)//sizeof(dtype))
        new_tsr.pos = self.pos
        idx = curr_mod._tmp_idx
        curr_mod._tmp_idx += 1
        new_tsr.name = 'tmptsr_%d'%idx
        curr_mod.append(Instruction('REINTERPRET', dst=new_tsr, src=self))
        return new_tsr

    def __getitem__(self, idx: Union[int, Var]):
        from ..tileop import TileOpModule
        new_tensor = Tensor(self.dtype, self.length)
        if isinstance(context.active_vec, TileOpModule):
            new_tensor.name = self.name + f'[{str(idx)}]'
        else:
            new_tensor.name = self.name + f'[{str(idx)}]'
        new_tensor.pos = self.pos
        return new_tensor

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'[Tensor] name={self.name} dtype={self.dtype} length={str(self.length)} pos={self.pos}'


class DBuff():
    dtype: DTYPE
    length: Union[int, Var]
    name: str
    pos: Pos

    def __init__(self, dtype: DTYPE, length: Union[int, Var], position: Optional[Pos] = None):
        self.dtype = dtype
        self.length = length
        self.name = ''
        # get position
        if position is not None:
            self.pos = position
        else:
            if context.active_cube is not None:
                self.pos = Position.L1
            elif context.active_vec is not None:
                self.pos = Position.UB
            else:
                raise Exception('DBuff must be created inside vector or cube')
        # assign to module
        active_mod = context.active_cube or context.active_vec
        if active_mod is not None:
            if active_mod._curr_phase=='computing':
                active_mod.add_buffer(self)

    def get(self, i: Union[int, Var]):
        new_tensor = Tensor(self.dtype, self.length, auto_declare=False)
        new_tensor.name = self.name + f'.get({str(i)})'
        new_tensor.pos = self.pos
        return new_tensor

    def __getitem__(self, idx: Union[int, Var]):
        return self.get(idx)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'[DBuff] name={self.name} dtype={self.dtype} length={str(self.length)}'

