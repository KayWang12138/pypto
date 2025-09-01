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

from .instruction import Instruction
from .std_vec import Vector
from .tensor import Tensor
from .utilfuncs import get_vartype_str
from .. import context


class CustStruct():
    _previdx: list[int]
    idx: int = -1

    def __init__(self, **kwargs):
        self._previdx = []

        self._configs = {}  # map var name to type

        annotations = self.__class__.__dict__.get('__annotations__', {})

        for name, type_ in annotations.items():
            if not (type_ in [int, bool, float, str, Tensor, Vector] \
                or isinstance(type_, Vector) or issubclass(type_, CustStruct)):
                raise Exception()

            if name in kwargs:
                raise NotImplementedError("Only used to define cust struct")
            self._configs[name] = type_

        for name in kwargs.keys():
            if not hasattr(self, name):
                raise TypeError(f"Unexpected argument '{name}'")

    def __str__(self):
        return f'CustStruct[{self.idx}]'

    def __repr__(self):
        instance_fields = []
        for name in getattr(self.__class__, '__annotations__', {}):
            instance_fields.append(f"{name}={getattr(self, name)!r}")
        return f"{self.__class__.__name__}({', '.join(instance_fields)})"

    def __getattr__(self, config: str):
        if config in self._configs:
            type_ = self._configs[config]
            if context.active_module is None:
                raise Exception()
            if type_ in [int, bool, float, str]:
                new_attr = context.active_module.create_var(dtype=get_vartype_str(type_))
                context.active_module.add_inst(Instruction('get_attr', [self, config], new_attr))
            elif type_ in [Tensor]:
                new_attr = context.active_module.create_tensor()
                context.active_module.add_inst(Instruction('get_attr_tensor', [self, config], new_attr))
            elif type_ in [Vector]:
                new_attr = context.active_module.create_vector(dtypes=['int'])
                context.active_module.add_inst(Instruction('get_attr_vector', [self, config], new_attr))
            elif isinstance(type_, Vector):
                new_attr = context.active_module.create_vector(dtypes=type_.dtype)
                context.active_module.add_inst(Instruction('get_attr_vector', [self, config], new_attr))
            return new_attr
        else:
            raise NotImplementedError("Attribute doesn't exist")

    def get_configs(self):
        return self._configs

    def get_idx(self):
        return self.idx

    def set_idx(self, idx: int):
        self.idx = idx

    def push(self):
        self._previdx.append(self.idx)

    def pop(self):
        self.idx = self._previdx.pop(-1)

    def get_type(self):
        return self.__class__.__name__
