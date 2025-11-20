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
from .instruction import Instruction
from .utilfuncs import get_vartype_str
from .. import context


class ConfigMap():
    idx: int = -1
    _previdx: list[int]

    def __init__(self, **kwargs):
        self._previdx = []

        self._configs = {}  # map var name to type

        annotations = getattr(self.__class__, '__annotations__', {})

        for name, type_ in annotations.items():
            if name in kwargs:
                raise NotImplementedError("Modifying config is not supported")
            if name in self.__class__.__dict__:
                delattr(self.__class__, name)
            self._configs[name] = type_

        for name in kwargs.keys():
            if not hasattr(self, name):
                raise TypeError(f"Unexpected argument '{name}'")
        super().__init__()

    def __repr__(self):
        fields = []
        for name in getattr(self.__class__, '__annotations__', {}):
            fields.append(f"{name}={getattr(self, name)!r}")
        return f"{self.__class__.__name__}({', '.join(fields)})"

    def __getattr__(self, n: str):
        if n in self._configs:
            type_ = self._configs[n]
            if context.active_module is None:
                raise Exception()
            if type_ not in [int, float, bool, str]:
                raise Exception()
            new_var = context.active_module.create_var(dtype=get_vartype_str(type_))
            context.active_module.add_inst(Instruction('get_attr_config', [self, n, type_], new_var))
            return new_var
        else:
            raise NotImplementedError("Attribute doesn't exist")

    @classmethod
    def get_type(cls):
        return 'std::map<std::string, std::variant<bool, int, float, std::string> >'

    def set_idx(self, idx: int):
        self.idx = idx

    def push(self):
        self._previdx.append(self.idx)

    def pop(self):
        self.idx = self._previdx.pop(-1)


