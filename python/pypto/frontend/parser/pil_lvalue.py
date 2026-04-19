#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


import abc

from enum import Enum
from collections import UserList, UserDict
from collections.abc import Iterable


from irbuilder import IRBuilder
from pil_runtime import (
    ScalarValue, TensorValue,
    is_const, is_scalar, is_tensor)


class LValueHolder(abc.ABC):
    pass

LValue = LValueHolder | object

class LValueMissing:
    pass

class LValueTypeMismatch:
    def __init__(self, then_value, else_value):
        self._then_value = then_value
        self._else_value = else_value

class LValueList(LValueHolder, UserList):

    def __init__(self, initlist=None):
        super().__init__(initlist)

    def __repr__(self):
        return super().__repr__()

    def __lt__(self, other):
        return super().__lt__(other)

    def __le__(self, other):
        return super().__le__(other)

    def __eq__(self, other):
        return super().__eq__(other)

    def __gt__(self, other):
        return super().__gt__(other)

    def __ge__(self, other):
        return super().__ge__(other)

    def __contains__(self, item: LValue) -> bool:
        return super().__contains__(item)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, i) -> LValue:
        return super().__getitem__(i)

    def __setitem__(self, i, item: LValue) -> None:
        super().__setitem__(i, item)

    def __delitem__(self, i):
        super().__delitem__(i)

    def __add__(self, other):
        return super().__add__(other)

    def __radd__(self, other):
        return super().__radd__(other)

    def __iadd__(self, other):
        return super().__iadd__(other)

    def __mul__(self, n):
        return super().__mul__(n)

    def __rmul__(self, n):
        return super().__rmul__(n)

    def __imul__(self, n):
        return super().__imul__(n)

    def __copy__(self):
        return super().__copy__()

    def append(self, item: LValue) -> None:
        super().append(item)

    def insert(self, i: int, item: LValue) -> None:
        super().insert(i, item)

    def pop(self, i: int = -1) -> LValue:
        return super().pop(i)

    def remove(self, item: LValue) -> None:
        super().remove(item)

    def clear(self) -> None:
        super().clear()

    def copy(self) -> 'LValueList':
        return super().copy()

    def count(self, item: LValue) -> int:
        return super().count(item)

    def index(self, item: LValue, *args) -> int:
        return super().index(item, *args)

    def reverse(self) -> None:
        super().reverse()

    def sort(self, /, *args, **kwds) -> None:
        super().sort(*args, **kwds)

    def extend(self, other: 'Iterable[LValue]') -> None:
        super().extend(other)


class LValueDict(LValueHolder, UserDict):

    def __init__(self, dict=None, /, **kwargs):
        super().__init__(dict, **kwargs)

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, key) -> LValue:
        return super().__getitem__(key)

    def __setitem__(self, key, item: LValue) -> None:
        super().__setitem__(key, item)

    def __delitem__(self, key) -> None:
        super().__delitem__(key)

    def __iter__(self):
        return super().__iter__()

    def __contains__(self, key) -> bool:
        return super().__contains__(key)

    def __repr__(self) -> str:
        return super().__repr__()

    def __or__(self, other) -> 'LValueDict':
        return super().__or__(other)

    def __ror__(self, other) -> 'LValueDict':
        return super().__ror__(other)

    def __ior__(self, other) -> 'LValueDict':
        return super().__ior__(other)

    def __copy__(self) -> 'LValueDict':
        return super().__copy__()

    def get(self, key, default=None) -> LValue | None:
        return super().get(key, default)

    def copy(self) -> 'LValueDict':
        return super().copy()

    @classmethod
    def fromkeys(cls, iterable, value=None) -> 'LValueDict':
        return super().fromkeys(iterable, value)


class LValuePath:
    def __init__(self, seglist: list[str | int | object]):
        self._seglist = tuple(seglist)

    def __iter__(self):
        return iter(self._seglist)

    def __getitem__(self, index):
        return self._seglist[index]

    def __repr__(self):
        return repr(self._seglist)

    def __hash__(self):
        return hash(self._seglist)

    def __eq__(self, other):
        return self._seglist == other._seglist

class LValueMapping:

    def __init__(self):
        self._var_dict = {}

    def set_var(self, var: str, value: LValue):
        self._var_dict[var] = value

    def get_var(self, var: str) -> LValue:
        return self._var_dict[var]

    def clone(self) -> 'LValueMapping':
        def _clone_lvalue(value: LValue) -> LValue:
            if isinstance(value, LValueList):
                return LValueList([_clone_lvalue(item) for item in value])
            if isinstance(value, LValueDict):
                return LValueDict({k: _clone_lvalue(v) for k, v in value.items()})
            return value

        new_mapping = LValueMapping()
        new_mapping._var_dict = {k: _clone_lvalue(v) for k, v in self._var_dict.items()}
        return new_mapping

    def clone_symbolic(self) -> 'LValueMapping':
        def _clone_symbolic_lvalue(value: LValue | object) -> LValue:
            if isinstance(value, LValueList):
                return LValueList([_clone_symbolic_lvalue(item) for item in value])
            if isinstance(value, LValueDict):
                return LValueDict({k: _clone_symbolic_lvalue(v) for k, v in value.items()})
            if is_const(value) or is_scalar(value):
                return ScalarValue.create()
            if is_tensor(value):
                return TensorValue.create()
            return value

        new_mapping = LValueMapping()
        new_mapping._var_dict = {k: _clone_symbolic_lvalue(v) for k, v in self._var_dict.items()}
        return new_mapping

    def get_path_set(self) -> set[LValuePath]:
        path_set = set()
        def _travel(prefix: list, value: LValue):
            if isinstance(value, LValueList):
                for i, item in enumerate(value):
                    _travel(prefix + [i], item)
            elif isinstance(value, LValueDict):
                for k, v in value.items():
                    _travel(prefix + [k], v)
            else:
                path_set.add(LValuePath(prefix))
        for var, value in self._var_dict.items():
            _travel([var], value)
        return path_set

    def get_value(self, path: LValuePath, default: LValue = LValueMissing) -> LValue:
        segs = list(path)
        if segs[0] not in self._var_dict:
            return default
        value = self._var_dict[segs[0]]
        for seg in segs[1:]:
            if isinstance(value, LValueList):
                if seg < 0 or seg >= len(value):
                    return default
            elif isinstance(value, LValueDict):
                if seg not in value:
                    return default
            value = value[seg]
        return value

    def set_value(self, path: LValuePath, value):
        segs = list(path)
        if len(segs) == 1:
            self._var_dict[segs[0]] = value
            return
        container = self._var_dict[segs[0]]
        for seg in segs[1:-1]:
            container = container[seg]
        container[segs[-1]] = value

    def __repr__(self):
        return repr(self._var_dict)

g_context = LValueMapping()
