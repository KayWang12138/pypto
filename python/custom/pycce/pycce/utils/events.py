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
from typing import Union, TYPE_CHECKING
from .pipe import PipeInst
from .instruction import Instruction

if TYPE_CHECKING:
    from ..cube import CubeModule
    from ..vec import VecModule


class SEvent():
    def __init__(self, src: PipeInst, dst: PipeInst):
        self.src = src
        self.dst = dst

    def set_module(self, module: Union['CubeModule', 'VecModule']):
        self.module = module

    def set_name(self, name: str):
        self.name = name

    def set_id(self, idd: int):
        self.idd = idd

    def set(self):
        self.module.append(Instruction('EVENTSET', name=self.name))

    def wait(self):
        self.module.append(Instruction('EVENTWAIT', name=self.name))

    def setall(self):
        self.module.append(Instruction('EVENTSETALL', name=self.name))

    def release(self):
        self.module.append(Instruction('EVENTRELEASE', name=self.name))

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'[SEvent]  name:{self.name}  src:{self.src}  dst:{self.dst}  id:{self.idd}'


class DEvent():
    def __init__(self, src: PipeInst, dst: PipeInst):
        self.src = src
        self.dst = dst

    def set_module(self, module: Union['CubeModule', 'VecModule']):
        self.module = module

    def set_name(self, name: str):
        self.name = name

    def set_id(self, idd: int):
        self.idd = idd

    def set(self):
        self.module.append(Instruction('EVENTSET', name=self.name))

    def wait(self):
        self.module.append(Instruction('EVENTWAIT', name=self.name))

    def setall(self):
        self.module.append(Instruction('EVENTSETALL', name=self.name))

    def release(self):
        self.module.append(Instruction('EVENTRELEASE', name=self.name))

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'[DEvent]  name:{self.name}  src:{self.src}  dst:{self.dst}  id:{self.idd}'
