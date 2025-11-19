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
from typing import Union

from .instruction import Instruction
from .tensor import Tensor
from .var import Var
from .. import context


class Tuple(Var):
    dtypes: list[str]

    def __init__(self, dtypes: list[str], is_declare: bool = True):
        self.dtypes = dtypes  # list of types when stored in C++
        super(Tuple, self).__init__(self.get_type(), is_declare)

    def __getitem__(self, idx: int) -> Union[Tensor, Var]:
        if context.active_module is None:
            raise Exception()
        if idx < 0:
            idx = idx + len(self.dtypes)
        if self.dtypes[idx] == "Tensor":
            res = context.active_module.create_tensor()
        else:
            res = context.active_module.create_var()
        context.active_module.add_inst(Instruction('tuple_get', [self, idx], res))
        return res

    def __setitem__(self, idx: int, data: Union[Tensor, Var, int]):
        raise NotImplementedError()

    def get_type(self) -> str:
        return f"std::tuple<{', '.join(self.dtypes)}>"
