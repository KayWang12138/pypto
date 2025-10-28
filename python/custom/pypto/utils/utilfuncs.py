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

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .var import Var
    from .tensor import Tensor


def get_vartype_str(v: type) -> str:
    from .tensor import Tensor
    if v == int:
        return "int"
    elif v == float:
        return "float"
    elif v == bool:
        return "bool"
    elif v == str:
        return "std::string"
    elif v == Tensor:
        return "Tensor"
    else:
        raise NotImplementedError()


def get_obj_dtype(v: Union['Var', int, float, bool, str, 'Tensor']):
    from .var import Var
    if isinstance(v, Var):
        return v.dtype
    else:
        return get_vartype_str(type(v))
