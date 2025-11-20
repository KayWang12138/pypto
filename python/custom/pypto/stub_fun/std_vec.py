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
from typing import Union

from .. import context
from ..utils import Tensor, Vector, Var, Instruction


def vec_size(a: Vector) -> Var:
    if context.active_module is None:
        raise Exception()
    result = context.active_module.create_var(dtype='int')
    context.active_module.add_inst(Instruction('get_vec_size', [a], result))
    return result


def vec_emplace_back(a: Vector, val: Union[Var, int, Tensor]):
    if context.active_module is None:
        raise Exception()
    context.active_module.add_inst(Instruction('vec_emplace_back', [a, val], None))
