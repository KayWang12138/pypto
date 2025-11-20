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
from ...utils import Var, Instruction, DT
from ... import context
from typing import Union


def set_mask(maskHigh: Union[int, Var], maskLow: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None
    if isinstance(maskHigh, Var):
        assert maskHigh.dtype==DT.uint64
    if isinstance(maskLow, Var):
        assert maskLow.dtype==DT.uint64

    g_vec.append(Instruction('SETMASK', low=maskLow, high=maskHigh))


def reset_mask():
    g_vec = context.active_vec
    assert g_vec is not None

    g_vec.append(Instruction('RESETMASK'))


def set_continuous_mask(count: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None
    if isinstance(count, Var) and count.dtype != DT.uint64:
        raise TypeError("The dtype of Var count in set_continuous_mask must be uint64!")

    g_vec.append(Instruction('SET_CONTINUOUS_MASK', count = count))
