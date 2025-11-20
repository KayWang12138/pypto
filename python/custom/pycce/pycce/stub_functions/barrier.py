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
from .. import context
from ..utils import Instruction, PIPE, PipeInst


def barrier(pipe: PipeInst):
    active_mod = context.active_cube or context.active_vec
    assert active_mod is not None, 'Barrier should run in either cube forward or vec forward'
    active_mod.append(Instruction('BAR', pipe=pipe))


def bar_m():
    barrier(PIPE.M)

def bar_v():
    barrier(PIPE.V)

def bar_mte2():
    barrier(PIPE.MTE2)

def bar_mte1():
    barrier(PIPE.MTE1)

def bar_mte3():
    barrier(PIPE.MTE3)

def bar_fix():
    barrier(PIPE.FIX)

def bar_all():
    barrier(PIPE.ALL)

