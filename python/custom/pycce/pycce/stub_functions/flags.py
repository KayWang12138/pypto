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
from .. import context
from ..utils import Instruction, PipeInst, Var
from typing import Union


def setflag(src: PipeInst, dst: PipeInst, event_id: Union[int, Var]):
    active_mod = context.active_cube or context.active_vec
    assert active_mod is not None, 'setflag should run in either cube forward or vec forward'
    active_mod.append(Instruction('SETFLAG', src=src, dst=dst, idd=event_id))


def waitflag(src: PipeInst, dst: PipeInst, event_id: Union[int, Var]):
    active_mod = context.active_cube or context.active_vec
    assert active_mod is not None, 'waitflag should run in either cube forward or vec forward'
    active_mod.append(Instruction('WAITFLAG', src=src, dst=dst, idd=event_id))

