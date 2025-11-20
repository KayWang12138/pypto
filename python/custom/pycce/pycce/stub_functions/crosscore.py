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
from ..utils import Instruction, PipeInst, PIPE


def cube_ready(flag: int=0, pipe: PipeInst=PIPE.FIX):
    assert pipe in [PIPE.MTE2, PIPE.MTE1, PIPE.FIX, PIPE.M]
    if context.active_cube is not None:
        context.active_cube.append(Instruction('CUBEREADY', flag=flag, pipe=pipe))
    else:
        raise NotImplementedError

def wait_vec(flag: int=0, pipe: PipeInst=PIPE.S):
    assert pipe in [PIPE.MTE2, PIPE.MTE1, PIPE.FIX, PIPE.M, PIPE.S]
    if context.active_cube is not None:
        context.active_cube.append(Instruction('WAITVEC', flag=flag, pipe=pipe))
    else:
        raise NotImplementedError

def vec_ready(flag: int=0, pipe: PipeInst=PIPE.MTE3):
    assert pipe in [PIPE.MTE2, PIPE.MTE3]
    if context.active_vec is not None:
        context.active_vec.append(Instruction('VECREADY', flag=flag, pipe=pipe))
    else:
        raise NotImplementedError

def wait_cube(flag: int=0, pipe: PipeInst=PIPE.S):
    assert pipe in [PIPE.MTE2, PIPE.MTE3, PIPE.S]
    if context.active_vec is not None:
        context.active_vec.append(Instruction('WAITCUBE', flag=flag, pipe=pipe))
    else:
        raise NotImplementedError

def allcube_ready(flag: int=0, pipe: PipeInst=PIPE.FIX):
    assert pipe in [PIPE.MTE2, PIPE.MTE1, PIPE.FIX, PIPE.M]
    if context.active_cube is not None:
        context.active_cube.append(Instruction('ALLCUBEREADY', flag=flag, pipe=pipe))
    else:
        raise NotImplementedError

def allcube_wait(flag: int=0, pipe: PipeInst=PIPE.S):
    assert pipe in [PIPE.MTE2, PIPE.MTE1, PIPE.FIX, PIPE.M, PIPE.S]
    if context.active_cube is not None:
        context.active_cube.append(Instruction('ALLCUBEWAIT', flag=flag, pipe=pipe))
    else:
        raise NotImplementedError

def allvec_wait(flag: int=0, pipe: PipeInst=PIPE.S):
    assert pipe in [PIPE.MTE2, PIPE.MTE3, PIPE.S]
    if context.active_vec is not None:
        context.active_vec.append(Instruction('ALLVECWAIT', flag=flag, pipe=pipe))
    else:
        raise NotImplementedError

def allvec_ready(flag: int=0, pipe: PipeInst=PIPE.MTE3):
    assert pipe in [PIPE.MTE2, PIPE.MTE3]
    if context.active_vec is not None:
        context.active_vec.append(Instruction('ALLVECREADY', flag=flag, pipe=pipe))
    else:
        raise NotImplementedError





