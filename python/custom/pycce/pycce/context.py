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
from typing import TYPE_CHECKING, Optional, Literal, Union


if TYPE_CHECKING:
    from .utils import Pos
    from .kernel import KernelBase
    from .vec import VecModule
    from .cube import CubeModule
    from .tileop import TileOpModule

active_vec: Optional[Union['VecModule', 'TileOpModule']] = None
active_cube: Optional['CubeModule'] = None
active_kernel: Optional['KernelBase'] = None
tsr_idx: int = 0

num_cores: int = 20
L1_size = 512*1024
L0A_size = 64*1024
L0B_size = 64*1024
L0C_size = 128*1024
UB_size = 192*1024
device_type = '910b3'

def get_max_size(pos: 'Pos'):
    from .utils import Position
    if pos==Position.UB:
        return UB_size
    if pos==Position.L1:
        return L1_size
    if pos==Position.L0A:
        return L0A_size
    if pos==Position.L0B:
        return L0B_size
    if pos==Position.L0C:
        return L0C_size
    return 0

def set_core_num(n: int):
    global num_cores
    num_cores = n

def set_device_type(device_type_i: Literal['david', '910b4', '910b3', '910b2', '910b1']):
    global num_cores, L1_size, L0A_size, L0B_size, L0C_size, UB_size, device_type
    if device_type_i=='david':
        num_cores = 32
        L1_size = 512*1024
        L0A_size = 64*1024
        L0B_size = 64*1024
        L0C_size = 256*1024
        UB_size = 256*1024
        device_type = device_type_i
    elif device_type_i=='910b1' or device_type_i=='910b2':
        num_cores = 24
        L1_size = 512*1024
        L0A_size = 64*1024
        L0B_size = 64*1024
        L0C_size = 128*1024
        UB_size = 192*1024
        device_type = device_type_i
    elif device_type_i=='910b4' or device_type_i=='910b3':
        num_cores = 20
        L1_size = 512*1024
        L0A_size = 64*1024
        L0B_size = 64*1024
        L0C_size = 128*1024
        UB_size = 192*1024
        device_type = device_type_i
    else:
        raise NotImplementedError

