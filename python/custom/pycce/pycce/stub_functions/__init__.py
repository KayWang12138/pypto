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
from .. import context
context.set_device_type('910b3')

from ..autosync import auto_sync
from ..flowcontrol import Loop, If, Else, Elif, Range

from .flags import setflag, waitflag
from .barrier import barrier, bar_all, bar_fix, bar_m, bar_mte1, bar_mte2, bar_mte3, bar_v
from .consts import ComputeOffset, Max, Min, Align128, Align16, Align256, Align32, Align64, scalar_sqrt, CeilDiv, GetCubeIdx, GetCubeNum, GetSubBlockIdx, GetVecIdx, GetVecNum
from .cube import gm_to_l1_nd2nz, l1_to_l0, l1_to_l0_nz2nz, l1_to_l0_nz2zn, l1_to_l0_nz2nn, l1_to_l0_nz2zz, l0c_to_gm_nz2nd, mad
from .crosscore import cube_ready, wait_cube, vec_ready, wait_vec, allvec_ready, allvec_wait, allcube_ready, allcube_wait

from .vec.vecmask import set_mask, reset_mask, set_continuous_mask
from .vec.datamove import gm_to_ub, ub_to_gm, ub_to_ub
from .vec.unary import exp, ln, abs, rec, sqrt, rsqrt, relu
from .vec.binary import add, sub, mul, div, vmax, vmin, vand, vor
from .vec.unaryscalar import adds, muls, vmaxs, vmins, lrelu, axpy
from .vec.cast import cast
from .vec.group import cadd, cgadd, cpadd, cmax, cgmax, cmin, cgmin
from .vec.dupbrcb import dup, brcb

from ..utils import RoundMode, PIPE, DATATYPE, DT, Position
from ..utils import Var, Tensor, GMTensor, DBuff, SEvent, DEvent

from ..cube import CubeModule
from ..vec import VecModule
from ..kernel import KernelBase
from ..decorators import vec_func, cube_func, kernel_func, tileop_func
