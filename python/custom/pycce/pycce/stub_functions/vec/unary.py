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
from ...utils import Tensor, Var, Instruction, Position, DT
from ... import context
from typing import Union


def exp(dst: Tensor, src: Tensor, \
        repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], \
        dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('EXP', dst=dst, src=src, \
                             repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, \
                             dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def ln(dst: Tensor, src: Tensor, \
       repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], \
       dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('LN', dst=dst, src=src, \
                             repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, \
                             dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def abs(dst: Tensor, src: Tensor, \
        repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], \
        dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('ABS', dst=dst, src=src, \
                             repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, \
                             dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def rec(dst: Tensor, src: Tensor, \
        repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], \
        dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('REC', dst=dst, src=src, \
                             repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, \
                             dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def sqrt(dst: Tensor, src: Tensor, \
         repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], \
         dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('SQRT', dst=dst, src=src, \
                             repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, \
                             dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def rsqrt(dst: Tensor, src: Tensor, \
          repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], \
          dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('RSQRT', dst=dst, src=src, \
                             repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, \
                             dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def relu(dst: Tensor, src: Tensor, \
         repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], \
         dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('RELU', dst=dst, src=src, \
                             repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, \
                             dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))

