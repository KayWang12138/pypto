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
from ...utils import Tensor, Var, Instruction, Position, DT
from ... import context
from typing import Union


def dup(dst: Tensor, value: Union[float, int, Var], repeat: Union[int, Var], dst_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    supported_dtype = [DT.float, DT.half, DT.int16, DT.uint16, DT.int, DT.int32, DT.uint32, DT.bfloat16]
    assert dst.dtype in supported_dtype, f'Not supported type {dst.dtype}. Supported: {supported_dtype}'

    g_vec.append(Instruction('DUP', dst=dst, src=value, repeat=repeat, dst_blk_stride=dst_blk_stride, dst_rep_stride=dst_rep_stride))


def brcb(dst: Tensor, src: Tensor, dst_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], repeat: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('BRCB', dst=dst, src=src, dst_blk_stride=dst_blk_stride, dst_rep_stride=dst_rep_stride, repeat=repeat))

