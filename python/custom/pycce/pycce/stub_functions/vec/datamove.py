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
from ...utils import Tensor, GMTensor, Var, Instruction, Position
from ... import context
from typing import Union


# Data move
def gm_to_ub(dst: Tensor, src: GMTensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call gm_to_ub within vector forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, GMTensor), 'src must be GMTensor on GM'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('GM2UB', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride))


def ub_to_gm(dst: GMTensor, src: Tensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_gm within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, GMTensor), 'dst must be GMTensor on GM'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2GM', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride))


def ub_to_ub(dst: Tensor, src: Tensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_ub within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2UB', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride))

