#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""

from typing import List
import pto
from pto import pto_impl
from .pto_utils import convert_to_symbolic

reshape = pto_impl.Reshape
assemble = pto_impl.Assemble
view = pto_impl.View


def assemble(input: pto.Tensor, offsets: List[int], out: pto.Tensor) -> None:
    """
    Assembles a small Tensor into a larger Tensor based on specified offsets.

    Parameters
    ---------
    input: Tensor
        The small input tensor to be assembled into the larger tensor
        
    offsets : List[int]
        List of offset values indicating where the input tensor should be placed in the output tensor. 
        It is required that the offsets is smaller than the shape of out.
        
    out: Tensor
        The larger output tensor that will contain the assembled input tensor
    Examples
    ---------
    >>> import pto
    >>> x = pto.tensor([2, 2], pto.data_type.DT_FP32)  # 2x2 tensor with all 1s
    >>> out = pto.tensor([4, 4], pto.data_type.DT_FP32)  # 4x4 tensor with all 0s
    >>> pto.assemble(x, [0, 0], out) 
    >>> print(out)
    [[1 1 0 0]
    [1 1 0 0]
    [0 0 0 0]
    [0 0 0 0]]
    """
    pto_impl.Assemble(input, offsets, out)