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
"""PyPTO"""
from typing import List
from .. import pto_impl

from ..op_wrapper import op_wrapper
from ..tensor import Tensor


@op_wrapper
def concat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    Concatenate multiple tensors according to the specified dimension.

    Parameters
    ---------
    tensors: Tensors
        tensor to be spliced.

    dim : int
        specified dimensions.

    out: Tensor
        The concatenated tensor
    Examples
    ---------
    x = pto.tensor([2, 2], pto.data_type.DT_FP32)  # 2x2 tensor with all 1s
    y = pto.tensor([2, 2], pto.data_type.DT_FP32)  # 2x2 tensor with all 0s
    dim = 0
    out = pto.concat([x, y], dim)

    Input  x : [[1 1],
                [1 1]]
           y : [[0 0],
                [0 0]]
    Output out:[[1 1],
                [1 1],
                [0 0],
                [0 0]]
    """
    return pto_impl.Cat(tensors, dim)
