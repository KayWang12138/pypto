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
# pyright: reportReturnType=false
# pyright: reportArgumentType=false
from pto import pto_impl
from .tensor import Tensor
from .operation import op_wrapper


@op_wrapper
def softmax(tensor: Tensor) -> Tensor:
    """
    Computes softmax activations along the specified axis.

    Args:
        tensor (Tensor): The input tensor.

    Returns:
        Tensor: The output tensor with softmax activations.
    """
    return pto_impl.softmax(tensor)

