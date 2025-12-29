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


"""The PTO Python Front-end Parser."""
from . import parser
from .parser import jit, function
from ..tensor import Scalar


def dynamic(name: str) -> Scalar:
    """Create a dynamic (symbolic) dimension.

    This function creates a symbolic scalar that can be used to define
    dynamic dimensions in tensor shapes. The symbolic scalar represents
    a runtime value that is not known at compile time.

    Parameters
    ----------
    name : str
        The name of the dynamic dimension.

    Returns
    -------
    Scalar
        A symbolic scalar representing the dynamic dimension.

    Examples
    --------
    >>> BS = pypto.dynamic("BS")
    >>> input_tensor = pypto.Tensor((BS, 128), pypto.DT_FP16)
    """
    return Scalar("int32", name)

