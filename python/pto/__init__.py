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
import sys

import pkg_resources

sys.path.append(str(pkg_resources.resource_filename(__package__, "")))

del pkg_resources
del sys


from .config import *  # noqa
from .controller import *  # noqa
from .element import Element
from .enum import *  # noqa
from .operation import *  # noqa
from .operator import cos, sin, sigmoid, softmax
from .pto_utils import ceildiv
from .runtime import jit
from .symbolic_scalar import SymbolicScalar
from .tensor import Tensor, mark_dynamic


def dump() -> str:
    return pto_impl.Dump()


def reset():
    pto_impl.Reset()


tensor = Tensor
element = Element
symbolic_scalar = SymbolicScalar

