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
"""PyPTO
"""


def load_shared_libs():
    import os
    import ctypes
    from importlib import metadata

    pypto_dir = metadata.distribution("pypto").locate_file("pypto")
    lib_dir = os.path.join(f"{pypto_dir}", "lib64")
    if not os.path.exists(lib_dir):
        lib_dir = os.path.join(f"{pypto_dir}", "lib")

    calc_path = os.path.join(lib_dir, "libtile_fwk_calculator.so")
    if os.path.exists(calc_path):
        import torch

        # calculator.so depends on torch, so load it after torch
        ctypes.CDLL(calc_path, mode=ctypes.RTLD_LOCAL)

    libs = ["libtile_fwk_interface.so", "libtile_fwk_codegen.so",
            "libtile_fwk_compiler.so", "libtile_fwk_runtime.so"]
    for lib in libs:
        if os.path.exists(os.path.join(lib_dir, lib)):
            ctypes.CDLL(os.path.join(lib_dir, lib), mode=ctypes.RTLD_GLOBAL)


load_shared_libs()


from .config import *  # noqa
from .controller import *  # noqa
from .element import Element
from .enum import *  # noqa
from .op import *  # noqa
from .operation import *  # noqa
from .operator import *  # noqa
from .pass_config import * # noqa
from .pto_utils import ceil, bytes_of
from .runtime import jit, verify
from .symbolic_scalar import SymbolicScalar
from .tensor import Tensor, mark_dynamic


tensor = Tensor
element = Element
symbolic_scalar = SymbolicScalar

verify_enable = pto_impl.IsVerifyEnabled
