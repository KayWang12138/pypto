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
"""PyPTO
"""
import ctypes
import platform
import logging
from typing import List
from pathlib import Path


def load_shared_libs() -> bool:
    lib_dir: Path = Path(Path(__file__).parent, "lib")
    lib_suffix: str = "dylib" if platform.system() == 'Darwin' else "so"
    lib_names: List[str] = ["tile_fwk_interface", "tile_fwk_codegen", "tile_fwk_compiler", "tile_fwk_runtime"]
    libs: List[Path] = [Path(lib_dir, f"lib{n}.{lib_suffix}").resolve() for n in lib_names]
    for lib in libs:
        if not lib.exists():
            logging.debug("%s not exist, skip pre load shared libraries process.", lib)
            return False
    for lib in libs:
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError as err:
            logging.error("Failed to load %s: %s", lib, err)
            return False
    logging.debug("Success Load Shared Libs: %s", libs)
    return True


load_shared_libs()


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
