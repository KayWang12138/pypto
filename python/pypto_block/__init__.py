# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO - Python Tensor Operations Library

This package provides Python bindings for the PyPTO C++ library.
"""

import sys

from . import pypto_core as _pypto_core

sys.modules.setdefault("pypto_block.pypto_core", _pypto_core)
for _sub in ("ir", "backend", "codegen", "passes", "logging", "testing"):
    if hasattr(_pypto_core, _sub):
        sys.modules.setdefault(f"pypto_block.pypto_core.{_sub}", getattr(_pypto_core, _sub))

from . import backend as backend

sys.modules.setdefault("pypto_block.backend", backend)

from . import ir as ir

sys.modules.setdefault("pypto_block.ir", ir)

from . import language as language

sys.modules.setdefault("pypto_block.language", language)

from .pypto_core import (
    DataType,
    InternalError,
    LogLevel,
    check,
    codegen,
    internal_check,
    log_debug,
    log_error,
    log_event,
    log_fatal,
    log_info,
    log_warn,
    passes,
    set_log_level,
    testing,
)

# Re-export DataType constants from ir module (single source of truth)
from .ir import (
    BOOL as DT_BOOL,
    INT4 as DT_INT4,
    INT8 as DT_INT8,
    INT16 as DT_INT16,
    INT32 as DT_INT32,
    INT64 as DT_INT64,
    UINT4 as DT_UINT4,
    UINT8 as DT_UINT8,
    UINT16 as DT_UINT16,
    UINT32 as DT_UINT32,
    UINT64 as DT_UINT64,
    FP4 as DT_FP4,
    FP8E4M3FN as DT_FP8E4M3FN,
    FP8E5M2 as DT_FP8E5M2,
    FP16 as DT_FP16,
    FP32 as DT_FP32,
    BF16 as DT_BF16,
    HF4 as DT_HF4,
    HF8 as DT_HF8,
    INDEX as DT_INDEX,
)

__all__ = [
    # Modules
    "codegen",
    "ir",
    "language",
    "passes",
    "testing",
    # Logging framework
    "InternalError",
    "LogLevel",
    "set_log_level",
    "log_debug",
    "log_info",
    "log_warn",
    "log_error",
    "log_fatal",
    "log_event",
    "check",
    "internal_check",
    # DataType class
    "DataType",
    # Dtype constants
    "DT_BOOL",
    "DT_INT4",
    "DT_INT8",
    "DT_INT16",
    "DT_INT32",
    "DT_INT64",
    "DT_UINT4",
    "DT_UINT8",
    "DT_UINT16",
    "DT_UINT32",
    "DT_UINT64",
    "DT_FP4",
    "DT_FP8E4M3FN",
    "DT_FP8E5M2",
    "DT_FP16",
    "DT_FP32",
    "DT_BF16",
    "DT_HF4",
    "DT_HF8",
    "DT_INDEX",
]

__version__ = "0.1.0"
