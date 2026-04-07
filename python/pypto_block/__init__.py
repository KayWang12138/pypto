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

import importlib
import sys
from typing import cast


def _alias_namespace(source_prefix: str, target_prefix: str) -> None:
    """Mirror loaded modules between the pypto and pypto_block namespaces."""
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if name == source_prefix or name.startswith(source_prefix + "."):
            alias = target_prefix + name[len(source_prefix):]
            sys.modules.setdefault(alias, module)

# Let source-compatible modules keep their original absolute imports (`pypto.*`)
# without colliding with the native target-repo package in separate test runs.
sys.modules.setdefault("pypto", sys.modules[__name__])

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
for _compat_mod in ("language.manual", "language.op.unified_ops"):
    _module = importlib.import_module(f".{_compat_mod}", __name__)
    sys.modules.setdefault(f"pypto.{_compat_mod}", _module)
_alias_namespace("pypto_block", "pypto")
_alias_namespace("pypto", "pypto_block")

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

# Convenient dtype constants
DT_BOOL: DataType = cast(DataType, DataType.BOOL)
DT_INT4: DataType = cast(DataType, DataType.INT4)
DT_INT8: DataType = cast(DataType, DataType.INT8)
DT_INT16: DataType = cast(DataType, DataType.INT16)
DT_INT32: DataType = cast(DataType, DataType.INT32)
DT_INT64: DataType = cast(DataType, DataType.INT64)
DT_UINT4: DataType = cast(DataType, DataType.UINT4)
DT_UINT8: DataType = cast(DataType, DataType.UINT8)
DT_UINT16: DataType = cast(DataType, DataType.UINT16)
DT_UINT32: DataType = cast(DataType, DataType.UINT32)
DT_UINT64: DataType = cast(DataType, DataType.UINT64)
DT_FP4: DataType = cast(DataType, DataType.FP4)
DT_FP8E4M3FN: DataType = cast(DataType, DataType.FP8E4M3FN)
DT_FP8E5M2: DataType = cast(DataType, DataType.FP8E5M2)
DT_FP16: DataType = cast(DataType, DataType.FP16)
DT_FP32: DataType = cast(DataType, DataType.FP32)
DT_BF16: DataType = cast(DataType, DataType.BF16)
DT_HF4: DataType = cast(DataType, DataType.HF4)
DT_HF8: DataType = cast(DataType, DataType.HF8)
DT_INDEX: DataType = cast(DataType, DataType.INDEX)

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
