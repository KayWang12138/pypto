# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO Language module - Type-safe DSL API for writing IR functions."""

from types import SimpleNamespace
from typing import Any

from pypto_block.pypto_core import DataType
from pypto_block.pypto_core.ir import ForKind, FunctionType, MemorySpace, MemRef, PipeType, TensorLayout

from . import parser
from .buffer import (
    Buffer,
    BufferSlot,
    L0ABuffer,
    L0ANBuffer,
    L0BBuffer,
    L0BNBuffer,
    L0CBuffer,
    L0CNBuffer,
    L1Buffer,
    L1NBuffer,
    NBuffer,
    TileSpec,
    UBBuffer,
    UBNBuffer,
)
from .dsl_api import (
    cond,
    const,
    incore,
    parallel,
    range,
    section_cube,
    section_vector,
    unroll,
    while_,
    yield_,
)
from .op import manual as block
from .op import mutex_ops as mutex
from .op import system_ops as system
from .op.manual import *  # noqa: F401, F403
from .op.manual import __all__ as _manual_all
from .op.ptr_ops import addptr, make_tensor
from .parser.decorator import InlineFunction, KernelFunction, func, function, inline, program
from .parser.text_parser import loads, loads_program, parse, parse_program
from .typing import DynVar, InOut, IntLike, Out, Ptr, Scalar, Tensor, Tile, dynamic
from .typing.dynamic import dynamic as dynamic
from .typing.tensor import TensorViewSpec as view


def struct(**kwargs: Any) -> Any:
    """Create a compile-time struct grouping IR expressions for attribute access."""
    return SimpleNamespace(**kwargs)


def StructArray(size: int, **kwargs: Any) -> Any:
    """Create a struct array of ``size`` identical structs."""
    return [SimpleNamespace(**kwargs) for _ in range(size)]


# Re-export TensorLayout constants for convenience
ND = TensorLayout.ND
DN = TensorLayout.DN
NZ = TensorLayout.NZ

# Re-export DataType constants from ir module (single source of truth)
from pypto_block.ir import (  # noqa: E402
    BF16,
    BOOL,
    FP4,
    FP8E4M3FN,
    FP8E5M2,
    FP16,
    FP32,
    HF4,
    HF8,
    INDEX,
    INT4,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT4,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
)

__all__ = [
    "function",
    "func",
    "inline",
    "program",
    "InlineFunction",
    "KernelFunction",
    "parse",
    "parser",
    "loads",
    "parse_program",
    "loads_program",
    "Tensor",
    "Tile",
    "Ptr",
    "Scalar",
    "DynVar",
    "InOut",
    "IntLike",
    "Out",
    "dynamic",
    "const",
    "range",
    "parallel",
    "unroll",
    "while_",
    "yield_",
    "cond",
    "incore",
    "section_vector",
    "section_cube",
    "block",
    "system",
    "mutex",
    "view",
    *_manual_all,
    "make_tensor",
    "addptr",
    "struct",
    "StructArray",
    "DataType",
    "FunctionType",
    "ForKind",
    "MemRef",
    "MemorySpace",
    "PipeType",
    "TensorLayout",
    "ND",
    "DN",
    "NZ",
    "FP4",
    "FP8E4M3FN",
    "FP8E5M2",
    "FP16",
    "FP32",
    "BF16",
    "HF4",
    "HF8",
    "INT4",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "UINT4",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "BOOL",
    "INDEX",
    "Buffer",
    "BufferSlot",
    "NBuffer",
    "TileSpec",
    "UBBuffer",
    "UBNBuffer",
    "L1Buffer",
    "L1NBuffer",
    "L0ABuffer",
    "L0ANBuffer",
    "L0BBuffer",
    "L0BNBuffer",
    "L0CBuffer",
    "L0CNBuffer",
]
