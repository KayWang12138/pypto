# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manual-op pipeline metadata used by parser features that need pipe IDs."""

from __future__ import annotations

from pypto_block.pypto_core.ir import MemorySpace, PipeType

_MANUAL_OP_TO_PIPE: dict[str, PipeType] = {
    "load": PipeType.MTE2,
    "load_tile": PipeType.MTE2,
    "insert": PipeType.MTE3,
    "matmul": PipeType.M,
    "matmul_acc": PipeType.M,
    "matmul_bias": PipeType.M,
    "gemv": PipeType.M,
    "gemv_acc": PipeType.M,
    "gemv_bias": PipeType.M,
    "neg": PipeType.V,
    "exp": PipeType.V,
    "sqrt": PipeType.V,
    "rsqrt": PipeType.V,
    "recip": PipeType.V,
    "log": PipeType.V,
    "abs": PipeType.V,
    "relu": PipeType.V,
    "not_": PipeType.V,
    "cast": PipeType.V,
    "add": PipeType.V,
    "sub": PipeType.V,
    "mul": PipeType.V,
    "div": PipeType.V,
    "rem": PipeType.V,
    "maximum": PipeType.V,
    "minimum": PipeType.V,
    "and_": PipeType.V,
    "or_": PipeType.V,
    "shl": PipeType.V,
    "shr": PipeType.V,
    "adds": PipeType.V,
    "subs": PipeType.V,
    "muls": PipeType.V,
    "divs": PipeType.V,
    "rems": PipeType.V,
    "ands": PipeType.V,
    "ors": PipeType.V,
    "shls": PipeType.V,
    "shrs": PipeType.V,
    "maxs": PipeType.V,
    "mins": PipeType.V,
    "lrelu": PipeType.V,
    "xor": PipeType.V,
    "xors": PipeType.V,
    "prelu": PipeType.V,
    "addc": PipeType.V,
    "subc": PipeType.V,
    "addsc": PipeType.V,
    "subsc": PipeType.V,
    "sel": PipeType.V,
    "sels": PipeType.V,
    "cmp": PipeType.V,
    "cmps": PipeType.V,
    "row_max": PipeType.V,
    "row_sum": PipeType.V,
    "col_max": PipeType.V,
    "col_sum": PipeType.V,
    "row_min": PipeType.V,
    "row_expand": PipeType.V,
    "row_expand_add": PipeType.V,
    "row_expand_sub": PipeType.V,
    "row_expand_mul": PipeType.V,
    "row_expand_div": PipeType.V,
    "col_expand": PipeType.V,
    "col_expand_mul": PipeType.V,
    "col_expand_div": PipeType.V,
    "col_expand_sub": PipeType.V,
    "expands": PipeType.V,
    "reshape": PipeType.V,
    "transpose": PipeType.V,
    "ub_copy": PipeType.V,
    "full": PipeType.V,
    "fillpad": PipeType.V,
    "fillpad_expand": PipeType.V,
}


def get_move_pipe(
    src_memory: MemorySpace | None,
    target_memory: MemorySpace | None,
) -> PipeType:
    """Determine the hardware pipe for a manual ``move`` operation."""
    if src_memory == MemorySpace.Acc and target_memory == MemorySpace.Vec:
        return PipeType.FIX
    if src_memory == MemorySpace.Mat:
        if target_memory in (MemorySpace.Left, MemorySpace.Right):
            return PipeType.MTE1
        if target_memory == MemorySpace.Vec:
            return PipeType.V
        return PipeType.FIX
    if src_memory == MemorySpace.Vec and target_memory == MemorySpace.Mat:
        return PipeType.FIX
    return PipeType.V


def get_store_pipe(src_memory: MemorySpace | None) -> PipeType:
    """Determine the hardware pipe for a manual ``store`` / ``store_tile`` operation."""
    if src_memory == MemorySpace.Acc:
        return PipeType.FIX
    return PipeType.MTE3
