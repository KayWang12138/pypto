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
"""PyPTO operation wrapper with autograd hook support."""
import functools
from typing import Any, Callable, Dict, Tuple

from . import pypto_impl
from ._element import Element
from ._utils import clear_source_location, set_source_location
from .symbolic_scalar import SymbolicScalar
from .tensor import Tensor


def _to_base(arg):
    if isinstance(arg, (Tensor, Element, SymbolicScalar)):
        return arg.base()
    elif isinstance(arg, (list, tuple)):
        return [_to_base(a) for a in arg]
    elif isinstance(arg, dict):
        return {k: _to_base(v) for k, v in arg.items()}
    else:
        return arg


def _from_base(out):
    if isinstance(out, pypto_impl.Tensor):
        return Tensor.from_base(out)
    elif isinstance(out, (list, tuple)):
        return [_from_base(a) for a in out]
    elif isinstance(out, dict):
        return {k: _from_base(v) for k, v in out.items()}
    else:
        return out


def _call_autograd_hook(
    func: Callable,
    op_name: str,
    original_args: Tuple[Any, ...],
    original_kwargs: Dict[str, Any],
    result: Any,
) -> None:
    """
    Call the autograd hook if tracing is active.

    This is a lazy import to avoid circular dependencies and ensure
    zero overhead when autograd is not used.
    """
    try:
        from .autograd.context import is_tracing
        if not is_tracing():
            return

        from .autograd.tracer import _autograd_hook_op
        _autograd_hook_op(
            op_callable=func,
            op_name=op_name,
            args=original_args,
            kwargs=original_kwargs,
            outputs=result,
        )
    except ImportError:
        # autograd module not available, skip
        pass


def _snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part)


def _infer_autograd_op_name(
    func: Callable,
    original_args: Tuple[Any, ...],
    original_kwargs: Dict[str, Any],
    result: Any,
) -> str:
    """
    Infer the canonical op name used by the autograd VJP registry.

    Notes:
    - Python wrappers are typically snake_case (e.g., add/matmul), while VJP rules
      are registered with the underlying op name in PascalCase (e.g., Add/Matmul).
    - Some Python wrappers map to different underlying ops (e.g., clone -> Assign,
      matmul -> Matmul/BatchMatmul depending on input dims).
    """
    name = getattr(func, "__name__", "")

    # Special cases
    if name == "clone":
        return "Assign"

    if name == "matmul":
        # matmul wrapper dispatches to Matmul / BatchMatmul based on dims
        try:
            if len(original_args) >= 2:
                a, b = original_args[0], original_args[1]
                if hasattr(a, "dim") and hasattr(b, "dim"):
                    a_dim = a.dim
                    b_dim = b.dim
                    if a_dim == b_dim == 2:
                        return "Matmul"
                    if a_dim == b_dim and a_dim in (3, 4):
                        return "BatchMatmul"
        except Exception:
            # Fall back to default mapping
            pass
        return "Matmul"

    return _snake_to_pascal(name)


def op_wrapper(func):
    """
    Decorator that wraps PyPTO operations.

    Handles:
    1. Converting Python wrapper types to C++ base types
    2. Setting source location for debugging
    3. Converting C++ results back to Python wrapper types
    4. Calling autograd hook for gradient tracking (when tracing is active)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Keep original args for autograd hook (before _to_base conversion)
        original_args = args
        original_kwargs = kwargs

        # Convert to base types for C++ call
        args = _to_base(args)
        kwargs = _to_base(kwargs)
        if not isinstance(args, (list, tuple)):
            raise TypeError(f"args must be list or tuple, but got {type(args)}.")

        # Execute the operation
        set_source_location()
        out = func(*args, **kwargs)
        clear_source_location()

        # Convert result back to Python types
        if out is None:
            result = None
        else:
            result = _from_base(out)

        # Call autograd hook (no-op if not tracing)
        _call_autograd_hook(
            func=func,
            op_name=_infer_autograd_op_name(
                func=func,
                original_args=original_args,
                original_kwargs=original_kwargs,
                result=result,
            ),
            original_args=original_args,
            original_kwargs=original_kwargs,
            result=result,
        )

        return result

    return wrapper
