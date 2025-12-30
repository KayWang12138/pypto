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
"""Core autograd state management with zero dependencies."""
import threading
from contextlib import contextmanager
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tracer import Tracer


class _AutogradContext(threading.local):
    """Thread-local autograd context."""

    def __init__(self):
        super().__init__()
        self.tracer: Optional[Any] = None
        self.tracing_enabled: bool = True


_context = _AutogradContext()


def is_tracing() -> bool:
    """Check if autograd tracing is currently active."""
    return _context.tracer is not None and _context.tracing_enabled


def get_current_tracer() -> Optional["Tracer"]:
    """Get the current tracer, or None if not tracing."""
    if not is_tracing():
        return None
    return _context.tracer


def set_current_tracer(tracer: Optional["Tracer"]) -> Optional["Tracer"]:
    """Set the current tracer and return the old one."""
    old_tracer = _context.tracer
    _context.tracer = tracer
    return old_tracer


@contextmanager
def trace(tracer: "Tracer"):
    """Context manager to enable autograd tracing with a specific tracer."""
    _ensure_rules_registered()
    old_tracer = set_current_tracer(tracer)
    try:
        yield tracer
    finally:
        set_current_tracer(old_tracer)


@contextmanager
def no_trace():
    """Context manager to temporarily disable autograd tracing."""
    old_enabled = _context.tracing_enabled
    _context.tracing_enabled = False
    try:
        yield
    finally:
        _context.tracing_enabled = old_enabled


_rules_registered = False


def _ensure_rules_registered():
    """Ensure VJP rules are registered (lazy initialization)."""
    global _rules_registered
    if not _rules_registered:
        from . import rules  # noqa: F401
        _rules_registered = True
