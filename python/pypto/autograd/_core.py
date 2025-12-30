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
"""
Core autograd state management - ZERO DEPENDENCIES.

This module contains only the minimal state management for autograd,
with no dependencies on other pypto modules. This allows it to be
safely imported at module load time without causing circular imports.

Other modules (tensor.py, _op_wrapper.py, _controller.py) can import
from this module at the top level.
"""
import threading
from contextlib import contextmanager
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tracer import Tracer


class _AutogradContext(threading.local):
    """Thread-local autograd context.

    Uses thread-local storage to ensure each thread has its own tracer state.
    This is important for multi-threaded applications.
    """

    def __init__(self):
        super().__init__()
        self.tracer: Optional[Any] = None  # Use Any to avoid import
        self.tracing_enabled: bool = True


# Global thread-local context instance
_context = _AutogradContext()


def is_tracing() -> bool:
    """Check if autograd tracing is currently active.

    Returns True only if:
    1. A tracer is set (we're inside a trace() context)
    2. Tracing is enabled (we're not inside a no_trace() context)

    This function is designed to be very fast when tracing is not active,
    as it's called on every operation.
    """
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
    """Context manager to enable autograd tracing with a specific tracer.

    Args:
        tracer: The Tracer instance to use for recording operations.

    Example:
        tracer = Tracer()
        with trace(tracer):
            # Operations here will be recorded
            y = pypto.add(a, b)
    """
    # Ensure VJP rules are registered on first use
    _ensure_rules_registered()

    old_tracer = set_current_tracer(tracer)
    try:
        yield tracer
    finally:
        set_current_tracer(old_tracer)


@contextmanager
def no_trace():
    """Context manager to temporarily disable autograd tracing.

    Useful for operations that should not be recorded (e.g., debug prints,
    validation checks, or operations that don't need gradients).

    Example:
        with trace(tracer):
            y = pypto.add(a, b)  # Recorded
            with no_trace():
                debug_val = pypto.sum(y, dim=0)  # NOT recorded
            z = pypto.mul(y, 2)  # Recorded
    """
    old_enabled = _context.tracing_enabled
    _context.tracing_enabled = False
    try:
        yield
    finally:
        _context.tracing_enabled = old_enabled


# ============================================================================
# Lazy initialization of VJP rules
# ============================================================================

_rules_registered = False


def _ensure_rules_registered():
    """Ensure VJP rules are registered (lazy initialization).

    This is called the first time trace() is used, which triggers
    the import of rules modules that register VJP functions.
    This avoids circular imports during module initialization.
    """
    global _rules_registered
    if not _rules_registered:
        # Import rules to trigger registration
        from . import rules  # noqa: F401
        _rules_registered = True
