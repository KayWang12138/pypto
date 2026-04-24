# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System operations for PyPTO IR.

System operations handle hardware synchronization primitives:
- sync_src / sync_dst: Set/Wait flag-based synchronization between pipes
- bar_v / bar_m / bar_all: Barrier synchronization for vector, matrix, or all units
"""

from pypto_block.pypto_core import ir as _ir_core
from pypto_block.pypto_core.ir import Call, Expr, PipeType, Span

from ..utils import _get_span_or_capture


def _create_sync_op(
    op_name: str,
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: int | Expr,
    span: Span | None,
) -> Call:
    """Create a flag-based synchronization operation.

    Args:
        op_name: Operation name (e.g., "system.sync_src")
        set_pipe: Pipe that sets the flag
        wait_pipe: Pipe that waits on the flag
        event_id: Event identifier (int for static, Expr for dynamic)
        span: Optional source span for debugging
    """
    actual_span = _get_span_or_capture(span, frame_offset=2)
    if isinstance(event_id, Expr):
        kwargs = {"set_pipe": set_pipe, "wait_pipe": wait_pipe}
        return _ir_core.create_op_call(op_name + "_dyn", [event_id], kwargs, actual_span)
    kwargs = {"set_pipe": set_pipe, "wait_pipe": wait_pipe, "event_id": event_id}
    return _ir_core.create_op_call(op_name, [], kwargs, actual_span)


def _create_barrier_op(op_name: str, *, span: Span | None) -> Call:
    """Create a barrier synchronization operation.

    Args:
        op_name: Operation name (e.g., "system.bar_v")
        span: Optional source span for debugging
    """
    actual_span = _get_span_or_capture(span, frame_offset=2)
    return _ir_core.create_op_call(op_name, [], {}, actual_span)


def sync_src(
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: int | Expr,
    span: Span | None = None,
) -> Call:
    """Send a synchronization signal (Set Flag).

    Args:
        set_pipe: Pipe that sets the flag
        wait_pipe: Pipe that will wait on the flag
        event_id: Event identifier
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.sync_src
    """
    return _create_sync_op(
        "system.sync_src", set_pipe=set_pipe, wait_pipe=wait_pipe, event_id=event_id, span=span
    )


def sync_dst(
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: int | Expr,
    span: Span | None = None,
) -> Call:
    """Wait for a synchronization signal (Wait Flag).

    Args:
        set_pipe: Pipe that sets the flag
        wait_pipe: Pipe that waits on the flag
        event_id: Event identifier
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.sync_dst
    """
    return _create_sync_op(
        "system.sync_dst", set_pipe=set_pipe, wait_pipe=wait_pipe, event_id=event_id, span=span
    )


def bar_v(*, span: Span | None = None) -> Call:
    """Vector unit barrier."""
    return _create_barrier_op("system.bar_v", span=span)


def bar_m(*, span: Span | None = None) -> Call:
    """Matrix unit barrier."""
    return _create_barrier_op("system.bar_m", span=span)


def bar_all(*, span: Span | None = None) -> Call:
    """Global barrier synchronization."""
    return _create_barrier_op("system.bar_all", span=span)

def set_cross_core(
    *,
    pipe: PipeType,
    event_id: int | Expr,
    max_event_id: int = 16,
    mode_id: int = 2,
    span: Span | None = None,
) -> Call:
    """Set for a synchronization signal (Cross core).

    Args:
        pipe: Pipe that sets the flag
        event_id: Event identifier (int for static, Expr for dynamic)
        max_event_id: Upper bound of event_id range (reduces codegen branches)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.set_cross_core
    """
    actual_span = _get_span_or_capture(span)
    if isinstance(event_id, Expr):
        return _ir_core.create_op_call(
            "system.set_cross_core_dyn", [event_id],
            {"pipe": pipe, "max_event_id": max_event_id}, actual_span)
    kwargs = {"pipe": pipe, "event_id": event_id, "mode_id": mode_id}
    return _ir_core.create_op_call("system.set_cross_core", [], kwargs, actual_span)

def wait_cross_core(
    *,
    pipe: PipeType,
    event_id: int | Expr,
    max_event_id: int = 16,
    span: Span | None = None,
) -> Call:
    """Wait for a synchronization signal (Cross core).

    Args:
        pipe: Pipe that waits the flag
        event_id: Event identifier (int for static, Expr for dynamic)
        max_event_id: Upper bound of event_id range (reduces codegen branches)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.wait_cross_core
    """
    actual_span = _get_span_or_capture(span)

    if isinstance(event_id, Expr):
        return _ir_core.create_op_call(
            "system.wait_cross_core_dyn", [event_id],
            {"pipe": pipe, "max_event_id": max_event_id}, actual_span)
    kwargs = {"pipe": pipe, "event_id": event_id}
    return _ir_core.create_op_call("system.wait_cross_core", [], kwargs, actual_span)

def sync_all(
    *,
    aiv_only: bool = True,
    trigger_pipe: PipeType = PipeType.ALL,
    wait_pipe: PipeType = PipeType.ALL,
    span: Span | None = None,
) -> Call:
    """Matrix multiplication with optional transpose.

    Args:
        aiv_only: True for AIV-ONLY sync, False for AIC and AIV sync
        trigger_pipe: Pipe type to trigger sync flag (Supported: PIPE_ALL, PIPE_MTE2, PIPE_MTE)
        wait_pipe: Pipe type to wait for sync flag (Supported: PIPE_ALL, PIPE_MTE2, PIPE_MTE)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for matrix multiplication
    """
    kwargs = {
        "aiv_only": aiv_only,
        "trigger_pipe": trigger_pipe,
        "wait_pipe": wait_pipe,
    }
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("system.sync_all", [], kwargs, actual_span)


# ============================================================================
# Mutex (Buffer-ID Token) — A5 only
# ----------------------------------------------------------------------------
# Alternative to event-id based sync_src/sync_dst: uses a buffer-id token
# (MutexID, range 0-31) to enforce ordering between pipes. Lowered to
# pto.get_buf / pto.rls_buf in the PTO backend.
# ============================================================================


def mutex_lock(
    pipe: PipeType,
    mutex_id: int | Expr,
    *,
    mode: int = 0,
    max_mutex_id: int = 2,
    buf_id_values: tuple | list | None = None,
    span: Span | None = None,
) -> Call:
    """Acquire a Mutex buffer-id token on ``pipe`` (A5).

    Blocks the ``pipe`` instruction queue until the previous holder of
    ``mutex_id`` releases it via :func:`mutex_unlock`.

    Args:
        pipe: PipeType for which to acquire the lock (e.g. PipeType.MTE2).
        mutex_id: MutexID (0-31, per Ascend C Mutex ISASI spec).
            May be a static int or a dynamic IR Expr; when dynamic, the
            PTO codegen emits an if-chain of static `pto.get_buf` using
            ``buf_id_values`` as the comparison targets.
        mode: Optional mode attribute (default 0).
        max_mutex_id: Upper bound of the unrolled range when ``mutex_id``
            is dynamic. Defaults to 2 (ping-pong double buffering).
        buf_id_values: Actual buf_id integer values for PTO if-chain
            (e.g. (2, 3)). When None, defaults to (0, 1, ..., max_mutex_id-1).
        span: Optional source span (auto-captured when omitted).

    Returns:
        Call expression for system.mutex_lock / system.mutex_lock_dyn.
    """
    actual_span = _get_span_or_capture(span, frame_offset=2)
    if isinstance(mutex_id, Expr):
        kwargs: dict = {"pipe": pipe, "mode": mode, "max_mutex_id": max_mutex_id}
        if buf_id_values is not None:
            kwargs["buf_id_values"] = list(buf_id_values)
        return _ir_core.create_op_call("system.mutex_lock_dyn", [mutex_id], kwargs, actual_span)
    kwargs = {"pipe": pipe, "mutex_id": mutex_id, "mode": mode}
    return _ir_core.create_op_call("system.mutex_lock", [], kwargs, actual_span)


def mutex_unlock(
    pipe: PipeType,
    mutex_id: int | Expr,
    *,
    mode: int = 0,
    max_mutex_id: int = 2,
    buf_id_values: tuple | list | None = None,
    span: Span | None = None,
) -> Call:
    """Release a previously acquired Mutex buffer-id token on ``pipe`` (A5).

    Must be paired with :func:`mutex_lock` on the same ``pipe`` and
    ``mutex_id``.

    Args:
        pipe: PipeType for which to release the lock.
        mutex_id: MutexID passed to the paired :func:`mutex_lock`.
        mode: Optional mode attribute (default 0).
        max_mutex_id: Upper bound of the unrolled range when dynamic.
        buf_id_values: Actual buf_id integer values for PTO if-chain.
        span: Optional source span (auto-captured when omitted).

    Returns:
        Call expression for system.mutex_unlock / system.mutex_unlock_dyn.
    """
    actual_span = _get_span_or_capture(span, frame_offset=2)
    if isinstance(mutex_id, Expr):
        kwargs: dict = {"pipe": pipe, "mode": mode, "max_mutex_id": max_mutex_id}
        if buf_id_values is not None:
            kwargs["buf_id_values"] = list(buf_id_values)
        return _ir_core.create_op_call("system.mutex_unlock_dyn", [mutex_id], kwargs, actual_span)
    kwargs = {"pipe": pipe, "mutex_id": mutex_id, "mode": mode}
    return _ir_core.create_op_call("system.mutex_unlock", [], kwargs, actual_span)
