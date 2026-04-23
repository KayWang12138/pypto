#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static check: reading tensor storage before any write is undefined and must be rejected.

Allocator calls (e.g. ``pypto.tensor`` / ``pypto.Tensor``) yield storage that is not
initialized until it is written (e.g. slice assignment ``t[:] = ...``). Reading that
storage first triggers a machine-side assert; this pass reports a ParserError earlier.

Strategy: prefer *under-reporting* over *false positives*. Branch merges use the
intersection of per-branch uninit sets, and loop bodies are analyzed with any name
that may be written inside the body pre-cleared from the entry state.
"""

from __future__ import annotations

import ast
from typing import Callable, Dict, Iterator, Optional, Set, Type

from pypto.error import ParserError

_MSG = (
    "read from tensor storage before it is written; assign to the tensor (e.g. "
    "`t[:] = ...`) before reading elements"
)


# ---------------------------------------------------------------------------
# Module-level helpers (stateless)
# ---------------------------------------------------------------------------


def _is_tensor_allocator_call(expr: ast.expr) -> bool:
    """Return True if ``expr`` looks like ``Tensor(...)`` / ``tensor(...)`` / ``x.Tensor(...)`` / ``x.tensor(...)``."""
    if not isinstance(expr, ast.Call):
        return False
    func = expr.func
    if isinstance(func, ast.Name) and func.id in ("Tensor", "tensor"):
        return True
    if isinstance(func, ast.Attribute) and func.attr in ("Tensor", "tensor"):
        return True
    return False


def _collect_param_names(node: ast.FunctionDef) -> Set[str]:
    names = {a.arg for a in node.args.args}
    names |= {a.arg for a in node.args.kwonlyargs}
    if node.args.vararg:
        names.add(node.args.vararg.arg)
    if node.args.kwarg:
        names.add(node.args.kwarg.arg)
    return names


def _target_root_names(target: ast.expr) -> Set[str]:
    """Return the root Name ids touched by an assignment target.

    Handles ``Name``, ``Subscript``, ``Attribute`` (both chained), and flat/nested
    tuple/list unpacking. Unknown shapes produce an empty set.
    """
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Subscript):
        cur: ast.expr = target
        while isinstance(cur, ast.Subscript):
            cur = cur.value
        return {cur.id} if isinstance(cur, ast.Name) else set()
    if isinstance(target, ast.Attribute):
        cur = target
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        return {cur.id} if isinstance(cur, ast.Name) else set()
    if isinstance(target, (ast.Tuple, ast.List)):
        result: Set[str] = set()
        for elt in target.elts:
            inner = elt.value if isinstance(elt, ast.Starred) else elt
            result |= _target_root_names(inner)
        return result
    return set()


def _iter_unpack_names(target: ast.expr) -> Iterator[str]:
    """Yield every plain ``ast.Name`` id inside (possibly nested) tuple/list unpacking.

    Starred elements are skipped; they bind a list of unknown length and are not
    valid allocator receivers.
    """
    if isinstance(target, ast.Name):
        yield target.id
        return
    if not isinstance(target, (ast.Tuple, ast.List)):
        return
    for elt in target.elts:
        if isinstance(elt, ast.Starred):
            continue
        yield from _iter_unpack_names(elt)


def _collect_writable_roots(stmts: list) -> Set[str]:
    """Conservative over-approximation of names that may be (re)written inside ``stmts``.

    Used at loop entry to simulate "later iteration" state: any name that *might* be
    assigned inside the loop body is cleared from the uninit set before body analysis.
    """
    roots: Set[str] = set()
    for stmt in stmts:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    roots |= _target_root_names(tgt)
            elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                roots |= _target_root_names(sub.target)
            elif isinstance(sub, ast.For):
                roots |= _target_root_names(sub.target)
    return roots


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class _Checker:
    """Walks a statement block, maintaining ``uninit`` — names that are known to
    reference tensor storage which has not yet been written.

    A read of such a name raises :class:`ParserError`. Control-flow joins use the
    intersection of branch results so that a mere *possibility* of a prior write
    is enough to suppress the diagnostic (no false positives).
    """

    # Populated lazily so bound-method handlers see instance state.
    _dispatch: Dict[Type[ast.stmt], Callable[["_Checker", ast.stmt], None]]

    def __init__(self, exempt: Set[str]) -> None:
        self._exempt = exempt
        self.uninit: Set[str] = set()
        self._dispatch = {
            ast.Assign: _Checker._handle_assign,
            ast.AnnAssign: _Checker._handle_ann_assign,
            ast.AugAssign: _Checker._handle_aug_assign,
            ast.Expr: _Checker._handle_expr,
            ast.Return: _Checker._handle_return,
            ast.If: _Checker._handle_if,
            ast.While: _Checker._handle_while,
            ast.For: _Checker._handle_for,
            ast.With: _Checker._handle_with,
            ast.AsyncWith: _Checker._handle_with,
            ast.Try: _Checker._handle_try,
            ast.Assert: _Checker._handle_assert,
            ast.FunctionDef: _Checker._handle_function_def,
            ast.AsyncFunctionDef: _Checker._handle_function_def,
            ast.Delete: _Checker._handle_delete,
        }

    # ----- public API ----------------------------------------------------

    def process(self, stmts: list) -> None:
        """Walk a list of statements, mutating ``self.uninit`` as we go."""
        for stmt in stmts:
            handler = self._dispatch.get(type(stmt))
            if handler is not None:
                handler(self, stmt)

    def check_reads(self, expr: ast.expr) -> None:
        """Raise ``ParserError`` if ``expr`` loads any currently-uninit name."""
        for child in ast.walk(expr):
            if not isinstance(child, ast.Name):
                continue
            if not isinstance(child.ctx, ast.Load):
                continue
            if child.id in self._exempt:
                continue
            if child.id in self.uninit:
                raise ParserError(child, ValueError(_MSG))

    # ----- sub-block helpers (keep child-checker access encapsulated) ----

    def _eval_block(self, init_uninit: Set[str], stmts: list) -> Set[str]:
        sub = _Checker(self._exempt)
        sub.uninit = set(init_uninit)
        sub.process(stmts)
        return sub.uninit

    def _eval_handler(
        self, before: Set[str], handler: ast.ExceptHandler
    ) -> Set[str]:
        sub = _Checker(self._exempt)
        sub.uninit = set(before)
        if handler.type is not None:
            sub.check_reads(handler.type)
        exc_name = handler.name
        if isinstance(exc_name, ast.Name):
            sub.uninit.discard(exc_name.id)
        elif isinstance(exc_name, str):
            sub.uninit.discard(exc_name)
        sub.process(handler.body)
        return sub.uninit

    # ----- write bookkeeping --------------------------------------------

    def _mark_name(self, name: str, is_alloc: bool) -> None:
        if is_alloc:
            self.uninit.add(name)
        else:
            self.uninit.discard(name)

    def _apply_write_target(self, target: ast.expr, is_alloc: bool) -> None:
        """Update ``uninit`` according to a single assignment target.

        * Plain name:         mark as uninit iff RHS is an allocator call.
        * Subscript/Attribute: treated as a definition write — clear the root name.
        * Tuple / list unpack: flatten to plain names (allocator result cannot be
          unpacked, so treated as ordinary writes).
        """
        if isinstance(target, ast.Name):
            self._mark_name(target.id, is_alloc)
            return
        if isinstance(target, (ast.Subscript, ast.Attribute)):
            for name in _target_root_names(target):
                self.uninit.discard(name)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for name in _iter_unpack_names(target):
                self._mark_name(name, is_alloc)

    # ----- per-statement handlers ---------------------------------------

    def _handle_assign(self, node: ast.Assign) -> None:
        self.check_reads(node.value)
        is_alloc = _is_tensor_allocator_call(node.value)
        for target in node.targets:
            self._apply_write_target(target, is_alloc)

    def _handle_ann_assign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        self.check_reads(node.value)
        is_alloc = _is_tensor_allocator_call(node.value)
        self._apply_write_target(node.target, is_alloc)

    def _handle_aug_assign(self, node: ast.AugAssign) -> None:
        self.check_reads(node.target)
        self.check_reads(node.value)
        for name in _target_root_names(node.target):
            self.uninit.discard(name)

    def _handle_expr(self, node: ast.Expr) -> None:
        self.check_reads(node.value)

    def _handle_return(self, node: ast.Return) -> None:
        if node.value is not None:
            self.check_reads(node.value)

    def _handle_assert(self, node: ast.Assert) -> None:
        self.check_reads(node.test)
        if node.msg is not None:
            self.check_reads(node.msg)

    def _handle_delete(self, node: ast.Delete) -> None:
        for tgt in node.targets:
            self.check_reads(tgt)

    def _handle_with(self, node: ast.stmt) -> None:
        for item in node.items:  # type: ignore[attr-defined]
            self.check_reads(item.context_expr)
        self.process(node.body)  # type: ignore[attr-defined]

    def _handle_function_def(self, node: ast.stmt) -> None:
        check_uninitialized_tensor_storage_reads(node, exempt_names=self._exempt)  # type: ignore[arg-type]

    def _handle_if(self, node: ast.If) -> None:
        self.check_reads(node.test)
        before = set(self.uninit)
        then_uninit = self._eval_block(before, node.body)
        else_uninit = self._eval_block(before, node.orelse)
        # Conservative: only names uninit on *all* branches survive as uninit.
        self.uninit = then_uninit & else_uninit

    def _handle_while(self, node: ast.While) -> None:
        self.check_reads(node.test)
        before = set(self.uninit)
        enter = before - _collect_writable_roots(node.body)
        body_uninit = self._eval_block(enter, node.body)
        # 0-iteration keeps ``before``; any completed body yields ``body_uninit``.
        self.uninit = before & body_uninit

    def _handle_for(self, node: ast.For) -> None:
        self.check_reads(node.iter)
        before = set(self.uninit)
        writable = _collect_writable_roots(node.body) | _target_root_names(node.target)
        enter = before - writable
        body_uninit = self._eval_block(enter, node.body)
        merged = before & body_uninit
        # ``orelse`` runs only on normal completion; analyze from the merged state.
        self.uninit = self._eval_block(merged, node.orelse)

    def _handle_try(self, node: ast.Try) -> None:
        before = set(self.uninit)
        body_uninit = self._eval_block(before, node.body)
        merged = set(body_uninit)
        for handler in node.handlers:
            handler_uninit = self._eval_handler(before, handler)
            merged &= handler_uninit
        # ``orelse`` follows normal body completion; ``finalbody`` runs afterwards.
        after_orelse = self._eval_block(merged, node.orelse)
        self.uninit = self._eval_block(after_orelse, node.finalbody)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def check_uninitialized_tensor_storage_reads(
    function_node: ast.FunctionDef,
    exempt_names: Optional[Set[str]] = None,
) -> None:
    """Raise ``ParserError`` when tensor storage may be read before any write.

    Parameters
    ----------
    function_node : ast.FunctionDef
        Kernel / script function to analyze.
    exempt_names : Optional[Set[str]]
        Names treated as externally initialized (e.g. parser extra vars or
        outer-scope bindings). Function parameters are always exempt.
    """
    exempt = set(exempt_names) if exempt_names else set()
    exempt |= _collect_param_names(function_node)
    checker = _Checker(exempt)
    checker.process(function_node.body)
