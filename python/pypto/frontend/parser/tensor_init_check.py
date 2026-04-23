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
"""

from __future__ import annotations

import ast
from typing import Optional, Set

from pypto.error import ParserError

_MSG = (
    "read from tensor storage before it is written; assign to the tensor (e.g. "
    "`t[:] = ...`) before reading elements"
)


def _is_tensor_allocator_call(expr: ast.expr) -> bool:
    if not isinstance(expr, ast.Call):
        return False
    func = expr.func
    if isinstance(func, ast.Name) and func.id in ("Tensor", "tensor"):
        return True
    if isinstance(func, ast.Attribute) and func.attr in ("Tensor", "tensor"):
        return True
    return False


def _collect_param_names(node: ast.FunctionDef) -> set[str]:
    names = {a.arg for a in node.args.args}
    names |= {a.arg for a in node.args.kwonlyargs}
    if node.args.vararg:
        names.add(node.args.vararg.arg)
    if node.args.kwarg:
        names.add(node.args.kwarg.arg)
    return names


def _roots_written_by_assign_target(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Subscript):
        cur: ast.expr = target
        while isinstance(cur, ast.Subscript):
            cur = cur.value
        if isinstance(cur, ast.Name):
            return [cur.id]
        return []
    return []


class _TensorStorageInitChecker:
    def __init__(self, exempt: Set[str]) -> None:
        self._exempt = exempt
        self.uninit: Set[str] = set()

    def visit_assign(self, node: ast.Assign) -> None:
        self._check_reads(node.value)
        val = node.value
        is_alloc = _is_tensor_allocator_call(val)
        ru = self._rhs_reads_uninit(val)
        for target in node.targets:
            roots = _roots_written_by_assign_target(target)
            if isinstance(target, (ast.Tuple, ast.List)):
                self._apply_unpack_names(target, is_alloc, ru)
                continue
            if isinstance(target, ast.Name):
                if is_alloc:
                    self.uninit.add(target.id)
                elif ru:
                    self.uninit.add(target.id)
                else:
                    self.uninit.discard(target.id)
                continue
            if isinstance(target, ast.Subscript):
                for r in roots:
                    self.uninit.discard(r)

    def visit_ann_assign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        self._check_reads(node.value)
        val = node.value
        is_alloc = _is_tensor_allocator_call(val)
        ru = self._rhs_reads_uninit(val)
        target = node.target
        roots = _roots_written_by_assign_target(target)
        if isinstance(target, ast.Name):
            if is_alloc:
                self.uninit.add(target.id)
            elif ru:
                self.uninit.add(target.id)
            else:
                self.uninit.discard(target.id)
        elif isinstance(target, ast.Subscript):
            for r in roots:
                self.uninit.discard(r)

    def visit_aug_assign(self, node: ast.AugAssign) -> None:
        self._check_reads(node.target)
        self._check_reads(node.value)
        cur: ast.expr = node.target
        while isinstance(cur, ast.Subscript):
            cur = cur.value
        if isinstance(cur, ast.Name):
            self.uninit.discard(cur.id)

    def visit_block(self, stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            self.visit_stmt(stmt)

    def visit_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            self.visit_assign(stmt)
            return
        if isinstance(stmt, ast.AnnAssign):
            self.visit_ann_assign(stmt)
            return
        if isinstance(stmt, ast.AugAssign):
            self.visit_aug_assign(stmt)
            return
        if isinstance(stmt, ast.Expr):
            self._check_reads(stmt.value)
            return
        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                self._check_reads(stmt.value)
            return
        if isinstance(stmt, ast.If):
            self._check_reads(stmt.test)
            before = set(self.uninit)
            ct = _TensorStorageInitChecker(self._exempt)
            ct.uninit = set(before)
            ct.visit_block(stmt.body)
            ce = _TensorStorageInitChecker(self._exempt)
            ce.uninit = set(before)
            ce.visit_block(stmt.orelse)
            self.uninit.clear()
            self.uninit.update(ct.uninit | ce.uninit)
            return
        if isinstance(stmt, ast.While):
            self._check_reads(stmt.test)
            before = set(self.uninit)
            cw = _TensorStorageInitChecker(self._exempt)
            cw.uninit = set(before)
            cw.visit_block(stmt.body)
            self.uninit.clear()
            self.uninit.update(before & cw.uninit)
            return
        if isinstance(stmt, ast.For):
            self._check_reads(stmt.iter)
            before = set(self.uninit)
            cf = _TensorStorageInitChecker(self._exempt)
            cf.uninit = set(before)
            cf.visit_block(stmt.body)
            self.uninit.clear()
            self.uninit.update(before & cf.uninit)
            cf_else = _TensorStorageInitChecker(self._exempt)
            cf_else.uninit = set(self.uninit)
            cf_else.visit_block(stmt.orelse)
            self.uninit.clear()
            self.uninit.update(cf_else.uninit)
            return
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                self._check_reads(item.context_expr)
            self.visit_block(stmt.body)
            return
        if isinstance(stmt, ast.Try):
            before = set(self.uninit)
            ct = _TensorStorageInitChecker(self._exempt)
            ct.uninit = set(before)
            ct.visit_block(stmt.body)
            merged = set(ct.uninit)
            for h in stmt.handlers:
                ch = _TensorStorageInitChecker(self._exempt)
                ch.uninit = set(before)
                if h.type is not None:
                    ch._check_reads(h.type)
                exc_name = h.name
                if isinstance(exc_name, ast.Name):
                    ch.uninit.discard(exc_name.id)
                elif isinstance(exc_name, str):
                    ch.uninit.discard(exc_name)
                ch.visit_block(h.body)
                merged |= ch.uninit
            self.uninit.clear()
            self.uninit.update(merged)
            co = _TensorStorageInitChecker(self._exempt)
            co.uninit = set(self.uninit)
            co.visit_block(stmt.orelse)
            self.uninit.clear()
            self.uninit.update(co.uninit)
            cf = _TensorStorageInitChecker(self._exempt)
            cf.uninit = set(self.uninit)
            cf.visit_block(stmt.finalbody)
            self.uninit.clear()
            self.uninit.update(cf.uninit)
            return
        if isinstance(stmt, ast.Assert):
            self._check_reads(stmt.test)
            if stmt.msg is not None:
                self._check_reads(stmt.msg)
            return
        if isinstance(stmt, ast.FunctionDef):
            check_uninitialized_tensor_storage_reads(stmt, exempt_names=self._exempt)
            return
        if isinstance(stmt, ast.Delete):
            for t in stmt.targets:
                self._check_reads(t)
            return

    def _rhs_reads_uninit(self, rhs: ast.expr) -> bool:
        for child in ast.walk(rhs):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id in self._exempt:
                    continue
                if child.id in self.uninit:
                    return True
        return False

    def _maybe_raise_name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id in self._exempt:
                return
            if node.id in self.uninit:
                raise ParserError(node, ValueError(_MSG))

    def _check_reads(self, expr: ast.expr) -> None:
        for child in ast.walk(expr):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                self._maybe_raise_name(child)

    def _apply_unpack_names(
        self, target: ast.expr, is_alloc: bool, ru: bool
    ) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                if isinstance(elt, ast.Starred):
                    continue
                if isinstance(elt, (ast.Tuple, ast.List)):
                    self._apply_unpack_names(elt, is_alloc, ru)
                elif isinstance(elt, ast.Name):
                    if is_alloc:
                        self.uninit.add(elt.id)
                    elif ru:
                        self.uninit.add(elt.id)
                    else:
                        self.uninit.discard(elt.id)
            return
        if isinstance(target, ast.Name):
            if is_alloc:
                self.uninit.add(target.id)
            elif ru:
                self.uninit.add(target.id)
            else:
                self.uninit.discard(target.id)


def check_uninitialized_tensor_storage_reads(
    function_node: ast.FunctionDef,
    exempt_names: Optional[set[str]] = None,
) -> None:
    """Raise ParserError when tensor storage may be read before any write.

    Parameters
    ----------
    function_node : ast.FunctionDef
        Kernel / script function to analyze.
    exempt_names : Optional[set[str]]
        Names treated as externally initialized (e.g. parser extra vars).
    """
    exempt_names = exempt_names or set()
    exempt = exempt_names | _collect_param_names(function_node)
    checker = _TensorStorageInitChecker(exempt)
    checker.visit_block(function_node.body)
