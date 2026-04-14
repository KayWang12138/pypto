# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AutoSyncHelper: encapsulates all sync_tracker interactions for the AST parser.

This module centralizes automatic pipeline synchronization logic that was
previously scattered across ast_parser.py.  The ASTParser delegates to
AutoSyncHelper via single-line calls at control-flow boundaries (loop
enter/exit, if/else enter/exit, manual op emission).
"""

from __future__ import annotations

import ast
import copy
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pypto_block.pypto_core import ir
from pypto_block.pypto_core.ir import MemorySpace, PipeType

from pypto_block.frontend.sync_tracker import (
    BackwardDep,
    LoopContext,
    SyncPair,
    SyncTracker,
    TileRegion,
    _OP_TILE_ACCESS,
    _OP_TO_PIPE,
    emit_backward_sync_dst,
    emit_backward_sync_src,
    emit_sync_pair,
    get_move_pipe,
    get_store_pipe,
    prescan_loop_backward_deps,
)
from pypto_block.frontend.sync_tracker.data_structures import BackwardDep as BD

if TYPE_CHECKING:
    from pypto_block.ir import IRBuilder
    from .scope_manager import ScopeManager

_MEMORY_SPACE_MAP: dict[str, MemorySpace] = {
    "Left": MemorySpace.Left,
    "Right": MemorySpace.Right,
    "Vec": MemorySpace.Vec,
    "Mat": MemorySpace.Mat,
    "Acc": MemorySpace.Acc,
}


# ---------------------------------------------------------------------------
# Module-level sync utilities (moved from ast_parser.py)
# ---------------------------------------------------------------------------

def _arch_needs_same_pipe_sync(npu_arch: str | None) -> bool:
    """Return True if the architecture requires same-pipeline sync insertion.

    dav-2201 (a2 / a3): V pipeline does not guarantee intra-pipe ordering.
    dav-3510 (a5): Hardware provides the ordering guarantee automatically.
    """
    if npu_arch is None:
        return False
    arch = npu_arch.lower()
    return "dav-2201" in arch or arch in ("a2", "a3")


def _loop_body_has_bar_all(body: list[ast.stmt]) -> bool:
    """Return True if the loop body contains a ``pl.system.bar_all()`` call."""
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if (
                isinstance(func, ast.Attribute) and func.attr == "bar_all"
                and isinstance(func.value, ast.Attribute) and func.value.attr == "system"
            ):
                return True
    return False


def _loop_body_backward_sync_redundant(body: list[ast.stmt]) -> bool:
    """Return True if the outer loop's backward sync is redundant.

    Redundant when bar_all() is present or all tile ops are inside nested loops.
    """
    if _loop_body_has_bar_all(body):
        return True
    for stmt in body:
        if isinstance(stmt, ast.For):
            continue
        if isinstance(stmt, ast.With):
            if not _loop_body_backward_sync_redundant(stmt.body):
                return False
            continue
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "plm"):
                return False
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "plm"):
                return False
            continue
    return True


# ---------------------------------------------------------------------------
# AutoSyncHelper class
# ---------------------------------------------------------------------------

class AutoSyncHelper:
    """Encapsulates all automatic pipeline synchronization logic.

    Created by ASTParser when ``auto_sync=True``.  Holds the SyncTracker
    instance and all pending backward-wait state that was previously
    scattered across ASTParser attributes.

    Args:
        builder: The IRBuilder used to emit sync IR ops.
        scope_manager: For looking up tile variables by name.
        npu_arch: Target architecture string for same-pipe sync detection.
        parse_expr_fn: Callback to parse an AST expression into an IR Expr
            (avoids reverse dependency on ASTParser).
        tile_tuple_registry: Reference to ASTParser's tile-tuple registry dict.
    """

    def __init__(
        self,
        builder: IRBuilder,
        scope_manager: ScopeManager,
        npu_arch: str | None,
        parse_expr_fn: Callable[[ast.expr], ir.Expr],
        tile_tuple_registry: dict[str, list[str]],
    ) -> None:
        same_pipe = _arch_needs_same_pipe_sync(npu_arch)
        self.tracker = SyncTracker(same_pipe_sync=same_pipe)
        self.builder = builder
        self.scope_manager = scope_manager
        self._parse_expr = parse_expr_fn
        self._tile_tuple_registry = tile_tuple_registry

        # Deferred backward waits: pipe → list of deps to emit before first op on that pipe
        self._pending_backward_waits: dict[PipeType, list[BackwardDep]] = {}
        self._pending_backward_wait_slot_expr: ir.Expr | None = None
        self._pre_if_pending_backward_waits: dict[PipeType, list[BackwardDep]] | None = None

    # -- tile registration ---------------------------------------------------

    def register_tile_region(self, var_name: str, var: ir.Var) -> None:
        """Extract MemRef from a tile Var and register with sync tracker."""
        var_type = var.type
        if not isinstance(var_type, ir.TileType):
            return
        memref = var_type.memref
        if memref is None:
            return
        addr_offset: int | None = None
        if isinstance(memref.addr_, ir.ConstInt):
            addr_offset = memref.addr_.value
        region = TileRegion(
            memory_space=memref.memory_space_,
            addr_offset=addr_offset,
            byte_size=memref.size_,
        )
        self.tracker.register_tile(var_name, region)

    # -- loop lifecycle ------------------------------------------------------

    def on_loop_pre_enter(
        self,
        body_stmts: list[ast.stmt],
        lookup_var: Callable[[str], Any],
        span: ir.Span,
    ) -> list[BackwardDep]:
        """Pre-scan loop body for backward deps, emit priming sync, enter loop.

        Must be called BEFORE the for_loop builder context so waits are
        emitted at the outer loop level.

        Returns the list of backward deps (empty if none or redundant).
        """
        self._flush_all_pending_backward_waits(span)

        backward_deps = prescan_loop_backward_deps(
            body_stmts,
            lookup_var,
            self.tracker._event_allocator,
            loop_depth=self.tracker.get_loop_depth(),
            tile_tuple_registry=self._tile_tuple_registry,
        )
        if _loop_body_backward_sync_redundant(body_stmts):
            backward_deps = []

        # Priming: emit sync_src before the loop so the first iteration's
        # wait_flag has a matching set_flag.
        for dep in backward_deps:
            if dep.n_slots > 1:
                for slot in range(dep.n_slots):
                    eid = (dep.event_id + slot) % 8
                    slot_dep = BackwardDep(dep.first_pipe, dep.last_pipe, dep.tile_name, eid, dep.loop_depth)
                    emit_backward_sync_src(self.builder, slot_dep, span)
            else:
                emit_backward_sync_src(self.builder, dep, span)

        self.tracker.enter_loop()
        return backward_deps

    def on_loop_body_start(
        self,
        backward_deps: list[BackwardDep],
        loop_var: ir.Var,
        step: ir.Expr,
        span: ir.Span,
    ) -> None:
        """Register deferred backward waits at loop body start.

        Defers each wait until the first op on its first_pipe runs, enabling
        pipeline overlap (e.g., load overlaps with previous matmul).
        """
        db_slot_expr = None
        if backward_deps and any(dep.n_slots > 1 for dep in backward_deps):
            db_slot_expr = self._build_db_slot_expr(loop_var, step, backward_deps[0].n_slots, span)
        self._pending_backward_waits = {}
        self._pending_backward_wait_slot_expr = db_slot_expr
        for dep in backward_deps:
            pipe = dep.first_pipe
            self._pending_backward_waits.setdefault(pipe, []).append(dep)

    def on_loop_body_end(
        self,
        backward_deps: list[BackwardDep],
        loop_var: ir.Var,
        step: ir.Expr,
        span: ir.Span,
    ) -> None:
        """Emit backward sync_src (set_flag) at end of loop body."""
        for dep in backward_deps:
            if dep.n_slots > 1:
                slot_expr = self._build_db_slot_expr(loop_var, step, dep.n_slots, span)
                self._emit_backward_db_sync_chain(dep, slot_expr, emit_backward_sync_src, span)
            else:
                emit_backward_sync_src(self.builder, dep, span)

    def on_loop_exit(
        self,
        backward_deps: list[BackwardDep],
        span: ir.Span,
    ) -> None:
        """Exit loop: restore pre-loop state, verify deps, emit drain sync_dst."""
        loop_ctx = self.tracker.exit_loop()
        self._verify_backward_deps(backward_deps, loop_ctx)

        # Drain: emit wait_flag to consume the last iteration's body-end set_flag
        for dep in backward_deps:
            if dep.n_slots > 1:
                for slot in range(dep.n_slots):
                    eid = (dep.event_id + slot) % 8
                    slot_dep = BackwardDep(dep.first_pipe, dep.last_pipe, dep.tile_name, eid, dep.loop_depth)
                    emit_backward_sync_dst(self.builder, slot_dep, span)
            else:
                emit_backward_sync_dst(self.builder, dep, span)

    # -- if/else lifecycle ---------------------------------------------------

    def on_if_enter(self) -> None:
        """Save pre-if buffer states and pending backward waits."""
        self.tracker.enter_if_branch()
        self._pre_if_pending_backward_waits = copy.deepcopy(self._pending_backward_waits)

    def on_else_enter(self) -> None:
        """Save then-branch states, restore pre-if for else analysis."""
        self.tracker.enter_else_branch()
        if self._pre_if_pending_backward_waits is not None:
            self._pending_backward_waits = copy.deepcopy(self._pre_if_pending_backward_waits)

    def on_if_exit(self) -> None:
        """Merge branch states conservatively."""
        self.tracker.exit_if()

    # -- pipeline fence ------------------------------------------------------

    def on_pipeline_fence(self) -> None:
        """Mark all local pipes as complete (e.g., after wait_cross_core)."""
        self.tracker.pipeline_fence()

    # -- forward sync emission -----------------------------------------------

    def emit_forward_syncs(self, op_name: str, call: ast.Call, span: ir.Span) -> None:
        """Check for cross-pipeline deps and emit sync_src/sync_dst before a manual op."""
        pipe = self._resolve_op_pipe(op_name, call)
        if pipe is None:
            return

        self._flush_pending_backward_waits(pipe, span)

        access = _OP_TILE_ACCESS.get(op_name)
        if access is None:
            return

        read_raw: list[str | list[str] | None] = [
            self._extract_tile_name_from_ast(call, i) for i in access.read_indices
        ]
        write_raw: list[str | list[str] | None] = [
            self._extract_tile_name_from_ast(call, i) for i in access.write_indices
        ]

        has_db = any(isinstance(n, list) for n in read_raw + write_raw)

        if not has_db:
            read_names = [n for n in read_raw if isinstance(n, str)]
            write_names = [n for n in write_raw if isinstance(n, str)]
            pairs = self.tracker.record_op(pipe, read_names, write_names)
            for pair in pairs:
                emit_sync_pair(self.builder, pair, span)
        else:
            self._emit_db_forward_syncs(pipe, read_raw, write_raw, call, span)

    # -- backward wait flushing ----------------------------------------------

    def _flush_pending_backward_waits(self, pipe: PipeType, span: ir.Span) -> None:
        """Emit deferred backward waits for *pipe* and remove from pending."""
        if pipe not in self._pending_backward_waits:
            return
        slot_expr = self._pending_backward_wait_slot_expr
        for dep in self._pending_backward_waits.pop(pipe):
            if dep.n_slots > 1 and slot_expr is not None:
                self._emit_backward_db_sync_chain(dep, slot_expr, emit_backward_sync_dst, span)
            else:
                emit_backward_sync_dst(self.builder, dep, span)

    def _flush_all_pending_backward_waits(self, span: ir.Span) -> None:
        """Emit ALL remaining deferred backward waits."""
        if not self._pending_backward_waits:
            return
        slot_expr = self._pending_backward_wait_slot_expr
        for pipe in list(self._pending_backward_waits.keys()):
            for dep in self._pending_backward_waits.pop(pipe):
                if dep.n_slots > 1 and slot_expr is not None:
                    self._emit_backward_db_sync_chain(dep, slot_expr, emit_backward_sync_dst, span)
                else:
                    emit_backward_sync_dst(self.builder, dep, span)

    # -- pipe resolution -----------------------------------------------------

    def _resolve_op_pipe(self, op_name: str, call: ast.Call) -> PipeType | None:
        """Determine the pipeline for a manual op."""
        if op_name == "move":
            return self._resolve_move_pipe(call)
        if op_name in ("store", "store_tile"):
            return self._resolve_store_pipe(call, op_name)
        return _OP_TO_PIPE.get(op_name)

    def _resolve_move_pipe(self, call: ast.Call) -> PipeType:
        """Resolve the pipeline for a ``move`` op."""
        src_memory: MemorySpace | None = None
        target_memory: MemorySpace | None = None
        for kw in call.keywords:
            if kw.arg == "src_memory" and isinstance(kw.value, ast.Attribute):
                src_memory = _MEMORY_SPACE_MAP.get(kw.value.attr)
            elif kw.arg == "target_memory" and isinstance(kw.value, ast.Attribute):
                target_memory = _MEMORY_SPACE_MAP.get(kw.value.attr)
        if target_memory is None and len(call.args) >= 1:
            target_memory = self._resolve_tile_arg_memory_space(call.args[0])
        if src_memory is None and len(call.args) >= 2:
            src_memory = self._resolve_tile_arg_memory_space(call.args[1])
        return get_move_pipe(src_memory, target_memory)

    def _resolve_store_pipe(self, call: ast.Call, op_name: str) -> PipeType:
        """Resolve the pipeline for a ``store`` / ``store_tile`` op."""
        src_memory: MemorySpace | None = None
        if 1 < len(call.args):
            src_memory = self._resolve_tile_arg_memory_space(call.args[1])
        return get_store_pipe(src_memory)

    def _resolve_tile_arg_memory_space(self, arg: ast.expr) -> MemorySpace | None:
        """Resolve the memory space of a tile argument (Name or Subscript)."""
        def _get_memory_space_from_var(var_name: str) -> MemorySpace | None:
            var = self.scope_manager.lookup_var(var_name)
            if var is None:
                return None
            var_type = getattr(var, "type", None)
            if var_type is None:
                return None
            memref = getattr(var_type, "memref", None)
            if memref is not None:
                return getattr(memref, "memory_space_", None)
            return getattr(var_type, "memory_space", None)

        if isinstance(arg, ast.Name):
            return _get_memory_space_from_var(arg.id)
        if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
            tuple_name = arg.value.id
            if tuple_name in self._tile_tuple_registry:
                first_tile_name = self._tile_tuple_registry[tuple_name][0]
                return _get_memory_space_from_var(first_tile_name)
        return None

    # -- tile name extraction ------------------------------------------------

    def _extract_tile_name_from_ast(self, call: ast.Call, idx: int) -> str | list[str] | None:
        """Extract tile variable name(s) from a positional arg at *idx*."""
        if idx >= len(call.args):
            return None
        arg = call.args[idx]
        if isinstance(arg, ast.Name):
            var = self.scope_manager.lookup_var(arg.id)
            if var is None:
                return None
            var_type = getattr(var, "type", None)
            if var_type is None or not isinstance(var_type, ir.TileType):
                return None
            return arg.id
        if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
            tuple_name = arg.value.id
            if tuple_name in self._tile_tuple_registry:
                return self._tile_tuple_registry[tuple_name]
        return None

    @staticmethod
    def _extract_plm_or_block_op_name(call: ast.Call) -> str | None:
        """Extract op name from a ``plm.xxx(...)`` call, or None."""
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "plm"
        ):
            return func.attr
        return None

    # -- DB forward sync -----------------------------------------------------

    def _emit_db_forward_syncs(
        self,
        pipe: PipeType,
        read_raw: list[str | list[str] | None],
        write_raw: list[str | list[str] | None],
        call: ast.Call,
        span: ir.Span,
    ) -> None:
        """Emit forward sync for double-buffer tile tuples."""
        tracker = self.tracker
        n_slots = 2
        for n in read_raw + write_raw:
            if isinstance(n, list):
                n_slots = len(n)
                break

        index_expr = self._extract_db_index_expr(call)
        if index_expr is None:
            return

        saved_states = copy.deepcopy(tracker._buffer_states)
        slot_pairs: list[list[SyncPair]] = []

        for slot in range(n_slots):
            tracker._buffer_states = copy.deepcopy(saved_states)
            read_names = [self._resolve_slot_name(n, slot) for n in read_raw]
            write_names = [self._resolve_slot_name(n, slot) for n in write_raw]
            read_names = [n for n in read_names if n is not None]
            write_names = [n for n in write_names if n is not None]
            pairs = tracker.record_op(pipe, read_names, write_names)
            slot_pairs.append(pairs)

        merged = copy.deepcopy(saved_states)
        for slot in range(n_slots):
            tracker._buffer_states = copy.deepcopy(saved_states)
            read_names = [self._resolve_slot_name(n, slot) for n in read_raw]
            write_names = [self._resolve_slot_name(n, slot) for n in write_raw]
            read_names = [n for n in read_names if n is not None]
            write_names = [n for n in write_names if n is not None]
            tracker.record_op(pipe, read_names, write_names)
            for tile_name, state in tracker._buffer_states.items():
                if tile_name not in saved_states or state != saved_states.get(tile_name):
                    merged[tile_name] = copy.deepcopy(state)
        tracker._buffer_states = merged

        all_pipe_keys: list[tuple[PipeType, PipeType]] = []
        seen: set[tuple[PipeType, PipeType]] = set()
        for pairs in slot_pairs:
            for p in pairs:
                key = (p.set_pipe, p.wait_pipe)
                if key not in seen:
                    seen.add(key)
                    all_pipe_keys.append(key)

        for set_p, wait_p in all_pipe_keys:
            slot_event_ids: list[int] = []
            for slot, pairs in enumerate(slot_pairs):
                matching = [p for p in pairs if p.set_pipe == set_p and p.wait_pipe == wait_p]
                if matching:
                    base_eid = tracker._event_allocator.forward_event_id(set_p, wait_p, n_slots=n_slots)
                    slot_event_ids.append((base_eid + slot) % tracker._event_allocator.MAX_EVENTS)
                else:
                    slot_event_ids.append(-1)
            self._build_db_sync_chain(set_p, wait_p, slot_event_ids, index_expr, 0, span)

    def _extract_db_index_expr(self, call: ast.Call) -> ir.Expr | None:
        """Extract the buffer-index IR expression from the first Subscript arg."""
        for arg in call.args:
            if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
                if arg.value.id in self._tile_tuple_registry:
                    return self._parse_expr(arg.slice)
        return None

    @staticmethod
    def _resolve_slot_name(raw: str | list[str] | None, slot: int) -> str | None:
        """Resolve a raw tile name to a specific slot's tile name."""
        if raw is None:
            return None
        if isinstance(raw, str):
            return raw
        if slot < len(raw):
            return raw[slot]
        return None

    # -- DB sync chain builders ----------------------------------------------

    def _build_db_sync_chain(
        self,
        set_pipe: PipeType,
        wait_pipe: PipeType,
        slot_event_ids: list[int],
        index_expr: ir.Expr,
        level: int,
        span: ir.Span,
    ) -> None:
        """Recursively build if-else chain emitting sync_src+sync_dst per slot."""
        n = len(slot_event_ids)
        eid = slot_event_ids[level]

        if level == n - 1:
            if eid >= 0:
                pair = SyncPair(set_pipe, wait_pipe, "raw", eid)
                emit_sync_pair(self.builder, pair, span)
            return

        if eid < 0:
            cond = index_expr == level
            with self.builder.if_stmt(cond, span) as if_b:
                pass
                if_b.else_()
                self._build_db_sync_chain(set_pipe, wait_pipe, slot_event_ids, index_expr, level + 1, span)
            return

        cond = index_expr == level
        with self.builder.if_stmt(cond, span) as if_b:
            pair = SyncPair(set_pipe, wait_pipe, "raw", eid)
            emit_sync_pair(self.builder, pair, span)
            if_b.else_()
            self._build_db_sync_chain(set_pipe, wait_pipe, slot_event_ids, index_expr, level + 1, span)

    def _build_db_slot_expr(
        self, loop_var: ir.Var, step: ir.Expr, n_slots: int, span: ir.Span,
    ) -> ir.Expr:
        """Build ``(loop_var / step) % n_slots`` as an IR expression."""
        div_expr = loop_var // step
        mod_expr = div_expr % n_slots
        return mod_expr

    # -- backward DB sync chain ----------------------------------------------

    def _emit_backward_db_sync_chain(
        self,
        dep: BackwardDep,
        slot_expr: ir.Expr,
        emit_fn: Callable,
        span: ir.Span,
    ) -> None:
        """Emit per-slot backward sync via if-else chain on *slot_expr*."""
        self._build_backward_db_chain(dep, slot_expr, emit_fn, 0, span)

    def _build_backward_db_chain(
        self,
        dep: BackwardDep,
        slot_expr: ir.Expr,
        emit_fn: Callable,
        level: int,
        span: ir.Span,
    ) -> None:
        n = dep.n_slots
        eid = (dep.event_id + level) % 8
        slot_dep = BD(dep.first_pipe, dep.last_pipe, dep.tile_name, eid, dep.loop_depth)

        if level == n - 1:
            emit_fn(self.builder, slot_dep, span)
            return

        cond = slot_expr == level
        with self.builder.if_stmt(cond, span) as if_b:
            emit_fn(self.builder, slot_dep, span)
            if_b.else_()
            self._build_backward_db_chain(dep, slot_expr, emit_fn, level + 1, span)

    # -- verification --------------------------------------------------------

    def _verify_backward_deps(self, prescan_deps: list, loop_ctx: object) -> None:
        """Warn if prescan backward deps differ from actual loop body access."""
        if not isinstance(loop_ctx, LoopContext):
            return
        actual_deps: set[tuple] = set()
        for tile_name in loop_ctx.first_access:
            first = loop_ctx.first_access[tile_name]
            last = loop_ctx.last_access.get(tile_name, first)
            if first != last:
                actual_deps.add((first, last, tile_name))

        prescan_set = {(d.first_pipe, d.last_pipe, d.tile_name) for d in prescan_deps}
        missed = actual_deps - prescan_set
        if missed:
            for first, last, name in missed:
                warnings.warn(
                    f"Auto-sync: prescan missed backward dep for tile '{name}' "
                    f"(first={first.name}, last={last.name}). "
                    f"Consider adding manual sync.",
                    stacklevel=2,
                )
