# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Buffer/NBuffer parsing helpers for ASTParser."""

from __future__ import annotations

import ast
from typing import Any

from pypto_block.pypto_core import DataType, ir

from .diagnostics import ParserSyntaxError, ParserTypeError
from .models import _StructVar


class BufferParserMixin:
    """Mixin containing Python-level Buffer and NBuffer descriptor helpers."""

    def _parse_buffer_descriptor_call(self, class_name: str, call: ast.Call) -> Any:
        """Parse pl.NBuffer(...) / pl.Buffer(...) / pl.L1NBuffer(...) etc.

        These are pure Python descriptors — not IR operations. All kwargs
        (shape, dtype, memory, base_addr, buf_ids, ...) are Python literals
        or closure constants, so we evaluate them via closure rather than
        through the IR-expression path (which would convert tuples into
        MakeTuple nodes, for example).
        """
        import pypto_block.language.buffer as _buf_mod

        cls = getattr(_buf_mod, class_name, None)
        if cls is None:
            raise ParserSyntaxError(
                f"Unknown buffer class: pl.{class_name}",
                span=self.span_tracker.get_span(call),
            )

        call_span = self.span_tracker.get_span(call)

        # Resolve positional args (canndsl convention: base_addr, dtype).
        pos_args = []
        for arg in call.args:
            try:
                pos_args.append(self.expr_evaluator.eval_expr(arg))
            except Exception:
                # dtype may need type_resolver
                try:
                    pos_args.append(self.type_resolver.resolve_dtype(arg))
                except Exception as exc2:
                    raise ParserTypeError(
                        f"pl.{class_name}() positional arg must be a static value: {exc2}",
                        span=self.span_tracker.get_span(arg),
                    ) from exc2

        kwargs: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is None:
                raise ParserSyntaxError(
                    f"pl.{class_name}() does not support **kwargs",
                    span=call_span,
                )
            # Dtype needs the type-resolver path so pl.FP16 / pl.FP32 resolve
            # correctly; everything else is a plain Python value.
            if kw.arg == "dtype":
                kwargs[kw.arg] = self.type_resolver.resolve_dtype(kw.value)
                continue
            try:
                kwargs[kw.arg] = self.expr_evaluator.eval_expr(kw.value)
            except Exception as exc:
                raise ParserTypeError(
                    f"pl.{class_name}() kwarg '{kw.arg}' must be a static "
                    f"Python value (constant, tuple, enum, ...): {exc}",
                    span=self.span_tracker.get_span(kw.value),
                ) from exc

        return cls(*pos_args, **kwargs)

    def _inject_nbuffer_cursor_struct(self, var_name: str, nbuf, span: ir.Span) -> None:
        """Inject an IR struct to track the auto-rotate cursor for a multi-slot NBuffer.

        Emits ``struct.declare`` + ``struct.set`` for the cursor field (init 0),
        and stores the struct name on the NBuffer object for later use by
        :meth:`_parse_nbuffer_method_call`.
        """
        struct_name = f"_nbuf_{var_name}_ctx"
        # Create _StructVar for scope tracking (cursor field is managed
        # entirely by struct.get/set — no separate IR variable needed).
        struct_var = _StructVar({"cursor": None}, name=struct_name)
        self.scope_manager.define_python_var(struct_name, struct_var, span=span)

        # Emit struct.declare
        decl_call = ir.create_op_call(
            "struct.declare", [],
            {"array": struct_name, "size": 1, "fields": "cursor"},
            span,
        )
        self.builder.emit(ir.EvalStmt(decl_call, span))
        # No struct.set needed — cursor init is 0 (default)

        # Also materialize tile tuple as an IR MakeTuple so tuple-index works.
        tile_exprs = []
        for slot in nbuf.slots:
            managed = slot.tile  # _TileRef
            tile_exprs.append(managed.unwrap())
        tile_tuple_expr = ir.MakeTuple(tile_exprs, span)
        tile_tuple_var = self.builder.let(f"_nbuf_{var_name}_tiles", tile_tuple_expr, span=span)

        # Store metadata on the NBuffer for _parse_nbuffer_method_call
        nbuf._ir_struct_name = struct_name
        nbuf._ir_tile_tuple_var = tile_tuple_var
        nbuf._ir_tile_tuple_size = nbuf.num_slots

    def _parse_nbuffer_method_call(self, nbuf, method_name: str, call: ast.Call, span: ir.Span):
        """Generate IR for ``nbuf.current()`` / ``nbuf.previous()`` with auto-rotate.

        For ``current()``:
          1. cursor = struct.get(_cursor)
          2. buf_idx = cursor % num_slots
          3. tile = tiles[buf_idx]  (if-else chain)
          4. struct.set(_cursor, cursor + 1)  (advance for next call)
          5. Return BufferSlot(tile=_TileRef(tile_ir, buf_id_ir), buf_id=buf_id_ir)

        For ``previous()``:
          1. cursor = struct.get(_cursor)
          2. buf_idx = (cursor - 1) % num_slots
          3. tile = tiles[buf_idx]
          4. No cursor advance
          5. Return BufferSlot
        """
        from pypto_block.language.buffer import BufferSlot, _TileRef

        struct_name = nbuf._ir_struct_name
        tile_tuple_var = nbuf._ir_tile_tuple_var
        n_slots = nbuf._ir_tile_tuple_size

        # Read cursor from struct.
        # Note: keep as a raw Call — PTO codegen emits memref.load each time
        # it encounters this expression. Do NOT let-bind: PTO's AssignStmt
        # handler for backend ops doesn't register non-tile variable mappings.
        idx_zero = ir.ConstInt(0, DataType.INDEX, span)
        cursor_expr = ir.create_op_call(
            "struct.get", [idx_zero],
            {"array": struct_name, "field": "cursor"}, span,
        )

        # Compute buf_idx
        n_const = ir.ConstInt(n_slots, DataType.INDEX, span)
        if method_name == "current":
            buf_idx_raw = ir.create_op_call(
                "arith.mod", [cursor_expr, n_const], {}, span,
            ) if n_slots > 2 else (cursor_expr % n_const)
        else:  # previous
            one = ir.ConstInt(1, DataType.INDEX, span)
            shifted = cursor_expr - one
            # For 2-slot: (cursor - 1) % 2 ≡ cursor + 1 mod 2 ≡ 1 - cursor%2
            buf_idx_raw = shifted % n_const

        # Snapshot buf_idx into a let-bound Var BEFORE the cursor advance below.
        # This decouples the tile-selection index from the advance, so codegen
        # emits `bufidx = cursor % N; cursor = cursor + 1; ... use arr[bufidx]`
        # rather than advancing the cursor before any use reads `cursor % N`.
        buf_idx_name = f"_bufidx_{self._tuple_idx_counter}"
        self._tuple_idx_counter += 1
        buf_idx_expr = self.builder.let(buf_idx_name, buf_idx_raw, span=span)

        # Build tuple-index if-else chain for tile selection
        elem_type = tile_tuple_var.type.fields[0] if hasattr(tile_tuple_var.type, 'fields') else None
        if elem_type is None:
            # Fallback: get type from the first tile
            elem_type = ir.TupleGetItemExpr(tile_tuple_var, 0, span).type

        tile_ir = self._build_tuple_index_chain(
            tile_tuple_var, buf_idx_expr, elem_type, n_slots, 0, span,
        )

        # Build buf_id selection only when NBuffer has buf_ids (Mutex sync enabled)
        buf_id_ir = None
        if nbuf.has_buf_ids:
            buf_id_exprs = [ir.ConstInt(bid, DataType.INDEX, span) for bid in nbuf.buf_ids]
            buf_id_tuple = ir.MakeTuple(buf_id_exprs, span)
            buf_id_ir = self._build_tuple_index_chain(
                buf_id_tuple, buf_idx_expr, ir.ScalarType(DataType.INDEX), n_slots, 0, span,
            )

        # For current(): advance cursor (struct.set cursor = cursor + 1)
        if method_name == "current":
            one = ir.ConstInt(1, DataType.INDEX, span)
            next_cursor = cursor_expr + one
            set_call = ir.create_op_call(
                "struct.set", [idx_zero, next_cursor],
                {"array": struct_name, "field": "cursor"}, span,
            )
            self.builder.emit(ir.EvalStmt(set_call, span))

        # Return a BufferSlot with dynamic tile + optional buf_id
        slot_memory = nbuf.slots[0].spec.memory if nbuf.slots else None
        buf_id_values = nbuf.buf_ids if nbuf.has_buf_ids else None
        managed_tile = _TileRef(tile_ir, buf_id_ir, buf_id_values=buf_id_values, memory=slot_memory)
        return BufferSlot(managed_tile, buf_id_ir)

