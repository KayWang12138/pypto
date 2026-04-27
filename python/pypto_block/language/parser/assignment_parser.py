# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Assignment parsing helpers for ASTParser."""

from __future__ import annotations

import ast
from typing import Any

from pypto_block.pypto_core import ir

from .constants import _BUFFER_CLASS_NAMES
from .diagnostics import ParserSyntaxError, ParserTypeError, UnsupportedFeatureError


class AssignmentParserMixin:
    """Mixin containing assignment parsing methods for ``ASTParser``."""

    def parse_annotated_assignment(self, stmt: ast.AnnAssign) -> None:
        """Parse annotated assignment: var: type = value.

        Args:
            stmt: AnnAssign AST node
        """
        if not isinstance(stmt.target, ast.Name):
            raise ParserSyntaxError(
                "Only simple variable assignments supported",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use a simple variable name for assignment targets",
            )

        var_name = stmt.target.id
        span = self.span_tracker.get_span(stmt)

        # Check if this is a yield assignment: var: type = pl.yield_(...)
        if isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute) and func.attr == "yield_":
                # Handle yield assignment
                yield_exprs = []
                for arg in stmt.value.args:
                    expr = self.parse_expression(arg)
                    yield_exprs.append(expr)

                # Emit yield statement
                yield_span = self.span_tracker.get_span(stmt.value)
                self.builder.emit(ir.YieldStmt(yield_exprs, yield_span))

                # Track variable name for if statement output registration
                if hasattr(self, "_current_yield_vars") and self._current_yield_vars is not None:
                    self._current_yield_vars.append(var_name)

                # Capture yield expression type for unannotated yield inference
                # Use setdefault so the then-branch type takes precedence over else
                if hasattr(self, "_current_yield_types") and self._current_yield_types is not None:
                    if len(yield_exprs) == 1:
                        self._current_yield_types.setdefault(var_name, yield_exprs[0].type)

                # Don't register in scope yet - will be done when if statement completes
                return

        # Parse value expression
        if stmt.value is None:
            raise UnsupportedFeatureError(
                "Yield assignment with no value is not supported",
                span=self.span_tracker.get_span(stmt),
                hint="Provide a value for the assignment",
            )
        value_expr = self.parse_expression(stmt.value)

        # Use annotation type as override when it carries memref info
        annotation_type = self.type_resolver.resolve_type_if_memref(stmt.annotation)
        var = self.builder.let(var_name, value_expr, type=annotation_type, span=span)

        # Register in scope
        self.scope_manager.define_var(var_name, var, span=span)

    # ------------------------------------------------------------------
    # parse_assignment helpers (Phase 2 extraction)
    # ------------------------------------------------------------------

    def _parse_yield_name_assignment(self, var_name: str, call: ast.Call, stmt_span: Any) -> bool:
        """Handle ``var = pl.yield_(...)``. Returns True if handled."""
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "yield_"):
            return False
        # Handle yield assignment
        yield_exprs = []
        for arg in call.args:
            expr = self.parse_expression(arg)
            if not isinstance(expr, ir.Expr):
                raise ParserSyntaxError(
                    f"Yield argument must be an IR expression, got {type(expr)}",
                    span=self.span_tracker.get_span(arg),
                    hint="Ensure yield arguments are valid expressions",
                )
            yield_exprs.append(expr)

        # Emit yield statement
        yield_span = self.span_tracker.get_span(call)
        self.builder.emit(ir.YieldStmt(yield_exprs, yield_span))

        # Track variable name for loop/if output registration
        if hasattr(self, "_current_yield_vars") and self._current_yield_vars is not None:
            self._current_yield_vars.append(var_name)

        # Capture yield expression type for unannotated yield inference
        # Use setdefault so the then-branch type takes precedence over else
        if hasattr(self, "_current_yield_types") and self._current_yield_types is not None:
            if len(yield_exprs) == 1:
                self._current_yield_types.setdefault(var_name, yield_exprs[0].type)

        # Don't register in scope yet - will be done when loop/if completes
        return True

    def _register_assignment_metadata(self, var_name: str, var: ir.Var, stmt: ast.Assign) -> None:
        """Register assignment metadata used by later parser lowering."""
        # Register constant-integer tuples for sync-op event_id expansion
        if isinstance(stmt.value, ast.Tuple) and all(
            isinstance(elt, ast.Constant) and isinstance(elt.value, int)
            for elt in stmt.value.elts
        ):
            self._const_tuple_registry[var_name] = [elt.value for elt in stmt.value.elts]  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # parse_assignment dispatcher
    # ------------------------------------------------------------------

    def _parse_tuple_unpacking(self, target: ast.Tuple, stmt: ast.Assign) -> None:
        """Handle tuple unpacking: (a, b, c) = pl.yield_(...) or func(...)."""
        if isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute) and func.attr == "yield_":
                self.parse_yield_assignment(target, stmt.value)
                return

        span = self.span_tracker.get_span(stmt)
        value_expr = self.parse_expression(stmt.value)
        tuple_var = self.builder.let("_tuple_tmp", value_expr, span=span)
        for i, elt in enumerate(target.elts):
            if not isinstance(elt, ast.Name):
                raise ParserSyntaxError(
                    f"Tuple unpacking target must be a variable name, got {ast.unparse(elt)}",
                    span=self.span_tracker.get_span(elt),
                    hint="Use simple variable names in tuple unpacking: a, b, c = func()",
                )
            item_expr = ir.TupleGetItemExpr(tuple_var, i, span)
            var = self.builder.let(elt.id, item_expr, span=span)
            self.scope_manager.define_var(elt.id, var, span=span)

    def _parse_name_assignment(self, var_name: str, stmt: ast.Assign) -> None:
        """Handle simple name assignment: var = expr."""
        span = self.span_tracker.get_span(stmt)

        # TileType assignment
        if isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Name) and func.id == "TileType":
                tile_type = self._parse_tile_type_call(stmt.value)
                self.scope_manager.define_python_var(var_name, tile_type, span=span)
                return
            if isinstance(func, ast.Attribute) and func.attr == "TileType":
                tile_type = self._parse_tile_type_call(stmt.value)
                self.scope_manager.define_python_var(var_name, tile_type, span=span)
                return
            # pl.NBuffer(...) / pl.Buffer(...) / pl.L1NBuffer(...) etc.
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _BUFFER_CLASS_NAMES
                and isinstance(func.value, ast.Name)
                and func.value.id == "pl"
            ):
                buf_obj = self._parse_buffer_descriptor_call(func.attr, stmt.value)
                # Inject Python variable name for readable codegen tile names
                from pypto_block.language.buffer import Buffer as _BufferCls, NBuffer as _NBufferCls
                if isinstance(buf_obj, _BufferCls):
                    buf_obj._var_name = var_name
                elif isinstance(buf_obj, _NBufferCls):
                    for i, slot in enumerate(buf_obj.slots):
                        slot._var_name = f"{var_name}_{i}"
                self.scope_manager.define_python_var(var_name, buf_obj, span=span)
                # For multi-slot NBuffers, inject an IR struct to track the cursor
                if isinstance(buf_obj, _NBufferCls) and buf_obj.num_slots > 1:
                    self._inject_nbuffer_cursor_struct(var_name, buf_obj, span)
                return
            # Delegate to struct / struct-array / yield helpers
            if self._parse_struct_assignment(var_name, stmt.value, span):
                return
            if self._parse_struct_array_assignment(var_name, stmt.value, span):
                return
            if self._parse_yield_name_assignment(var_name, stmt.value, span):
                return

        self._check_struct_in_collection(var_name, stmt.value)
        if self._parse_struct_array_subscript_alias(var_name, stmt.value, span):
            return

        # Regular expression assignment
        value_expr = self.parse_expression(stmt.value)
        if value_expr is None:
            raise ParserTypeError(
                f"Cannot assign void inline function result to '{var_name}'",
                span=span,
                hint="Inline functions used as expressions must return a value",
            )
        # If parse_expression returned a non-IR Python object (e.g.
        # BufferSlot from nbuf.current()), store as python_var rather
        # than trying to emit an IR let-binding.
        if not isinstance(value_expr, ir.Expr):
            self.scope_manager.define_python_var(var_name, value_expr, span=span)
            return
        ir_var_name = self._inline_prefix + var_name if self._inline_prefix else var_name
        var = self.builder.let(ir_var_name, value_expr, span=span)
        self.scope_manager.define_var(var_name, var, span=span)
        self._register_assignment_metadata(var_name, var, stmt)

        # Auto-mutex: emit deferred mutex_unlock AFTER the assignment
        if self._auto_mutex:
            self._emit_auto_mutex_unlocks()

    def parse_assignment(self, stmt: ast.Assign) -> None:
        """Parse regular assignment: var = value or tuple unpacking.

        Args:
            stmt: Assign AST node
        """
        if len(stmt.targets) != 1:
            raise ParserSyntaxError(
                f"Unsupported assignment: {ast.unparse(stmt)}",
                span=self.span_tracker.get_span(stmt),
                hint="Use simple variable assignments or tuple unpacking with pl.yield_()",
            )

        target = stmt.targets[0]

        if isinstance(target, ast.Tuple):
            self._parse_tuple_unpacking(target, stmt)
            return

        if isinstance(target, ast.Name):
            self._parse_name_assignment(target.id, stmt)
            return

        if isinstance(target, ast.Attribute):
            if self._parse_struct_field_assignment(target, stmt):
                return
            if self._parse_struct_array_field_assignment(target, stmt):
                return

        raise ParserSyntaxError(
            f"Unsupported assignment: {ast.unparse(stmt)}",
            span=self.span_tracker.get_span(stmt),
            hint="Use simple variable assignments or tuple unpacking with pl.yield_()",
        )

    def parse_yield_assignment(self, target: ast.Tuple, value: ast.Call) -> None:
        """Parse yield assignment: (a, b) = pl.yield_(x, y).

        Args:
            target: Tuple of target variable names
            value: Call to pl.yield_()
        """
        # Parse yield expressions
        yield_exprs = []
        for arg in value.args:
            expr = self.parse_expression(arg)
            # Ensure it's an IR Expr
            if not isinstance(expr, ir.Expr):
                raise ParserSyntaxError(
                    f"Yield argument must be an IR expression, got {type(expr)}",
                    span=self.span_tracker.get_span(arg),
                    hint="Ensure yield arguments are valid expressions",
                )
            yield_exprs.append(expr)

        # Emit yield statement
        span = self.span_tracker.get_span(value)
        self.builder.emit(ir.YieldStmt(yield_exprs, span))

        # Track yielded variable names for if/for statement processing
        if hasattr(self, "_current_yield_vars") and self._current_yield_vars is not None:
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self._current_yield_vars.append(elt.id)

        # Capture yield expression types for unannotated yield inference
        # Use setdefault so the then-branch type takes precedence over else
        if hasattr(self, "_current_yield_types") and self._current_yield_types is not None:
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name) and i < len(yield_exprs):
                    self._current_yield_types.setdefault(elt.id, yield_exprs[i].type)

        # For tuple yields at the for/while loop level, register the variables
        # (they'll be available as loop.get_result().return_vars)
        if (self.in_for_loop or self.in_while_loop) and not self.in_if_stmt:
            # Register yielded variable names in scope
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name):
                    var_name = elt.id
                    # Will be resolved from loop outputs
                    self.scope_manager.define_var(var_name, f"loop_yield_{i}")
