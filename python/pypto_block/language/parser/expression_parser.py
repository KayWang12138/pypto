# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Expression parsing helpers for ASTParser."""

from __future__ import annotations

import ast
from typing import Any

from pypto_block.ir import op as ir_op
from pypto_block.pypto_core import DataType, ir

from .constants import _MEMORY_SPACE_MAP
from .diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)
from .models import _DynamicStructView, _StructArrayVar


class ExpressionParserMixin:
    """Mixin containing expression, attribute, and subscript parsing."""

    # -----------------------------------------------------------------------
    # Sync-op statement expander — event_id=tuple[index] support
    # -----------------------------------------------------------------------

    _SYNC_OP_NAMES: frozenset[str] = frozenset({"sync_src", "sync_dst"})

    def _expand_stmt_level(self, node: ast.expr) -> bool:
        """Try to expand a system sync-op call with a subscript event_id at statement level.

        When a sync op is written as:
            pl.system.sync_src(..., event_id=event_ids[buf_idx])
        where ``event_ids`` is a tuple of integer constants, this expands it
        into an if-else chain so each branch contains a sync call with a
        static constant event_id.

        Returns True if expansion was performed (caller should skip normal handling).

        Note: kept for potential future statement-level expansion use cases.
        """
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in self._SYNC_OP_NAMES
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "system"
        ):
            return False
        # Find event_id kwarg with a subscript value
        event_id_kw = next(
            (kw for kw in node.keywords if kw.arg == "event_id" and isinstance(kw.value, ast.Subscript)),
            None,
        )
        if event_id_kw is None:
            return False

        span = self.span_tracker.get_span(node)
        subscript = event_id_kw.value
        assert isinstance(subscript, ast.Subscript)

        # Resolve the tuple of integer constants
        constants = self._resolve_const_event_id_tuple(subscript.value, span)
        if constants is None:
            return False  # fall through to normal handling, which will error on ConvertKwargsDict

        # Parse index expression
        index_expr = self.parse_expression(subscript.slice)

        # Parse all other kwargs (excluding event_id)
        other_kwargs: dict[str, Any] = {
            kw.arg: self._resolve_single_kwarg(kw.arg, kw.value)
            for kw in node.keywords
            if kw.arg is not None and kw.arg != "event_id"
        }

        op_name = func.attr
        op_func = getattr(ir_op.system, op_name)
        self._build_sync_event_chain(op_func, other_kwargs, index_expr, constants, 0, span)
        return True

    def _resolve_const_event_id_tuple(self, node: ast.expr, span: ir.Span) -> list[int] | None:
        """Return the list of integer constants for a tuple AST node, or None if not resolvable."""
        # Case 1: literal tuple, e.g. (0, 1)
        if isinstance(node, ast.Tuple):
            if all(isinstance(elt, ast.Constant) and isinstance(elt.value, int) for elt in node.elts):
                return [elt.value for elt in node.elts]  # type: ignore[union-attr]
        # Case 2: name in constant registry, e.g. event_ids = (0, 1) assigned earlier
        if isinstance(node, ast.Name) and node.id in self._const_tuple_registry:
            return self._const_tuple_registry[node.id]
        return None

    def _build_sync_event_chain(
        self,
        op_func: Any,
        other_kwargs: dict[str, Any],
        index_expr: ir.Expr,
        constants: list[int],
        level: int,
        span: ir.Span,
    ) -> None:
        """Recursively build if-else chain of sync calls with constant event_ids.

        Generates:
            if index == 0: sync_op(event_id=constants[0])
            else:
              if index == 1: sync_op(event_id=constants[1])
              else: sync_op(event_id=constants[n-1])   # leaf
        """
        n = len(constants)
        if level == n - 1:
            # Leaf: always emit this variant
            call_expr = op_func(**other_kwargs, event_id=constants[level], span=span)
            self.builder.eval_stmt(call_expr, span)
            return

        cond = index_expr == level
        with self.builder.if_stmt(cond, span) as if_b:
            # Then branch
            call_expr = op_func(**other_kwargs, event_id=constants[level], span=span)
            self.builder.eval_stmt(call_expr, span)
            if_b.else_()
            # Else branch: recurse
            self._build_sync_event_chain(op_func, other_kwargs, index_expr, constants, level + 1, span)
            # No return_var — this is a non-value-returning scf.if

    def parse_evaluation_statement(self, stmt: ast.Expr) -> None:
        """Parse evaluation statement (EvalStmt).

        Evaluation statements represent operations executed for their side effects,
        with the return value discarded (e.g., synchronization barriers).

        Args:
            stmt: Expr AST node
        """
        expr = self.parse_expression(stmt.value)
        span = self.span_tracker.get_span(stmt)

        # Void inline functions or python_var method calls (e.g. nbuf.advance())
        # return None or a non-IR sentinel — nothing to emit.
        if expr is None or not isinstance(expr, ir.Expr):
            return

        # Emit EvalStmt using builder method
        self.builder.eval_stmt(expr, span)

        # Auto-mutex: emit deferred mutex_unlock AFTER the op
        if self._auto_mutex:
            self._emit_auto_mutex_unlocks()

    def parse_expression(self, expr: ast.expr) -> ir.Expr:
        """Parse expression and return IR Expr.

        Args:
            expr: AST expression node

        Returns:
            IR expression
        """
        if isinstance(expr, ast.Name):
            return self.parse_name(expr)
        elif isinstance(expr, ast.Constant):
            return self.parse_constant(expr)
        elif isinstance(expr, ast.BinOp):
            return self.parse_binop(expr)
        elif isinstance(expr, ast.Compare):
            return self.parse_compare(expr)
        elif isinstance(expr, ast.Call):
            return self.parse_call(expr)
        elif isinstance(expr, ast.Attribute):
            return self.parse_attribute(expr)
        elif isinstance(expr, ast.UnaryOp):
            return self.parse_unaryop(expr)
        elif isinstance(expr, ast.List):
            return self.parse_list(expr)
        elif isinstance(expr, ast.Tuple):
            return self.parse_tuple_literal(expr)
        elif isinstance(expr, ast.Subscript):
            return self.parse_subscript(expr)
        else:
            raise UnsupportedFeatureError(
                f"Unsupported expression type: {type(expr).__name__}",
                span=self.span_tracker.get_span(expr),
                hint="Use supported expressions like variables, constants, operations, or function calls",
            )

    def parse_name(self, name: ast.Name) -> ir.Expr | Any:
        """Parse variable name reference.

        Resolves names by checking the DSL scope first, then falling back
        to closure variables from the enclosing Python scope.

        Args:
            name: Name AST node

        Returns:
            IR expression (Var from scope, or constant/tuple from closure)
        """
        var_name = name.id
        # Check if it's a Python variable (non-IR value like TileType)
        python_var = self.scope_manager.get_python_var(var_name)
        if python_var is not None:
            return python_var

        # In inline mode, restrict IR variable lookup to the inline scope only.
        # Variables from the caller's scope must be passed as function arguments.
        # Exception: tile buffers (TileType, TupleType, TensorType) are allowed
        # from outer scope since they represent hardware resources, not scalars.
        if self._inline_mode:
            var = self.scope_manager.lookup_var_bounded(var_name, barrier="inline")
            if var is None:
                # Check if the variable exists in outer scope and is a tile/tensor/tuple
                outer_var = self.scope_manager.lookup_var(var_name)
                if outer_var is not None and hasattr(outer_var, "type"):
                    vtype = outer_var.type
                    if (isinstance(vtype, ir.TileType)
                            or isinstance(vtype, ir.TupleType)
                            or isinstance(vtype, ir.TensorType)):
                        var = outer_var
        else:
            var = self.scope_manager.lookup_var(var_name)

        if var is not None:
            return var

        # Fall back to closure variables
        result = self.expr_evaluator.try_eval_as_ir(name)
        if result is not None:
            return result

        raise UndefinedVariableError(
            f"Undefined variable '{var_name}'",
            span=self.span_tracker.get_span(name),
            hint="Check if the variable is defined before using it or is available in the enclosing scope",
        )

    def parse_constant(self, const: ast.Constant) -> ir.Expr:
        """Parse constant value.

        Args:
            const: Constant AST node

        Returns:
            IR constant expression
        """
        span = self.span_tracker.get_span(const)
        value = const.value

        if isinstance(value, bool):
            return ir.ConstBool(value, span)
        elif isinstance(value, int):
            return ir.ConstInt(value, DataType.INDEX, span)
        elif isinstance(value, float):
            return ir.ConstFloat(value, DataType.DEFAULT_CONST_FLOAT, span)
        else:
            raise ParserTypeError(
                f"Unsupported constant type: {type(value)}",
                span=self.span_tracker.get_span(const),
                hint="Use int, float, or bool constants",
            )

    def parse_binop(self, binop: ast.BinOp) -> ir.Expr:
        """Parse binary operation.

        Args:
            binop: BinOp AST node

        Returns:
            IR binary expression
        """
        span = self.span_tracker.get_span(binop)
        left = self.parse_expression(binop.left)
        right = self.parse_expression(binop.right)

        # Tile + offset in VF scope → TileOffsetExpr (pointer arithmetic)
        if isinstance(binop.op, ast.Add) and isinstance(left.type, ir.TileType):
            return ir.TileOffsetExpr(left, right, span)

        op_map = {
            ast.Add: ir.add,
            ast.Sub: ir.sub,
            ast.Mult: ir.mul,
            ast.Div: ir.truediv,
            ast.FloorDiv: ir.floordiv,
            ast.Mod: ir.mod,
        }

        op_type = type(binop.op)
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported binary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(binop),
                hint="Use supported operators: +, -, *, /, //, %",
            )

        return op_map[op_type](left, right, span)

    def parse_compare(self, compare: ast.Compare) -> ir.Expr:
        """Parse comparison operation.

        Args:
            compare: Compare AST node

        Returns:
            IR comparison expression
        """
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            raise ParserSyntaxError(
                "Only simple comparisons supported",
                span=self.span_tracker.get_span(compare),
                hint="Use single comparison operators like: a < b, not chained comparisons",
            )

        span = self.span_tracker.get_span(compare)
        left = self.parse_expression(compare.left)
        right = self.parse_expression(compare.comparators[0])

        op_map = {
            ast.Eq: ir.eq,
            ast.NotEq: ir.ne,
            ast.Lt: ir.lt,
            ast.LtE: ir.le,
            ast.Gt: ir.gt,
            ast.GtE: ir.ge,
        }

        op_type = type(compare.ops[0])
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported comparison: {op_type.__name__}",
                span=self.span_tracker.get_span(compare),
                hint="Use supported comparisons: ==, !=, <, <=, >, >=",
            )

        return op_map[op_type](left, right, span)

    def parse_unaryop(self, unary: ast.UnaryOp) -> ir.Expr:
        """Parse unary operation.

        Args:
            unary: UnaryOp AST node

        Returns:
            IR unary expression
        """
        span = self.span_tracker.get_span(unary)
        operand = self.parse_expression(unary.operand)

        op_map = {
            ast.USub: ir.neg,
            ast.Not: ir.bit_not,
        }

        op_type = type(unary.op)
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported unary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(unary),
                hint="Use supported unary operators: -, not",
            )

        return op_map[op_type](operand, span)

    def _parse_tiling_attribute(
        self, obj_name: str, field_name: str, span: ir.Span,
    ) -> ir.Expr | None:
        """Handle tiling registry lookup for attribute access.

        Returns the IR expression if obj_name is a tiling param, None otherwise.
        Raises ParserTypeError for invalid field access.
        """
        if obj_name not in self.tiling_registry:
            return None
        field_vars = self.tiling_registry[obj_name]
        if field_name in field_vars:
            val = field_vars[field_name]
            if isinstance(val, list):
                raise ParserTypeError(
                    f"Array field '{field_name}' must be accessed with an integer index",
                    span=span,
                    hint=f"Use {obj_name}.{field_name}[0] through "
                                 f"{obj_name}.{field_name}[{len(val) - 1}]",
                )
            return val  # scalar ir.Var
        raise ParserTypeError(
            f"Tiling parameter '{obj_name}' has no field '{field_name}'",
            span=span,
            hint=f"Valid fields are: {', '.join(field_vars.keys())}",
        )

    def parse_attribute(self, attr: ast.Attribute) -> ir.Expr:
        """Parse attribute access.

        Args:
            attr: Attribute AST node

        Returns:
            IR expression
        """
        span = self.span_tracker.get_span(attr)
        if isinstance(attr.value, ast.Name):
            obj_name = attr.value.id
            field_name = attr.attr

            struct_result = self._parse_struct_attribute(obj_name, field_name, span)
            if struct_result is not None:
                return struct_result

            tiling_result = self._parse_tiling_attribute(obj_name, field_name, span)
            if tiling_result is not None:
                return tiling_result

            if obj_name == "pl" and field_name in _MEMORY_SPACE_MAP:
                return ir.ConstInt(_MEMORY_SPACE_MAP[field_name].value, DataType.INT64, span)
        # Check for nested attribute access like pl.MemorySpace.Left
        if isinstance(attr.value, ast.Attribute):
            inner_attr = attr.value
            if isinstance(inner_attr.value, ast.Name):
                inner_obj_name = inner_attr.value.id
                inner_field_name = inner_attr.attr
                outer_field_name = attr.attr
                if inner_obj_name == "pl" and inner_field_name == "MemorySpace":
                    if outer_field_name in _MEMORY_SPACE_MAP:
                        return ir.ConstInt(_MEMORY_SPACE_MAP[outer_field_name].value, DataType.INT64, span)
        # Check for struct_array[idx].field compound pattern
        compound_result = self._parse_struct_array_compound_attribute(attr, span)
        if compound_result is not None:
            return compound_result

        # Generic fallback: try evaluating the whole attribute chain via
        # closure (augmented with python_vars from scope) and convert the
        # result to IR. Supports Buffer/NBuffer python_var attribute access
        # like nbuf.slots[0].tile or nbuf.buf_ids[0].
        ir_result = self._try_eval_python_var_attr_as_ir(attr, span)
        if ir_result is not None:
            return ir_result

        raise UnsupportedFeatureError(
            f"Standalone attribute access not supported: {ast.unparse(attr)}",
            span=span,
            hint="Attribute access is only supported for tiling parameters (e.g., tiling.x) "
                         "or within function calls",
        )

    def _try_eval_python_var_attr_as_ir(self, node: ast.expr, span: ir.Span):
        """Try to evaluate an attribute/subscript chain rooted at a python_var.

        When the root Name of an attribute chain (e.g. ``nbuf`` in
        ``nbuf.slots[0].tile``) is a python_var stored in scope_manager,
        we temporarily inject it into the expr_evaluator's closure and
        eval the whole expression.  The resulting Python value is then
        converted to an IR Expr via ``python_value_to_ir``.

        Returns an ir.Expr on success, None if the root is not a python_var
        or evaluation fails.
        """
        # First try via scope python_vars (kernel-internal NBuffer etc.)
        result = self._eval_with_python_vars(node, span)
        if result is not None:
            return result
        # Also try plain closure eval (external Buffer/NBuffer declarations).
        # First try raw eval to check for _TileRef cache.
        from pypto_block.language.buffer import _TileRef as _MT
        ok, raw_val = self.expr_evaluator.try_eval_expr(node)
        if ok:
            if isinstance(raw_val, _MT) and raw_val._ir_var is not None:
                return raw_val._ir_var
            try:
                result = self.expr_evaluator.python_value_to_ir(raw_val, span)
                result = self._ensure_tile_has_var(result, span,
                    tileref=raw_val if isinstance(raw_val, _MT) else None)
                if isinstance(raw_val, _MT) and isinstance(result, ir.Var):
                    raw_val._ir_var = result
                return result
            except Exception:
                pass
        return None

    def _try_eval_python_var_call_as_ir(self, node: ast.Call):
        """Try to evaluate a call rooted at a python_var (e.g. ``nbuf.current()``).

        Returns an ir.Expr on success. Returns a raw Python object (e.g.
        BufferSlot) when the result is not IR-convertible — the caller
        (parse_assignment) stores it as python_var. Returns the sentinel
        ``True`` for void calls (advance()). Returns None only if the
        root is not a python_var or evaluation fails entirely.
        """
        # Intercept NBuffer.current() / .previous() with IR cursor struct.
        from pypto_block.language.buffer import NBuffer as _NBufferCls
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("current", "previous"):
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                nbuf = self.scope_manager.get_python_var(obj_name)
                if isinstance(nbuf, _NBufferCls) and hasattr(nbuf, "_ir_struct_name"):
                    span = self.span_tracker.get_span(node)
                    return self._parse_nbuffer_method_call(
                        nbuf, node.func.attr, node, span,
                    )

        span = self.span_tracker.get_span(node)
        return self._eval_with_python_vars(node, span, allow_raw=True)

    def _eval_with_python_vars(self, node: ast.expr, span: ir.Span, *, allow_raw: bool = False):
        """Evaluate ``node`` with scope python_vars injected into the closure.

        Returns an ir.Expr on success, None if the root Name is not a
        python_var or evaluation/conversion fails. When ``allow_raw`` is True,
        returns raw Python objects that cannot be IR-converted (e.g. BufferSlot),
        and returns ``True`` (as a sentinel) for void results (Python None).
        """
        # Walk to the root Name.
        root = node
        while isinstance(root, (ast.Attribute, ast.Subscript, ast.Call)):
            if isinstance(root, ast.Call):
                root = root.func
            else:
                root = root.value
        if not isinstance(root, ast.Name):
            return None

        py_obj = self.scope_manager.get_python_var(root.id)
        if py_obj is None:
            return None

        saved = self.expr_evaluator.closure_vars
        try:
            augmented = dict(saved)
            augmented[root.id] = py_obj
            self.expr_evaluator.closure_vars = augmented
            ok, value = self.expr_evaluator.try_eval_expr(node)
            if not ok:
                return None
            # Void call (e.g. advance()) — signal the caller to skip emission.
            if value is None and allow_raw:
                return True  # sentinel: "eval succeeded, nothing to emit"
            try:
                # If value is a _TileRef with a cached IR Var, reuse it.
                from pypto_block.language.buffer import _TileRef as _MT
                if isinstance(value, _MT) and value._ir_var is not None:
                    return value._ir_var

                result = self.expr_evaluator.python_value_to_ir(value, span)
                # Wrap anonymous tile Call nodes in a let-binding so that
                # CCE codegen can resolve them to C++ variable names.
                result = self._ensure_tile_has_var(result, span,
                    tileref=value if isinstance(value, _MT) else None)

                # Cache the IR Var on the _TileRef for future reuse.
                if isinstance(value, _MT) and isinstance(result, ir.Var):
                    value._ir_var = result

                return result
            except Exception:
                return value if allow_raw else None
        except Exception:
            return None
        finally:
            self.expr_evaluator.closure_vars = saved

    def _ensure_tile_has_var(self, expr, span: ir.Span, tileref=None):
        """Wrap an anonymous tile ``ir.Call`` in a ``builder.let`` binding.

        CCE codegen resolves tile names via ``context_.GetVarName(Var)``.
        When a tile comes from ``_TileRef.unwrap()`` or closure eval, it
        may be a bare ``ir.Call("block.make_tile", ...)`` that was never
        assigned to a variable — producing an empty name in the generated C++.

        This method detects that case and emits a let-binding so the tile
        gets a proper IR variable name. If ``tileref`` has a ``_var_name``,
        uses that for readability; otherwise falls back to ``_buf_tile_N``.
        """
        if not isinstance(expr, ir.Expr):
            return expr
        # Only wrap Call nodes that produce a TileType and are NOT already Vars.
        if isinstance(expr, ir.Var):
            return expr
        if isinstance(expr, ir.Call) and isinstance(expr.type, ir.TileType):
            if tileref is not None and getattr(tileref, '_var_name', None):
                name = tileref._var_name
            else:
                name = f"_buf_tile_{self._buf_tile_counter}"
                self._buf_tile_counter += 1
            return self.builder.let(name, expr, span=span)
        return expr

    def parse_list(self, list_node: ast.List) -> ir.MakeTuple:
        """Parse list literal into MakeTuple IR expression.


        Args:
            list_node: List AST node

        Returns:
            MakeTuple IR expression
        """
        span = self.span_tracker.get_span(list_node)
        elements = [self.parse_expression(elt) for elt in list_node.elts]
        return ir.MakeTuple(elements, span)

    def parse_tuple_literal(self, tuple_node: ast.Tuple) -> ir.MakeTuple:
        """Parse tuple literal like (x, y, z).

        Args:
            tuple_node: Tuple AST node

        Returns:
            MakeTuple IR expression
        """
        span = self.span_tracker.get_span(tuple_node)
        elements = [self.parse_expression(elt) for elt in tuple_node.elts]
        return ir.MakeTuple(elements, span)

    def _build_tuple_index_chain(
        self,
        value_expr: ir.Expr,
        index_expr: ir.Expr,
        elem_type: ir.Type,
        n: int,
        level: int,
        span: ir.Span,
    ) -> ir.Var | None:
        """Recursively build nested if-else chain for variable tuple index access.

        Lowers `tuple[idx]` into a nested if-else structure at the IR level:
          if idx == 0: yield tuple[0]
          else:
            if idx == 1: yield tuple[1]
            else: yield tuple[2]  # leaf

        Args:
            value_expr: The tuple expression being indexed
            index_expr: Variable index expression
            elem_type: Type of each tuple element (must be homogeneous)
            n: Total number of tuple elements
            level: Current element index being tested (0-based)
            span: Source span for IR nodes

        Returns:
            The phi var from the outermost IfStmt, or None for the leaf case.
        """
        if level == n - 1:
            # Leaf: emit yield unconditionally (already in innermost else branch)
            self.builder.emit(ir.YieldStmt([ir.TupleGetItemExpr(value_expr, level, span)], span))
            return None

        cond = index_expr == level  # Expr.__eq__ produces a comparison Expr
        with self.builder.if_stmt(cond, span) as if_b:
            # Then branch: yield the element at this level
            self.builder.emit(ir.YieldStmt([ir.TupleGetItemExpr(value_expr, level, span)], span))
            if_b.else_()
            # Else branch: recurse to the next level
            inner_result = self._build_tuple_index_chain(
                value_expr, index_expr, elem_type, n, level + 1, span
            )
            if inner_result is not None:
                # Forward the inner phi var as this branch's yield
                self.builder.emit(ir.YieldStmt([inner_result], span))
            # Declare phi variable AFTER both branches, still inside the with block
            result_name = f"_tidx_{self._tuple_idx_counter}"
            self._tuple_idx_counter += 1
            if_b.return_var(result_name, elem_type, span)
        return if_b.output(0)

    def _parse_tiling_array_subscript(
        self, subscript: ast.Subscript, span: ir.Span,
    ) -> ir.Expr | None:
        """Handle tiling array field access: tiling.arr[i].

        Returns the IR expression if this is a tiling array subscript, None otherwise.
        """
        if not isinstance(subscript.value, ast.Attribute):
            return None
        attr = subscript.value
        if not (isinstance(attr.value, ast.Name) and attr.value.id in self.tiling_registry):
            return None

        obj_name = attr.value.id
        field_name = attr.attr
        field_val = self.tiling_registry[obj_name].get(field_name)
        if isinstance(field_val, list):
            if (not isinstance(subscript.slice, ast.Constant)
                    or not isinstance(subscript.slice.value, int)):
                raise UnsupportedFeatureError(
                    "Tiling array fields only support literal integer indices",
                    span=span,
                    hint=f"Use a constant index like tiling.{field_name}[0]",
                )
            idx = subscript.slice.value
            if idx < 0 or idx >= len(field_val):
                raise ParserTypeError(
                    f"Index {idx} out of bounds for array field '{field_name}' "
                    f"(size {len(field_val)})",
                    span=span,
                    hint=f"Valid indices are 0 to {len(field_val) - 1}",
                )
            return field_val[idx]
        if field_val is not None:
            # Scalar field accessed with subscript — helpful error
            raise ParserTypeError(
                f"Scalar field '{field_name}' does not support subscript access",
                span=span,
                hint=f"Use tiling.{field_name} directly (no index needed)",
            )
        return None

    def _parse_variable_index_subscript(
        self, value_expr: ir.Expr, subscript: ast.Subscript,
        index_expr: ir.Expr, span: ir.Span,
    ) -> ir.Expr:
        """Handle variable (non-constant) index subscript on tuple/tile values."""
        value_type = value_expr.type
        # TileType: tile[offset] → TileOffsetExpr (element offset)
        if isinstance(value_type, ir.TileType):
            return ir.TileOffsetExpr(value_expr, index_expr, span)
        if not isinstance(value_type, ir.TupleType):
            raise ParserTypeError(
                f"Subscript requires tuple type, got {type(value_type).__name__}",
                span=span,
                hint="Only tuple types support subscript access in this context",
            )

        elem_types = list(value_type.types)
        if not elem_types:
            raise ParserTypeError(
                "Cannot index into empty tuple",
                span=span,
            )

        # Variable indexing requires all elements to share the same type
        first_type = elem_types[0]
        for i, t in enumerate(elem_types[1:], 1):
            if not ir.structural_equal(t, first_type, enable_auto_mapping=False):
                raise ParserTypeError(
                    f"Variable tuple index requires all elements to have the same type, "
                    f"but element 0 has type {first_type} and element {i} has type {t}",
                    span=span,
                    hint="Use a constant index to access elements of different types",
                )

        # Cache lookup: same (tuple_var_name, index_ssa_var_name) → reuse existing phi var.
        # Applies to all tuple element types (tile, tensor, event ID, etc.).
        cache_key: tuple[str, str] | None = None
        if isinstance(subscript.value, ast.Name) and isinstance(index_expr, ir.Var):
            cache_key = (subscript.value.id, index_expr.name)
            if cache_key in self._tuple_select_cache:
                return self._tuple_select_cache[cache_key]

        result = self._build_tuple_index_chain(
            value_expr, index_expr, first_type, len(elem_types), 0, span
        )
        if result is None:
            # Single-element tuple: leaf emits directly, return TupleGetItemExpr
            return ir.TupleGetItemExpr(value_expr, 0, span)

        # Store in cache for subsequent uses of the same buf[idx]
        if cache_key is not None:
            self._tuple_select_cache[cache_key] = result
        return result

    def parse_subscript(self, subscript: ast.Subscript) -> ir.Expr:
        """Parse subscript expression like tuple[0].

        Args:
            subscript: Subscript AST node

        Returns:
            IR expression (TupleGetItemExpr for tuple access)

        Example Python syntax:
            first = my_tuple[0]      # Creates TupleGetItemExpr(my_tuple, 0)
            nested = my_tuple[1][2]  # Creates nested TupleGetItemExpr
        """
        span = self.span_tracker.get_span(subscript)

        # Check for tiling array field access: tiling.arr[i]
        tiling_result = self._parse_tiling_array_subscript(subscript, span)
        if tiling_result is not None:
            return tiling_result

        # Check for struct array subscript: ctx_arr[idx] → _DynamicStructView
        if isinstance(subscript.value, ast.Name):
            _sa_obj = self.scope_manager.get_python_var(subscript.value.id)
            if _sa_obj is None:
                _sa_obj = self.scope_manager.lookup_var(subscript.value.id)
            if isinstance(_sa_obj, _StructArrayVar):
                index_expr = self.parse_expression(subscript.slice)
                return _DynamicStructView(_sa_obj, index_expr)

        value_expr = self.parse_expression(subscript.value)

        # Parse index from slice
        if isinstance(subscript.slice, ast.Constant):
            index = subscript.slice.value
            if not isinstance(index, int):
                raise ParserSyntaxError(
                    "Tuple index must be an integer",
                    span=span,
                    hint="Use integer index like tuple[0]",
                )
        else:
            # Variable index: parse as IR expression and lower to an if-else chain
            index_expr = self.parse_expression(subscript.slice)
            return self._parse_variable_index_subscript(value_expr, subscript, index_expr, span)

        # Check if value is tuple type or tile type (runtime check)
        value_type = value_expr.type
        # TileType with constant index: tile[const] → TileOffsetExpr
        if isinstance(value_type, ir.TileType):
            const_offset = ir.ConstInt(index, DataType.INDEX, ir.Span.unknown())
            return ir.TileOffsetExpr(value_expr, const_offset, span)
        if not isinstance(value_type, ir.TupleType):
            raise ParserTypeError(
                f"Subscript requires tuple type, got {type(value_type).__name__}",
                span=span,
                hint="Only tuple types support subscript access in this context",
            )

        # Create TupleGetItemExpr
        return ir.TupleGetItemExpr(value_expr, index, span)

    def _resolve_yield_var_type(self, annotation: ast.expr | None) -> ir.Type:
        """Resolve type annotation for a yield variable.

        Args:
            annotation: Type annotation AST node, or None if not annotated

        Returns:
            Resolved IR type
        """
        if annotation is None:
            # Fallback to generic tensor type when no annotation present
            return ir.TensorType([1], DataType.INT32)

        resolved = self.type_resolver.resolve_type(annotation)
        # resolve_type can return list[Type] for tuple[...] annotations
        if isinstance(resolved, list):
            if len(resolved) == 0:
                # Empty tuple type - use fallback
                return ir.TensorType([1], DataType.INT32)
            if len(resolved) == 1:
                # Single element - unwrap
                return resolved[0]
            # Multiple elements - create TupleType
            return ir.TupleType(resolved)
        # Single type
        return resolved

    def _scan_for_yields(self, stmts: list[ast.stmt]) -> list[tuple[str, ast.expr | None]]:
        """Scan statements for yield assignments to determine output variable names and types.

        Args:
            stmts: List of statements to scan

        Returns:
            List of tuples (variable_name, type_annotation) where type_annotation is None if not annotated
        """
        yield_vars = []

        for stmt in stmts:
            # Check for annotated assignment with yield_: var: type = pl.yield_(...)
            if isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name) and isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yield_":
                        yield_vars.append((stmt.target.id, stmt.annotation))

            # Check for regular assignment with yield_: var = pl.yield_(...)
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    # Single variable assignment
                    if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Attribute) and func.attr == "yield_":
                            yield_vars.append((target.id, None))
                    # Tuple unpacking: (a, b) = pl.yield_(...)
                    elif isinstance(target, ast.Tuple) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Attribute) and func.attr == "yield_":
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    yield_vars.append((elt.id, None))

            # Recursively scan nested if statements
            elif isinstance(stmt, ast.If):
                yield_vars.extend(self._scan_for_yields(stmt.body))
                if stmt.orelse:
                    # Only take yields from else if they match then branch
                    # For simplicity, just take from then branch
                    pass

        return yield_vars
