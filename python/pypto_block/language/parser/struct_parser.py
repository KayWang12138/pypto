# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Struct parsing helpers for ASTParser."""

from __future__ import annotations

import ast
from typing import Any

from pypto_block.pypto_core import DataType, ir

from .diagnostics import ParserSyntaxError, ParserTypeError
from .models import _DynamicStructView, _StructArrayVar, _StructVar


class StructParserMixin:
    """Mixin containing ``pl.struct`` and ``pl.StructArray`` helpers."""

    def _parse_struct_assignment(self, var_name: str, call: ast.Call, span: Any) -> bool:
        """Handle ``var = pl.struct(field1=val1, ...)``. Returns True if handled."""
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "struct"):
            return False
        struct_var = self._parse_struct_call(call, var_name)
        self.scope_manager.define_python_var(var_name, struct_var, span=span)
        # Emit struct.declare for C++ struct codegen
        fields_csv = ",".join(struct_var.fields.keys())
        decl_call = ir.create_op_call(
            "struct.declare", [],
            {"array": var_name, "size": 1, "fields": fields_csv},
            span,
        )
        self.builder.emit(ir.EvalStmt(decl_call, span))
        # Emit struct.set for non-zero init values
        idx_zero = ir.ConstInt(0, DataType.INDEX, span)
        for fname, fval in struct_var.fields.items():
            if isinstance(fval, ir.Expr):
                # Skip trivial zero inits
                if isinstance(fval, ir.ConstInt) and fval.value == 0:
                    continue
                init_call = ir.create_op_call(
                    "struct.set", [idx_zero, fval],
                    {"array": var_name, "field": fname}, span,
                )
                self.builder.emit(ir.EvalStmt(init_call, span))
        return True

    def _parse_struct_array_assignment(self, var_name: str, call: ast.Call, span: Any) -> bool:
        """Handle ``var = pl.StructArray(N, field1=val1, ...)``. Returns True if handled."""
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "StructArray"):
            return False
        sa_call = call
        if not sa_call.args or not isinstance(sa_call.args[0], ast.Constant):
            raise ParserSyntaxError(
                "pl.StructArray() requires an integer size as first argument",
                span=span,
                hint="Use pl.StructArray(3, field1=0, field2=0, ...)",
            )
        arr_size = sa_call.args[0].value
        if not isinstance(arr_size, int) or arr_size < 1:
            raise ParserSyntaxError(
                f"pl.StructArray() size must be a positive integer, got {arr_size}",
                span=span,
            )
        # Parse field kwargs
        fields: dict[str, Any] = {}
        for kw in sa_call.keywords:
            if kw.arg is None:
                raise ParserSyntaxError("pl.StructArray() does not support **kwargs", span=span)
            fields[kw.arg] = self.parse_expression(kw.value)
        if not fields:
            raise ParserSyntaxError("pl.StructArray() requires at least one field", span=span)

        # Create _StructArrayVar with dummy _StructVar slots (for field validation)
        structs = [_StructVar(dict(fields), name=f"{var_name}_{i}") for i in range(arr_size)]
        struct_arr = _StructArrayVar(structs, name=var_name)
        self.scope_manager.define_python_var(var_name, struct_arr, span=span)

        # Emit struct.declare
        fields_csv = ",".join(fields.keys())
        decl_call = ir.create_op_call(
            "struct.declare", [],
            {"array": var_name, "size": arr_size, "fields": fields_csv},
            span,
        )
        self.builder.emit(ir.EvalStmt(decl_call, span))

        # Emit struct.set for non-zero init values (applied to all elements)
        for slot in range(arr_size):
            idx_expr = ir.ConstInt(slot, DataType.INDEX, span)
            for fname, fval in fields.items():
                if isinstance(fval, ir.Expr):
                    if isinstance(fval, ir.ConstInt) and fval.value == 0:
                        continue
                    init_call = ir.create_op_call(
                        "struct.set", [idx_expr, fval],
                        {"array": var_name, "field": fname}, span,
                    )
                    self.builder.emit(ir.EvalStmt(init_call, span))
        return True

    def _parse_struct_field_assignment(self, target: ast.Attribute, stmt: ast.Assign) -> bool:
        """Handle ``ctx.field = value`` and ``_DynamicStructView`` field write. Returns True if handled."""
        if not isinstance(target.value, ast.Name):
            return False
        obj_name = target.value.id
        field_name = target.attr
        span = self.span_tracker.get_span(stmt)

        obj = self.scope_manager.get_python_var(obj_name)
        if obj is None:
            obj = self.scope_manager.lookup_var(obj_name)
        if isinstance(obj, _StructVar):
            if field_name not in obj.fields:
                raise ParserTypeError(
                    f"Struct '{obj_name}' has no field '{field_name}'",
                    span=span,
                    hint=f"Available fields: {', '.join(obj.fields.keys())}",
                )
            value_expr = self.parse_expression(stmt.value)
            if obj.name and isinstance(value_expr, ir.Expr):
                # Named struct with C++ codegen — use struct.set
                idx_zero = ir.ConstInt(0, DataType.INDEX, span)
                call = ir.create_op_call(
                    "struct.set", [idx_zero, value_expr],
                    {"array": obj.name, "field": field_name}, span,
                )
                self.builder.emit(ir.EvalStmt(call, span))
            elif isinstance(value_expr, ir.Expr):
                # Unnamed struct — fallback to IR variable
                ir_name = f"_{obj.name}_{field_name}" if obj.name else field_name
                var = self.builder.let(ir_name, value_expr, span=span)
                obj.fields[field_name] = var
            else:
                obj.fields[field_name] = value_expr
            return True
        # _DynamicStructView field write (view passed as function arg)
        if isinstance(obj, _DynamicStructView):
            if field_name not in obj.array.field_names:
                raise ParserTypeError(
                    f"Struct array view has no field '{field_name}'",
                    span=span,
                    hint=f"Available fields: {', '.join(obj.array.field_names)}",
                )
            value_expr = self.parse_expression(stmt.value)
            self._struct_array_field_write(obj.array, obj.index_expr, field_name, value_expr, span, ref_name=obj.ref_name)
            return True
        return False

    def _parse_struct_array_field_assignment(self, target: ast.Attribute, stmt: ast.Assign) -> bool:
        """Handle ``ctx_arr[idx].field = value``. Returns True if handled."""
        if not (isinstance(target.value, ast.Subscript)
                and isinstance(target.value.value, ast.Name)):
            return False
        arr_name = target.value.value.id
        field_name = target.attr
        span = self.span_tracker.get_span(stmt)

        arr_obj = self.scope_manager.get_python_var(arr_name)
        if arr_obj is None:
            arr_obj = self.scope_manager.lookup_var(arr_name)
        if isinstance(arr_obj, _StructArrayVar):
            if field_name not in arr_obj.field_names:
                raise ParserTypeError(
                    f"Struct array '{arr_name}' has no field '{field_name}'",
                    span=span,
                    hint=f"Available fields: {', '.join(arr_obj.field_names)}",
                )
            index_expr = self.parse_expression(target.value.slice)
            value_expr = self.parse_expression(stmt.value)
            self._struct_array_field_write(arr_obj, index_expr, field_name, value_expr, span)
            return True
        return False

    def _check_struct_in_collection(self, var_name: str, value: ast.expr) -> None:
        """Raise if RHS is a list/tuple containing struct variables."""
        if not isinstance(value, (ast.List, ast.Tuple)) or not value.elts:
            return
        for elt in value.elts:
            if isinstance(elt, ast.Name):
                obj = self.scope_manager.get_python_var(elt.id)
                if isinstance(obj, _StructVar):
                    raise ParserSyntaxError(
                        "Putting pl.struct into a list/tuple is not supported. "
                        "Use pl.StructArray(N, field=val, ...) instead.",
                        span=self.span_tracker.get_span(elt),
                        hint=f"Example: {var_name} = pl.StructArray(N, field1=0, field2=0)",
                    )

    def _parse_struct_array_subscript_alias(
        self, var_name: str, value: ast.expr, span: ir.Span,
    ) -> bool:
        """Handle struct array subscript alias: ctx = ctx_arr[idx].

        Returns True if handled, False otherwise.
        """
        if not (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)):
            return False
        arr_obj = self.scope_manager.get_python_var(value.value.id)
        if arr_obj is None:
            arr_obj = self.scope_manager.lookup_var(value.value.id)
        if not isinstance(arr_obj, _StructArrayVar):
            return False
        index_expr = self.parse_expression(value.slice)
        view = _DynamicStructView(arr_obj, index_expr, ref_name=var_name)
        self.scope_manager.define_python_var(var_name, view, span=span)
        ref_call = ir.create_op_call(
            "struct.ref", [index_expr],
            {"array": arr_obj.name, "var": var_name}, span,
        )
        self.builder.emit(ir.EvalStmt(ref_call, span))
        return True

    def _parse_struct_call(self, call: ast.Call, struct_name: str = "") -> _StructVar:
        """Parse pl.struct(field1=val1, field2=val2, ...) into a _StructVar.

        Each field is emitted as an IR variable via ``builder.let`` so that
        subsequent mutations produce proper IR ``AssignStmt`` nodes.  This is
        critical for loop-carried variables: the codegen must see the
        reassignment inside the loop body to generate correct MLIR.

        IR variable names are prefixed with ``_{struct_name}_`` to avoid
        collisions with standalone variables of the same name in other scopes
        (e.g. cube vs. vector section both using ``q_count``).

        Args:
            call: The AST Call node for pl.struct(...)
            struct_name: LHS variable name (e.g. "ctx") used to prefix IR names

        Returns:
            _StructVar with fields mapping to IR Vars
        """
        span = self.span_tracker.get_span(call)
        if call.args:
            raise ParserSyntaxError(
                "pl.struct() only accepts keyword arguments",
                span=span,
                hint="Use pl.struct(name1=val1, name2=val2, ...)",
            )
        fields: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is None:
                raise ParserSyntaxError(
                    "pl.struct() does not support **kwargs",
                    span=span,
                )
            value_expr = self.parse_expression(kw.value)
            if struct_name:
                # Named struct: fields will use struct.get/set via C++ struct.
                # Don't emit IR variables — just record field names for validation.
                fields[kw.arg] = value_expr  # keep for validation only
            elif isinstance(value_expr, ir.Expr):
                # Unnamed struct: emit IR variable for codegen tracking
                ir_name = kw.arg
                var = self.builder.let(ir_name, value_expr, span=span)
                fields[kw.arg] = var
            else:
                fields[kw.arg] = value_expr
        return _StructVar(fields, name=struct_name)

    def _parse_struct_attribute(
        self, obj_name: str, field_name: str, span: ir.Span,
    ) -> ir.Expr | None:
        """Handle struct/dynamic-struct-view attribute access.

        Returns the IR expression if obj_name resolves to a struct, None otherwise.
        """
        obj = self.scope_manager.get_python_var(obj_name)
        if obj is None:
            obj = self.scope_manager.lookup_var(obj_name)
        if isinstance(obj, _StructVar):
            if field_name not in obj.fields:
                raise ParserTypeError(
                    f"Struct '{obj_name}' has no field '{field_name}'",
                    span=span,
                    hint=f"Available fields: {', '.join(obj.fields.keys())}",
                )
            if obj.name:
                idx_zero = ir.ConstInt(0, DataType.INDEX, span)
                return ir.create_op_call(
                    "struct.get", [idx_zero],
                    {"array": obj.name, "field": field_name}, span,
                )
            return obj.fields[field_name]
        if isinstance(obj, _DynamicStructView):
            if field_name not in obj.array.field_names:
                raise ParserTypeError(
                    f"Struct array view has no field '{field_name}'",
                    span=span,
                    hint=f"Available fields: {', '.join(obj.array.field_names)}",
                )
            return self._struct_array_field_read(obj, field_name, span)
        return None

    def _parse_struct_array_compound_attribute(
        self, attr: ast.Attribute, span: ir.Span,
    ) -> ir.Expr | None:
        """Handle struct_array[idx].field compound pattern.

        Returns the IR expression if matched, None otherwise.
        """
        if not (isinstance(attr.value, ast.Subscript) and isinstance(attr.value.value, ast.Name)):
            return None
        arr_name = attr.value.value.id
        arr_obj = self.scope_manager.get_python_var(arr_name)
        if arr_obj is None:
            arr_obj = self.scope_manager.lookup_var(arr_name)
        if not isinstance(arr_obj, _StructArrayVar):
            return None
        field_name = attr.attr
        if field_name not in arr_obj.field_names:
            raise ParserTypeError(
                f"Struct array '{arr_name}' has no field '{field_name}'",
                span=span,
                hint=f"Available fields: {', '.join(arr_obj.field_names)}",
            )
        index_expr = self.parse_expression(attr.value.slice)
        view = _DynamicStructView(arr_obj, index_expr)
        return self._struct_array_field_read(view, field_name, span)

    def _struct_array_field_read(
        self,
        view: "_DynamicStructView",
        field_name: str,
        span: "ir.Span",
    ) -> ir.Expr:
        """Read a field from a struct array at a dynamic index.

        If the view has a ref_name (from ``ctx = ctx_arr[idx]``), the codegen
        uses the C++ reference (``ctx.field``) instead of ``ctx_arr[idx].field``.
        """
        kwargs: dict[str, Any] = {"array": view.array.name, "field": field_name}
        if view.ref_name:
            kwargs["ref"] = view.ref_name
        return ir.create_op_call("struct.get", [view.index_expr], kwargs, span)

    def _struct_array_field_write(
        self,
        arr: "_StructArrayVar",
        index_expr: ir.Expr,
        field_name: str,
        value_expr: ir.Expr,
        span: "ir.Span",
        ref_name: str = "",
    ) -> None:
        """Write a field to one slot of a struct array at a dynamic index.

        If ref_name is set, the codegen uses the C++ reference (``ctx.field = val``)
        instead of ``ctx_arr[idx].field = val``.
        """
        kwargs: dict[str, Any] = {"array": arr.name, "field": field_name}
        if ref_name:
            kwargs["ref"] = ref_name
        call = ir.create_op_call("struct.set", [index_expr, value_expr], kwargs, span)
        self.builder.emit(ir.EvalStmt(call, span))
