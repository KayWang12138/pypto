# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AST parsing for converting Python DSL to IR builder calls."""

import ast
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pypto_block.ir import IRBuilder
from pypto_block.ir import op as ir_op
from pypto_block.pypto_core import DataType, ir
from pypto_block.pypto_core.ir import MemorySpace

_MEMORY_SPACE_MAP: dict[str, MemorySpace] = {
    "Left": MemorySpace.Left,
    "Right": MemorySpace.Right,
    "Vec": MemorySpace.Vec,
    "Mat": MemorySpace.Mat,
    "Acc": MemorySpace.Acc,
    "Scaling": MemorySpace.Scaling,
}

_BUFFER_CLASS_NAMES = frozenset({
    "NBuffer", "Buffer",
    "UBBuffer", "UBNBuffer",
    "L1Buffer", "L1NBuffer",
    "L0ABuffer", "L0ANBuffer",
    "L0BBuffer", "L0BNBuffer",
    "L0CBuffer", "L0CNBuffer",
})

from .diagnostics import (
    InvalidOperationError,
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)
from .expr_evaluator import ExprEvaluator
from .scope_manager import ScopeManager
from .span_tracker import SpanTracker
from .type_resolver import TypeResolver
from ..typing.tiling import ArrayFieldInfo, ScalarFieldInfo, get_tiling_fields, is_tiling_class

if TYPE_CHECKING:
    from .decorator import InlineFunction


def _is_const_int(value: object) -> bool:
    """Check if a value is a compile-time constant integer."""
    if isinstance(value, (int, ir.ConstInt)):
        return True
    return isinstance(value, ir.Neg) and isinstance(value.operand, ir.ConstInt)


def _const_int_value(value: object) -> int | None:
    """Extract integer value from a compile-time constant, or None."""
    if isinstance(value, int):
        return value
    if isinstance(value, ir.ConstInt):
        return value.value
    if isinstance(value, ir.Neg) and isinstance(value.operand, ir.ConstInt):
        return -value.operand.value
    return None


class _StructVar:
    """Compile-time grouping of IR expressions, accessed via attribute syntax.

    Created by ``pl.struct(field1=val1, field2=val2, ...)``.
    """

    def __init__(self, fields: dict[str, Any], name: str = "") -> None:
        self.fields = fields
        self.name = name


class _StructArrayVar:
    """List/tuple of homogeneous structs supporting dynamic index access.

    Created by assigning a list or tuple of ``_StructVar`` objects that share
    the same field names::

        ctx_arr = [ctx_0, ctx_1, ctx_2]   # all created via pl.struct(...)

    Dynamic indexing ``ctx_arr[idx]`` produces a ``_DynamicStructView`` which
    can be used for field reads (``view.field``), field writes
    (``view.field = val``), or as a function argument.
    """

    def __init__(self, structs: list[_StructVar], name: str = "") -> None:
        self.structs = structs
        self.field_names = list(structs[0].fields.keys())
        self.name = name  # C++ array variable name


class _DynamicStructView:
    """Runtime view into a ``_StructArrayVar`` at a dynamic IR index.

    Field reads lower to ``struct.get`` IR calls; field writes to ``struct.set``.
    The CCE codegen translates these to direct C++ array access: ``arr[idx].field``.
    """

    def __init__(self, array: _StructArrayVar, index_expr: "ir.Expr", ref_name: str = "") -> None:
        self.array = array
        self.index_expr = index_expr
        self.ref_name = ref_name  # C++ reference variable name (empty = no ref)


class ASTParser:
    """Parses Python AST and builds IR using IRBuilder."""

    def __init__(
        self,
        source_file: str,
        source_lines: list[str],
        line_offset: int = 0,
        col_offset: int = 0,
        global_vars: dict[str, ir.GlobalVar] | None = None,
        gvar_to_func: dict[ir.GlobalVar, ir.Function] | None = None,
        strict_ssa: bool = False,
        closure_vars: dict[str, Any] | None = None,
        auto_mutex: bool = False,
    ):
        """Initialize AST parser.

        Args:
            source_file: Path to source file
            source_lines: Lines of source code (dedented for parsing)
            line_offset: Line number offset to add to AST line numbers (for dedented code)
            col_offset: Column offset to add to AST column numbers (for dedented code)
            global_vars: Optional map of function names to GlobalVars for cross-function calls
            gvar_to_func: Optional map of GlobalVars to parsed Functions for type inference
            strict_ssa: If True, enforce SSA (single assignment). If False (default), allow reassignment.
            closure_vars: Optional variables from the enclosing scope for dynamic shape resolution
            auto_mutex: If True, automatically insert mutex lock/unlock around buffer-managed tile ops.
        """
        self.span_tracker = SpanTracker(source_file, source_lines, line_offset, col_offset)
        self.scope_manager = ScopeManager(strict_ssa=strict_ssa)
        self.expr_evaluator = ExprEvaluator(
            closure_vars=closure_vars or {},
            span_tracker=self.span_tracker,
        )
        self.type_resolver = TypeResolver(
            expr_evaluator=self.expr_evaluator,
            scope_lookup=self.scope_manager.lookup_var,
            span_tracker=self.span_tracker,
        )
        self.builder = IRBuilder()
        self.global_vars = global_vars or {}  # Track GlobalVars for cross-function calls
        self.gvar_to_func = gvar_to_func or {}  # Track parsed functions for type inference
        self.external_funcs: dict[str, ir.Function] = {}  # Track external functions referenced

        # Track context for handling yields and returns
        self.in_for_loop = False
        self.in_while_loop = False
        self.in_if_stmt = False
        self.current_if_builder = None
        self.current_loop_builder = None

        # Inline function expansion state
        self._inline_mode = False
        self._inline_return_expr: ir.Expr | None = None
        self._inline_prefix: str = ""  # unique prefix per inline expansion
        self._inline_counter: int = 0  # monotonic counter for unique prefixes

        # Cache for implicitly compiled functions (keyed by id(fn))
        self._implicit_func_cache: dict[int, Any] = {}

        # Registry mapping tiling param names to their flattened field vars.
        # Scalar fields map to a single ir.Var; array fields map to list[ir.Var].
        self.tiling_registry: dict[str, dict[str, ir.Var | list[ir.Var]]] = {}

        # Counter for generating unique names in variable tuple index lowering
        self._tuple_idx_counter: int = 0
        # Counter for anonymous buffer tile variables (auto-named _buf_tile_N).
        self._buf_tile_counter: int = 0

        # Registry mapping variable names to their constant-integer-tuple values.
        # Populated when a simple assignment like `event_ids = (0, 1)` is parsed.
        # Used by the sync-op statement expander to generate per-branch IfStmt chains.
        self._const_tuple_registry: dict[str, list[int]] = {}

        # Cache: (tuple_var_name, index_ssa_var_name) → phi ir.Var from _build_tuple_index_chain.
        # Applies to all tuple types (tile, tensor, event ID, etc.).
        # Prevents re-emitting an if-else chain when the same buf[idx] expression
        # appears multiple times in the same linear code region.
        self._tuple_select_cache: dict[tuple[str, str], ir.Var] = {}

        self._auto_mutex = auto_mutex

    def _validate_tiling_params(
        self, args_to_process: list[ast.arg], func_def: ast.FunctionDef,
    ) -> None:
        """Pre-validate tiling constraints: at most 1 tiling param, must be last."""
        tiling_param_names = [
            arg.arg for arg in args_to_process
            if arg.annotation is not None and self._resolve_tiling_class(arg.annotation) is not None
        ]
        if len(tiling_param_names) > 1:
            raise ParserSyntaxError(
                f"Function '{func_def.name}' has {len(tiling_param_names)} tiling parameters "
                f"({', '.join(tiling_param_names)}), but at most 1 is allowed",
                span=self.span_tracker.get_span(func_def),
                hint="A kernel may have at most one tiling parameter",
            )
        if len(tiling_param_names) == 1:
            if not args_to_process or args_to_process[-1].arg != tiling_param_names[0]:
                tiling_arg = next(a for a in args_to_process if a.arg == tiling_param_names[0])
                raise ParserSyntaxError(
                    f"Tiling parameter '{tiling_param_names[0]}' must be the last parameter",
                    span=self.span_tracker.get_span(tiling_arg),
                    hint="Move the tiling parameter to the last position",
                )

    def _parse_function_param(self, arg: ast.arg, f: Any) -> None:
        """Parse a single function parameter and register it in scope or tiling registry."""
        param_name = arg.arg

        if arg.annotation is None:
            raise ParserTypeError(
                f"Parameter '{param_name}' missing type annotation",
                span=self.span_tracker.get_span(arg),
                hint="Add a type annotation like: x: pl.Tensor[[64], pl.FP32]",
            )

        tiling_cls = self._resolve_tiling_class(arg.annotation)
        if tiling_cls is not None:
            param_span = self.span_tracker.get_span(arg)
            field_vars: dict[str, ir.Var | list[ir.Var]] = {}
            for field_name, field_info in get_tiling_fields(tiling_cls).items():
                if isinstance(field_info, ScalarFieldInfo):
                    flat_name = f"{param_name}_{field_name}"
                    flat_var = f.param(flat_name, ir.ScalarType(field_info.dtype), param_span)
                    field_vars[field_name] = flat_var
                else:  # ArrayFieldInfo
                    vars_list: list[ir.Var] = []
                    for i in range(field_info.size):
                        flat_name = f"{param_name}_{field_name}_{i}"
                        flat_var = f.param(flat_name, ir.ScalarType(field_info.dtype), param_span)
                        vars_list.append(flat_var)
                    field_vars[field_name] = vars_list
            self.tiling_registry[param_name] = field_vars
            return  # do NOT register tiling name itself in scope

        param_type, param_direction = self.type_resolver.resolve_param_type(arg.annotation)
        param_span = self.span_tracker.get_span(arg)
        param_var = f.param(param_name, param_type, param_span, direction=param_direction)
        self.scope_manager.define_var(param_name, param_var, allow_redef=True)

    def parse_function(
        self,
        func_def: ast.FunctionDef,
        func_type: ir.FunctionType = ir.FunctionType.Opaque,
    ) -> ir.Function:
        """Parse function definition and build IR.

        Args:
            func_def: AST FunctionDef node
            func_type: Function type (default: Opaque)

        Returns:
            IR Function object
        """
        func_name = func_def.name
        func_span = self.span_tracker.get_span(func_def)

        self.scope_manager.enter_scope("function")
        self.tiling_registry = {}

        # Collect args to process, filtering out bare 'self'
        args_to_process = [
            arg for arg in func_def.args.args
            if not (arg.arg == "self" and arg.annotation is None)
        ]

        self._validate_tiling_params(args_to_process, func_def)

        with self.builder.function(func_name, func_span, type=func_type) as f:
            for arg in args_to_process:
                self._parse_function_param(arg, f)

            # Parse return type
            if func_def.returns:
                return_type = self.type_resolver.resolve_type(func_def.returns)
                if isinstance(return_type, list):
                    for rt in return_type:
                        f.return_type(rt)
                else:
                    f.return_type(return_type)

            # Parse function body (skip docstrings)
            for i, stmt in enumerate(func_def.body):
                if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    if isinstance(stmt.value.value, str):
                        continue  # Skip docstring
                self.parse_statement(stmt)

        self.scope_manager.exit_scope()
        return f.get_result()

    def _resolve_tiling_class(self, annotation: ast.expr) -> type | None:
        """Return the tiling class if annotation refers to one in closure_vars, else None.

        Args:
            annotation: AST expression node for the annotation

        Returns:
            The resolved tiling class, or None if the annotation is not a tiling class
        """
        if not isinstance(annotation, ast.Name):
            return None
        cls = self.expr_evaluator.closure_vars.get(annotation.id)
        return cls if is_tiling_class(cls) else None

    def parse_statement(self, stmt: ast.stmt) -> None:
        """Parse a statement node.

        Args:
            stmt: AST statement node
        """
        if isinstance(stmt, ast.AnnAssign):
            self.parse_annotated_assignment(stmt)
        elif isinstance(stmt, ast.Assign):
            self.parse_assignment(stmt)
        elif isinstance(stmt, ast.For):
            self.parse_for_loop(stmt)
        elif isinstance(stmt, ast.While):
            self.parse_while_loop(stmt)
        elif isinstance(stmt, ast.If):
            self.parse_if_statement(stmt)
        elif isinstance(stmt, ast.With):
            self.parse_with_statement(stmt)
        elif isinstance(stmt, ast.Return):
            self.parse_return(stmt)
        elif isinstance(stmt, ast.Break):
            self.parse_break(stmt)
        elif isinstance(stmt, ast.Continue):
            self.parse_continue(stmt)
        elif isinstance(stmt, ast.Expr):
            self.parse_evaluation_statement(stmt)
        elif isinstance(stmt, ast.Pass):
            pass  # No-op: pass statements are valid in DSL functions
        else:
            raise UnsupportedFeatureError(
                f"Unsupported statement type: {type(stmt).__name__}",
                span=self.span_tracker.get_span(stmt),
                hint="Only assignments, for loops, while loops, if statements, "
                "with statements, returns, break, and continue are supported in DSL functions",
            )

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
                        f"Putting pl.struct into a list/tuple is not supported. "
                        f"Use pl.StructArray(N, field=val, ...) instead.",
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

    _VALID_ITERATORS = {"range", "parallel", "unroll", "while_"}
    _ITERATOR_ERROR = "For loop must use pl.range(), pl.parallel(), pl.unroll(), or pl.while_()"
    _ITERATOR_HINT = "Use pl.range(), pl.parallel(), pl.unroll(), or pl.while_() as the iterator"

    def _validate_for_loop_iterator(self, stmt: ast.For) -> tuple[ast.Call, str]:
        """Validate that for loop uses pl.range(), pl.parallel(), pl.unroll(), or pl.while_().

        Returns:
            Tuple of (call_node, iterator_type) where iterator_type is
            "range", "parallel", "unroll", or "while_"
        """
        if not isinstance(stmt.iter, ast.Call):
            raise ParserSyntaxError(
                self._ITERATOR_ERROR,
                span=self.span_tracker.get_span(stmt.iter),
                hint=self._ITERATOR_HINT,
            )

        iter_call = stmt.iter
        func = iter_call.func
        if isinstance(func, ast.Attribute) and func.attr in self._VALID_ITERATORS:
            return iter_call, func.attr

        raise ParserSyntaxError(
            self._ITERATOR_ERROR,
            span=self.span_tracker.get_span(stmt.iter),
            hint=self._ITERATOR_HINT,
        )

    def _parse_for_loop_target(self, stmt: ast.For) -> tuple[str, ast.AST | None, bool]:
        """Parse for loop target, returning (loop_var_name, iter_args_node, is_simple_for)."""
        if isinstance(stmt.target, ast.Name):
            return stmt.target.id, None, True

        if isinstance(stmt.target, ast.Tuple) and len(stmt.target.elts) == 2:
            loop_var_node = stmt.target.elts[0]
            iter_args_node = stmt.target.elts[1]

            if not isinstance(loop_var_node, ast.Name):
                raise ParserSyntaxError(
                    "Loop variable must be a simple name",
                    span=self.span_tracker.get_span(loop_var_node),
                    hint="Use a simple variable name for the loop counter",
                )
            return loop_var_node.id, iter_args_node, False

        raise ParserSyntaxError(
            "For loop target must be a simple name or: (loop_var, (iter_args...))",
            span=self.span_tracker.get_span(stmt.target),
            hint="Use: for i in pl.range(n) or for i, (var1,) in pl.range(n, init_values=(...,))",
        )

    def _setup_iter_args(self, loop: Any, iter_args_node: ast.AST, init_values: list) -> None:
        """Set up iter_args and return_vars for Pattern A loops."""
        if not isinstance(iter_args_node, ast.Tuple):
            raise ParserSyntaxError(
                "Iter args must be a tuple",
                span=self.span_tracker.get_span(iter_args_node),
                hint="Wrap iteration variables in parentheses: (var1, var2)",
            )

        if len(iter_args_node.elts) != len(init_values):
            raise ParserSyntaxError(
                f"Mismatch: {len(iter_args_node.elts)} iter_args but {len(init_values)} init_values",
                span=self.span_tracker.get_span(iter_args_node),
                hint=f"Provide exactly {len(init_values)} iteration variable(s) to match init_values",
            )

        for i, iter_arg_node in enumerate(iter_args_node.elts):
            if not isinstance(iter_arg_node, ast.Name):
                raise ParserSyntaxError(
                    "Iter arg must be a simple name",
                    span=self.span_tracker.get_span(iter_arg_node),
                    hint="Use simple variable names for iteration variables",
                )
            iter_arg_var = loop.iter_arg(iter_arg_node.id, init_values[i])
            self.scope_manager.define_var(iter_arg_node.id, iter_arg_var, allow_redef=True)

        for iter_arg_node in iter_args_node.elts:
            assert isinstance(iter_arg_node, ast.Name)
            loop.return_var(f"{iter_arg_node.id}_out")

    def _parse_for_loop_body(
        self, stmt: ast.For, loop: Any, loop_var: ir.Var, loop_var_name: str,
        is_simple_for: bool, iter_args_node: ast.Tuple | None,
        range_args: dict[str, Any], backward_deps: list, span: ir.Span,
    ) -> list[str]:
        """Parse the body of a for loop inside the loop context.

        Returns the list of yield output variable names.
        """
        self.current_loop_builder = loop
        self.in_for_loop = True
        self.scope_manager.enter_scope("for")
        self.scope_manager.define_var(loop_var_name, loop_var, allow_redef=True)

        if not is_simple_for:
            assert iter_args_node is not None
            self._setup_iter_args(loop, iter_args_node, range_args["init_values"])

        prev_yield_tracker = getattr(self, "_current_yield_vars", None)
        self._current_yield_vars = []
        prev_yield_types = getattr(self, "_current_yield_types", None)
        self._current_yield_types = {}

        for body_stmt in stmt.body:
            self.parse_statement(body_stmt)

        loop_output_vars = self._current_yield_vars[:]
        self._current_yield_vars = prev_yield_tracker
        self._current_yield_types = prev_yield_types

        should_leak = is_simple_for and not loop_output_vars
        self.scope_manager.exit_scope(leak_vars=should_leak)
        self.in_for_loop = False
        self.current_loop_builder = None
        return loop_output_vars

    def parse_for_loop(self, stmt: ast.For) -> None:
        """Parse for loop with pl.range(), pl.parallel(), pl.unroll(), or pl.while_()."""
        iter_call, iterator_type = self._validate_for_loop_iterator(stmt)

        if iterator_type == "while_":
            self._parse_while_as_for(stmt, iter_call)
            return

        _ITERATOR_TO_KIND = {
            "range": ir.ForKind.Sequential,
            "parallel": ir.ForKind.Parallel,
            "unroll": ir.ForKind.Unroll,
        }
        loop_var_name, iter_args_node, is_simple_for = self._parse_for_loop_target(stmt)
        range_args = self._parse_range_call(iter_call)
        self._validate_for_loop_args(iterator_type, range_args, iter_call, is_simple_for, stmt)

        chunk_expr = range_args.get("chunk")
        chunk_policy_str = range_args.get("chunk_policy", "leading_full")
        if chunk_expr is not None:
            self._validate_chunk_args(chunk_expr, range_args["init_values"], iter_call)

        kind = _ITERATOR_TO_KIND[iterator_type]
        loop_var = self.builder.var(loop_var_name, ir.ScalarType(DataType.INDEX))
        span = self.span_tracker.get_span(stmt)

        backward_deps: list = []

        with self.builder.for_loop(
            loop_var, range_args["start"], range_args["stop"], range_args["step"],
            span, kind, chunk_size=chunk_expr, chunk_policy=chunk_policy_str,
        ) as loop:
            loop_output_vars = self._parse_for_loop_body(
                stmt, loop, loop_var, loop_var_name, is_simple_for,
                iter_args_node, range_args, backward_deps, span,
            )

        if not is_simple_for:
            loop_result = loop.get_result()
            if hasattr(loop_result, "return_vars") and loop_result.return_vars and loop_output_vars:
                for i, var_name in enumerate(loop_output_vars):
                    if i < len(loop_result.return_vars):
                        self.scope_manager.define_var(var_name, loop_result.return_vars[i])

    def _validate_for_loop_args(
        self, iterator_type: str, range_args: dict[str, Any],
        iter_call: ast.Call, is_simple_for: bool, stmt: ast.For,
    ) -> None:
        """Validate for-loop arguments after range parsing.

        Checks init_values compatibility with simple-for and unroll loops,
        and validates compile-time constant bounds for unroll.
        """
        if is_simple_for and range_args["init_values"]:
            raise ParserSyntaxError(
                "For loop target must be a tuple when init_values is provided",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use: for i, (var1,) in pl.range(n, init_values=(val1,)) to include iter_args",
            )

        if iterator_type == "unroll" and range_args["init_values"]:
            raise ParserSyntaxError(
                "pl.unroll() cannot be combined with init_values",
                span=self.span_tracker.get_span(iter_call),
                hint="Unrolled loops do not support loop-carried values (init_values)",
            )

        # For pl.unroll(), require compile-time constant integer bounds
        # and reject step=0. Fail early with clear parser errors instead of
        # later generic failures in the UnrollLoops C++ pass.
        # Note: negative literals like -1 become ir.Neg(ir.ConstInt(1)).
        if iterator_type == "unroll":
            for _bound_name in ("start", "stop", "step"):
                _bound_value = range_args.get(_bound_name)
                if _bound_value is not None and not _is_const_int(_bound_value):
                    raise ParserSyntaxError(
                        "pl.unroll() requires compile-time constant integer bounds",
                        span=self.span_tracker.get_span(iter_call),
                        hint="Use integer literals for start, stop, and step in pl.unroll().",
                    )
            _step = range_args.get("step")
            if _const_int_value(_step) == 0:
                raise ParserSyntaxError(
                    "pl.unroll() step cannot be zero",
                    span=self.span_tracker.get_span(iter_call),
                    hint="Use a non-zero step in pl.unroll(start, stop, step).",
                )

    def _validate_chunk_args(self, chunk_expr: Any, init_values: list[Any], iter_call: ast.Call) -> None:
        """Validate chunk arguments for range/parallel/unroll loops."""
        if init_values:
            raise ParserSyntaxError(
                "chunk cannot be combined with init_values",
                span=self.span_tracker.get_span(iter_call),
                hint="Chunked loops do not support loop-carried values (init_values)",
            )
        if not _is_const_int(chunk_expr):
            raise ParserSyntaxError(
                "chunk must be a compile-time constant positive integer",
                span=self.span_tracker.get_span(iter_call),
                hint="Use an integer literal for chunk: chunk=5",
            )
        chunk_val = _const_int_value(chunk_expr)
        if chunk_val is not None and chunk_val <= 0:
            raise ParserSyntaxError(
                f"chunk must be a positive integer, got {chunk_val}",
                span=self.span_tracker.get_span(iter_call),
                hint="Use a positive integer for chunk: chunk=5",
            )

    def _parse_range_keywords(self, call: ast.Call) -> tuple[list, Any, str]:
        """Parse keyword arguments from a pl.range() call.

        Returns (init_values, chunk, chunk_policy).
        """
        init_values: list = []
        chunk = None
        chunk_policy = "leading_full"
        for keyword in call.keywords:
            if keyword.arg == "init_values":
                if isinstance(keyword.value, (ast.List, ast.Tuple)):
                    for elt in keyword.value.elts:
                        init_values.append(self.parse_expression(elt))
                else:
                    raise ParserSyntaxError(
                        "init_values must be a list or tuple",
                        span=self.span_tracker.get_span(keyword.value),
                        hint="Use a tuple for init_values: init_values=(var1, var2)",
                    )
            elif keyword.arg == "chunk":
                chunk = self.parse_expression(keyword.value)
            elif keyword.arg == "chunk_policy":
                if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    _VALID_CHUNK_POLICIES = {"leading_full"}
                    if keyword.value.value not in _VALID_CHUNK_POLICIES:
                        raise ParserSyntaxError(
                            f"Unsupported chunk_policy: {keyword.value.value!r}",
                            span=self.span_tracker.get_span(keyword.value),
                            hint=f"Supported values: {', '.join(sorted(_VALID_CHUNK_POLICIES))}",
                        )
                    chunk_policy = keyword.value.value
                else:
                    raise ParserSyntaxError(
                        "chunk_policy must be a string literal",
                        span=self.span_tracker.get_span(keyword.value),
                        hint='Use a string like chunk_policy="leading_full"',
                    )
            else:
                raise ParserSyntaxError(
                    f"Unknown keyword argument '{keyword.arg}' in range()",
                    span=self.span_tracker.get_span(keyword),
                    hint="Supported keywords: init_values, chunk, chunk_policy",
                )
        return init_values, chunk, chunk_policy

    def _parse_range_call(self, call: ast.Call) -> dict[str, Any]:
        """Parse pl.range() call arguments.

        Args:
            call: AST Call node for pl.range()

        Returns:
            Dictionary with start, stop, step, init_values
        """
        if len(call.args) < 1:
            raise ParserSyntaxError(
                "pl.range() requires at least 1 argument (stop)",
                span=self.span_tracker.get_span(call),
                hint="Provide at least the stop value: pl.range(10) or pl.range(0, 10)",
            )

        start = 0
        step = 1

        if len(call.args) == 1:
            stop = self.parse_expression(call.args[0])
        elif len(call.args) == 2:
            start = self.parse_expression(call.args[0])
            stop = self.parse_expression(call.args[1])
        elif len(call.args) >= 3:
            start = self.parse_expression(call.args[0])
            stop = self.parse_expression(call.args[1])
            step = self.parse_expression(call.args[2])

        init_values, chunk, chunk_policy = self._parse_range_keywords(call)

        return {
            "start": start,
            "stop": stop,
            "step": step,
            "init_values": init_values,
            "chunk": chunk,
            "chunk_policy": chunk_policy,
        }

    def _is_cond_call(self, stmt: ast.stmt) -> bool:
        """Check if statement is a pl.cond() call (without parsing).

        Args:
            stmt: AST statement node

        Returns:
            True if statement is pl.cond() call, False otherwise
        """
        if not isinstance(stmt, ast.Expr):
            return False

        call = stmt.value
        if not isinstance(call, ast.Call):
            return False

        # Check if this is pl.cond() or cond()
        if isinstance(call.func, ast.Attribute):
            # pl.cond() form
            return call.func.attr == "cond"
        elif isinstance(call.func, ast.Name):
            # cond() form (if imported directly)
            return call.func.id == "cond"

        return False

    def _extract_cond_call(self, stmt: ast.stmt) -> ir.Expr | None:
        """Extract condition from pl.cond() call statement.

        Args:
            stmt: AST statement node

        Returns:
            Parsed condition expression if statement is pl.cond(), None otherwise
        """
        if not self._is_cond_call(stmt):
            return None

        call = stmt.value  # type: ignore[union-attr]

        # Parse the condition argument
        if len(call.args) != 1:  # type: ignore[attr-defined]
            raise ParserSyntaxError(
                "pl.cond() requires exactly 1 argument",
                span=self.span_tracker.get_span(call),
                hint="Use: pl.cond(condition)",
            )

        return self.parse_expression(call.args[0])  # type: ignore[attr-defined]

    def _validate_while_call_args(self, while_call: ast.Call) -> None:
        """Validate that pl.while_() has no positional arguments."""
        if len(while_call.args) > 0:
            raise ParserSyntaxError(
                "pl.while_() takes no positional arguments",
                span=self.span_tracker.get_span(while_call),
                hint="Use: pl.while_(init_values=(...,)) with pl.cond(condition) as first statement in body",
            )

    def _parse_while_init_values(self, while_call: ast.Call) -> list[ir.Expr]:
        """Parse init_values from pl.while_() keyword arguments."""
        init_values = []
        for keyword in while_call.keywords:
            if keyword.arg == "init_values":
                if isinstance(keyword.value, (ast.List, ast.Tuple)):
                    for elt in keyword.value.elts:
                        init_values.append(self.parse_expression(elt))
                else:
                    raise ParserSyntaxError(
                        "init_values must be a tuple or list",
                        span=self.span_tracker.get_span(keyword.value),
                        hint="Use a tuple for init_values (lists also accepted): init_values=(var1, var2)",
                    )

        if not init_values:
            raise ParserSyntaxError(
                "pl.while_() requires init_values",
                span=self.span_tracker.get_span(while_call),
                hint="Provide init_values: pl.while_(init_values=(val1, val2))",
            )

        return init_values

    def _validate_while_body(self, stmt: ast.For) -> None:
        """Validate pl.while_() body structure."""
        if not stmt.body:
            raise ParserSyntaxError(
                "pl.while_() body cannot be empty",
                span=self.span_tracker.get_span(stmt),
                hint="Add pl.cond(condition) as first statement",
            )

        if not self._is_cond_call(stmt.body[0]):
            raise ParserSyntaxError(
                "First statement in pl.while_() body must be pl.cond(condition)",
                span=self.span_tracker.get_span(stmt.body[0]),
                hint="Add pl.cond(condition) as first statement",
            )

    def _validate_while_target(self, stmt: ast.For, init_values: list[ir.Expr]) -> ast.Tuple:
        """Validate and return pl.while_() target tuple."""
        if not isinstance(stmt.target, ast.Tuple):
            raise ParserSyntaxError(
                "While loop target must be a tuple for pl.while_()",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use: for (var1, var2) in pl.while_(init_values=(...,))",
            )

        iter_args_node = stmt.target

        if len(iter_args_node.elts) != len(init_values):
            raise ParserSyntaxError(
                f"Mismatch: {len(iter_args_node.elts)} iter_args but {len(init_values)} init_values",
                span=self.span_tracker.get_span(iter_args_node),
                hint=f"Provide exactly {len(init_values)} iteration variable(s) to match init_values",
            )

        return iter_args_node

    def _setup_while_iter_args(
        self, loop: Any, iter_args_node: ast.Tuple, init_values: list[ir.Expr]
    ) -> None:
        """Set up iter_args for pl.while_() loop."""
        for i, iter_arg_node in enumerate(iter_args_node.elts):
            if not isinstance(iter_arg_node, ast.Name):
                raise ParserSyntaxError(
                    "Iter arg must be a simple name",
                    span=self.span_tracker.get_span(iter_arg_node),
                    hint="Use simple variable names for iteration variables",
                )
            iter_arg_var = loop.iter_arg(iter_arg_node.id, init_values[i])
            self.scope_manager.define_var(iter_arg_node.id, iter_arg_var, allow_redef=True)

    def _parse_while_body_statements(self, stmt: ast.For) -> list[str]:
        """Parse body statements for pl.while_() loop, return yielded vars."""
        prev_yield_tracker = getattr(self, "_current_yield_vars", None)
        self._current_yield_vars = []
        prev_yield_types = getattr(self, "_current_yield_types", None)
        self._current_yield_types = {}

        # Parse body (skip first statement which is pl.cond())
        for i, body_stmt in enumerate(stmt.body):
            if i == 0:
                continue  # Skip the pl.cond() statement

            # Check if pl.cond() appears anywhere else in body
            if self._is_cond_call(body_stmt):
                raise ParserSyntaxError(
                    "pl.cond() can only be the first statement in a pl.while_() loop body",
                    span=self.span_tracker.get_span(body_stmt),
                    hint="Remove this pl.cond() - condition is already specified at the start",
                )

            self.parse_statement(body_stmt)

        loop_output_vars = self._current_yield_vars[:]
        self._current_yield_vars = prev_yield_tracker
        self._current_yield_types = prev_yield_types
        return loop_output_vars

    def _register_while_outputs(self, loop: Any, loop_output_vars: list[str]) -> None:
        """Register output variables from pl.while_() loop."""
        loop_result = loop.get_result()
        if hasattr(loop_result, "return_vars") and loop_result.return_vars and loop_output_vars:
            for i, var_name in enumerate(loop_output_vars):
                if i < len(loop_result.return_vars):
                    self.scope_manager.define_var(var_name, loop_result.return_vars[i])

    def _parse_while_as_for(self, stmt: ast.For, while_call: ast.Call) -> None:
        """Parse while loop using for...in pl.while_() pattern.

        Pattern: for (var1, var2) in pl.while_(init_values=(val1, val2)):
                     pl.cond(condition)
                     ...

        Args:
            stmt: For AST node
            while_call: Call to pl.while_()
        """
        # Validate and parse arguments
        self._validate_while_call_args(while_call)
        init_values = self._parse_while_init_values(while_call)
        self._validate_while_body(stmt)
        iter_args_node = self._validate_while_target(stmt, init_values)

        span = self.span_tracker.get_span(stmt)
        placeholder_condition = ir.ConstBool(True, span)

        with self.builder.while_loop(placeholder_condition, span) as loop:
            self.current_loop_builder = loop
            self.in_while_loop = True
            self.scope_manager.enter_scope("while")

            # Set up iter_args
            self._setup_while_iter_args(loop, iter_args_node, init_values)

            # Parse and set the condition (now that iter_args are in scope)
            condition = self._extract_cond_call(stmt.body[0])
            if condition is None:
                raise ParserSyntaxError(
                    "First statement in pl.while_() body must be pl.cond(condition)",
                    span=self.span_tracker.get_span(stmt.body[0]),
                    hint="Add pl.cond(condition) as first statement",
                )
            loop.set_condition(condition)

            # Add return_vars
            for iter_arg_node in iter_args_node.elts:
                assert isinstance(iter_arg_node, ast.Name)
                loop.return_var(f"{iter_arg_node.id}_out")

            # Parse body statements
            loop_output_vars = self._parse_while_body_statements(stmt)

            self.scope_manager.exit_scope(leak_vars=False)
            self.in_while_loop = False
            self.current_loop_builder = None

        # Register output variables
        self._register_while_outputs(loop, loop_output_vars)

    def parse_while_loop(self, stmt: ast.While) -> None:
        """Parse natural while loop syntax.

        Natural while syntax: while condition: body

        This creates a WhileStmt without iter_args (non-SSA form).
        The C++ ConvertToSSA pass will convert it to SSA form if needed.

        Args:
            stmt: While AST node
        """
        # Parse natural while syntax: while condition:
        condition = self.parse_expression(stmt.test)
        span = self.span_tracker.get_span(stmt)

        with self.builder.while_loop(condition, span) as loop:
            self.current_loop_builder = loop
            self.in_while_loop = True
            self.scope_manager.enter_scope("while")

            # Parse body statements
            for body_stmt in stmt.body:
                self.parse_statement(body_stmt)

            # Variables leak to outer scope (ConvertToSSA will handle)
            self.scope_manager.exit_scope(leak_vars=True)
            self.in_while_loop = False
            self.current_loop_builder = None

    def _scan_and_merge_yield_vars(
        self, stmt: ast.If,
    ) -> list[tuple[str, ast.expr | None]]:
        """Scan then/else branches for yield variable names and merge them.

        Returns merged list of (var_name, annotation) tuples. Then-branch
        takes precedence for type when a name appears in both branches.
        """
        yield_vars = self._scan_for_yields(stmt.body)
        if stmt.orelse:
            else_yield_vars = self._scan_for_yields(stmt.orelse)
            then_names = {name for name, _ in yield_vars}
            for name, annotation in else_yield_vars:
                if name not in then_names:
                    yield_vars.append((name, annotation))
        return yield_vars

    def _register_if_output_vars(
        self, if_builder: Any, yield_vars: list[tuple[str, ast.expr | None]],
    ) -> None:
        """Register output variables from an if statement into the outer scope."""
        if not yield_vars:
            return
        if_result = if_builder.get_result()
        if hasattr(if_result, "return_vars") and if_result.return_vars:
            for i, (var_name, _) in enumerate(yield_vars):
                if i < len(if_result.return_vars):
                    output_var = if_result.return_vars[i]
                    self.scope_manager.define_var(var_name, output_var)

    def parse_if_statement(self, stmt: ast.If) -> None:
        """Parse if statement with phi nodes.

        When pl.yield_() is used, phi nodes are created via return_vars.
        When no yields are used (plain syntax), variables leak to outer scope
        and the C++ ConvertToSSA pass handles creating phi nodes.

        Args:
            stmt: If AST node
        """
        condition = self.parse_expression(stmt.test)
        span = self.span_tracker.get_span(stmt)

        with self.builder.if_stmt(condition, span) as if_builder:
            self.current_if_builder = if_builder
            self.in_if_stmt = True

            prev_yield_tracker = getattr(self, "_current_yield_vars", None)
            self._current_yield_vars = []
            prev_yield_types = getattr(self, "_current_yield_types", None)
            self._current_yield_types = {}

            then_yield_vars = self._scan_and_merge_yield_vars(stmt)
            should_leak = not bool(then_yield_vars)

            saved_tuple_cache = dict(self._tuple_select_cache)
            self.scope_manager.enter_scope("if")
            for then_stmt in stmt.body:
                self.parse_statement(then_stmt)
            self.scope_manager.exit_scope(leak_vars=should_leak)

            if stmt.orelse:
                self._tuple_select_cache = saved_tuple_cache
                if_builder.else_()
                self.scope_manager.enter_scope("else")
                for else_stmt in stmt.orelse:
                    self.parse_statement(else_stmt)
                self.scope_manager.exit_scope(leak_vars=should_leak)

            for var_name, annotation in then_yield_vars:
                if annotation is not None:
                    var_type = self._resolve_yield_var_type(annotation)
                elif var_name in self._current_yield_types:
                    var_type = self._current_yield_types[var_name]
                else:
                    var_type = self._resolve_yield_var_type(None)
                if_builder.return_var(var_name, var_type)

            self._current_yield_vars = prev_yield_tracker
            self._current_yield_types = prev_yield_types

        self._tuple_select_cache = saved_tuple_cache

        self._register_if_output_vars(if_builder, then_yield_vars)

        self.in_if_stmt = False
        self.current_if_builder = None

    def parse_with_statement(self, stmt: ast.With) -> None:
        """Parse with statement for scope contexts.

        Currently supports:
        - with pl.incore(): ... (creates ScopeStmt with InCore scope)
        - with pl.section_vector(): ... (creates SectionStmt with Vector section)
        - with pl.section_cube(): ... (creates SectionStmt with Cube section)

        Args:
            stmt: With AST node
        """
        # Check that we have exactly one context manager
        if len(stmt.items) != 1:
            raise ParserSyntaxError(
                "Only single context manager supported in with statement",
                span=self.span_tracker.get_span(stmt),
                hint="Use 'with pl.incore():' or 'with pl.section_vector():' without multiple context managers",
            )

        item = stmt.items[0]
        context_expr = item.context_expr

        # Check if this is pl.incore(), pl.section_vector(), or pl.section_cube()
        if isinstance(context_expr, ast.Call):
            func = context_expr.func
            if isinstance(func, ast.Attribute):
                span = self.span_tracker.get_span(stmt)
                
                # Handle pl.incore() - creates ScopeStmt
                if func.attr == "incore":
                    with self.builder.scope(ir.ScopeKind.InCore, span):
                        self.scope_manager.enter_scope("scope")
                        for body_stmt in stmt.body:
                            self.parse_statement(body_stmt)
                        self.scope_manager.exit_scope(leak_vars=False)
                    return
                
                # Handle pl.section_vector() - creates SectionStmt
                if func.attr == "section_vector":
                    self._tuple_select_cache.clear()
                    with self.builder.section(ir.SectionKind.Vector, span):
                        self.scope_manager.enter_scope("section")
                        for body_stmt in stmt.body:
                            self.parse_statement(body_stmt)
                        self.scope_manager.exit_scope(leak_vars=False)
                    return

                # Handle pl.section_cube() - creates SectionStmt
                if func.attr == "section_cube":
                    self._tuple_select_cache.clear()
                    with self.builder.section(ir.SectionKind.Cube, span):
                        self.scope_manager.enter_scope("section")
                        for body_stmt in stmt.body:
                            self.parse_statement(body_stmt)
                        self.scope_manager.exit_scope(leak_vars=False)
                    return

                # Handle vf.vf_scope(name="...") - VF API code region marker
                if func.attr == "vf_scope":
                    # Extract name kwarg
                    vf_name = "vf_scope"
                    for kw in context_expr.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            vf_name = kw.value.value
                    # Emit scope_enter as EvalStmt
                    enter_call = ir.create_op_call(
                        "vf.vf_scope_enter", [], {"name": vf_name}, span
                    )
                    self.builder.emit(ir.EvalStmt(enter_call, span))
                    # Parse body normally (VF API calls inside)
                    self.scope_manager.enter_scope("vf_scope")
                    for body_stmt in stmt.body:
                        self.parse_statement(body_stmt)
                    self.scope_manager.exit_scope(leak_vars=True)
                    # Emit scope_exit as EvalStmt
                    exit_call = ir.create_op_call(
                        "vf.vf_scope_exit", [], {"name": vf_name}, span
                    )
                    self.builder.emit(ir.EvalStmt(exit_call, span))
                    return

        # Unsupported context manager
        raise UnsupportedFeatureError(
            "Unsupported context manager in with statement",
            span=self.span_tracker.get_span(stmt),
            hint="Only 'with pl.incore():', 'with pl.section_vector():', or 'with pl.section_cube():' are currently supported",
        )

    def parse_return(self, stmt: ast.Return) -> None:
        """Parse return statement.

        In inline mode, captures return expression instead of emitting ReturnStmt.

        Args:
            stmt: Return AST node
        """
        if self._inline_mode:
            if stmt.value is None:
                return  # void inline, no return value
            if isinstance(stmt.value, ast.Tuple):
                exprs = [self.parse_expression(elt) for elt in stmt.value.elts]
                self._inline_return_expr = ir.MakeTuple(exprs, self.span_tracker.get_span(stmt))
            else:
                self._inline_return_expr = self.parse_expression(stmt.value)
            return

        span = self.span_tracker.get_span(stmt)

        if stmt.value is None:
            self.builder.return_stmt(None, span)
            return

        # Handle tuple return
        if isinstance(stmt.value, ast.Tuple):
            return_exprs = []
            for elt in stmt.value.elts:
                return_exprs.append(self.parse_expression(elt))
            self.builder.return_stmt(return_exprs, span)
        else:
            # Single return value
            return_expr = self.parse_expression(stmt.value)
            self.builder.return_stmt([return_expr], span)

    def parse_break(self, stmt: ast.Break) -> None:
        """Parse break statement.

        Args:
            stmt: Break AST node
        """
        span = self.span_tracker.get_span(stmt)
        self.builder.break_stmt(span)

    def parse_continue(self, stmt: ast.Continue) -> None:
        """Parse continue statement.

        Args:
            stmt: Continue AST node
        """
        span = self.span_tracker.get_span(stmt)
        self.builder.continue_stmt(span)

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

    def parse_call(self, call: ast.Call) -> ir.Expr:
        """Parse function call.

        Args:
            call: Call AST node

        Returns:
            IR expression from call
        """
        func = call.func
        # Handle TileType(...) - type descriptor, not an operation
        if isinstance(func, ast.Name) and func.id == "TileType":
            return self._parse_tile_type_call(call)

        # Handle pl.yield_() specially
        if isinstance(func, ast.Attribute) and func.attr == "yield_":
            return self.parse_yield_call(call)

        # Handle cross-function calls via self.method_name() in @pl.program classes
        if isinstance(func, ast.Attribute):
            # Check for self.method_name pattern
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                method_name = func.attr
                if method_name in self.global_vars:
                    gvar = self.global_vars[method_name]
                    args = [self.parse_expression(arg) for arg in call.args]
                    span = self.span_tracker.get_span(call)

                    # Use return type from the parsed function if available
                    func_obj = self.gvar_to_func.get(gvar)
                    return_types = func_obj.return_types if func_obj else []
                    return self._make_call_with_return_type(gvar, args, return_types, span)
                else:
                    raise UndefinedVariableError(
                        f"Function '{method_name}' not defined in program",
                        span=self.span_tracker.get_span(call),
                        hint=f"Available functions: {list(self.global_vars.keys())}",
                    )

            # Handle pl.tensor.*, pl.block.*, and pl.* operation calls
            return self.parse_op_call(call)

        # Handle bare-name calls to external ir.Function or InlineFunction
        if isinstance(func, ast.Name):
            from .decorator import InlineFunction, KernelFunction  # noqa: PLC0415 (circular import)

            func_name = func.id
            resolved = self.expr_evaluator.closure_vars.get(func_name)
            if isinstance(resolved, ir.Function):
                return self._parse_external_function_call(func_name, resolved, call)
            if isinstance(resolved, InlineFunction):
                return self._parse_inline_call(func_name, resolved, call)
            if isinstance(resolved, KernelFunction):
                return self._parse_func_call(func_name, resolved, call)
            # Implicit func: annotated → func.call, unannotated → auto-inline
            if callable(resolved) and not isinstance(resolved, type):
                import inspect as _inspect  # noqa: PLC0415
                hints = getattr(resolved, "__annotations__", {})
                params = [p for p in _inspect.signature(resolved).parameters if p != "self"]
                # Annotated: all params have hints, OR no params but has return hint
                fully_annotated = (
                    (params and all(p in hints for p in params))
                    or (not params and "return" in hints)
                )
                if fully_annotated:
                    return self._implicit_func_call(func_name, resolved, call)
                return self._auto_inline_call(func_name, resolved, call)

        raise UnsupportedFeatureError(
            f"Unsupported function call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.* operations, pl.yield_(), self.method() for cross-function calls, "
            "or call an external @pl.function / @pl.inline by name",
        )

    def parse_yield_call(self, call: ast.Call) -> ir.Expr:
        """Parse pl.yield_() call.

        Args:
            call: Call to pl.yield_() or pl.yield_()

        Returns:
            IR expression (first yielded value for single yield)
        """
        span = self.span_tracker.get_span(call)
        yield_exprs = []

        for arg in call.args:
            expr = self.parse_expression(arg)
            yield_exprs.append(expr)

        # Emit yield statement
        self.builder.emit(ir.YieldStmt(yield_exprs, span))

        # Track yielded variables for if statement processing
        # This is for single assignment like: var = pl.yield_(expr)
        # We'll return a placeholder that gets resolved when if statement completes

        # Return first expression as the "value" of the yield
        # This handles: var = pl.yield_(expr)
        if len(yield_exprs) == 1:
            return yield_exprs[0]

        # For multiple yields, this should be handled as tuple assignment
        raise ParserSyntaxError(
            "Multiple yields should use tuple unpacking assignment",
            span=self.span_tracker.get_span(call),
            hint="Use tuple unpacking: (a, b) = pl.yield_(x, y)",
        )

    def parse_op_call(self, call: ast.Call) -> ir.Expr:
        """Parse operation call like pl.tensor.create_tensor() or pl.add().

        Args:
            call: Call AST node

        Returns:
            IR expression from operation
        """
        func = call.func

        # Navigate through attribute chain to find operation
        # e.g., pl.tensor.create_tensor -> ["pl", "tensor", "create_tensor"]
        # e.g., pl.add -> ["pl", "add"]
        attrs = []
        node = func
        while isinstance(node, ast.Attribute):
            attrs.insert(0, node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            attrs.insert(0, node.id)

        # pl.tensor.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] in ("pl", "plm") and attrs[1] == "tensor":
            op_name = attrs[2]
            return self._parse_tensor_op(op_name, call)

        # pl.block.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] in ("pl", "plm") and attrs[1] == "block":
            op_name = attrs[2]
            return self._parse_block_op(op_name, call)

        # pl.system.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "system":
            op_name = attrs[2]
            return self._parse_system_op(op_name, call)

        # pl.mutex.{operation} (3-segment) — A5 Mutex buffer-id tokens
        # Rewritten to ir_op.system.mutex_{lock,unlock} via the shared dispatcher.
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "mutex":
            op_name = "mutex_" + attrs[2]
            return self._parse_system_op(op_name, call)

        # pl.NBuffer(...) / pl.Buffer(...) / pl.L1NBuffer(...) etc.
        # Declarative buffer descriptors (Python-level objects; not IR ops).
        if len(attrs) == 2 and attrs[0] == "pl" and attrs[1] in _BUFFER_CLASS_NAMES:
            return self._parse_buffer_descriptor_call(attrs[1], call)

        # pl.const(value, dtype) — typed constant literal
        if len(attrs) >= 2 and attrs[0] in ("pl", "plm") and attrs[1] == "const":
            return self._parse_typed_constant(call)
        
        # plm.TileType(...) - type descriptor, not an operation
        if len(attrs) == 2 and attrs[0] == "plm" and attrs[1] == "TileType":
            return self._parse_tile_type_call(call)

        # plm.{operation} (2-segment) — manual (non-SSA) ops
        if len(attrs) == 2 and attrs[0] == "plm" and attrs[1] != "const":
            return self._parse_manual_op(attrs[1], call)

        # vf.{operation} (2-segment) — VF API ops (direct VF instruction)
        if len(attrs) == 2 and attrs[0] == "vf":
            return self._parse_vf_op(attrs[1], call)

        # pl.{operation} (2-segment, unified dispatch or promoted ops)
        if len(attrs) >= 2 and attrs[0] in ("pl", "plm") and attrs[1] not in ("tensor", "block", "system", "TileType"):
            op_name = attrs[1]
            return self._parse_unified_op(op_name, call)

        # Fallback: method call on a python_var (e.g. q_l1_db.current()).
        # Try evaluating the whole call via closure+scope and convert result.
        result = self._try_eval_python_var_call_as_ir(call)
        if result is not None:
            return result

        raise UnsupportedFeatureError(
            f"Unsupported operation call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.*, pl.tensor.*, pl.block.*, or pl.system.* operations",
        )

    def _make_call_with_return_type(
        self,
        gvar: ir.GlobalVar,
        args: list[ir.Expr],
        return_types: list[ir.Type],
        span: ir.Span,
    ) -> ir.Expr:
        """Create an ir.Call, attaching the return type when known.

        Args:
            gvar: GlobalVar identifying the callee
            args: Parsed argument expressions
            return_types: The callee's return type list (may be empty)
            span: Source span for the call
        """
        if not return_types:
            return ir.Call(gvar, args, span)
        if len(return_types) == 1:
            return ir.Call(gvar, args, return_types[0], span)
        return ir.Call(gvar, args, ir.TupleType(return_types), span)

    def _parse_external_function_call(
        self, _local_name: str, ext_func: ir.Function, call: ast.Call
    ) -> ir.Expr:
        """Parse a call to an externally-defined ir.Function.

        Args:
            _local_name: The name used in the caller's scope (may be aliased)
            ext_func: The external ir.Function object
            call: The AST Call node
        """
        func_name = ext_func.name
        span = self.span_tracker.get_span(call)

        # Validate no naming conflict with internal program functions
        if func_name in self.global_vars:
            raise ParserSyntaxError(
                f"External function '{func_name}' conflicts with program function '{func_name}'",
                span=span,
                hint="Rename either the external or program function to avoid the name conflict",
            )

        # Check for conflicting externals with same .name but different objects
        if func_name in self.external_funcs and self.external_funcs[func_name] is not ext_func:
            raise ParserSyntaxError(
                f"Conflicting external functions with name '{func_name}'",
                span=span,
                hint="External functions must have unique names; rename one of the functions",
            )

        # Track the external function
        self.external_funcs[func_name] = ext_func

        args = [self.parse_expression(arg) for arg in call.args]
        gvar = ir.GlobalVar(func_name)
        return self._make_call_with_return_type(gvar, args, ext_func.return_types, span)

    @staticmethod
    def _is_docstring(stmt: ast.stmt) -> bool:
        """Check if an AST statement is a docstring (string constant expression)."""
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )

    def _parse_inline_call(self, _local_name: str, inline_func: "InlineFunction", call: ast.Call) -> ir.Expr | None:
        """Parse a call to an InlineFunction, expanding its body in-place.

        Args:
            _local_name: The name used in the caller's scope
            inline_func: The InlineFunction object
            call: The AST Call node
        """
        span = self.span_tracker.get_span(call)

        expected = len(inline_func.param_names)
        got = len(call.args)
        if got < expected:
            raise ParserTypeError(
                f"Inline function '{inline_func.name}' expects {expected} argument(s), got {got}",
                span=span,
                hint=f"Check the inline function's parameter list: {inline_func.param_names}",
            )

        # Parse call arguments in the caller's context before entering inline scope
        arg_exprs = [self.parse_expression(arg) for arg in call.args]

        self.scope_manager.enter_scope("inline")
        for param_name, arg_expr in zip(inline_func.param_names, arg_exprs):
            self.scope_manager.define_var(param_name, arg_expr, allow_redef=True)

        # Handle extra arguments: inject them as closure variables
        extra_closure_vars = {}
        if got > expected:
            for i in range(expected, got):
                arg_node = call.args[i]
                if isinstance(arg_node, ast.Name):
                    var_name = arg_node.id
                    extra_closure_vars[var_name] = arg_exprs[i]

        # Save parser state and switch to the inline function's context
        prev_inline_state = (self._inline_mode, self._inline_return_expr, self._inline_prefix)
        self._inline_mode = True
        self._inline_return_expr = None
        self._inline_prefix = f"_il{self._inline_counter}_"
        self._inline_counter += 1

        prev_closure_vars = self.expr_evaluator.closure_vars
        self.expr_evaluator.closure_vars = {**inline_func.closure_vars, **prev_closure_vars, **extra_closure_vars}

        prev_span_state = (
            self.span_tracker.source_file,
            self.span_tracker.source_lines,
            self.span_tracker.line_offset,
            self.span_tracker.col_offset,
        )
        self.span_tracker.source_file = inline_func.source_file
        self.span_tracker.source_lines = inline_func.source_lines
        self.span_tracker.line_offset = inline_func.line_offset
        self.span_tracker.col_offset = inline_func.col_offset

        try:
            for i, stmt in enumerate(inline_func.func_def.body):
                if i == 0 and self._is_docstring(stmt):
                    continue
                self.parse_statement(stmt)
        finally:
            # Restore parser state
            (
                self.span_tracker.source_file,
                self.span_tracker.source_lines,
                self.span_tracker.line_offset,
                self.span_tracker.col_offset,
            ) = prev_span_state
            self.expr_evaluator.closure_vars = prev_closure_vars
            return_expr = self._inline_return_expr
            self._inline_mode, self._inline_return_expr, self._inline_prefix = prev_inline_state
            # Leak vars so inlined definitions are visible to the caller
            self.scope_manager.exit_scope(leak_vars=True)

        return return_expr

    @staticmethod
    def _check_no_nested_calls(func_def: ast.FunctionDef, func_name: str, span: Any) -> None:
        """Raise UnsupportedFeatureError if the function body contains bare-name function calls."""
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                raise UnsupportedFeatureError(
                    f"Auto-inlined function '{func_name}' cannot call other functions "
                    f"(found call to '{node.func.id}'). "
                    f"Add type annotations to use func.call, or use @pl.inline.",
                    span=span,
                )

    def _retrieve_function_source(
        self, func_name: str, fn: Callable, span: ir.Span, decorator_hint: str,
    ) -> tuple[str, list[str], int, int, ast.FunctionDef]:
        """Retrieve source, parse AST, and locate FunctionDef for a callable.

        Returns (source_file, source_lines, line_offset, col_offset, func_def).
        """
        import textwrap as _tw  # noqa: PLC0415

        from .decorator import _get_source_info  # noqa: PLC0415

        try:
            source_file, source_lines_raw, starting_line = _get_source_info(fn, "function")
        except Exception as e:
            raise UnsupportedFeatureError(
                f"Cannot compile '{func_name}': unable to retrieve source — {e}",
                span=span,
                hint=f"Define '{func_name}' in a .py file, or use {decorator_hint}",
            ) from e

        source_code = _tw.dedent("".join(source_lines_raw))
        col_offset = len(source_lines_raw[0]) - len(source_lines_raw[0].lstrip()) if source_lines_raw else 0
        line_offset = starting_line - 1
        source_lines = source_code.split("\n")

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise UnsupportedFeatureError(
                f"Cannot parse '{func_name}': {e}",
                span=span,
                hint=f"Use {decorator_hint} to explicitly mark '{func_name}'",
            ) from e

        func_def = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn.__name__),
            None,
        )
        if func_def is None:
            raise UnsupportedFeatureError(
                f"Cannot find function definition for '{func_name}' in source",
                span=span,
                hint=f"Use {decorator_hint} to explicitly mark '{func_name}'",
            )

        return source_file, source_lines, line_offset, col_offset, func_def

    def _build_function_closure(self, fn: Callable) -> dict[str, Any]:
        """Build closure dict from a callable's globals and free variables."""
        fn_closure: dict[str, Any] = {**fn.__globals__}
        if fn.__closure__ and fn.__code__.co_freevars:
            fn_closure.update(
                dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))
            )
        return fn_closure

    def _auto_inline_call(self, func_name: str, fn: Callable, call: ast.Call) -> ir.Expr | None:
        """Inline an unannotated plain Python function at the call site.

        The function body is expanded in-place (like @pl.inline). Nested bare-name
        function calls are forbidden; raise UnsupportedFeatureError if found.

        Args:
            func_name: Name used at the call site
            fn: The callable Python function (no DSL annotations)
            call: AST Call node

        Returns:
            IR expression (inlined return value)
        """
        from .decorator import InlineFunction  # noqa: PLC0415

        span = self.span_tracker.get_span(call)
        source_file, source_lines, line_offset, col_offset, func_def = (
            self._retrieve_function_source(func_name, fn, span, "@pl.inline")
        )

        # Constraint: no nested bare-name function calls
        self._check_no_nested_calls(func_def, func_name, span)

        fn_closure = self._build_function_closure(fn)
        param_names = [a.arg for a in func_def.args.args if a.arg != "self"]
        inline_func = InlineFunction(
            name=fn.__name__,
            func_def=func_def,
            param_names=param_names,
            source_file=source_file,
            source_lines=source_lines,
            line_offset=line_offset,
            col_offset=col_offset,
            closure_vars={**fn_closure, **self.expr_evaluator.closure_vars},
        )
        return self._parse_inline_call(func_name, inline_func, call)

    def _implicit_func_call(self, func_name: str, fn: Callable, call: ast.Call) -> ir.Expr:
        """Compile an annotated plain Python function as a KernelFunction and emit func.call.

        Functions with complete DSL type annotations are compiled on first encounter and
        cached by id(fn). Subsequent calls reuse the cached KernelFunction.

        Functions without annotations raise UnsupportedFeatureError with a helpful hint.

        Args:
            func_name: Name used at the call site
            fn: The callable Python function
            call: AST Call node

        Returns:
            IR expression (func.call result)
        """
        from .decorator import KernelFunction  # noqa: PLC0415
        from .diagnostics import ParserError, ParserTypeError as _ParserTypeError  # noqa: PLC0415

        span = self.span_tracker.get_span(call)

        fn_id = id(fn)
        if fn_id in self._implicit_func_cache:
            return self._parse_func_call(func_name, self._implicit_func_cache[fn_id], call)

        source_file, source_lines, line_offset, col_offset, func_def = (
            self._retrieve_function_source(func_name, fn, span, "@pl.func")
        )

        fn_closure = self._build_function_closure(fn)
        sub_parser = ASTParser(
            source_file,
            source_lines,
            line_offset,
            col_offset,
            closure_vars={**fn_closure, **self.expr_evaluator.closure_vars},
        )
        # Share cache so nested implicit calls are deduplicated
        sub_parser._implicit_func_cache = self._implicit_func_cache

        try:
            ir_func = sub_parser.parse_function(func_def, func_type=ir.FunctionType.Helper)
        except _ParserTypeError as e:
            raise UnsupportedFeatureError(
                f"'{func_name}' called from kernel but has no DSL type annotations. "
                f"Add annotations or use @pl.func.",
                span=span,
                hint=f"Example: def {func_name}(x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]: ...",
            ) from e
        except ParserError:
            raise

        gvar = ir.GlobalVar(fn.__name__)
        param_names = [a.arg for a in func_def.args.args if a.arg != "self"]
        kfunc = KernelFunction(
            name=fn.__name__,
            ir_function=ir_func,
            gvar=gvar,
            param_names=param_names,
        )
        self._implicit_func_cache[fn_id] = kfunc
        # Merge nested implicit functions discovered by the sub-parser
        self.external_funcs.update(sub_parser.external_funcs)
        # Register this function so its definition is included in the program
        self.external_funcs[fn.__name__] = ir_func
        return self._parse_func_call(func_name, kfunc, call)

    def _parse_func_call(self, func_name: str, kfunc: "KernelFunction", call: ast.Call) -> ir.Expr:
        """Parse a call to a @pl.func function, emitting an ir.Call for func.call generation.

        Args:
            func_name: Name used at the call site
            kfunc: KernelFunction holding the compiled ir.Function
            call: AST Call node

        Returns:
            ir.Call expression with the function's GlobalVar and parsed arguments
        """
        from .decorator import KernelFunction  # noqa: PLC0415

        span = self.span_tracker.get_span(call)
        expected = len(kfunc.param_names)
        got = len(call.args)
        if got != expected:
            raise ParserTypeError(
                f"Function '{func_name}' expects {expected} argument(s), got {got}",
                span=span,
                hint=f"Parameters: {kfunc.param_names}",
            )

        arg_exprs = [self.parse_expression(arg) for arg in call.args]
        return_types = list(kfunc.ir_function.return_types)
        return self._make_call_with_return_type(kfunc.gvar, arg_exprs, return_types, span)

    def _parse_tile_type_call(self, call: ast.Call) -> Any:
        """Parse TileType(...) as a dataclass instantiation."""
        from pypto_block.language.op.manual.op.manual_ops import TileType

        kwargs = {}
        for kw in call.keywords:
            kwargs[kw.arg] = self._resolve_single_kwarg(kw.arg, kw.value)

        return TileType(**kwargs)

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

    def _parse_op_kwargs(self, call: ast.Call) -> dict[str, Any]:
        """Parse keyword arguments for an operation call.

        Shared helper for tensor, block, system, and unified op parsing.

        Args:
            call: Call AST node

        Returns:
            Dictionary of keyword argument names to values
        """
        return {kw.arg: self._resolve_single_kwarg(kw.arg, kw.value) for kw in call.keywords}

    def _resolve_single_kwarg(self, key: str, value: ast.expr) -> Any:
        """Resolve a single keyword argument value to a Python or IR value.

        Args:
            key: Keyword argument name
            value: AST expression for the value

        Returns:
            Resolved Python or IR value
        """
        if key == "dtype":
            return self.type_resolver.resolve_dtype(value)
        elif isinstance(value, ast.Constant):
            return value.value
        elif isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            return self._resolve_unary_kwarg(value)
        elif isinstance(value, ast.Name):
            return self._resolve_name_kwarg(value)
        elif isinstance(value, ast.Attribute):
            return self._resolve_attribute_kwarg(value)
        elif isinstance(value, ast.List):
            return self._resolve_list_kwarg(value)
        else:
            return self.parse_expression(value)

    def _resolve_unary_kwarg(self, value: ast.UnaryOp) -> Any:
        """Resolve a unary op kwarg value (e.g., -1)."""
        if isinstance(value.operand, ast.Constant) and isinstance(value.operand.value, (int, float)):
            return -value.operand.value
        return self.parse_expression(value)

    def _resolve_name_kwarg(self, value: ast.Name) -> Any:
        """Resolve a Name kwarg value via scope lookup or closure eval."""
        if value.id in ["True", "False"]:
            return value.id == "True"
        if self.scope_manager.lookup_var(value.id) is not None:
            return self.parse_expression(value)  # IR var from scope
        # Not in IR scope — evaluate from closure (raises ParserTypeError if undefined)
        return self.expr_evaluator.eval_expr(value)

    def _resolve_attribute_kwarg(self, value: ast.Attribute) -> Any:
        """Resolve an Attribute kwarg value (e.g., pl.FP32, config.field)."""
        try:
            return self.type_resolver.resolve_dtype(value)
        except ParserTypeError:
            # Not a dtype — evaluate as a general expression from closure.
            # Use eval_expr (not try_eval_expr) so failures surface expression-specific
            # errors instead of the misleading dtype error from above.
            return self.expr_evaluator.eval_expr(value)

    def _resolve_list_kwarg(self, value: ast.List) -> Any:
        """Resolve a List kwarg value, trying closure eval first."""
        # If any element refers to a name in IR scope, parse as IR expressions
        # (mirrors _resolve_name_kwarg: IR scope takes priority over closure)
        if any(
            isinstance(elt, ast.Name) and self.scope_manager.lookup_var(elt.id) is not None
            for elt in value.elts
        ):
            return self.parse_list(value)
        success, result = self.expr_evaluator.try_eval_expr(value)
        if success and isinstance(result, list):
            return result
        return self.parse_list(value)

    def _parse_tensor_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse tensor operation.

        Args:
            op_name: Name of tensor operation
            call: Call AST node

        Returns:
            IR expression from tensor operation
        """
        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)

        # Map language-level operation name to IR-level name if needed
        ir_op_name = self._TENSOR_OP_NAME_MAP.get(op_name, op_name)

        # Call the appropriate tensor operation
        if hasattr(ir_op.tensor, ir_op_name):
            op_func = getattr(ir_op.tensor, ir_op_name)
            call_span = self.span_tracker.get_span(call)
            return op_func(*args, **kwargs, span=call_span)

        raise InvalidOperationError(
            f"Unknown tensor operation: {op_name}",
            span=self.span_tracker.get_span(call),
            hint=f"Check if '{op_name}' is a valid tensor operation",
        )

    def _parse_block_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse block operation.

        Args:
            op_name: Name of block operation
            call: Call AST node

        Returns:
            IR expression from block operation
        """
        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)

        # Special handling for make_tile with TileType
        if op_name == "make_tile":
            from pypto_block.language.op.manual.op.manual_ops import TileType
            if len(args) >= 1 and isinstance(args[0], TileType):
                tile_type = args[0]
                # Extract parameters from TileType
                kwargs.setdefault("shape", tile_type.shape)
                kwargs.setdefault("dtype", tile_type.dtype)
                kwargs.setdefault("target_memory", tile_type.target_memory)
                if tile_type.valid_shape is not None:
                    kwargs.setdefault("valid_shape", tile_type.valid_shape)
                if tile_type.blayout is not None:
                    kwargs.setdefault("blayout", tile_type.blayout)
                if tile_type.slayout is not None:
                    kwargs.setdefault("slayout", tile_type.slayout)
                if tile_type.fractal is not None:
                    kwargs.setdefault("fractal", tile_type.fractal)
                if tile_type.pad is not None:
                    kwargs.setdefault("pad", tile_type.pad)
                if tile_type.compact is not None:
                    kwargs.setdefault("compact", tile_type.compact)
                # Remove TileType from args, keep addr and size
                args = args[1:]

        # Call the appropriate block operation
        if hasattr(ir_op.block, op_name):
            op_func = getattr(ir_op.block, op_name)
            call_span = self.span_tracker.get_span(call)
            return op_func(*args, **kwargs, span=call_span)

        raise InvalidOperationError(
            f"Unknown block operation: {op_name}",
            span=self.span_tracker.get_span(call),
            hint=f"Check if '{op_name}' is a valid block operation",
        )

    def _parse_system_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse system operation.

        Args:
            op_name: Name of system operation
            call: Call AST node

        Returns:
            IR expression from system operation
        """
        # Mutex ops take positional (pipe, mutex_id) but pipe must be resolved
        # as a Python PipeType enum rather than an IR expression — so we use
        # kwarg-style resolution for the positional args.
        if op_name in ("mutex_lock", "mutex_unlock"):
            return self._parse_mutex_op(op_name, call)

        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)

        if hasattr(ir_op.system, op_name):
            op_func = getattr(ir_op.system, op_name)
            call_span = self.span_tracker.get_span(call)
            return op_func(*args, **kwargs, span=call_span)

        raise InvalidOperationError(
            f"Unknown system operation: {op_name}",
            span=self.span_tracker.get_span(call),
            hint=f"Check if '{op_name}' is a valid system operation",
        )

    def _parse_assert_op(self, call: ast.Call, call_span: ir.Span) -> ir.Expr:
        """Parse assert_ debug operation."""
        if call.keywords:
            raise ParserSyntaxError(
                "assert_ does not accept keyword arguments",
                span=call_span,
            )
        if len(call.args) < 1:
            raise ParserSyntaxError(
                f"assert_ requires at least 1 argument (condition), got {len(call.args)}",
                span=call_span,
            )

        condition = self.parse_expression(call.args[0])
        condition_text = self.span_tracker.get_source_text(call.args[0])

        if len(call.args) == 1:
            return ir_op.debug.assert_(condition, condition_text=condition_text, span=call_span)

        format_node = call.args[1]
        if not isinstance(format_node, ast.Constant) or not isinstance(format_node.value, str):
            raise ParserTypeError(
                "assert_ message must be a string literal",
                span=self.span_tracker.get_span(format_node),
                hint='Use a literal like plm.assert_(cond, "bad state") or plm.assert_(cond, "x=%d", x)',
            )

        args = [self.parse_expression(arg) for arg in call.args[2:]]
        return ir_op.debug.assert_(
            condition,
            format_node.value,
            *args,
            condition_text=condition_text,
            span=call_span,
        )


    def _parse_printf_op(self, call: ast.Call, call_span: ir.Span) -> ir.Expr:
        """Parse printf debug operation."""
        if call.keywords:
            raise ParserSyntaxError(
                "printf does not accept keyword arguments",
                span=call_span,
            )
        if len(call.args) < 1:
            raise ParserSyntaxError(
                f"printf requires at least a format string, got {len(call.args)} arguments",
                span=call_span,
            )

        format_node = call.args[0]
        if not isinstance(format_node, ast.Constant) or not isinstance(format_node.value, str):
            raise ParserTypeError(
                "printf format must be a string literal",
                span=self.span_tracker.get_span(format_node),
                hint='Use a literal like plm.printf("hello\\n") or plm.printf("x=%d\\n", value)',
            )

        args = [self.parse_expression(arg) for arg in call.args[1:]]
        return ir_op.debug.printf(format_node.value, *args, span=call_span)


    def _parse_mutex_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse pl.mutex.lock / pl.mutex.unlock.

        Supports two forms:
            pl.mutex.lock(pl.PipeType.MTE2, 0)            # positional
            pl.mutex.lock(pipe=pl.PipeType.MTE2, mutex_id=0)  # keyword

        The first positional arg (pipe) is always resolved via the closure
        (must be a static PipeType enum). The second positional arg
        (mutex_id) is resolved as a Python int when it is a constant, or as
        an IR Expr (for dynamic buf_id tuple subscripts) otherwise.
        """
        call_span = self.span_tracker.get_span(call)
        op_func = getattr(ir_op.system, op_name)

        # Collect pipe and mutex_id from either positional or keyword args.
        pipe_val = None
        mutex_id_val = None
        extra_kwargs: dict[str, Any] = {}

        positional = list(call.args)
        if len(positional) >= 1:
            pipe_val = self._resolve_mutex_pipe_arg(positional[0])
        if len(positional) >= 2:
            mutex_id_val = self._resolve_mutex_id_arg(positional[1])
        if len(positional) > 2:
            raise ParserSyntaxError(
                f"pl.mutex.{op_name.split('_', 1)[1]} takes at most 2 positional "
                f"arguments (pipe, mutex_id), got {len(positional)}",
                span=call_span,
            )

        for kw in call.keywords:
            if kw.arg == "pipe":
                pipe_val = self._resolve_mutex_pipe_arg(kw.value)
            elif kw.arg == "mutex_id":
                mutex_id_val = self._resolve_mutex_id_arg(kw.value)
            else:
                extra_kwargs[kw.arg] = self._resolve_single_kwarg(kw.arg, kw.value)

        if pipe_val is None:
            raise ParserSyntaxError(
                f"pl.mutex.{op_name.split('_', 1)[1]} missing required argument 'pipe'",
                span=call_span,
            )
        if mutex_id_val is None:
            raise ParserSyntaxError(
                f"pl.mutex.{op_name.split('_', 1)[1]} missing required argument 'mutex_id'",
                span=call_span,
            )

        return op_func(pipe_val, mutex_id_val, span=call_span, **extra_kwargs)

    def _resolve_mutex_pipe_arg(self, node: ast.expr):
        """Resolve a ``pipe`` argument to a static PipeType enum.

        Uses closure eval (not parse_expression) because PipeType is a
        Python-level constant, not an IR value.
        """
        try:
            return self.expr_evaluator.eval_expr(node)
        except Exception as exc:
            raise ParserTypeError(
                f"pl.mutex.lock/unlock 'pipe' must be a static PipeType enum: {exc}",
                span=self.span_tracker.get_span(node),
                hint="Pass e.g. pl.PipeType.MTE2",
            ) from exc

    def _resolve_mutex_id_arg(self, node: ast.expr):
        """Resolve a ``mutex_id`` argument.

        Returns an int when the AST is a constant, otherwise an IR Expr for
        the dynamic variant (e.g. tuple subscript ``buf_ids[buf_idx]``).
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
                return -node.operand.value
        # Try closure eval augmented with python_vars (for cur.buf_id etc).
        root = node
        while isinstance(root, (ast.Attribute, ast.Subscript, ast.Call)):
            if isinstance(root, ast.Call):
                root = root.func
            else:
                root = root.value
        if isinstance(root, ast.Name):
            py_obj = self.scope_manager.get_python_var(root.id)
            if py_obj is not None:
                saved = self.expr_evaluator.closure_vars
                try:
                    augmented = dict(saved)
                    augmented[root.id] = py_obj
                    self.expr_evaluator.closure_vars = augmented
                    ok, val = self.expr_evaluator.try_eval_expr(node)
                    if ok and isinstance(val, int):
                        return val
                finally:
                    self.expr_evaluator.closure_vars = saved
        # Try plain closure eval.
        ok, val = self.expr_evaluator.try_eval_expr(node)
        if ok and isinstance(val, int):
            return val
        return self.parse_expression(node)


    def _parse_debug_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse debug operation."""
        call_span = self.span_tracker.get_span(call)

        if op_name in {"dump_tensor", "dump_tile"}:
            args = [self.parse_expression(arg) for arg in call.args]
            kwargs = self._parse_op_kwargs(call)
            if hasattr(ir_op.debug, op_name):
                op_func = getattr(ir_op.debug, op_name)
                return op_func(*args, **kwargs, span=call_span)

            raise InvalidOperationError(
                f"Unknown debug operation: {op_name}",
                span=call_span,
                hint=f"Check if '{op_name}' is a valid debug operation",
            )

        if op_name == "assert_":
            kwargs = self._parse_op_kwargs(call)
            unknown_kwargs = sorted(key for key in kwargs if key != "loc")
            if unknown_kwargs:
                raise ParserSyntaxError(
                    "assert_ only accepts keyword argument 'loc', got "
                    + ", ".join(unknown_kwargs),
                    span=call_span,
                )
            if len(call.args) < 1:
                raise ParserSyntaxError(
                    f"assert_ requires at least 1 argument (condition), got {len(call.args)}",
                    span=call_span,
                )
            loc = kwargs.get("loc", False)
            if not isinstance(loc, bool):
                raise ParserSyntaxError("assert_ keyword argument 'loc' must be bool", span=call_span)

            condition = self.parse_expression(call.args[0])
            condition_text = self.span_tracker.get_source_text(call.args[0])

            if len(call.args) == 1:
                return ir_op.debug.assert_(
                    condition, condition_text=condition_text, loc=loc, span=call_span
                )

            format_node = call.args[1]
            if not isinstance(format_node, ast.Constant) or not isinstance(format_node.value, str):
                raise ParserTypeError(
                    "assert_ message must be a string literal",
                    span=self.span_tracker.get_span(format_node),
                    hint='Use a literal like plm.assert_(cond, "bad state") or plm.assert_(cond, "x=%d", x)',
                )

            args = [self.parse_expression(arg) for arg in call.args[2:]]
            return ir_op.debug.assert_(
                condition,
                format_node.value,
                *args,
                condition_text=condition_text,
                loc=loc,
                span=call_span,
            )

        if op_name == "trap":
            if call.keywords:
                raise ParserSyntaxError(
                    "trap does not accept keyword arguments",
                    span=call_span,
                )
            if call.args:
                raise ParserSyntaxError(
                    f"trap takes no arguments, got {len(call.args)}",
                    span=call_span,
                )

            return ir_op.debug.trap(span=call_span)

        if op_name == "printf":
            kwargs = self._parse_op_kwargs(call)
            unknown_kwargs = sorted(key for key in kwargs if key != "loc")
            if unknown_kwargs:
                raise ParserSyntaxError(
                    "printf only accepts keyword argument 'loc', got " + ", ".join(unknown_kwargs),
                    span=call_span,
                )
            if len(call.args) < 1:
                raise ParserSyntaxError(
                    f"printf requires at least a format string, got {len(call.args)} arguments",
                    span=call_span,
                )
            loc = kwargs.get("loc", False)
            if not isinstance(loc, bool):
                raise ParserSyntaxError("printf keyword argument 'loc' must be bool", span=call_span)

            format_node = call.args[0]
            if not isinstance(format_node, ast.Constant) or not isinstance(format_node.value, str):
                raise ParserTypeError(
                    "printf format must be a string literal",
                    span=self.span_tracker.get_span(format_node),
                    hint='Use a literal like plm.printf("hello\\n") or plm.printf("x=%d\\n", value)',
                )

            args = [self.parse_expression(arg) for arg in call.args[1:]]
            return ir_op.debug.printf(format_node.value, *args, loc=loc, span=call_span)

        raise InvalidOperationError(
            f"Unknown debug operation: {op_name}",
            span=call_span,
            hint=f"Check if '{op_name}' is a valid debug operation",
        )

    def _parse_ptr_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse pointer operation (ptoas scene: make_tensor, addptr).

        Args:
            op_name: Name of pointer operation
            call: Call AST node

        Returns:
            IR expression from pointer operation
        """
        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)
        if hasattr(ir_op.ptr, op_name):
            op_func = getattr(ir_op.ptr, op_name)
            call_span = self.span_tracker.get_span(call)
            return op_func(*args, **kwargs, span=call_span)

        raise InvalidOperationError(
            f"Unknown ptr operation: {op_name}",
            span=self.span_tracker.get_span(call),
            hint=f"Check if '{op_name}' is a valid ptr operation",
        )

    # Manual ops that share block SSA semantics (no explicit output tile arg).
    # These are routed to _parse_block_op directly.
    _MANUAL_AS_BLOCK_OPS: frozenset[str] = frozenset({
        "make_tile",  # allocation — same IR op as SSA
    })

    _MANUAL_AS_DEBUG_OPS: frozenset[str] = frozenset({
        "assert_",
        "dump_tensor",
        "dump_tile",
        "printf",
        "trap",
    })
    _INTERNAL_ONLY_MANUAL_OPS: frozenset[str] = frozenset({
        "move_fp",
        "store_fp",
    })
    _INTERNAL_ONLY_MANUAL_OP_HINTS: dict[str, str] = {
        "move_fp": "Use plm.move(..., fp_tile=...) instead",
        "store_fp": "Use plm.store(..., fp_tile=...) instead",
    }

    def _parse_vf_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse a VF API operation call: vf.{op_name}(...).

        VF ops directly emit VF instructions. Arguments and kwargs are passed
        through to ir.create_op_call with the "vf." prefix.

        Args:
            op_name: Name of the VF operation (without ``vf.`` prefix).
            call: Call AST node.

        Returns:
            IR expression for the VF op call.
        """
        span = self.span_tracker.get_span(call)
        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)
        return ir.create_op_call(f"vf.{op_name}", args, kwargs, span)

    # --- auto_mutex helpers ---------------------------------------------------

    def _try_resolve_tileref(self, node: ast.expr):
        """Try to resolve an AST node to a _TileRef without unwrapping to IR.

        Handles: ``cur.tile``, ``buf.tile``, ``nbuf.slots[i].tile``,
        and direct _TileRef references in closure/scope.
        Returns _TileRef or None.
        """
        from pypto_block.language.buffer import _TileRef
        # Walk to root Name to check python_var / closure
        root = node
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        if not isinstance(root, ast.Name):
            return None

        # Check scope python_var
        py_obj = self.scope_manager.get_python_var(root.id)
        if py_obj is not None:
            saved = self.expr_evaluator.closure_vars
            try:
                augmented = dict(saved)
                augmented[root.id] = py_obj
                self.expr_evaluator.closure_vars = augmented
                ok, val = self.expr_evaluator.try_eval_expr(node)
                if ok and isinstance(val, _TileRef):
                    return val
            finally:
                self.expr_evaluator.closure_vars = saved
            return None

        # Check closure
        ok, val = self.expr_evaluator.try_eval_expr(node)
        if ok and isinstance(val, _TileRef):
            return val
        return None

    def _resolve_auto_mutex_pipe(self, op_name: str, tilerefs: list):
        """Determine the pipe for auto_mutex from op_name and tile memory spaces."""
        from .op_pipeline import _MANUAL_OP_TO_PIPE, get_move_pipe, get_store_pipe

        if op_name == "move":
            # move(dst, src) → DSL arg0=dst, arg1=src
            dst_mem = tilerefs[0]._memory if tilerefs[0] else None
            src_mem = tilerefs[1]._memory if len(tilerefs) > 1 and tilerefs[1] else None
            if src_mem is not None and dst_mem is not None:
                return get_move_pipe(src_mem, dst_mem)
        if op_name in ("store", "store_tile"):
            # store(tensor, tile, offsets) → DSL arg1=tile (the source)
            src_mem = tilerefs[1]._memory if len(tilerefs) > 1 and tilerefs[1] else None
            if src_mem is not None:
                return get_store_pipe(src_mem)
        return _MANUAL_OP_TO_PIPE.get(op_name)

    def _emit_auto_mutex(self, op_name: str, call: ast.Call, span: ir.Span):
        """Emit mutex_lock before and mutex_unlock after a manual op.

        Scans call.args for _TileRef-backed tiles, determines the op pipe,
        and emits lock/unlock for each unique _TileRef.
        Returns None — the caller still parses the op normally.
        """
        from pypto_block.ir.op.system_ops import mutex_lock, mutex_unlock

        # 1. Scan args for _TileRef objects (before parse_expression)
        tilerefs = [self._try_resolve_tileref(arg) for arg in call.args]
        unique_refs = []
        seen = set()
        for tref in tilerefs:
            if tref is None or not tref.has_buf_id:
                continue
            key = id(tref)
            if key in seen:
                continue
            seen.add(key)
            unique_refs.append(tref)

        if not unique_refs:
            return

        # 2. Determine pipe
        pipe = self._resolve_auto_mutex_pipe(op_name, tilerefs)
        if pipe is None:
            return

        # 3. Emit lock for each unique _TileRef
        for tref in unique_refs:
            lock_expr = mutex_lock(pipe, tref._buf_id, buf_id_values=tref._buf_id_values, span=span)
            self.builder.emit(ir.EvalStmt(lock_expr, span))

        # Store for post-op unlock emission
        self._pending_mutex_unlocks = (unique_refs, pipe, span)

    def _emit_auto_mutex_unlocks(self):
        """Emit mutex_unlock calls queued by _emit_auto_mutex."""
        if not hasattr(self, "_pending_mutex_unlocks") or self._pending_mutex_unlocks is None:
            return
        from pypto_block.ir.op.system_ops import mutex_unlock

        unique_refs, pipe, span = self._pending_mutex_unlocks
        for tref in unique_refs:
            unlock_expr = mutex_unlock(pipe, tref._buf_id, buf_id_values=tref._buf_id_values, span=span)
            self.builder.emit(ir.EvalStmt(unlock_expr, span))
        self._pending_mutex_unlocks = None


    def _parse_manual_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse a manual (non-SSA) operation call: plm.{op_name}(..., dst=tile).

        Manual ops differ from block ops in two ways:
          1. They receive a pre-allocated output tile via ``dst=`` or ``out=`` kwarg.
          2. The dst variable is rebound in scope to the returned SSA value so that
             subsequent reads of that variable see the updated data.

        Args:
            op_name: Name of the manual operation (without ``manual.`` prefix).
            call: Call AST node.

        Returns:
            IR expression for the manual op call.
        """
        span = self.span_tracker.get_span(call)

        if op_name in self._INTERNAL_ONLY_MANUAL_OPS:
            raise InvalidOperationError(
                f"Unknown manual operation: {op_name}",
                span=span,
                hint=self._INTERNAL_ONLY_MANUAL_OP_HINTS[op_name],
            )

        # Ops with SSA block semantics — no explicit output tile needed.
        if op_name in self._MANUAL_AS_BLOCK_OPS:
            return self._parse_block_op(op_name, call)
        if op_name in self._MANUAL_AS_DEBUG_OPS:
            return self._parse_debug_op(op_name, call)

        # Auto-mutex: emit mutex_lock before op (and queue unlock for after)
        if self._auto_mutex:
            self._emit_auto_mutex(op_name, call, span)

        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)
        # Dispatch to ir_op.manual.<op_name> when a handler exists.
        if hasattr(ir_op.manual, op_name):
            op_func = getattr(ir_op.manual, op_name)
            result = op_func(*args, **kwargs, span=span)
        else:
            # first args is out, we need push out from first to last when create op
            result = ir.create_op_call(
                f"manual.{op_name}", args[1:] + args[0:1], kwargs, span
            )

        # Auto-mutex unlocks are deferred — emitted by parse_evaluation_statement
        # AFTER the op's EvalStmt is emitted, to ensure correct IR ordering:
        #   mutex_lock → op → mutex_unlock

        # Do NOT rebind the dst variable in scope.  The tile Var created by
        # make_tile remains the canonical buffer handle for the whole kernel;
        # manual ops (load, add, …) are side-effects on that buffer.  Re-binding
        # to the Call node would cause subsequent uses of the variable to resolve
        # to a Call rather than a Var, breaking GetExprAsCode lookups in the
        # PTO backend (which expects Var nodes for tile arguments).

        return result

    # Maps unified op names to the scalar variant for block ops.
    # Only binary arithmetic ops have scalar auto-dispatch.
    _BLOCK_SCALAR_OPS: dict[str, str] = {
        "add": "adds",
        "sub": "subs",
        "mul": "muls",
        "div": "divs",
    }

    # Maps unified op names to ir scalar expression functions.
    _SCALAR_BINARY_OPS: dict[str, str] = {
        "min": "min_",
        "max": "max_",
    }

    _SCALAR_UNARY_OPS: dict[str, str] = {}

    _SCALAR_BLOCK_OPS = {"index_cast"}

    # Maps language-level tensor operation names to IR-level names.
    _TENSOR_OP_NAME_MAP: dict[str, str] = {
        "create_tensor": "create",
    }

    # Ops that exist only in one module (no dispatch needed).
    _TENSOR_ONLY_OPS = {
        "create_tensor",
        "dim",
        "assemble",
        "add_scalar",
        "sub_scalar",
        "mul_scalar",
        "div_scalar",
    }

    # Ops that only exist in the ptr module (ptoas scene).
    _PTR_ONLY_OPS = {"make_tensor", "addptr"}
    _BLOCK_ONLY_OPS = {
        "load",
        "store",
        "move",
        "neg",
        "sqrt",
        "rsqrt",
        "recip",
        "log",
        "relu",
        "matmul_acc",
        "minimum",
        "cmp",
        "cmps",
        "adds",
        "subs",
        "muls",
        "divs",
        "sum",
        "row_min",
        "row_expand",
        "row_expand_add",
        "row_expand_sub",
        "row_expand_mul",
        "row_expand_div",
        "col_expand",
        "col_expand_mul",
        "col_expand_div",
        "col_expand_sub",
        "col_max",
        "col_sum",
        "expands",
        "matmul_bias",
        "gemv",
        "gemv_acc",
        "gemv_bias",
        "abs",
        "make_tile",
    }

    def _parse_unified_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse unified operation call (pl.{op_name}).

        Dispatches to tensor or block IR op based on the first argument's type.

        Args:
            op_name: Name of the operation
            call: Call AST node

        Returns:
            IR expression from the dispatched operation
        """
        # Short-circuit for ops that only exist in one module
        if op_name in self._PTR_ONLY_OPS:
            return self._parse_ptr_op(op_name, call)
        if op_name in self._TENSOR_ONLY_OPS:
            return self._parse_tensor_op(op_name, call)
        if op_name in self._BLOCK_ONLY_OPS:
            return self._parse_block_op(op_name, call)

        call_span = self.span_tracker.get_span(call)

        if not call.args:
            raise InvalidOperationError(
                f"Unified operation '{op_name}' requires at least one argument for type dispatch",
                span=call_span,
                hint="Provide a Tensor or Tile as the first argument",
            )

        # Parse only the first arg to determine dispatch target
        first_arg = self.parse_expression(call.args[0])
        first_type = first_arg.type

        if isinstance(first_type, ir.TensorType):
            return self._parse_tensor_op(op_name, call)

        if isinstance(first_type, ir.TileType):
            # For binary arithmetic ops, check if rhs is scalar → use scalar variant
            scalar_op = self._BLOCK_SCALAR_OPS.get(op_name)
            if scalar_op and len(call.args) >= 2:
                rhs_arg = self.parse_expression(call.args[1])
                if isinstance(rhs_arg.type, ir.ScalarType):
                    return self._parse_block_op(scalar_op, call)

            return self._parse_block_op(op_name, call)

        if isinstance(first_type, ir.ScalarType):
            return self._parse_scalar_op(op_name, call, call_span)

        raise InvalidOperationError(
            f"Cannot dispatch '{op_name}': first argument has type {type(first_type).__name__}, "
            f"expected TensorType, TileType, or ScalarType",
            span=call_span,
            hint="Use pl.tensor.* or pl.block.* for explicit dispatch; for pointer ops use pl.make_tensor or pl.addptr",
        )

    def _parse_typed_constant(self, call: ast.Call) -> ir.Expr:
        """Parse pl.const(value, dtype) → ConstInt or ConstFloat.

        Args:
            call: Call AST node for pl.const(value, dtype)

        Returns:
            ConstInt or ConstFloat with the specified dtype
        """
        span = self.span_tracker.get_span(call)

        if len(call.args) != 2:
            raise ParserSyntaxError(
                "pl.const() requires exactly 2 arguments: value and dtype",
                span=span,
                hint="Use pl.const(42, pl.INT32) or pl.const(1.0, pl.FP16)",
            )

        # Extract numeric value from first argument (handles Constant and -Constant)
        value_node = call.args[0]
        negate = False
        if isinstance(value_node, ast.UnaryOp) and isinstance(value_node.op, ast.USub):
            negate = True
            value_node = value_node.operand

        if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, (int, float)):
            raise ParserSyntaxError(
                "pl.const() first argument must be a numeric literal",
                span=span,
                hint="Use an int or float literal: pl.const(42, pl.INT32)",
            )

        value = value_node.value
        if negate:
            value = -value

        # Resolve dtype from second argument
        dtype = self.type_resolver.resolve_dtype(call.args[1])

        if isinstance(value, float):
            return ir.ConstFloat(value, dtype, span)
        else:
            return ir.ConstInt(value, dtype, span)

    def _parse_scalar_op(self, op_name: str, call: ast.Call, call_span: ir.Span) -> ir.Expr:
        """Parse scalar operation (e.g. pl.min(s1, s2) where s1, s2 are scalars).

        Args:
            op_name: Name of the operation
            call: Call AST node
            call_span: Source span for error reporting

        Returns:
            IR scalar expression
        """
        if call.keywords:
            raise InvalidOperationError(
                f"Scalar operation '{op_name}' does not accept keyword arguments",
                span=call_span,
            )

        if op_name in self._SCALAR_BINARY_OPS:
            if len(call.args) != 2:
                raise InvalidOperationError(
                    f"Scalar binary operation '{op_name}' requires exactly 2 arguments, got {len(call.args)}",
                    span=call_span,
                )
            lhs = self.parse_expression(call.args[0])
            rhs = self.parse_expression(call.args[1])
            ir_func_name = self._SCALAR_BINARY_OPS[op_name]
            ir_func = getattr(ir, ir_func_name)
            return ir_func(lhs, rhs, call_span)

        if op_name in self._SCALAR_UNARY_OPS:
            if len(call.args) != 1:
                raise InvalidOperationError(
                    f"Scalar unary operation '{op_name}' requires exactly 1 argument, got {len(call.args)}",
                    span=call_span,
                )
            arg = self.parse_expression(call.args[0])
            ir_func_name = self._SCALAR_UNARY_OPS[op_name]
            ir_func = getattr(ir, ir_func_name)
            return ir_func(arg, call_span)

        if op_name in self._SCALAR_BLOCK_OPS:
            if len(call.args) != 1:
                raise InvalidOperationError(
                    f"Scalar block operation '{op_name}' requires exactly 1 argument, got {len(call.args)}",
                    span=call_span,
                )
            arg = self.parse_expression(call.args[0])
            ir_func = getattr(ir_op.block, op_name)
            return ir_func(arg, call_span)

        raise InvalidOperationError(
            f"Operation '{op_name}' is not supported for scalar arguments",
            span=call_span,
            hint="Supported scalar ops: min, max, index_cast",
        )

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


__all__ = ["ASTParser"]
