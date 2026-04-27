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
from typing import Any

from pypto_block.ir import IRBuilder
from pypto_block.pypto_core import ir

from .assignment_parser import AssignmentParserMixin
from .buffer_parser import BufferParserMixin
from .call_parser import CallParserMixin
from .control_flow_parser import ControlFlowParserMixin
from .expression_parser import ExpressionParserMixin
from .struct_parser import StructParserMixin
from .diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    UnsupportedFeatureError,
)
from .expr_evaluator import ExprEvaluator
from .scope_manager import ScopeManager
from .span_tracker import SpanTracker
from .type_resolver import TypeResolver
from ..typing.tiling import ScalarFieldInfo, get_tiling_fields, is_tiling_class


class ASTParser(
    AssignmentParserMixin,
    ControlFlowParserMixin,
    ExpressionParserMixin,
    StructParserMixin,
    BufferParserMixin,
    CallParserMixin,
):
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



__all__ = ["ASTParser"]
