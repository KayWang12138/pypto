# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Control-flow parsing helpers for ASTParser."""

from __future__ import annotations

import ast
from typing import Any

from pypto_block.pypto_core import DataType, ir

from .diagnostics import ParserSyntaxError, UnsupportedFeatureError
from .utils import _const_int_value, _is_const_int


class ControlFlowParserMixin:
    """Mixin containing loop, branch, with-scope, and return parsing."""

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
