# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Call and operation parsing helpers for ASTParser."""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pypto_block.ir import op as ir_op
from pypto_block.pypto_core import ir

from .constants import _BUFFER_CLASS_NAMES
from .diagnostics import (
    InvalidOperationError,
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)

if TYPE_CHECKING:
    from .decorator import InlineFunction, KernelFunction


class CallParserMixin:
    """Mixin containing call and operation parsing methods for ``ASTParser``."""

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
        from .ast_parser import ASTParser  # noqa: PLC0415

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
        from pypto_block.ir.op.system_ops import mutex_lock

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
