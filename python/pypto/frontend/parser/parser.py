#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""PTO Script Parser."""
from collections.abc import Iterator
import re
from typing import Any, Optional, Union

import pypto
from pypto.symbolic_scalar import SymbolicScalar
from . import doc
from .context import Context
from .diagnostics import DiagnosticLevel, Diagnostics, Source
from .error import ParserError, RenderedParserError
from .evaluator import ExprEvaluator
from .liveness import LivenessAnalyzer

DEFAULT_VISIT = {
    "Interactive",
    "Module",
    "Expression",
    "Pass",
}


class Parser(doc.NodeVisitor):
    """The TVMScript parser

    Parameters
    ----------
    diag : Diagnostics
        The diagnostics for error reporting.

    context : Context
        The context for parsing.

    """

    diag: Diagnostics
    context: Context
    delete_after: dict[int, set[str]]
    _parsed_node: Optional[doc.AST]
    _parsed_extra_vars: dict[str, Any]
    _result: Optional[Any]
    _signature_cache: Optional[
        tuple[list[pypto.Tensor], list[tuple[str, type]], list[pypto.Tensor]]
    ]

    def __init__(
        self, source: Source, extra_vars: Optional[dict[str, Any]] = None
    ) -> None:
        self.diag = Diagnostics(source)
        self.context = Context()
        self.delete_after = {}
        self._parsed_node = None
        self._parsed_extra_vars = extra_vars or {}
        self._result = None
        self._signature_cache = None
        self._bound_dim_values: Optional[dict[str, int]] = None

    def parse(self) -> "Parser":
        """The main parse method for parser (lazy mode).

        Prepares the AST but defers actual parsing until execute() is called.

        Returns
        -------
        res : Parser
            Returns self for chaining.
        """
        node = self.diag.source.as_ast()
        analyzer = LivenessAnalyzer()
        exempt_vars = set(self._parsed_extra_vars.keys())
        self.delete_after = analyzer.analyze(node, exempt_vars)

        # Store for later execution (lazy mode)
        self._parsed_node = node
        return self

    def match_input_shapes(
        self,
        input_shapes: list[list[int]],
        input_tensor_defs: Optional[list[pypto.Tensor]] = None,
    ) -> dict[str, int]:
        """Match input tensors to symbolic dimensions.

        Creates a mapping from SymbolicScalar objects (found in tensor shapes or
        as symbolic parameters) to their concrete values based on actual input data.

        Parameters
        ----------
        input_shapes : list[list[int]]
            List of input shapes.

        input_tensor_defs : Optional[list[pypto.Tensor]]
            List of input tensor definitions.

        Returns
        -------
        dict[str, int]
            Mapping from SymbolicScalar objects to their concrete values.
        """
        dim_value_map = {}

        # Get the signature to know which inputs have symbolic dimensions
        if input_tensor_defs is None:
            input_tensor_defs, _, _ = self.get_signature()

        def _assign_dim_value(dim: pypto.SymbolicScalar, actual_value: int) -> None:
            if dim_value_map.get(str(dim), actual_value) != actual_value:
                raise ValueError(
                    f"Symbolic scalar {dim} has multiple concrete values: {dim_value_map[dim]} and {actual_value}"
                )
            dim_value_map[str(dim)] = actual_value

        # Iterate through both the actual inputs and their definitions
        for actual_input_shape, tensor_def in zip(input_shapes, input_tensor_defs):
            if isinstance(actual_input_shape, list):
                # For Tensor inputs, map each symbolic dimension to its concrete shape value
                for axis, dim in enumerate(tensor_def.shape):
                    if isinstance(dim, pypto.SymbolicScalar):
                        # Extract the actual shape value from the input tensor
                        actual_value = actual_input_shape[axis]
                        if isinstance(actual_value, int):
                            _assign_dim_value(dim, actual_value)
            else:
                raise TypeError(
                    f"Invalid input shape type: {type(actual_input_shape)}, expected list"
                )

        return dim_value_map

    def bind_dynamic_dims_from_inputs(self, inputs: list[list[int]]) -> None:
        """Bind symbolic dimensions to concrete values using sample inputs.

        Parameters
        ----------
        inputs : list[list[int]]
            Concrete sample inputs whose shapes/values are used to resolve
            dynamic (symbolic) dimensions.

        """

        self._bound_dim_values = self.match_input_shapes(inputs)

    def bind_non_tensor_args(self, args: dict[str, Any]) -> None:
        """Inject concrete non-tensor argument values for parsing/execution."""
        self._parsed_extra_vars.update(args)

    def _apply_bound_dim_values_to_context_frame(self) -> None:
        """Replace symbolic scalars in the current frame with bound concrete values."""
        if not self._bound_dim_values:
            return
        if not self.context.frames:
            return

        current_frame = self.context.frames[-1]
        for var_name in list(current_frame.vars):
            values_stack = self.context.name2value.get(var_name, [])
            if not values_stack:
                continue
            current_value = values_stack[-1]
            if (
                isinstance(current_value, SymbolicScalar)
                and str(current_value) in self._bound_dim_values
            ):
                concrete_value = self._bound_dim_values[str(current_value)]
                self.context.add(var_name, concrete_value, allow_update=True)

    def get_signature(
        self,
    ) -> tuple[list[pypto.Tensor], list[tuple[str, type]], list[pypto.Tensor]]:
        """Extract function signature (inputs and outputs) without full parsing.

        Returns
        -------
        res : tuple[list[pypto.Tensor], list[pypto.Tensor]]
            A tuple of (input_tensors, output_tensors).

        Raises
        ------
        RuntimeError
            If parse() was not called before get_signature().
        """

        if self._signature_cache is not None:
            return self._signature_cache

        node = self.diag.source.as_ast()

        # Find the function definition node
        if isinstance(node, doc.Module):
            for item in node.body:
                if isinstance(item, doc.FunctionDef):
                    function_node = item
                    break
            else:
                raise RuntimeError("No function definition found in parsed AST")
        else:
            raise RuntimeError("Expected Module AST node")

        # Temporarily set up context to parse signature
        with self.context.with_frame():
            for k, v in self._parsed_extra_vars.items():
                self.context.add(k, v)
            # If sample inputs were provided, use them to concretize symbolic dims.
            self._apply_bound_dim_values_to_context_frame()

            # Get input arguments
            tensor_input_args, non_tensor_input_args = self.visit_arguments(
                function_node.args
            )

            # Get and validate output arguments
            output_expr = self.visit_expr(function_node.returns)
            output_tensors = self._normalize_output_annotation(
                output_expr, function_node.returns
            )

            self._signature_cache = (
                tensor_input_args,
                non_tensor_input_args,
                output_tensors,
            )

            return self._signature_cache

    def execute(self) -> Any:
        """Execute the deferred parsing.

        Returns
        -------
        res : Any
            The doc AST node visiting result.

        Raises
        ------
        RuntimeError
            If parse() was not called before execute().
        """
        if self._parsed_node is None:
            raise RuntimeError("parse() must be called before execute()")

        # Return cached result if already executed
        if self._result is not None:
            return self._result

        # Execute the deferred parsing
        with self.context.with_frame():
            for k, v in self._parsed_extra_vars.items():
                self.context.add(k, v)
            # Apply any concrete bindings for symbolic dimensions
            self._apply_bound_dim_values_to_context_frame()
            self._result = self.visit(self._parsed_node)
        return self._result

    def _normalize_output_annotation(
        self, output_expr: Any, node: doc.AST
    ) -> list[pypto.Tensor]:
        """Validate and normalize return annotations into a tensor list."""
        if isinstance(output_expr, pypto.Tensor):
            return [output_expr]
        if isinstance(output_expr, (list, tuple)):
            tensors: list[pypto.Tensor] = []
            for idx, item in enumerate(output_expr):
                if not isinstance(item, pypto.Tensor):
                    raise ParserError(
                        node,
                        TypeError(
                            f"Return annotation at index {idx} must be a tensor, "
                            f"but got {type(item).__name__}."
                        ),
                    )
                tensors.append(item)
            return tensors

        raise ParserError(
            node,
            TypeError(
                "Return annotation must be a tensor or a list/tuple of tensors, "
                f"but got {type(output_expr).__name__}."
            ),
        )

    def eval_expr(
        self,
        node: Union[doc.Expression, doc.expr],
        extra_vars: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Expression evaluation when parsing.

        Parameters
        ----------
        node : Union[doc.expr, doc.Expression]
            The root node of AST tree node of expression to evaluate.

        extra_vars : Optional[dict[str, Any]]
            The optional global value table for expression evaluation.

        Returns
        -------
        res : Any
            The evaluation result.
        """
        var_values = self.context.get()
        if extra_vars is not None:
            for k, v in extra_vars.items():
                var_values[k] = v
        return ExprEvaluator.eval(node, var_values, self.diag)

    def visit(self, node: doc.AST) -> Any:
        """The general visiting method.

        Parameters
        ----------
        node : doc.AST
            The doc AST node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        if isinstance(node, (list, tuple)):
            result = None
            for item in node:
                res = self.visit(item)
                if res is not None:
                    result = res
            return result
        if not isinstance(node, doc.AST):
            return
        name = node.__class__.__name__.split(".")[-1]

        if name in DEFAULT_VISIT:
            func = self.generic_visit
        else:
            # Convert CamelCase to snake_case for function names
            snake_case_name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
            func = getattr(self, f"visit_{snake_case_name}", None)
        if func is None:
            raise ParserError(
                node,
                f"{name} is not supported by the PTO parser yet. "
                f"Please check the documentation for supported Python features.",
            )
        try:
            return func(node)
        except RenderedParserError:  # prevent the bug from being reported again
            # If it is a rendered parser error, do not report it again.
            raise
        except ParserError as err:
            # If it is a parser error, report it.
            self.diag.error(err.node, str(err))
        except Exception as err:  # pylint: disable=broad-except
            # If it is a python native error, report it as a bug. Note that all users error should be reported as ParserError.
            self.diag.bug(node, str(err))

    def generic_visit(self, node: doc.AST) -> Any:
        """Generic visit method that visits all child nodes.

        This is called when no specific visit_* method exists for a node type.
        It recursively visits all fields of the node and returns the last non-None result.

        Parameters
        ----------
        node : doc.AST
            The node whose children should be visited.

        Returns
        -------
        res : Any
            The last non-None result from visiting child nodes.
        """
        result = None
        for field in doc.get_cls_fields(node.__class__):
            value = getattr(node, field, None)
            if value is None:
                pass
            elif isinstance(value, (doc.AST, list, tuple)):
                res = self.visit(value)
                if res is not None:
                    result = res
        return result

    def visit_body(self, node: list[doc.stmt]) -> Any:
        """The general body visiting method.

        Parameters
        ----------
        node : list[doc.stmt]
            The list of statements in body.

        Returns
        -------
        res : Any
            The visiting result.
        """
        for stmt in node:
            self.visit(stmt)
            self._auto_cleanup_after_stmt(stmt)

    def _validate_return_statements(self, node: doc.FunctionDef) -> None:
        """Validate that return statements only appear once at the end of the function.

        Parameters
        ----------
        node : doc.FunctionDef
            The function definition node to validate.

        Raises
        ------
        ParserError
            If there are multiple return statements or a return statement not at the end.
        """

        def find_returns(stmts: list[doc.stmt]) -> list[tuple[int, doc.Return]]:
            """Find all return statements with their positions."""
            returns = []
            for idx, stmt in enumerate(stmts):
                if isinstance(stmt, doc.Return):
                    returns.append((idx, stmt))
                # Check for returns in nested if statements
                elif isinstance(stmt, doc.If):
                    # Check in if body
                    nested_returns = find_returns(stmt.body)
                    if nested_returns:
                        for _, ret in nested_returns:
                            returns.append((idx, ret))
                    # Check in else/elif body
                    if stmt.orelse:
                        nested_returns = find_returns(stmt.orelse)
                        if nested_returns:
                            for _, ret in nested_returns:
                                returns.append((idx, ret))
                # Check for returns in for loops
                elif isinstance(stmt, doc.For):
                    nested_returns = find_returns(stmt.body)
                    if nested_returns:
                        for _, ret in nested_returns:
                            returns.append((idx, ret))
                    if stmt.orelse:
                        nested_returns = find_returns(stmt.orelse)
                        if nested_returns:
                            for _, ret in nested_returns:
                                returns.append((idx, ret))
            return returns

        returns = find_returns(node.body)

        if len(returns) > 1:
            raise ParserError(
                returns[1][1],
                ValueError(
                    "Only one return statement is allowed in a function. "
                    f"Found {len(returns)} return statements."
                ),
            )

        if len(returns) == 1:
            idx, return_node = returns[0]
            # Check if return is at the last position of the function body
            if idx != len(node.body) - 1:
                raise ParserError(
                    return_node,
                    ValueError(
                        "Return statement must be at the last position of the function body. "
                        f"Found return at position {idx}, but function body has {len(node.body)} statements."
                    ),
                )

    def _extract_return_names(self, node: doc.FunctionDef) -> Optional[list[str]]:
        """Extract the variable names being returned from the function.

        Parameters
        ----------
        node : doc.FunctionDef
            The function definition node.

        Returns
        -------
        res : Optional[list[str]]
            List of variable names being returned, or None if no return statement.
        """
        # Find the return statement (should be at the end)
        if not node.body:
            return None

        last_stmt = node.body[-1]
        if not isinstance(last_stmt, doc.Return):
            return None

        if last_stmt.value is None:
            return None

        # Extract variable names from the return value
        return_value = last_stmt.value
        if isinstance(return_value, doc.Name):
            # Single return value: return x
            return [return_value.id]
        elif isinstance(return_value, (doc.Tuple, doc.List)):
            # Multiple return values: return x, y
            names = []
            for elt in return_value.elts:
                if isinstance(elt, doc.Name):
                    names.append(elt.id)
                else:
                    # If return contains expressions (not just names), we can't use this optimization
                    return None
            return names
        else:
            # Return contains an expression, not just variable names
            raise ParserError(
                node,
                ValueError(
                    "Return value must be a variable name or a tuple/list of variable names."
                ),
            )

    def _validate_output_args(
        self, output_args: list[pypto.Tensor], node: doc.FunctionDef
    ) -> None:
        """Validate that output arguments are valid tensors.

        Parameters
        ----------
        output_args : list[pypto.Tensor]
            The output arguments to validate.
        node : doc.FunctionDef
            The function definition node for error reporting.

        Raises
        ------
        ParserError
            If any output argument is invalid.
        """
        for output_arg in output_args:
            # Valid output includes:
            # - pypto.Tensor: -> pypto.Tensor(shape, dtype)
            # - tuple/list of pypto.Tensor: -> tuple/list of pypto.Tensor(shape, dtype)
            # - TODO: type(pypto.Tensor): -> pypto.Tensor

            if output_arg is None:
                raise ParserError(
                    node.returns,
                    ValueError(
                        "Return value must be a tensor with shape and dtype specified."
                    ),
                )

            if output_arg is pypto.Tensor:
                raise ParserError(
                    node.returns,
                    ValueError(
                        "Return value must be a tensor with shape and dtype specified."
                    ),
                )

            if output_arg is not None and not isinstance(output_arg, pypto.Tensor):
                raise ParserError(
                    node.returns,
                    TypeError(
                        f"Return value must be a tensor, but got {type(output_arg)}."
                    ),
                )

    def _mark_dynamic_dimensions(
        self, tensors: list[pypto.Tensor]
    ) -> list[pypto.Tensor]:
        """Mark dynamic dimensions for tensors.

        Dynamic dimensions are those specified with pypto.dynamic() and represented
        as SymbolicScalar objects. This method marks them in the PTO IR.

        Parameters
        ----------
        tensors : list[pypto.Tensor]
            List of tensors to process.

        Returns
        -------
        list[pypto.Tensor]
            List of tensors with dynamic dimensions marked.
        """
        result = []
        for tensor in tensors:
            if isinstance(tensor, pypto.Tensor):
                shape = [
                    -1 if isinstance(dim, pypto.SymbolicScalar) else dim
                    for dim in tensor.shape
                ]
                result.append(pypto.Tensor(shape, tensor.dtype, tensor.name))
            else:
                raise ParserError(
                    tensor,
                    TypeError(
                        f"Tensor must be a pypto.Tensor, but got {type(tensor)}."
                    ),
                )
        return result

    def _add_tensor_args_to_context(self, tensor_args: list[pypto.Tensor]) -> None:
        """Add tensor arguments to the parsing context.

        Parameters
        ----------
        tensor_args : list[pypto.Tensor]
            List of tensor arguments to add to context.
        """
        for arg in tensor_args:
            if isinstance(arg, pypto.Tensor):
                self.context.add(arg.name, arg)

    def _setup_output_var_mapping(
        self, node: doc.FunctionDef, output_args: list[pypto.Tensor]
    ) -> dict[str, pypto.Tensor]:
        """Set up mapping from return variable names to output tensors.

        This method extracts the variable names being returned and maps them to
        the pre-defined output tensors, then adds them to the context.

        Parameters
        ----------
        node : doc.FunctionDef
            The function definition node.
        output_args : list[pypto.Tensor]
            List of output tensors.

        Returns
        -------
        dict[str, pypto.Tensor]
            Mapping from variable names to output tensors.

        Raises
        ------
        ParserError
            If the number of return values doesn't match the number of output tensors.
        """
        return_names = self._extract_return_names(node)
        output_var_mapping = {}

        if return_names is not None:
            if len(return_names) != len(output_args):
                raise ParserError(
                    node.body[-1] if node.body else node,
                    ValueError(
                        f"Return statement has {len(return_names)} values but function signature "
                        f"specifies {len(output_args)} output tensors."
                    ),
                )
            # Map return variable names to pre-defined output tensors
            # and add them to context so they're used directly
            for name, output_tensor in zip(return_names, output_args):
                output_tensor.name = name
                output_var_mapping[name] = output_tensor
                # Add to context so the variable refers to the pre-defined tensor
                self.context.add(name, output_tensor)

        return output_var_mapping

    def _add_metadata_to_context(
        self, func_name: str, output_var_mapping: dict[str, pypto.Tensor]
    ) -> None:
        """Add function metadata to context for use in other visit methods.

        Parameters
        ----------
        func_name : str
            The function name.
        output_var_mapping : dict[str, pypto.Tensor]
            Mapping from variable names to output tensors.
        """
        # Store function name for use in visit_return
        # TODO: Move to ir builder context once it is implemented.
        self.context.add("__func_name__", func_name)
        # Add output variable mapping to context
        # This tells the parser to use pre-defined output tensors for these variables
        self.context.add("__output_var_mapping__", output_var_mapping)

    def visit_function_def(self, node: doc.FunctionDef) -> pypto.Function:
        """The general function definition visit method.

        Parameters
        ----------
        node : doc.FunctionDef
            The doc FunctionDef node.

        Note
        ----
        FunctionDef node structure:
            name: str
            args: arguments
            body: list[stmt]
            decorator_list: list[expr]
            returns: Optional[expr]
        """
        # Validate return statements in function body
        self._validate_return_statements(node)

        with self.context.with_frame():
            # Step 1: Extract function signature
            tensor_input_args, _, output_args = self.get_signature()

            # Step 2: Validate output arguments
            self._validate_output_args(output_args, node)

            # Step 3: Set up output variable mapping
            # Note that the naming process should be done before the dynamic dimension marking,
            output_var_mapping = self._setup_output_var_mapping(node, output_args)

            # # Step 4: Mark dynamic dimensions for input tensors
            # self._mark_dynamic_dimensions(tensor_input_args)
            # self._mark_dynamic_dimensions(output_args)

            # Step 5: Add arguments to parsing context
            self._add_tensor_args_to_context(tensor_input_args)

            # Step 6: Add metadata to context
            self._add_metadata_to_context(node.name, output_var_mapping)

            # Step 7: Create PTO function and parse body
            with pypto.function(node.name, tensor_input_args, output_args):
                for _ in pypto.loop(1):
                    self.visit_body(node.body)

        return pypto.functions.get_last_function()

    def visit_arg(self, node: doc.arg) -> Union[list, tuple]:
        """The general arg visiting method.

        Parameters
        ----------
        node : doc.arg
            The doc AST arg node.

        Returns
        -------
        res : tuple[str, Any, bool]
            A tuple of (name, value, is_tensor) where:
            - name: the argument name
            - value: the argument value (Tensor or type)
            - is_tensor: True if it's a tensor argument, False otherwise

        Note
        ----
        arg node structure:
            arg: str
            annotation: expr
        """
        if isinstance(node, (doc.Tuple, doc.List)):
            return [self.visit_arg(arg) for arg in node.elts]
        name = node.arg
        if node.annotation is None:
            raise ParserError(
                node, ValueError("Annotation is required for function arguments.")
            )
        anno = self.visit_expr(node.annotation)
        if isinstance(anno, pypto.Tensor):
            anno.name = name
            return name, anno
        elif isinstance(anno, type):
            # Non-tensor type annotation (e.g., bool, int, str, etc.)
            return name, anno
        else:
            raise ParserError(
                node,
                TypeError(
                    f"Annotation must be a tensor or type, but got {type(anno)}."
                ),
            )

    def visit_arguments(
        self, node: doc.arguments
    ) -> tuple[list[pypto.Tensor], list[tuple[str, type]]]:
        """The general arguments visiting method.

        Parameters
        ----------
        node : doc.arguments
            The doc AST arguments node.

        Returns
        -------
        res : tuple[list[pypto.Tensor], list[tuple[str, type]]]
            A tuple of (tensor_args, non_tensor_args) where:
            - tensor_args: list of Tensor arguments
            - non_tensor_args: list of (name, type) tuples for non-tensor arguments
        """
        if node.vararg is not None:
            raise ParserError(
                node,
                NotImplementedError(
                    "Variable-length arguments (*args) are not supported. "
                    "Please use a fixed number of arguments."
                ),
            )
        if len(node.kwonlyargs) > 0:
            raise ParserError(
                node,
                NotImplementedError(
                    "Keyword-only arguments are not supported. "
                    "Please use regular positional or keyword arguments."
                ),
            )
        if len(node.kw_defaults) > 0:
            raise ParserError(
                node,
                NotImplementedError(
                    "Keyword argument defaults are not supported. "
                    "All arguments must be explicitly provided."
                ),
            )
        if node.kwarg is not None:
            raise ParserError(
                node,
                NotImplementedError(
                    "Keyword argument packing (**kwargs) is not supported. "
                    "Please use explicit keyword arguments."
                ),
            )
        if len(node.defaults) > 0:
            raise ParserError(
                node,
                NotImplementedError(
                    "Default argument values are not supported. "
                    "All arguments must be explicitly provided."
                ),
            )
        if len(node.posonlyargs) > 0:
            raise ParserError(
                node,
                NotImplementedError(
                    "Position-only arguments are not supported. "
                    "Please use regular arguments."
                ),
            )

        # Process all arguments and separate tensors from non-tensors
        tensor_args = []
        non_tensor_args = []

        for arg in node.args:
            result = self.visit_arg(arg)
            if isinstance(result, tuple):
                name, value = result
                if isinstance(value, pypto.Tensor):
                    tensor_args.append(value)
                else:
                    non_tensor_args.append((name, value))
            elif isinstance(result, list):
                # Handle nested tuples/lists if needed
                for item in result:
                    if isinstance(item, tuple):
                        name, value = item
                        if isinstance(value, pypto.Tensor):
                            tensor_args.append(value)
                        else:
                            non_tensor_args.append((name, value))

        return tensor_args, non_tensor_args

    def visit_for(self, node: doc.For) -> Any:
        """The general for visiting method.

        Parameters
        ----------
        node : doc.For
            The doc AST for node.

        Returns
        -------
        res : Any
            The visiting result.

        Note
        ----
        For node structure:
            target: expr (loop variable)
            iter: expr (iterator expression, e.g., range(10))
            body: list[stmt] (loop body)
            orelse: list[stmt] (else clause, not supported)
        """
        # Check for unsupported else clause
        if node.orelse:
            raise ParserError(
                node,
                NotImplementedError(
                    "For-else clauses are not supported. "
                    "Consider using a separate if statement after the loop."
                ),
            )

        # Extract the loop variable name
        if not isinstance(node.target, doc.Name):
            raise ParserError(
                node.target,
                TypeError(
                    f"Loop variable must be a simple name, but got {type(node.target).__name__}."
                ),
            )
        loop_var_name = node.target.id

        # Try to evaluate the iterator expression (e.g., range(10))
        # This works even with symbolic values in range bounds because
        # Python's range() is lazily evaluated
        iter_expr = self.eval_expr(node.iter)

        # Support range() calls - extract start, stop, step parameters
        # These parameters can be concrete values or symbolic expressions
        iterator = None
        if isinstance(iter_expr, range):
            # Extract start, stop, step from range object
            start = iter_expr.start
            stop = iter_expr.stop
            step = iter_expr.step
            iterator = pypto.loop(
                start, stop, step, name="Dynamic", idx_name=loop_var_name
            )
        elif isinstance(iter_expr, Iterator):
            iterator = iter_expr
        else:
            raise ParserError(
                node.iter,
                TypeError(
                    f"Loop iterator must be a range object or Iterator, but got {type(iter_expr).__name__}."
                ),
            )

        # Create the loop using pypto.loop, which generates the appropriate IR for iteration.
        # The loop variable is created by the pypto.loop iterator and added to the context
        # so it can be used within the loop body.
        # Create a new frame for the loop body scope
        with self.context.with_frame():
            # The loop variable is yielded by the iterator
            for loop_var in iterator:
                # Add the loop variable to the context
                self.context.add(loop_var_name, loop_var)
                # Visit the loop body
                self.visit_body(node.body)

    def _assign_target(self, target: doc.expr, expr: Any) -> None:
        """Helper method to assign an expression to a target.

        Parameters
        ----------
        target : doc.expr
            The assignment target (Name, Tuple, List, or Subscript).
        expr : Any
            The value to assign.
        """
        if isinstance(target, doc.Name):
            # Simple assignment: a = expr
            # Handle output tensors first to avoid recreating them
            output_var_mapping = self.context.get().get("__output_var_mapping__", {})
            output_tensor = output_var_mapping.get(target.id)
            if output_tensor is not None:
                output_tensor[:] = expr
                return
            # Set the tensor name if expr is a tensor
            if isinstance(expr, pypto.Tensor):
                expr.name = target.id
            self.context.add(target.id, expr)
        elif isinstance(target, (doc.Tuple, doc.List)):
            # Unpacking assignment: a, b = expr
            if not isinstance(expr, (list, tuple)):
                raise ParserError(
                    target,
                    TypeError(f"Cannot unpack non-sequence type {type(expr).__name__}"),
                )
            if len(target.elts) != len(expr):
                raise ParserError(
                    target,
                    ValueError(
                        f"Cannot unpack {len(expr)} values into {len(target.elts)} targets"
                    ),
                )
            for t, e in zip(target.elts, expr):
                self._assign_target(t, e)
        elif isinstance(target, doc.Subscript):
            # Subscript assignment: b[:] = expr or b[0] = expr
            # This handles in-place tensor updates using Python's subscript syntax.
            # Evaluate the value (e.g., b) to get the tensor being assigned to
            tensor = self.eval_expr(target.value)
            # Evaluate the slice (e.g., : or 0) to get the slice/index object
            slice_obj = self.eval_expr(target.slice)
            # Perform the assignment using __setitem__, which translates to
            # the appropriate PTO IR operation for tensor element/slice updates
            tensor[slice_obj] = expr
        else:
            raise ParserError(
                target,
                TypeError(
                    f"Assignment target must be a name, tuple, or subscript, "
                    f"but got {type(target).__name__}."
                ),
            )

    def visit_assign(self, node: doc.Assign) -> None:
        """The general assign visiting method.

        Parameters
        ----------
        node : doc.Assign
            The doc AST assign node.

        Returns
        -------
        res : None
            The visiting result. None.

        Note
        ----
        Assign node structure:
            targets: list[expr]
            value: expr
        """
        expr = self.visit_expr(node.value)

        for target in node.targets:
            self._assign_target(target, expr)

    def visit_ann_assign(self, node: doc.AnnAssign) -> Any:
        """The general annotated assign visiting method.

        Parameters
        ----------
        node : doc.Assign
            The doc AST annotated assign node.

        Returns
        -------
        res : Any
            The visiting result.

        Note
        ----
        AnnAssign node structure:
            target: expr
            annotation: expr
            value: Optional[expr]
        """
        # Reuse the assign visiting method to visit the annotated assign node.
        return self.visit_assign(node)

    def visit_expr(self, node: doc.Expr) -> Any:
        """The general expression visiting method.

        Parameters
        ----------
        node : doc.Expr
            The doc AST expression node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return self.eval_expr(node)

    def visit_if(self, node: doc.If) -> Any:
        """The general if visiting method.

        Parameters
        ----------
        node : doc.If
            The doc AST if node.

        Returns
        -------
        res : Any
            The visiting result.

        Note
        ----
        If node structure:
            test: expr (condition expression)
            body: list[stmt] (if body)
            orelse: list[stmt] (else/elif body)
        """
        # Evaluate the test condition
        test_expr = self.eval_expr(node.test)

        if isinstance(test_expr, pypto.SymbolicScalar):
            cond = pypto.cond(test_expr)
        elif isinstance(test_expr, bool):
            cond = test_expr
        else:
            raise ParserError(
                node.test,
                TypeError(
                    f"Test condition must be a symbolic scalar or boolean, but got {type(test_expr).__name__}."
                ),
            )

        # Execute the if statement using the condition as a context manager
        if cond:
            # Visit the if body
            self.visit_body(node.body)
        else:
            # Visit the else body (if it exists)
            if node.orelse:
                self.visit_body(node.orelse)

    def visit_return(self, node: doc.Return) -> Any:
        """The general return visiting method.

        Parameters
        ----------
        node : doc.Return
            The doc AST return node.

        Returns
        -------
        res : Union[None, pypto.Tensor, list[pypto.Tensor]]
            The visiting result: None, a single tensor, or a list of tensors.
        """
        if node.value is None:
            # Case 1: return without any value
            return None

        expr = self.visit_expr(node.value)

        # Case 2: return None (explicit)
        if expr is None:
            return None

        # Get function name and output variable mapping from context
        func_name = self.context.get().get("__func_name__", "")

        # Case 3: return a single tensor
        if isinstance(expr, pypto.Tensor):
            expr.name = f"output_{func_name}"
            result = expr
        # Case 4: return a tuple or list of tensors
        elif isinstance(expr, (list, tuple)):
            result = []
            for idx, elem in enumerate(expr):
                if not isinstance(elem, pypto.Tensor):
                    raise ParserError(
                        node,
                        TypeError(
                            f"Return value at index {idx} must be a tensor, "
                            f"but got {type(elem).__name__}."
                        ),
                    )
                # Name each tensor in the list for better traceability
                if elem.name is None or elem.name == "":
                    elem.name = f"output_{func_name}_{idx}"
                result.append(elem)
        # Case 5: invalid return type
        else:
            raise ParserError(
                node,
                TypeError(
                    f"Return value must be None, a tensor, or a list/tuple of tensors, "
                    f"but got {type(expr).__name__}."
                ),
            )

        return result

    def visit_delete(self, node: doc.Delete) -> None:
        """The general delete visiting method.

        Parameters
        ----------
        node : doc.Delete
            The doc AST delete node.

        Returns
        -------
        res : None
            The visiting result. None.

        Note
        ----
        Delete node structure:
            targets: list[expr]
        """
        for target in node.targets:
            if isinstance(target, doc.Name):
                # Delete a simple variable
                try:
                    self.context.delete(target.id)
                except NameError as e:
                    raise ParserError(target, e) from e
                except ValueError as e:
                    raise ParserError(target, e) from e
            else:
                raise ParserError(
                    target,
                    TypeError(
                        f"Delete target must be a name, "
                        f"but got {type(target).__name__}."
                    ),
                )

    def _auto_cleanup_after_stmt(self, stmt: doc.stmt) -> None:
        """Automatically cleanup variables after statement if enabled.

        Parameters
        ----------
        stmt : doc.stmt
            The statement that was just visited.
        """

        stmt_id = id(stmt)
        if stmt_id in self.delete_after:
            vars_to_delete = self.delete_after[stmt_id]
            self.context.mark_for_deletion(vars_to_delete)
            self.context.cleanup_marked()

    # ============================================================
    # Reporting methods
    # ============================================================
    def report(self, node: doc.AST, msg: str, level: DiagnosticLevel) -> None:
        """Report a diagnostic."""
        self.diag.emit(node, msg, level)
