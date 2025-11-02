#! /usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""Entry point for PTO Script Parser.

This module provides the main entry points for parsing PTO scripts,
including the parse function and JIT decorator.
"""

import inspect
from typing import Any, Callable, List, Optional, Union

import torch
import pypto
from pypto import pypto_impl
from pypto.frontend.parser.diagnostics import Source
from pypto.frontend.parser.parser import Parser
from pypto.converter import _dtype_from, _torch_dtype_from


def _default_globals() -> dict[str, Any]:
    """Get the default global variables for parsing.

    Returns
    -------
    dict[str, Any]
        Dictionary containing default global variables (pto module).
    """
    return {
        "pypto": pypto,
    }


def parse(program: Source, extra_vars: Optional[dict[str, Any]] = None) -> Any:
    """Parse a PTO script program.

    This function parses a PTO script source and returns the parsed result,
    typically a pypto.Function object.

    Parameters
    ----------
    program : Source
        The source code to parse.
    extra_vars : Optional[dict[str, Any]], optional
        Additional variables to make available during parsing.
        These are merged with default globals (pto module).

    Returns
    -------
    Any
        The parsed result, typically a pypto.Function object.

    Examples
    --------
    >>> source = Source("def foo(): ...")
    >>> func = parse(source)
    >>> isinstance(func, pypto.Function)
    True
    """
    if extra_vars is None:
        merged_vars = _default_globals()
    else:
        merged_vars = {**_default_globals(), **extra_vars}
    parser = Parser(program, merged_vars)
    parser.parse()
    return parser.execute()


def _to_tensor_data(tensors: List[torch.Tensor]):
    """Convert torch tensors to PTO tensor data."""
    datas = []
    for t in tensors:
        data = pypto_impl.DeviceTensorData(
            _dtype_from(t.dtype),
            t.data_ptr(),
            list(t.shape),
        )
        datas.append(data)
    return datas


class JitCallableWrapper:
    """Callable wrapper for pypto.Function that integrates frontend parsing with runtime execution.

    This class wraps a pypto.Function and makes it callable with torch tensors,
    integrating the frontend JIT compilation with the runtime execution mechanism.
    Parsing is deferred until the first __call__ invocation (lazy mode).
    """

    def __init__(
        self,
        pto_function: pypto.Function,
        original_func: Callable,
        handler: int,
        codegen_options: Optional[dict[str, Any]] = None,
        host_options: Optional[dict[str, Any]] = None,
    ):
        """Initialize the wrapper.

        Parameters
        ----------
        pto_function : pypto.Function
            The parsed PTO function (None initially in lazy mode).
        original_func : Callable
            The original Python function.
        handler : int
            The runtime handler (None initially in lazy mode).
        """
        self._pto_function = pto_function
        self._original_func = original_func
        self._handler = handler
        self._is_compiled = pto_function is not None
        self._compiled_non_tensor_args: Optional[dict[str, Any]] = None
        self._parser = None  # Store parser for lazy parsing
        self._codegen_options = (
            None if codegen_options is None else dict(codegen_options)
        )
        self._host_options = None if host_options is None else dict(host_options)

        # Copy metadata from the original function
        if hasattr(original_func, "__name__"):
            self.__name__ = original_func.__name__
        if hasattr(original_func, "__doc__"):
            self.__doc__ = original_func.__doc__

    def _create_parser(self) -> Parser:
        """Create and prepare a parser for the wrapped function."""
        source = Source(self._original_func)
        closure_vars = inspect.getclosurevars(self._original_func)
        captured_vars = {**closure_vars.nonlocals, **closure_vars.globals}
        parser = Parser(source, captured_vars)
        return parser

    def _extract_non_tensor_args(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract concrete non-tensor arguments from the call."""
        signature = inspect.signature(self._original_func)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return {
            name: value
            for name, value in bound.arguments.items()
            if not isinstance(value, torch.Tensor)
        }

    def _non_tensor_args_match(self, non_tensor_args: dict[str, Any]) -> bool:
        """Check whether we already compiled for the given non-tensor args."""
        if self._compiled_non_tensor_args is None:
            return False
        try:
            return self._compiled_non_tensor_args == non_tensor_args
        except Exception:
            # If args are not directly comparable, force recompilation.
            return False

    def _compile_if_needed(
        self,
        concrete_input_shapes: list[list[int]],
        non_tensor_args: dict[str, Any],
    ):
        """Compile the function on first call if not already compiled."""
        if self._is_compiled and self._non_tensor_args_match(non_tensor_args):
            return

        # Re-create parser so compilation matches current non-tensor args
        self._parser = self._create_parser()
        self._parser.parse()

        # Bind concrete non-tensor arguments so parsing can use runtime values
        self._parser.bind_non_tensor_args(non_tensor_args)

        # Get function signature (inputs and outputs) for OperatorBegin
        input_tensors, _, output_tensors = self._parser.get_signature()

        # Initialize backend for compilation
        pypto_impl.DeviceInit()
        handler = pypto_impl.OperatorBegin(
            [t.base() for t in input_tensors], [t.base() for t in output_tensors]
        )

        # Set options AFTER OperatorBegin() to match @pypto.jit behavior
        if self._codegen_options:
            pypto.set_codegen_options(**self._codegen_options)
        if self._host_options:
            pypto.set_host_options(**self._host_options)

        # Bind dynamic dimensions from concrete inputs
        if concrete_input_shapes:
            self._parser.bind_dynamic_dims_from_inputs(concrete_input_shapes)

        # Execute the deferred parsing (happens on first __call__)
        self._pto_function = self._parser.execute()
        pypto_impl.OperatorEnd(handler)
        self._handler = handler
        self._compiled_non_tensor_args = dict(non_tensor_args)
        self._is_compiled = True

    def __call__(self, *args, **kwargs):
        """Execute the function with torch tensors.

        Parameters
        ----------
        *args : torch.Tensor
            Input tensors followed by any additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Union[torch.Tensor, tuple[torch.Tensor, ...]]
            Output tensor(s).
        """

        in_tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        non_tensor_args = self._extract_non_tensor_args(args, kwargs)

        # Validate input tensors are contiguous
        for in_tensor in in_tensors:
            if not in_tensor.is_contiguous():
                raise RuntimeError(
                    "pypto.frontend.jit requires that all input tensors "
                    "must be contiguous."
                )

        # Use output tensors from parser signature to allocate output tensors
        out_tensors = []

        # Create output tensors with the same device as input tensors
        if in_tensors:
            device = in_tensors[0].device
            for tensor in in_tensors[1:]:
                if tensor.device != device:
                    raise RuntimeError(
                        f"pypto.frontend.jit requires that all input tensors "
                        f"must be on the same device. Got tensors on devices: "
                        f"{device} and {tensor.device}"
                    )
        else:
            raise RuntimeError("pypto.frontend.jit requires at least one input tensor")

        # Resolve symbolic dimensions using current input shapes so outputs
        # allocated below match the runtime dynamic sizes.
        concrete_input_shapes = [list(in_tensor.shape) for in_tensor in in_tensors]
        self._compile_if_needed(concrete_input_shapes, non_tensor_args)
        symbolic_dim_value_map = {}
        tmp_parser = self._create_parser()
        tmp_parser.bind_non_tensor_args(non_tensor_args)
        input_tensor_defs, _, output_tensor_defs = tmp_parser.get_signature()
        symbolic_dim_value_map = tmp_parser.match_input_shapes(
            concrete_input_shapes, input_tensor_defs
        )

        for out_tensor_def in output_tensor_defs:
            shape_list = []
            # Build shape by resolving symbolic dimensions from the output tensor definition
            for dim in out_tensor_def.shape:
                if isinstance(dim, pypto.SymbolicScalar):
                    dim_value = symbolic_dim_value_map.get(str(dim))
                    if dim_value is None:
                        raise ValueError(
                            f"Dynamic dimension {dim} not found in symbolic_dim_value_map"
                        )
                    shape_list.append(dim_value)
                else:
                    # Static dimension
                    shape_list.append(dim)

            shape = tuple(shape_list)
            dtype = _torch_dtype_from(out_tensor_def.dtype)
            out_tensor = torch.empty(shape, dtype=dtype, device=device)
            out_tensors.append(out_tensor)

        # Execute the function
        workspace_size = pypto_impl.GetWorkSpaceSize(self._handler)
        workspace_tensor = torch.zeros(workspace_size, device=device)

        pypto_impl.OperatorDeviceRunOnceDataFromDevice(
            self._handler,
            _to_tensor_data(in_tensors),
            _to_tensor_data(out_tensors),
            torch.npu.current_stream().npu_stream,
            workspace_tensor.data_ptr(),
        )
        # Return single tensor or tuple based on number of outputs
        if len(out_tensors) == 1:
            return out_tensors[0]
        return tuple(out_tensors)

    @property
    def function(self) -> pypto.Function:
        """Get the underlying pypto.Function."""
        return self._pto_function

    @property
    def handler(self):
        """Get the runtime handler."""
        return self._handler


def jit(
    func: Optional[Callable] = None,
    *,
    host_options: Optional[dict[str, Any]] = None,
    codegen_options: Optional[dict[str, Any]] = None,
) -> Union[Callable, Callable[[Callable], JitCallableWrapper]]:
    """JIT decorator for compiling Python functions to PTO IR.

    This decorator compiles a Python function into PTO's intermediate representation
    at decoration time. The decorated function will be replaced with the compiled
    PTO function.

    Parameters
    ----------
    func : Optional[Callable], optional
        The function to decorate. If None, returns a decorator function.
        This allows both @jit and @jit() syntax.

    host_options : Optional[dict[str, Any]], optional
        Options passed to ``pypto.set_host_options``. Defaults to
        ``{"only_codegen": True}`` if not provided.
    codegen_options : Optional[dict[str, Any]], optional
        Options passed to ``pypto.set_codegen_options``. Defaults to
        ``{"support_dynamic_unaligned": True}`` if not provided.

    Returns
    -------
    Union[Callable, Callable[[Callable], Callable]]
        Either the decorated function (if func is provided) or a decorator function.

    Raises
    ------
    TypeError
        If the decorator is applied to a non-function object.

    Examples
    --------
    >>> @jit()
    ... def my_kernel(x: pypto.Tensor([16], "float32")) -> pypto.Tensor([16], "float32"):
    ...     return x + 1
    >>> isinstance(my_kernel, pypto.Function)
    True

    >>> @jit
    ... def my_kernel2(x: pypto.Tensor([16], "float32")) -> pypto.Tensor([16], "float32"):
    ...     return x * 2
    >>> isinstance(my_kernel2, pypto.Function)
    True

    Notes
    -----
    The decorator extracts closure variables (nonlocals and globals) from the
    original function and makes them available during parsing. The resulting
    PTO function preserves the original function's name and docstring.
    """

    def decorator_wrapper(f: Callable) -> JitCallableWrapper:
        if not inspect.isfunction(f):
            raise TypeError("jit decorator can only be used on functions")

        # Create wrapper without compiling - defer to first call
        # This matches the behavior of @pypto.jit and avoids backend initialization
        # during module load time
        wrapper = JitCallableWrapper(
            None,
            f,
            None,
            codegen_options=codegen_options,
            host_options=host_options,
        )
        return wrapper

    if func is None:
        # Called with parentheses: @jit()
        return decorator_wrapper

    # Called without parentheses: @jit
    return decorator_wrapper(func)
