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
from typing import Any, Callable, Optional, Union

import torch
import pypto
from pypto import pypto_impl
from pypto.frontend.parser.diagnostics import Source
from pypto.frontend.parser.parser import Parser, NestedFunctionMarker
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


def _to_tensor_data(tensors: list[torch.Tensor]) -> list[pypto_impl.DeviceTensorData]:
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

    @staticmethod
    def _get_func_nonlocals(func: Callable) -> dict[str, Any]:
        """A modified version of `inspect.getclosurevars`"""

        if inspect.ismethod(func):
            func = func.__func__

        if not inspect.isfunction(func):
            raise TypeError(f"{func!r} is not a Python function")

        code = func.__code__
        # Nonlocal references are named in co_freevars and resolved
        # by looking them up in __closure__ by positional index
        nonlocal_vars = {}
        if func.__closure__ is not None:
            for var, cell in zip(code.co_freevars, func.__closure__):
                try:
                    nonlocal_vars[var] = cell.cell_contents
                except ValueError as err:
                    # cell_contents may raise ValueError if the cell is empty.
                    if "empty" not in str(err):
                        raise
        return nonlocal_vars

    def _create_parser(self) -> Parser:
        """Create and prepare a parser for the wrapped function."""
        source = Source(self._original_func)
        captured_vars = {
            **self._original_func.__globals__,
            **self._get_func_nonlocals(self._original_func),
        }
        parser = Parser(source, captured_vars)
        return parser

    def _compile_if_needed(
        self,
        concrete_input_shapes: list[list[int]],
    ):
        """Compile the function on first call if not already compiled."""
        if self._is_compiled:
            return

        # Re-create parser for compilation
        self._parser = self._create_parser()
        self._parser.parse()

        # Get function signature (inputs and outputs) for OperatorBegin
        # input_tensors, output_tensors = self._parser.get_signature()

        # Initialize backend for compilation
        pypto_impl.DeviceInit()
        handler = pypto_impl.OperatorBegin()

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
        self._is_compiled = True

    def __call__(self, *args, **kwargs):
        """Execute the function with torch tensors.

        Parameters
        ----------
        *args : torch.Tensor
            Input tensors (all arguments must be torch.Tensor).
        **kwargs : Any
            Not supported - all arguments must be positional tensors.

        Returns
        -------
        Union[torch.Tensor, tuple[torch.Tensor, ...]]
            Output tensor(s).
        """

        # Validate that all arguments are tensors
        if kwargs:
            raise RuntimeError(
                "pypto.frontend.jit requires that all arguments must be tensors. "
                "Keyword arguments are not supported."
            )

        for i, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                raise RuntimeError(
                    f"pypto.frontend.jit requires that all arguments must be pypto.tensor. "
                    f"Argument at position {i} is {type(arg).__name__}, not a tensor."
                )

        in_tensors = list(args)

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
        self._compile_if_needed(concrete_input_shapes)
        symbolic_dim_value_map = {}
        tmp_parser = self._create_parser()
        input_tensor_defs, output_tensor_defs = tmp_parser.get_signature()
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
        in_tensor_data = _to_tensor_data(in_tensors)
        out_tensor_data = _to_tensor_data(out_tensors)
        workspace_size = pypto_impl.GetWorkSpaceSize(
            self._handler, in_tensor_data, out_tensor_data
        )
        print([x.GetShape() for x in in_tensor_data])
        print([x.GetShape() for x in out_tensor_data])
        workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
        runtime_error_msg = pypto_impl.OperatorDeviceRunOnceDataFromDevice(
            self._handler,
            in_tensor_data + out_tensor_data,
            list(),
            torch.npu.current_stream().npu_stream,
            workspace_tensor.data_ptr(),
        )
        if runtime_error_msg != "":
            raise RuntimeError(runtime_error_msg)
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


def function(
    func: Optional[Callable] = None,
) -> Union[Callable, NestedFunctionMarker]:
    """Decorator to mark a function as eligible for nested inline execution."""

    if func is None:

        def decorator(f: Callable) -> NestedFunctionMarker:
            marker = NestedFunctionMarker()
            marker._original_func = f
            marker._func_name = f.__name__
            return marker

        return decorator

    marker = NestedFunctionMarker()
    marker._original_func = func
    marker._func_name = func.__name__
    return marker


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
