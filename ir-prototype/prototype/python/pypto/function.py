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

"""Function management for PTO IR.

This module provides context managers and utilities for creating and managing
PTO Function objects during parsing.
"""

from contextlib import contextmanager
from typing import List, Optional

from . import pypto_impl
from .tensor import Tensor


class Function:
    """Python wrapper for C++ Func class."""

    def __init__(self, base: Optional[pypto_impl.Function] = None):
        """Initialize a Function wrapper.

        Args:
            base: The underlying C++ Function object. If None, this creates
                  an empty wrapper (not typically used directly).
        """
        self._base = base

    @property
    def base(self) -> pypto_impl.Function:
        """Get the underlying C++ Function object.

        Returns:
            The C++ Function object.
        """
        if self._base is None:
            raise RuntimeError("Function base is None")
        return self._base

    @classmethod
    def from_base(cls, base: pypto_impl.Function) -> "Function":
        """Create a Function wrapper from a C++ Function object.

        Args:
            base: The C++ Function object to wrap.

        Returns:
            A new Function wrapper instance.
        """
        obj = cls.__new__(cls)
        obj._base = base
        return obj
        
    def __repr__(self) -> str:
        """String representation of the function."""
        return self._cpp_func.Print()


class FunctionManager:
    """Manager for tracking created functions during parsing."""

    def __init__(self):
        """Initialize the function manager."""
        self._functions: List[Function] = []

    def add_function(self, func: Function) -> None:
        """Add a function to the manager.

        Parameters
        ----------
        func : Function
            The function to add.
        """
        self._functions.append(func)

    def get_last_function(self) -> Optional[Function]:
        """Get the last added function.

        Returns
        -------
        Optional[Function]
            The last function, or None if no functions exist.
        """
        if not self._functions:
            return None
        return self._functions[-1]

    def push_function(self, func: Function) -> None:
        """Push a function onto the stack.

        Parameters
        ----------
        func : Function
            The function to push.
        """
        self._functions.append(func)

    def pop_function(self) -> None:
        """Pop a function from the stack."""
        if not self._functions:
            raise RuntimeError("Function stack is empty")
        self._functions.pop()

    def clear(self) -> None:
        """Clear all functions."""
        self._functions.clear()


# Global function manager instance
_functions = FunctionManager()


@contextmanager
def function(
    name: str,
    input_tensors: List[Tensor],
    output_tensors: List[Tensor],
    kind: pypto_impl.FunctionKind = pypto_impl.FunctionKind.ControlFlow,
):
    """Context manager for creating a PTO function.

    Parameters
    ----------
    name : str
        The name of the function.
    input_tensors : List[Tensor]
        List of input tensor definitions.
    output_tensors : List[Tensor]
        List of output tensor definitions.
    kind : pypto_impl.FunctionKind, optional
        The function kind (default: ControlFlow).

    Yields
    ------
    Function
        The created function object.
    """
    # Import IRBuilder here to avoid circular dependency
    from .builder import get_default_builder
    from . import pypto_impl as impl
    
    # Get the default IRBuilder
    builder = get_default_builder()
    
    # Ensure builder has a module
    if builder._module is None:
        # Create a default module if none exists
        module = impl.ProgramModule("default")
        builder.set_module(module)
    
    # Convert Python Tensor objects to C++ ValuePtr
    input_values = []
    for tensor in input_tensors:
        if tensor._base is None:
            raise ValueError(f"Tensor {tensor} does not have an underlying IR tensor")
        # pybind11 automatically upcasts std::shared_ptr<Tensor> to std::shared_ptr<Value>
        input_values.append(tensor._base)
    
    output_values = []
    for tensor in output_tensors:
        if tensor._base is None:
            raise ValueError(f"Tensor {tensor} does not have an underlying IR tensor")
        # pybind11 automatically upcasts std::shared_ptr<Tensor> to std::shared_ptr<Value>
        output_values.append(tensor._base)

    # Create FunctionSignature
    signature = impl.FunctionSignature()
    signature.arguments = input_values
    signature.results = output_values

    # Create function using IRBuilder
    cpp_func = builder._impl.CreateFunction(name, kind, signature, False)
    
    # Enter function body
    builder._impl.EnterFunctionBody(cpp_func)

    # Wrap in Python Function
    func = Function(cpp_func)

    # Add to manager
    _functions.add_function(func)

    try:
        yield func
    finally:
        # Cleanup if needed
        pass


def get_last_function() -> Optional[Function]:
    """Get the last created function.

    Returns
    -------
    Optional[Function]
        The last function, or None if no functions exist.
    """
    return _functions.get_last_function()


# Expose functions module-like interface
class FunctionsModule:
    """Module-like interface for function management."""

    @property
    def get_last_function(self):
        """Get the last created function."""
        return get_last_function

    def push_function(self, func: Function) -> None:
        """Push a function onto the stack."""
        _functions.push_function(func)

    def pop_function(self) -> None:
        """Pop a function from the stack."""
        _functions.pop_function()


functions = FunctionsModule()

