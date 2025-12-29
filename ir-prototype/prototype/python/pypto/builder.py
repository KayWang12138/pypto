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

"""IRBuilder Python wrapper for PTO-IR construction.

This module provides a high-level Python API for constructing PTO-IR using the C++ IRBuilder.
The IRBuilder manages insertion points, scopes, and provides context managers for structured
control flow (functions, loops, conditionals).
"""

from contextlib import contextmanager
from typing import List, Optional, Iterator, Union
from . import pypto_impl
from .tensor import Tensor, Scalar, SymInt
from . import utils

__all__ = ["IRBuilder", "get_default_builder"]


class IRBuilder:
    """Python wrapper for C++ IRBuilder.
    
    The IRBuilder is the primary interface for constructing PTO-IR. It maintains
    the current insertion point (function, scope, block) and provides high-level
    methods for creating IR nodes with proper scope management.
    
    Example usage:
        >>> from pypto.builder import IRBuilder
        >>> from pypto import Tensor, DT_FP32
        >>> 
        >>> builder = IRBuilder()
        >>> module = pypto_impl.ProgramModule("my_module")
        >>> builder.set_module(module)
        >>> 
        >>> with builder.function("foo", [Tensor((4,), DT_FP32)], 
        ...                      [Tensor((4,), DT_FP32)]) as (b, inputs, outputs):
        ...     x = inputs[0]
        ...     with b.for_loop("i", 0, 4, 1) as i:
        ...         # Loop body
        ...         pass
    """
    
    def __init__(self, module: Optional[pypto_impl.ProgramModule] = None):
        """Initialize IRBuilder.
        
        Parameters
        ----------
        module : ProgramModule, optional
            The program module to build IR into. If not provided, must be set
            later via set_module().
        """
        if module is not None:
            self._impl = pypto_impl.IRBuilder(module)
        else:
            self._impl = pypto_impl.IRBuilder()
        self._module = module
        self._current_function = None
        self._scope_guards = []  # Stack of active scope guards
    
    def set_module(self, module: pypto_impl.ProgramModule) -> None:
        """Set the program module for IR construction.
        
        Parameters
        ----------
        module : ProgramModule
            The program module to build IR into.
        """
        self._module = module
        self._impl.SetModule(module)
    
    @contextmanager
    def function(
        self,
        name: str,
        inputs: List[Tensor],
        outputs: List[Tensor],
        kind: pypto_impl.FunctionKind = pypto_impl.FunctionKind.ControlFlow,
        set_as_entry: bool = False
    ):
        """Context manager for creating a PTO function.
        
        This creates a new function and sets it as the current insertion point.
        All IR construction within the context will be added to this function.
        
        Parameters
        ----------
        name : str
            The name of the function.
        inputs : List[Tensor]
            List of input tensor definitions.
        outputs : List[Tensor]
            List of output tensor definitions.
        kind : FunctionKind, optional
            The function kind (default: ControlFlow).
        set_as_entry : bool, optional
            Whether to set this as the program entry point (default: False).
        
        Yields
        ------
        tuple
            (builder, input_values, output_values) where:
            - builder: This IRBuilder instance
            - input_values: List of input Value objects from the function signature
            - output_values: List of output Value objects from the function signature
        
        Example
        -------
        >>> with builder.function("foo", [x], [y]) as (b, inputs, outputs):
        ...     # Function body
        ...     result = b.emit_op(...)
        """
        if self._module is None:
            raise RuntimeError("IRBuilder.function: module is not set. Call set_module() first.")
        
        # Convert Python Tensor objects to C++ ValuePtr
        # pybind11 automatically upcasts std::shared_ptr<Tensor> to std::shared_ptr<Value>
        input_values = []
        for tensor in inputs:
            if tensor._base is None:
                raise ValueError(f"Input tensor {tensor} does not have an underlying IR tensor")
            input_values.append(tensor._base)
        
        output_values = []
        for tensor in outputs:
            if tensor._base is None:
                raise ValueError(f"Output tensor {tensor} does not have an underlying IR tensor")
            output_values.append(tensor._base)
        
        # Create FunctionSignature
        signature = pypto_impl.FunctionSignature()
        signature.arguments = input_values
        signature.results = output_values
        
        # Create function
        func_cpp = self._impl.CreateFunction(name, kind, signature, set_as_entry)
        self._current_function = func_cpp
        
        # Enter function body
        guard = self._impl.EnterFunctionBody(func_cpp)
        
        try:
            yield (self, input_values, output_values)
        finally:
            # Restore previous state (C++ IRBuilder maintains this internally)
            self._current_function = None
            del guard
    
    @contextmanager
    def for_loop(
        self,
        iteration_var: str,
        lower_bound: Union[int, Scalar],
        upper_bound: Union[int, Scalar],
        step: Union[int, Scalar] = 1
    ) -> Iterator[Scalar]:
        """Context manager for creating a for loop.
        
        Creates a ForStatement and enters its body scope. The loop index variable
        is yielded as a Scalar for use in the loop body.
        
        Parameters
        ----------
        iteration_var : str
            Name of the loop iteration variable.
        lower_bound : int or Scalar
            Lower bound of the loop range.
        upper_bound : int or Scalar
            Upper bound of the loop range.
        step : int or Scalar, optional
            Step size (default: 1).
        
        Yields
        ------
        Scalar
            A symbolic scalar representing the loop iteration variable.
        
        Example
        -------
        >>> with builder.for_loop("i", 0, 10, 1) as i:
        ...     # Loop body can use i as a Scalar
        ...     offset = i * 2
        """
        # Convert loop bounds to Scalar objects
        def _to_scalar(value: Union[int, Scalar]) -> Scalar:
            if isinstance(value, Scalar):
                return value
            elif isinstance(value, int):
                # Create a constant Scalar from int
                return Scalar(value)
            else:
                raise TypeError(f"Cannot convert {type(value)} to Scalar")
        
        start_scalar = _to_scalar(lower_bound)
        end_scalar = _to_scalar(upper_bound)
        step_scalar = _to_scalar(step)
        
        # Create symbolic scalar for loop index variable
        iteration_var_scalar = Scalar("int32", iteration_var, pypto_impl.ScalarValueKind.Symbolic)
        
        # Create ForStatement with Scalar objects
        for_stmt = self._impl.CreateForStmt(iteration_var_scalar._base, start_scalar._base, end_scalar._base, step_scalar._base)
        
        # Enter for body
        scope_guard = self._impl.EnterForBody(for_stmt)
        self._scope_guards.append(scope_guard)
        
        # Use the same scalar as loop index
        loop_index = iteration_var_scalar
        
        try:
            yield loop_index
        finally:
            # Exit scope
            if self._scope_guards:
                self._scope_guards.pop()
            # Process loop-carried variables and build yield/result
            self._impl.ExitForStatement(for_stmt)
    
    @contextmanager
    def if_then(self, condition: Union[str, Scalar, SymInt]):
        """Context manager for creating an if-then branch.
        
        Creates an IfStatement and enters its then-branch scope.
        
        Parameters
        ----------
        condition : str, Scalar, or SymInt
            The condition expression.
        
        Yields
        ------
        IRBuilder
            This builder instance for chaining.
        
        Example
        -------
        >>> with builder.if_then(cond) as b:
        ...     # Then branch
        ...     result = b.emit_op(...)
        """
        # Convert condition to string
        if isinstance(condition, str):
            cond_str = condition
        elif isinstance(condition, Scalar):
            if hasattr(condition._base, 'GetSSAName'):
                name = condition._base.GetSSAName()
                cond_str = name if name else str(condition)
            else:
                cond_str = str(condition)
        else:
            cond_str = str(condition)
        
        # Create IfStatement
        if_stmt = self._impl.CreateIfStmt(cond_str)
        
        # Store for potential else branch
        if not hasattr(self, '_current_if_stmt'):
            self._current_if_stmt = []
        self._current_if_stmt.append(if_stmt)
        
        # Enter then branch
        scope_guard = self._impl.EnterIfThen(if_stmt)
        self._scope_guards.append(scope_guard)
        
        try:
            yield self
        finally:
            # Exit scope
            if self._scope_guards:
                self._scope_guards.pop()
    
    @contextmanager
    def if_else(self):
        """Context manager for creating an if-else branch.
        
        Must be called after if_then() to create the else branch of the most
        recent if statement.
        
        Yields
        ------
        IRBuilder
            This builder instance for chaining.
        
        Example
        -------
        >>> with builder.if_then(cond) as b:
        ...     # Then branch
        ...     pass
        >>> with builder.if_else() as b:
        ...     # Else branch
        ...     pass
        """
        if not hasattr(self, '_current_if_stmt') or not self._current_if_stmt:
            raise RuntimeError("if_else() must be called after if_then()")
        
        if_stmt = self._current_if_stmt[-1]
        
        # Enter else branch
        scope_guard = self._impl.EnterIfElse(if_stmt)
        self._scope_guards.append(scope_guard)
        
        try:
            yield self
        finally:
            # Exit scope and remove if statement from stack
            if self._scope_guards:
                self._scope_guards.pop()
            if self._current_if_stmt:
                self._current_if_stmt.pop()
    
    def tensor(
        self,
        shape: List[Union[int, Scalar]],
        dtype: pypto_impl.DataType,
        name: str = ""
    ) -> Tensor:
        """Create a new tensor in the current scope.
        
        Parameters
        ----------
        shape : List[int or Scalar]
            The shape of the tensor.
        dtype : DataType
            The data type of the tensor.
        name : str, optional
            The name of the tensor.
        
        Returns
        -------
        Tensor
            A new Tensor wrapper around the created IR tensor.
        """
        # Convert shape to list of Scalars
        scalar_shape = utils.to_syms(shape)
        
        # Create tensor using C++ IRBuilder
        # This returns a shared_ptr<Tensor> which is now directly bound to Python
        ir_tensor = self._impl.CreateTensor(scalar_shape, dtype, name)
        
        # Wrap the C++ Tensor in our Python Tensor wrapper
        return Tensor.from_base(ir_tensor)
    
    def emit_op(
        self,
        opcode: pypto_impl.Opcode,
        inputs: List[Union[Tensor, Scalar]],
        payload: Optional[object] = None,
        name: str = ""
    ) -> List[Union[Tensor, Scalar]]:
        """Emit an operation and return its outputs.
        
        This is the unified op construction interface. It creates an operation
        using the C++ IRBuilder's CreateOp method and returns the output values.
        
        Parameters
        ----------
        opcode : Opcode
            The operation opcode.
        inputs : List[Tensor or Scalar]
            The input values.
        payload : OpPayload, optional
            Optional operation-specific payload.
        name : str, optional
            The name for the operation.
        
        Returns
        -------
        List[Tensor or Scalar]
            The output values from the operation.
        """
        # Convert Python Tensor/Scalar to C++ ValuePtr
        # pybind11 automatically handles upcasting to ValuePtr
        input_values = []
        for inp in inputs:
            if isinstance(inp, Tensor):
                input_values.append(inp._base)
            elif isinstance(inp, Scalar):
                input_values.append(inp._base)
            else:
                raise TypeError(f"Input must be Tensor or Scalar, got {type(inp)}")
        
        # Create operation
        output_values = self._impl.CreateOp(opcode, input_values, payload, name)
        
        # Wrap outputs back to Python types
        outputs = []
        for out_val in output_values:
            # Try to determine if it's a Tensor or Scalar
            # C++ Tensor and Scalar are now directly bound to Python
            if hasattr(out_val, 'GetShape'):  # Likely a Tensor
                outputs.append(Tensor.from_base(out_val))
            else:  # Likely a Scalar
                outputs.append(Scalar.from_base(out_val))
        
        return outputs
    
    def get_active_block(self) -> Optional[pypto_impl.BlockStatement]:
        """Get the current active block statement.
        
        Returns
        -------
        BlockStatement or None
            The current active block, or None if no block is active.
        """
        try:
            return self._impl.GetOrCreateActiveBlock()
        except Exception:
            return None
    
    def get_current_function(self) -> Optional[pypto_impl.Function]:
        """Get the current function being built.
        
        Returns
        -------
        Function or None
            The current function, or None if no function is active.
        """
        try:
            return self._impl.GetCurrentFunction()
        except Exception:
            return None
        
    def get_current_scope(self) -> Optional[pypto_impl.Scope]:
        """Get the current scope

        Returns
        -------
        Scope or None
            The current scope, or None if not in any scope. 
        """
        try:
            return self._impl.GetCurrentScope()
        except Exception:
            return None

# Global default builder instance
_default_builder: Optional[IRBuilder] = None


def get_default_builder() -> IRBuilder:
    """Get or create the default global IRBuilder instance.
    
    Returns
    -------
    IRBuilder
        The default builder instance.
    """
    global _default_builder
    if _default_builder is None:
        _default_builder = IRBuilder()
    return _default_builder


def reset_default_builder():
    """Reset the default builder to a new instance."""
    global _default_builder
    _default_builder = None

