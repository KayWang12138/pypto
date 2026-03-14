#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""PTO Python Frontend - JIT Compilation and Parsing Interface.

This module provides the public API for the PTO frontend, which enables JIT
compilation of Python functions into optimized PTO kernels. The frontend parses
Python functions decorated with @jit or @function and converts them to PTO
intermediate representation (IR) for execution on NPU hardware or in simulation.

Public API
----------
jit : decorator
    JIT compilation decorator for converting Python functions to PTO kernels.
    Use @pypto.frontend.jit or @pypto.frontend.jit() to decorate functions.

function : decorator
    Nested function decorator for inline expansion. Functions decorated with
    @pypto.frontend.function are inlined when called from JIT-compiled kernels.

dynamic : function
    Create symbolic dimensions for dynamic tensor shapes. Returns a SymbolicScalar
    that represents a runtime-determined dimension value.

parser : module
    The parser submodule containing the core parsing implementation. Generally
    not accessed directly by users.

Usage Examples
--------------
Basic JIT compilation:
    >>> @pypto.frontend.jit()
    ... def my_kernel(x: pypto.Tensor((16,), pypto.DT_FP32)):
    ...     return pypto.add(x, x)

Nested functions:
    >>> @pypto.frontend.function
    ... def helper(x):
    ...     return pypto.add(x, x)
    >>>
    >>> @pypto.frontend.jit
    ... def kernel(x):
    ...     return helper(x)  # Inlined during compilation

See Also
--------
developer_doc.md : Comprehensive developer documentation
parser.entry : Entry points and wrapper classes
parser.parser : Main parser implementation
"""
from . import parser
from .parser import jit, function