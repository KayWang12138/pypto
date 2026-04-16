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

"""Error classes and exception handling for PyPTO.

This module defines custom exception classes and error handling utilities
for the PyPTO framework. It provides a consistent error interface with
error codes for different error categories.

Classes:
    _ErrCode: Enumeration of error codes for different error types.
    PyptoError: Base exception class for all PyPTO errors.
    ParserError: Parser exception with AST node association and error codes.
    RenderedParserError: Parser error indicating pre-rendered error message.
"""
import ast
from typing import Union, Optional


class _ErrCode:
    """Error code enumeration for PyPTO exceptions.
    
    This class defines unique hexadecimal error codes for different
    exception categories, enabling consistent error identification
    and tracking across the PyPTO framework.
    """
    
    TYPE_ERROR: int = 0xF00001
    VALUE_ERROR: int = 0xF00002
    RUNTIME_ERROR: int = 0xF00003
    NAME_ERROR: int = 0xF00004
    NOT_IMPLEMENTED_ERROR: int = 0xF00005
    UNKNOWN: int = 0xF1FFFFF


def _get_err_code(msg: Exception) -> int:
    """Get error code based on exception type.
    
    Args:
        msg: Exception object.
        
    Returns:
        Error code corresponding to the exception type.
    """
    if isinstance(msg, TypeError):
        return _ErrCode.TYPE_ERROR
    elif isinstance(msg, ValueError):
        return _ErrCode.VALUE_ERROR
    elif isinstance(msg, NotImplementedError):
        return _ErrCode.NOT_IMPLEMENTED_ERROR
    elif isinstance(msg, RuntimeError):
        return _ErrCode.RUNTIME_ERROR
    elif isinstance(msg, NameError):
        return _ErrCode.NAME_ERROR
    else:
        return _ErrCode.UNKNOWN


class PyptoError(Exception):
    """Base exception class for all PyPTO errors.

    This is the common base class for all PyPTO-specific exceptions.
    It provides a consistent interface for error handling throughout
    the PyPTO framework.
    
    Args:
        msg: Exception object or error message string. Formatted as
             "ErrCode: XXX, ExceptionType: message" or "ErrCode: XXX, message".
        err_code: Error code (int). Defaults to _ErrCode.UNKNOWN.
                 
    Attributes:
        node: Optional AST node reference for source location tracking.
              Initialized to None and can be set by subclasses.
    """

    def __init__(self, msg: Union[str, Exception], err_code: Optional[int] = None):
        if err_code is None:
            err_code = _ErrCode.UNKNOWN
        if isinstance(msg, Exception):
            msg = f"ErrCode: {err_code:X}, {type(msg).__name__}: {msg}"
        else:
            msg = f"ErrCode: {err_code:X}, {msg}"
        super().__init__(msg)
        self.node: Optional[ast.AST] = None


class ParserError(PyptoError):
    """Base exception class for parser errors with AST node association.

    This error class associates errors with specific AST nodes for better
    source location reporting. It is used throughout the parser to provide
    detailed error messages with precise location information.

    For Exception inputs, error code is automatically determined based on
    the exception type. For string inputs, defaults to _ErrCode.UNKNOWN.

    Args:
        node: AST node where the error occurred, used for source location
              reporting and error message formatting.
        msg: Exception object or error message string.

    Attributes:
        node: The AST node where the error occurred.
    """

    def __init__(self, node: ast.AST, msg: Union[str, Exception]):
        err_code = _get_err_code(msg) if isinstance(msg, Exception) else None
        super().__init__(msg, err_code=err_code)
        self.node = node


class RenderedParserError(ParserError):
    """Special error class indicating that the error message has been rendered.

    This error class is used to signal that the error message has already
    been formatted and displayed to the user, preventing duplicate error
    output. It inherits from ParserError and maintains the same node
    association for source location tracking.
    
    This is typically used internally by the parser when error messages
    have already been processed and should not be re-formatted or re-displayed.
    
    Args:
        node: AST node where the error occurred.
        msg: Exception object or error message string (already rendered).
    """