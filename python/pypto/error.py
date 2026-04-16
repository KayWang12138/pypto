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


def _get_err_code(msg: Union[str, Exception]) -> tuple[_ErrCode, str]:
    """Extract error code and format message from exception or string.
    
    Maps Python built-in exception types to corresponding PyPTO error codes.
    For Exception inputs, returns the appropriate error code and a formatted
    message containing the exception type and message. For string inputs,
    returns UNKNOWN error code with the original string.
    
    Args:
        msg: Exception object or error message string.
        
    Returns:
        Tuple of (error_code, formatted_message).
    """
    if isinstance(msg, Exception):
        if isinstance(msg, TypeError):
            err_code = _ErrCode.TYPE_ERROR
        elif isinstance(msg, ValueError):
            err_code = _ErrCode.VALUE_ERROR
        elif isinstance(msg, NotImplementedError):
            # Must be made before the RuntimeError, because this error class is its subclass.
            err_code = _ErrCode.NOT_IMPLEMENTED_ERROR
        elif isinstance(msg, RuntimeError):
            err_code = _ErrCode.RUNTIME_ERROR
        elif isinstance(msg, NameError):
            err_code = _ErrCode.NAME_ERROR
        else:
            err_code = _ErrCode.UNKNOWN
        msg = f"{type(msg).__name__}: {msg}"
    else:
        err_code = _ErrCode.UNKNOWN
    return err_code, msg


class PyptoError(Exception):
    """Base exception class for all PyPTO errors.

    This is the common base class for all PyPTO-specific exceptions.
    It provides a consistent interface for error handling throughout
    the PyPTO framework.
    
    Args:
        msg: Exception object or error message string. Exception objects
             will be formatted as "ExceptionType: message".
             
    Attributes:
        node: Optional AST node reference for source location tracking.
              Initialized to None and can be set by subclasses.
    """

    def __init__(self, msg: Union[str, Exception]):
        if isinstance(msg, Exception):
            msg = f"{type(msg).__name__}: {msg}"
        super().__init__(msg)
        self.node: Optional[ast.AST] = None


class ParserError(PyptoError):
    """Base exception class for parser errors with AST node association.

    This error class associates errors with specific AST nodes for better
    source location reporting. It is used throughout the parser to provide
    detailed error messages with precise location information.

    The error code is automatically determined based on the exception type:
        - TypeError -> _ErrCode.TYPE_ERROR
        - ValueError -> _ErrCode.VALUE_ERROR
        - RuntimeError -> _ErrCode.RUNTIME_ERROR
        - NameError -> _ErrCode.NAME_ERROR
        - NotImplementedError -> _ErrCode.NOT_IMPLEMENTED_ERROR

    Args:
        node: AST node where the error occurred, used for source location
              reporting and error message formatting.
        msg: Exception object or error message string. Exception objects will
             be formatted with error code as "ErrCode: XXX, ExceptionType: message".

    Attributes:
        node: The AST node where the error occurred.
    """

    def __init__(self, node: ast.AST, msg: Union[str, Exception]):
        if isinstance(msg, Exception):
            err_code, formatted_msg = _get_err_code(msg)
            super().__init__(f"ErrCode: {err_code:X}, {formatted_msg}")
        else:
            super().__init__(f"{msg}")
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