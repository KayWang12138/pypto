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

Key Components:
    - _ErrCode: Enumeration of error codes for different error types
    - PyptoError: Base exception class for all PyPTO errors
    - Specialized error types: PTOTypeError, PTOValueError, PTORuntimeError,
      PTONameError, PTONotImplementedError for different error categories

All error classes inherit from PyptoError and include an error code for
identification and tracking purposes.
"""
import ast
from typing import Union, Optional


class _ErrCode:
    TYPE_ERROR: int = 0xF00001
    VALUE_ERROR: int = 0xF00002
    RUNTIME_ERROR: int = 0xF00003
    NAME_ERROR: int = 0xF00004
    NOT_IMPLEMENTED_ERROR: int = 0xF00005
    
    UNKNOWN: int = 0xF1FFFFF


def _get_err_code(msg: Union[str, Exception]) -> tuple[_ErrCode, str]:
    """Get error code and formatted message based on message type."""
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
    """

    def __init__(self, msg: Union[str, Exception]):
        if isinstance(msg, Exception):
            msg = f"{type(msg).__name__}: {msg}"
        super().__init__(msg)
        self.node: Optional[ast.AST] = None


class ParserError(PyptoError):
    """Base exception class for parser errors.

    This error class associates errors with specific AST nodes for better
    source location reporting. It is used throughout the parser to provide
    detailed error messages with precise location information.

    The error code is automatically determined based on the exception type:
    - TypeError -> _ErrCode.TYPE_ERROR
    - ValueError -> _ErrCode.VALUE_ERROR
    - RuntimeError -> _ErrCode.RUNTIME_ERROR
    - NameError -> _ErrCode.NAME_ERROR
    - NotImplementedError -> _ErrCode.NOT_IMPLEMENTED_ERROR

    Attributes:
        node: The AST node where the error occurred, used for source location
              reporting and error message formatting
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
    """