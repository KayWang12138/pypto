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

"""Error classes and exception handling for PTO Script Parser.

This module defines custom exception classes and error handling utilities
for the PTO Script Parser. It provides enhanced error reporting with optional
backtraces controlled via environment variables.

Key Components:
    - PTOError: Base exception class for all PyPTO errors
    - ParserError: Base exception class for parser errors, associates errors
      with specific AST nodes for better source location reporting
    - RenderedParserError: A special error class indicating that the error
      message has already been formatted and displayed
    - Specialized error types: PTOTypeError, PTOValueError, PTORuntimeError
      for different error categories
    - Exception hook wrapper: Customizes Python's exception handling to
      provide cleaner error output and cleanup multiprocessing resources

Environment Variables:
    - PTO_BACKTRACE: Set to 1 to enable full Python backtraces for parser errors.
      By default (0), only the user-friendly error message is shown without
      internal stack traces, making errors more readable for end users.

The exception hook automatically cleans up multiprocessing child processes
when the parser is interrupted, preventing orphaned processes.
"""
import ast
import enum
from typing import Union, Optional

class ErrCode(enum.Enum):
    TYPE_ERROR = 0xF12001
    VALUE_ERROR = 0xF12002
    RUNTIME_ERROR = 0xF12003
    NAME_ERROR = 0xF12004
    NOT_IMPLEMENTED_ERROR = 0xF12005

class PTOError(Exception):
    """Base exception class for all PyPTO errors.

    This is the common base class for all PyPTO-specific exceptions.
    It provides a consistent interface for error handling throughout
    the PyPTO framework.
    """

    def __init__(self, msg: Union[str, Exception], err_code: ErrCode):
        if isinstance(msg, Exception):
            msg = f"{type(msg).__name__}: {msg}"
        super().__init__(f"ErrCode: {err_code.value:X}, {msg}")
        self.node: Optional[ast.AST] = None


class PTOTypeError(PTOError):
    """Error raised for type-related errors in PyPTO operations."""

    def __init__(self, msg: Union[str, Exception]):
        super().__init__(msg, ErrCode.TYPE_ERROR)


class PTOValueError(PTOError):
    """Error raised for value-related errors in PyPTO operations."""

    def __init__(self, msg: Union[str, Exception]):
        super().__init__(msg, ErrCode.VALUE_ERROR)


class PTORuntimeError(PTOError):
    """Error raised for runtime errors in PyPTO operations."""

    def __init__(self, msg: Union[str, Exception]):
        super().__init__(msg, ErrCode.RUNTIME_ERROR)


class PTONameError(PTOError):
    """Error raised for runtime errors in PyPTO operations."""

    def __init__(self, msg: Union[str, Exception]):
        super().__init__(msg, ErrCode.NAME_ERROR)


class PTONotImplementedError(PTOError):
    """Error raised for runtime errors in PyPTO operations."""

    def __init__(self, msg: Union[str, Exception]):
        super().__init__(msg, ErrCode.NOT_IMPLEMENTED_ERROR)