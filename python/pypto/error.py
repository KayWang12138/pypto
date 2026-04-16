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
    - ErrCode: Enumeration of error codes for different error types
    - PTOError: Base exception class for all PyPTO errors
    - Specialized error types: PTOTypeError, PTOValueError, PTORuntimeError,
      PTONameError, PTONotImplementedError for different error categories

All error classes inherit from PTOError and include an error code for
identification and tracking purposes.
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
    """Error raised for name-related errors in PyPTO operations."""

    def __init__(self, msg: Union[str, Exception]):
        super().__init__(msg, ErrCode.NAME_ERROR)


class PTONotImplementedError(PTOError):
    """Error raised for not implemented errors in PyPTO operations."""

    def __init__(self, msg: Union[str, Exception]):
        super().__init__(msg, ErrCode.NOT_IMPLEMENTED_ERROR)