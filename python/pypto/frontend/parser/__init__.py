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


"""PTO Script Parser.

This module provides the PTO Script Parser, which parses Python functions decorated
with @pypto.jit and converts them to PTO intermediate representation (IR). The parser
is inspired by TVM's Script Parser and provides a high-level interface for compiling
Python code to PTO operations.

Key Components:
    - jit: Decorator for JIT compilation of Python functions
    - Parser: Main parser class for converting Python AST to PTO IR
    - AST utilities: doc_core and doc modules for AST manipulation
    - Error handling: error module for parser error reporting

Example:
    @pypto.jit
    def my_kernel(x: pypto.Tensor([16], "float32")) -> pypto.Tensor([16], "float32"):
        return x + 1
"""

from . import doc_core, doc, error
from .entry import jit