#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for uninitialized tensor storage read interception."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CHECKER_PATH = _REPO_ROOT / "python" / "pypto" / "frontend" / "parser" / "tensor_init_check.py"


def _load_tensor_init_check():
    """Load tensor_init_check without importing the full pypto package tree."""
    stub_error = types.ModuleType("pypto.error")

    class ParserError(Exception):
        def __init__(self, node: ast.AST, msg):
            self.node = node
            super().__init__(msg if isinstance(msg, str) else str(msg))

    stub_error.ParserError = ParserError
    saved_pypto = sys.modules.get("pypto")
    saved_err = sys.modules.get("pypto.error")
    sys.modules.setdefault("pypto", types.ModuleType("pypto"))
    sys.modules["pypto.error"] = stub_error

    spec = importlib.util.spec_from_file_location(
        "tensor_init_check_under_test",
        _CHECKER_PATH,
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        if saved_err is not None:
            sys.modules["pypto.error"] = saved_err
        else:
            sys.modules.pop("pypto.error", None)
        if saved_pypto is None:
            sys.modules.pop("pypto", None)
        else:
            sys.modules["pypto"] = saved_pypto
        sys.modules.pop("tensor_init_check_under_test", None)

    return mod


_MOD = _load_tensor_init_check()
_check = _MOD.check_uninitialized_tensor_storage_reads


def _parse_fn(code: str) -> ast.FunctionDef:
    tree = ast.parse(code)
    stmt = tree.body[0]
    assert isinstance(stmt, ast.FunctionDef)
    return stmt


def test_detects_read_inside_loop_before_write():
    fn = _parse_fn(
        """
def foo(x):
    t = pypto.tensor((1,), pypto.DT_FP32, "t")
    for i in range(10):
        y = t[0]
"""
    )
    with pytest.raises(_MOD.ParserError):
        _check(fn)


def test_allows_write_then_read_in_loop():
    fn = _parse_fn(
        """
def foo(x):
    t = pypto.tensor((1,), pypto.DT_FP32, "t")
    t[:] = x
    for i in range(10):
        y = t[0]
"""
    )
    _check(fn)


def test_linear_read_before_write_raises():
    fn = _parse_fn(
        """
def foo(x):
    t = pypto.tensor((1,), pypto.DT_FP32, "t")
    a = t[0]
"""
    )
    with pytest.raises(_MOD.ParserError):
        _check(fn)
