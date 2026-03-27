# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0
"""Unit tests for ``pypto.export.helpers``."""

from __future__ import annotations

import textwrap

import pytest

from pypto.export.helpers import (
    _camel_case_to_snake_case,
    _get_renamed_func_source,
    _snake_case_to_camel_case,
    _torch_dtype_to_ge_dtype,
    _torch_dtype_to_ir_dtype,
    _unwrap_decorated_func_name,
    _unwrap_decorated_func_source,
)


def test_unwrap_decorated_func_source_strips_leading_decorator_block():
    raw = textwrap.dedent(
        '''\
        @some_decorator
        @another
        def foo(a, b):
            return a + b
        '''
    )
    out = _unwrap_decorated_func_source(raw)
    assert out.startswith("def foo(")
    assert "@some_decorator" not in out
    assert "return a + b" in out


def test_unwrap_decorated_func_source_plain_def_unchanged_prefix():
    raw = textwrap.dedent(
        '''\
        def bar():
            return 42
        '''
    )
    assert _unwrap_decorated_func_source(raw) == raw


def test_unwrap_decorated_func_name_simple():
    assert _unwrap_decorated_func_name("infer_shape") == "infer_shape"


def test_unwrap_decorated_func_name_first_token_only():
    assert _unwrap_decorated_func_name("infer_shape wrapped") == "infer_shape"


@pytest.mark.parametrize(
    ("snake", "expected_camel"),
    [
        ("infer_shape", "inferShape"),
        ("calc_workspace", "calcWorkspace"),
        ("a", "a"),
        ("word", "word"),
        ("foo_bar_baz", "fooBarBaz"),
    ],
)
def test_snake_case_to_camel_case(snake: str, expected_camel: str):
    assert _snake_case_to_camel_case(snake) == expected_camel


@pytest.mark.parametrize(
    ("pascal_or_camel", "expected_snake"),
    [
        ("Add", "add"),
        ("AddPyptoCustomOp", "add_pypto_custom_op"),
        ("inferShape", "infer_shape"),
        ("XMLParser", "xml_parser"),
        ("a", "a"),
    ],
)
def test_camel_case_to_snake_case(pascal_or_camel: str, expected_snake: str):
    assert _camel_case_to_snake_case(pascal_or_camel) == expected_snake


def _sample_kernel_for_rename(x, y):
    return x + y


def test_get_renamed_func_source_replaces_first_function_name_only():
    out = _get_renamed_func_source(_sample_kernel_for_rename, "renamed_kernel")
    assert out.startswith("def renamed_kernel(")
    assert "def _sample_kernel_for_rename(" not in out
    assert "return x + y" in out


def test_get_renamed_func_source_does_not_rename_later_occurrences_in_body():
    def func_with_self_reference():
        """func_with_self_reference docstring."""
        return func_with_self_reference

    out = _get_renamed_func_source(func_with_self_reference, "new_name")
    assert out.startswith("def new_name(")
    # Docstring still mentions old name; only first def name token is replaced once.
    assert "func_with_self_reference docstring" in out


def test_torch_dtype_to_ge_dtype_float_variants_map_to_float16():
    assert _torch_dtype_to_ge_dtype("torch.float16") == "ge::DT_FLOAT16"
    assert _torch_dtype_to_ge_dtype("torch.float32") == "ge::DT_FLOAT16"
    assert _torch_dtype_to_ge_dtype("torch.bfloat16") == "ge::DT_FLOAT16"


@pytest.mark.parametrize("bad", ["float16", "fp16", "ge::DT_FLOAT16", "int"])
def test_torch_dtype_to_ge_dtype_rejects_non_torch_prefix(bad: str):
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _torch_dtype_to_ge_dtype(bad)


def test_torch_dtype_to_ge_dtype_rejects_non_float_torch_dtype():
    with pytest.raises(ValueError, match="Only torch float dtypes"):
        _torch_dtype_to_ge_dtype("torch.int32")


def test_torch_dtype_to_ir_dtype_matches_expected_members():
    import pypto_ir

    assert _torch_dtype_to_ir_dtype("torch.float16") == pypto_ir.DataType.FP16
    assert _torch_dtype_to_ir_dtype("torch.float32") == pypto_ir.DataType.FP32
    assert _torch_dtype_to_ir_dtype("torch.int32") == pypto_ir.DataType.INT32


def test_torch_dtype_to_ir_dtype_rejects_non_torch_prefix():
    with pytest.raises(ValueError, match="unsupported dtype"):
        _torch_dtype_to_ir_dtype("float32")
