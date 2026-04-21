# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0
"""Unit tests for ``pypto.export.helpers``."""

from __future__ import annotations

import textwrap

import pytest
import torch

from pypto.export.dtype_mapping import (
    _ge_data_type_enum_value_to_element_size,
    _ge_data_type_enum_value_to_torch_dtype,
    _ge_dtype_token_from_base,
    _GE_DATA_TYPE_VALUE_TO_TORCH_BASE,
    _torch_dtype_to_ge_dtype,
    _torch_dtype_to_ir_dtype,
)
from pypto.export.helpers import (
    _camel_case_to_snake_case,
    _get_renamed_func_source,
    _snake_case_to_camel_case,
    _unwrap_decorated_func_name,
    _unwrap_decorated_func_source,
)
from pypto.export import meta_schema
from pypto.export.pypto_op import (
    pypto_op_calc_workspace,
    pypto_op_infer_dtype,
    pypto_op_infer_shape,
    pypto_op_kernel,
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


def test_torch_dtype_to_ge_dtype_mappings():
    # Floats
    assert _torch_dtype_to_ge_dtype("torch.float16") == "ge::DT_FLOAT16"
    assert _torch_dtype_to_ge_dtype("torch.float32") == "ge::DT_FLOAT"
    assert _torch_dtype_to_ge_dtype("torch.float64") == "ge::DT_DOUBLE"
    assert _torch_dtype_to_ge_dtype("torch.bfloat16") == "ge::DT_BF16"
    # Signed ints
    assert _torch_dtype_to_ge_dtype("torch.int8") == "ge::DT_INT8"
    assert _torch_dtype_to_ge_dtype("torch.int16") == "ge::DT_INT16"
    assert _torch_dtype_to_ge_dtype("torch.int32") == "ge::DT_INT32"
    assert _torch_dtype_to_ge_dtype("torch.int64") == "ge::DT_INT64"
    # Unsigned ints
    assert _torch_dtype_to_ge_dtype("torch.uint8") == "ge::DT_UINT8"
    assert _torch_dtype_to_ge_dtype("torch.uint16") == "ge::DT_UINT16"
    assert _torch_dtype_to_ge_dtype("torch.uint32") == "ge::DT_UINT32"
    assert _torch_dtype_to_ge_dtype("torch.uint64") == "ge::DT_UINT64"
    # Bool
    assert _torch_dtype_to_ge_dtype("torch.bool") == "ge::DT_BOOL"
    # Complex
    assert _torch_dtype_to_ge_dtype("torch.complex32") == "ge::DT_COMPLEX32"
    assert _torch_dtype_to_ge_dtype("torch.complex64") == "ge::DT_COMPLEX64"
    assert _torch_dtype_to_ge_dtype("torch.complex128") == "ge::DT_COMPLEX128"
    # Quantized storage dtypes
    assert _torch_dtype_to_ge_dtype("torch.qint8") == "ge::DT_QINT8"
    assert _torch_dtype_to_ge_dtype("torch.qint16") == "ge::DT_QINT16"
    assert _torch_dtype_to_ge_dtype("torch.qint32") == "ge::DT_QINT32"
    assert _torch_dtype_to_ge_dtype("torch.quint8") == "ge::DT_QUINT8"
    assert _torch_dtype_to_ge_dtype("torch.quint16") == "ge::DT_QUINT16"


@pytest.mark.parametrize("bad", ["float16", "fp16", "ge::DT_FLOAT16", "int"])
def test_torch_dtype_to_ge_dtype_rejects_non_torch_prefix(bad: str):
    with pytest.raises(ValueError, match="Unsupported dtype for GE mapping"):
        _torch_dtype_to_ge_dtype(bad)


def test_torch_dtype_to_ge_dtype_rejects_unsupported_torch_dtype():
    # e.g. float8: not mapped to ge::DT_HIFLOAT8 without a confirmed 1:1 semantics
    with pytest.raises(ValueError, match="Unsupported torch dtype for GE mapping"):
        _torch_dtype_to_ge_dtype("torch.float8_e4m3fn")


def test_ge_data_type_enum_value_to_element_size_examples():
    assert _ge_data_type_enum_value_to_element_size(0) == 4  # float32
    assert _ge_data_type_enum_value_to_element_size(1) == 2  # float16
    assert _ge_data_type_enum_value_to_element_size(27) == 2  # bfloat16


def test_ge_data_type_enum_value_to_element_size_rejects_unmapped():
    with pytest.raises(ValueError, match="element size"):
        _ge_data_type_enum_value_to_element_size(13)  # DT_STRING


def test_ge_data_type_enum_value_to_torch_dtype_int16_example():
    import torch

    assert _ge_data_type_enum_value_to_torch_dtype(6) is torch.int16


@pytest.mark.parametrize(
    ("value", "basename"),
    sorted(_GE_DATA_TYPE_VALUE_TO_TORCH_BASE.items(), key=lambda x: x[0]),
)
def test_ge_data_type_enum_value_to_torch_dtype_matches_basename(value: int, basename: str):
    import torch

    if not hasattr(torch, basename):
        pytest.skip(f"torch has no dtype {basename!r}")
    td = _ge_data_type_enum_value_to_torch_dtype(value)
    assert str(td) == f"torch.{basename}"


@pytest.mark.parametrize(
    ("value", "basename"),
    sorted(_GE_DATA_TYPE_VALUE_TO_TORCH_BASE.items(), key=lambda x: x[0]),
)
def test_ge_data_type_enum_value_forward_ge_token_matches(value: int, basename: str):
    import torch

    if not hasattr(torch, basename):
        pytest.skip(f"torch has no dtype {basename!r}")
    td = getattr(torch, basename)
    assert _torch_dtype_to_ge_dtype(str(td)) == _ge_dtype_token_from_base(basename)


def test_ge_data_type_enum_value_to_torch_dtype_rejects_unmapped():
    with pytest.raises(ValueError, match="Unsupported ge::DataType enum value"):
        _ge_data_type_enum_value_to_torch_dtype(13)  # DT_STRING in gert_ge_minimal.hpp


def test_torch_dtype_to_ir_dtype_matches_expected_members():
    pypto_ir = pytest.importorskip("pypto_ir")

    assert _torch_dtype_to_ir_dtype("torch.float16") == pypto_ir.DataType.FP16
    assert _torch_dtype_to_ir_dtype("torch.float32") == pypto_ir.DataType.FP32
    assert _torch_dtype_to_ir_dtype("torch.int32") == pypto_ir.DataType.INT32


def test_torch_dtype_to_ir_dtype_rejects_non_torch_prefix():
    pytest.importorskip("pypto_ir")
    with pytest.raises(ValueError, match="unsupported dtype"):
        _torch_dtype_to_ir_dtype("float32")


def test_pypto_op_calc_workspace_decorator_wiring():
    @pypto_op_kernel(
        kernel_name="ws_test_kernel",
        vec_tile_shapes=(1, 1),
    )
    def kernel_body(x):
        return x

    @pypto_op_infer_shape(pypto_op_kernel=kernel_body)
    def ws_infer_shape(x_shape: tuple[int, int]) -> tuple[int, int]:
        return x_shape

    @pypto_op_calc_workspace(pypto_op_kernel=kernel_body)
    def ws_calc_workspace(x_shape: tuple[int, int], x_dtype_size: int) -> int:
        return 0

    @pypto_op_infer_dtype(pypto_op_kernel=kernel_body)
    def ws_infer_dtype(x_dtype):
        return x_dtype

    assert hasattr(kernel_body, "__pypto_calc_workspace_fn__")
    assert kernel_body.__pypto_calc_workspace_fn__ is ws_calc_workspace

    meta = getattr(kernel_body, "__pypto_meta__", {})
    assert meta_schema._META_KEY__CALC_WORKSPACE_SOURCE in meta
