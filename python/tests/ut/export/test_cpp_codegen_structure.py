# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0
"""Python-only structural checks for pypto.export.cpp codegen (infer_shape + domi plugin + custom executor)."""

from __future__ import annotations

import importlib.util
import typing
from pathlib import Path

import pytest

from pypto.export import cpp as cpp_mod
from pypto.export.pypto_op import _FRAMEWORK_TYPE__ONNX

_SAMPLES_PATH = Path(__file__).resolve().parent / "infer_shape_samples.py"


def _load_samples():
    spec = importlib.util.spec_from_file_location("infer_shape_samples_ut", _SAMPLES_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


samples = _load_samples()


@pytest.mark.parametrize(
    ("fn", "expected_in_dims", "expected_out_dims"),
    [
        (samples.infer_shape_two_by_two, [2, 2], 2),
        (samples.infer_shape_4d_broadcast, [4, 4], 4),
        (samples.infer_shape_sum_last, [3, 3], 3),
    ],
)
def test_parse_infer_shape_metadata(fn, expected_in_dims, expected_out_dims):
    meta = cpp_mod._parse_infer_shape_for_codegen(fn)
    assert meta["input_dims"] == expected_in_dims
    assert meta["output_dims"] == expected_out_dims
    assert meta["cpp_bind_name"] == "inferShape"


# TODO confirm the fields to verify
@pytest.mark.parametrize("fn", [samples.infer_shape_two_by_two, samples.infer_shape_4d_broadcast])
def test_infer_shape_ge_impl_body_contains_expected_ops(fn):
    meta = cpp_mod._parse_infer_shape_for_codegen(fn)
    body = cpp_mod._infer_shape_ge_impl_body(meta)
    n_in = len(meta["input_dims"])
    for i in range(n_in):
        assert f"context->GetInputShape({i})" in body
        assert f"in{i}_shape->GetDimNum()" in body
        assert "std::make_tuple" in body
    assert "context->GetOutputShape(0)" in body
    assert "gert::Shape{" in body
    assert f"{meta['cpp_bind_name']}(" in body
    assert "GRAPH_SUCCESS" in body
    assert "GRAPH_FAILED" in body


# TODO confirm the fields to verify
def test_embedded_pybind_contains_exec_and_cast_tuple():
    fn = samples.infer_shape_two_by_two
    block = cpp_mod._generate_infer_shape_pybind_embedded(fn, embed_in_host=True)
    assert "namespace {" in block
    assert "py::exec" in block
    assert "inferShape" in block
    assert ".cast<std::tuple<int64_t, int64_t>>" in block
    assert "py::gil_scoped_acquire" in block


def test_to_cpp_type_int_and_tuple():
    assert cpp_mod._to_cpp_type(int) == "int64_t"
    assert cpp_mod._to_cpp_type(typing.Tuple[int, int]) == "std::tuple<int64_t, int64_t>"


def test_infer_shape_host_tu_for_test_is_single_ge_namespace():
    text = cpp_mod._generate_infer_shape_host_tu_for_test(samples.infer_shape_two_by_two)
    assert 'namespace ge {' in text
    assert "InferShapeGeImpl" in text
    assert "register/op_def_registry" not in text
    assert "AddCustom" not in text
    assert "OP_ADD" not in text


# --- domi plugin (REGISTER_CUSTOM_OP) ---


def test_generate_op_custom_plugin_cpp_has_expected_preamble_and_includes():
    text = cpp_mod._generate_op_custom_plugin_cpp(
        "Add", framework_type=_FRAMEWORK_TYPE__ONNX
    )
    assert text.startswith("// Auto-generated\n")
    assert '#include "register/register.h"' in text


def test_generate_op_custom_plugin_cpp_domi_namespace_and_parse_param_shape():
    text = cpp_mod._generate_op_custom_plugin_cpp(
        "Add", framework_type=_FRAMEWORK_TYPE__ONNX
    )
    assert "namespace domi {" in text
    assert "Status ParseParamAdd(const Message* op_src, ge::Operator& op_dest)" in text
    assert "return SUCCESS;" in text
    assert 'REGISTER_CUSTOM_OP("Add")' in text
    assert f".FrameworkType({_FRAMEWORK_TYPE__ONNX})" in text
    assert '.OriginOpType("Add")' in text
    assert ".ParseParamsByOperator(ParseParamAdd)" in text


@pytest.mark.parametrize(
    "op_type",
    ["MyOp", "AddPyptoCustomOp"],
)
def test_generate_op_custom_plugin_cpp_op_type_parameterizes_names(op_type: str):
    text = cpp_mod._generate_op_custom_plugin_cpp(
        op_type, framework_type=_FRAMEWORK_TYPE__ONNX
    )
    assert f"ParseParam{op_type}" in text
    assert f'REGISTER_CUSTOM_OP("{op_type}")' in text
    assert f'.OriginOpType("{op_type}")' in text
    assert f".ParseParamsByOperator(ParseParam{op_type})" in text


# TODO confirm the fields to verify
def test_generate_custom_executor_cpp_has_system_includes_and_class():
    full = cpp_mod._generate_custom_executor_cpp("Add")
    assert '#include "graph/custom_op.h"' in full
    assert '#include "exe_graph/runtime/sinkable_op_execution_context.h"' in full
    assert '#include "acl/acl_rt.h"' in full
    assert "class Add : public SinkableExecuteOp" in full
    assert "REG_AUTO_MAPPING_OP(Add)" in full


# TODO confirm the fields to verify
def test_custom_executor_prepare_execute_structure_invariants():
    full = cpp_mod._generate_custom_executor_cpp("Add")
    assert "[output0][input0]...[inputN-1][workspace]" in full
    assert "argsSize = sizeof(int64_t) * (1 + inputNum + 1)" in full
    assert "for (size_t i = 0; i < inputNum; ++i)" in full
    assert "GetInputTensor(i)" in full
    assert "p[0] = reinterpret_cast<int64_t>(ctx->GetOutputTensor(0)->GetAddr())" in full
    assert "p[1 + inputNum] = reinterpret_cast<int64_t>(workspaceAddr)" in full
    assert "std::vector<size_t> inputOffsets(inputNum)" in full
    assert "SpecifyToOffset(SinkableOpIo::kInput, inputOffsets.data(), inputNum)" in full
    assert "SpecifyToOffset(SinkableOpIo::kOutput, 0, outputNum)" in full
    assert "GetInputTensor(1)" not in full


def test_custom_executor_class_fragment_is_embedded_in_full_output():
    fragment = cpp_mod._custom_executor_class_cpp_for_test("Add")
    full = cpp_mod._generate_custom_executor_cpp("Add")
    assert fragment in full
    assert '#include "graph/custom_op.h"' not in fragment


def test_op_type_parameterizes_plugin_host_and_executor_cpp():
    op_type = "MyOp"
    plugin = cpp_mod._generate_op_custom_plugin_cpp(
        op_type, framework_type=_FRAMEWORK_TYPE__ONNX
    )
    assert "ParseParamMyOp" in plugin
    assert 'REGISTER_CUSTOM_OP("MyOp")' in plugin
    assert ".FrameworkType(ONNX)" in plugin
    assert '.OriginOpType("MyOp")' in plugin

    host = cpp_mod._generate_op_custom_def_cpp(op_type=op_type)
    assert "class MyOp : public OpDef" in host
    assert "explicit MyOp(const char *name)" in host
    assert "OP_ADD(MyOp)" in host

    exe = cpp_mod._generate_custom_executor_cpp(op_type)
    assert "class MyOp : public SinkableExecuteOp" in exe
    assert "REG_AUTO_MAPPING_OP(MyOp)" in exe


def test_generate_op_custom_plugin_cpp_framework_type_argument():
    text = cpp_mod._generate_op_custom_plugin_cpp("MyOp", framework_type="CAFFE")
    assert ".FrameworkType(CAFFE)" in text
    assert ".FrameworkType(ONNX)" not in text


def test_invalid_op_type_raises():
    with pytest.raises(ValueError, match="op_type"):
        cpp_mod._validate_op_type_identifier("")
    with pytest.raises(ValueError, match="op_type"):
        cpp_mod._generate_custom_executor_cpp("1Bad")
    with pytest.raises(ValueError, match="op_type"):
        cpp_mod._generate_op_custom_plugin_cpp(
            "1Bad", framework_type=_FRAMEWORK_TYPE__ONNX
        )