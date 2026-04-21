# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0
"""Python-only structural checks for pypto.export.cpp.codegen (infer_shape + domi plugin + custom executor)."""

from __future__ import annotations

import importlib.util
import typing
from pathlib import Path

import pytest
import torch

from pypto.export.cpp import codegen as cpp_mod
from pypto.export.pypto_op import _FRAMEWORK_TYPE__ONNX

_EXPORT_TEST_DIR = Path(__file__).resolve().parent


def _load_samples_module(unique_name: str, filename: str):
    """Load a sibling ``*.py`` sample module by path (works without a parent package)."""
    path = _EXPORT_TEST_DIR / filename
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


infer_shape_samples = _load_samples_module("infer_shape_samples_ut_structure", "infer_shape_samples.py")
calc_workspace_samples = _load_samples_module("calc_workspace_samples_ut_structure", "calc_workspace_samples.py")


@pytest.mark.parametrize(
    ("fn", "expected_in_dims", "expected_out_dims"),
    [
        (infer_shape_samples.infer_shape_two_by_two, [2, 2], [2]),
        (infer_shape_samples.infer_shape_4d_broadcast, [4, 4], [4]),
        (infer_shape_samples.infer_shape_sum_last, [3, 3], [3]),
        (infer_shape_samples.infer_shape_nd_identity, [None], [None]),
    ],
)
def test_parse_infer_shape_metadata(fn, expected_in_dims, expected_out_dims):
    meta = cpp_mod._parse_infer_shape_for_codegen(fn)
    assert meta.input_dims == expected_in_dims
    assert meta.output_dims == expected_out_dims
    assert meta.cpp_bind_name == "inferShape"


def test_parse_infer_shape_variadic_modes_and_elem_cpp():
    meta = cpp_mod._parse_infer_shape_for_codegen(infer_shape_samples.infer_shape_nd_identity)
    assert meta.input_modes == ["variadic"]
    assert meta.input_elem_cpp == ["int64_t"]
    assert meta.output_modes == ["variadic"]
    assert meta.output_elem_cpp == ["int64_t"]


def test_parse_infer_shape_two_outputs_metadata():
    meta = cpp_mod._parse_infer_shape_for_codegen(infer_shape_samples.infer_shape_two_outputs)
    assert meta.input_dims == [2, 2]
    assert meta.output_dims == [2, 2]
    assert len(meta.output_modes) == 2
    assert meta.output_modes == ["fixed", "fixed"]


# TODO confirm the fields to verify
@pytest.mark.parametrize("fn", [infer_shape_samples.infer_shape_two_by_two, infer_shape_samples.infer_shape_4d_broadcast])
def test_infer_shape_ge_impl_body_contains_expected_ops(fn):
    meta = cpp_mod._parse_infer_shape_for_codegen(fn)
    body = cpp_mod._infer_shape_ge_impl_body(meta)
    n_in = len(meta.input_modes)
    for i in range(n_in):
        assert f"context->GetInputShape({i})" in body
        assert f"in{i}_shape->GetDimNum()" in body
        assert "std::make_tuple" in body
    for k in range(len(meta.output_modes)):
        assert f"context->GetOutputShape({k})" in body
    assert "gert::Shape{" in body
    assert f"{meta.cpp_bind_name}(" in body
    assert "GRAPH_SUCCESS" in body
    assert "GRAPH_FAILED" in body


def test_infer_shape_ge_impl_body_variadic_uses_vector_and_setdims():
    meta = cpp_mod._parse_infer_shape_for_codegen(infer_shape_samples.infer_shape_nd_identity)
    body = cpp_mod._infer_shape_ge_impl_body(meta)
    assert "std::vector<int64_t> in0_shape_vec" in body
    assert "in0_shape_vec.push_back" in body
    assert "SetDimNum(out_shape_vec.size())" in body
    assert "(*out_shape)[j]" in body
    assert "std::make_tuple" not in body


# TODO confirm the fields to verify
def test_embedded_pybind_contains_exec_and_cast_tuple():
    fn = infer_shape_samples.infer_shape_two_by_two
    block = cpp_mod._generate_infer_shape_pybind_embedded(fn, embed_in_host=True)
    assert "namespace {" in block
    assert "py::exec" in block
    assert "inferShape" in block
    assert ".cast<std::tuple<int64_t, int64_t>>" in block
    assert "py::gil_scoped_acquire" in block


def test_embedded_pybind_variadic_casts_vector():
    fn = infer_shape_samples.infer_shape_nd_identity
    block = cpp_mod._generate_infer_shape_pybind_embedded(fn, embed_in_host=True)
    assert ".cast<std::vector<int64_t>>" in block


def test_embedded_pybind_two_outputs_nested_tuple_cast():
    fn = infer_shape_samples.infer_shape_two_outputs
    block = cpp_mod._generate_infer_shape_pybind_embedded(fn, embed_in_host=True)
    assert ".cast<std::tuple<std::tuple<int64_t, int64_t>, std::tuple<int64_t, int64_t>>>" in block


def test_to_cpp_type_int_and_tuple():
    assert cpp_mod._to_cpp_type(int) == "int64_t"
    assert cpp_mod._to_cpp_type(typing.Tuple[int, int]) == "std::tuple<int64_t, int64_t>"
    assert cpp_mod._to_cpp_type(typing.Tuple[int, ...]) == "std::vector<int64_t>"
    assert cpp_mod._to_cpp_type(typing.Tuple[float, ...]) == "std::vector<float>"


def test_infer_shape_host_tu_for_test_is_single_ge_namespace():
    text = cpp_mod._generate_infer_shape_host_tu_for_test(infer_shape_samples.infer_shape_two_by_two)
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
    assert "Status ParseParamAdd(const ge::Operator& op_src, ge::Operator& op_dest)" in text
    assert "return SUCCESS;" in text
    assert 'REGISTER_CUSTOM_OP("Add")' in text
    assert f".FrameworkType({_FRAMEWORK_TYPE__ONNX})" in text
    assert '.OriginOpType("Add")' in text
    assert ".ParseParamsByOperatorFn(ParseParamAdd)" in text


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
    assert f".ParseParamsByOperatorFn(ParseParam{op_type})" in text


def _dummy_calc_workspace(shape0: typing.Tuple[int, int], dtype_size0: int) -> int:
    del dtype_size0
    return 0


def test_generate_custom_executor_cpp_has_system_includes_and_class():
    # TODO confirm the fields to verify
    full = cpp_mod._generate_custom_executor_cpp("Add", calc_workspace_func=_dummy_calc_workspace)
    assert "GeDataTypeElementSize" in full
    assert "pypto.export.dtype_mapping" not in full
    assert '#include "graph/custom_op.h"' in full
    assert '#include "exe_graph/runtime/sinkable_op_execution_context.h"' in full
    assert '#include "acl/acl_rt.h"' in full
    assert "class Add : public SinkableExecuteOp" in full
    assert "REG_AUTO_MAPPING_OP(Add)" in full


def test_generated_ge_dtype_element_size_switch_matches_dtype_mapping():
    """Emitted ``GeDataTypeElementSize`` cases stay aligned with ``dtype_mapping`` (spot-check)."""
    full = cpp_mod._generate_custom_executor_cpp(
        "Add", calc_workspace_func=_dummy_calc_workspace, for_compile_test=True
    )
    assert "case ge::DT_FLOAT: return 4;" in full
    assert "case ge::DT_FLOAT16: return 2;" in full
    assert "default: return -1;" in full


def test_generate_custom_executor_cpp_for_compile_test_omits_ge_runtime_includes():
    full = cpp_mod._generate_custom_executor_cpp(
        "Add", calc_workspace_func=_dummy_calc_workspace, for_compile_test=True
    )
    assert '#include "graph/custom_op.h"' not in full
    assert '#include "exe_graph/runtime/sinkable_op_execution_context.h"' not in full
    assert '#include "acl/acl_rt.h"' not in full
    assert "#include <pybind11/pybind11.h>" in full
    assert "class Add : public SinkableExecuteOp" in full


def test_custom_executor_prepare_execute_structure_invariants():
    # TODO confirm the fields to verify
    full = cpp_mod._generate_custom_executor_cpp("Add", calc_workspace_func=_dummy_calc_workspace)
    assert "[out0]...[out_{K-1}][in0]...[in_{M-1}][workspace]" in full
    assert "argsSize = sizeof(int64_t) * (outputNum + inputNum + 1)" in full
    assert "for (size_t i = 0; i < inputNum; ++i)" in full
    assert "GetInputTensor(i)" in full
    assert "for (size_t k = 0; k < outputNum; ++k)" in full
    assert "GetOutputTensor(k)->GetAddr()" in full
    assert "p[outputNum + inputNum] = reinterpret_cast<int64_t>(workspaceAddr)" in full
    assert "std::vector<size_t> inputOffsets(inputNum)" in full
    assert "SpecifyIoOffset(SinkableOpIo::kInput, inputOffsets.data(), inputNum)" in full
    assert "SpecifyIoOffset(SinkableOpIo::kOutput, nullptr, outputNum)" in full
    assert "GetInputTensor(1)" not in full


def test_custom_executor_class_fragment_is_embedded_in_full_output():
    meta = cpp_mod._parse_calc_workspace_for_codegen(_dummy_calc_workspace)
    workspace_block = cpp_mod._generate_workspace_block_for_calc_workspace(meta)
    fragment = cpp_mod._custom_executor_class_cpp("Add", workspace_block)
    full = cpp_mod._generate_custom_executor_cpp("Add", calc_workspace_func=_dummy_calc_workspace)
    assert fragment in full
    assert '#include "graph/custom_op.h"' not in fragment


def test_workspace_block_wired_into_executor_cpp():
    meta = cpp_mod._parse_calc_workspace_for_codegen(calc_workspace_samples.calc_workspace_fixed)
    workspace_block = cpp_mod._generate_workspace_block_for_calc_workspace(meta)
    full = cpp_mod._generate_custom_executor_cpp("Add", calc_workspace_func=calc_workspace_samples.calc_workspace_fixed)
    # Check that at least the key call site and guard logic from the workspace block
    # appear in the generated executor translation unit.
    assert "auto ws = calcWorkspace(" in full
    assert "if (ws < 0)" in full
    assert workspace_block in full


def test_op_type_parameterizes_plugin_host_and_executor_cpp():
    op_type = "MyOp"
    plugin = cpp_mod._generate_op_custom_plugin_cpp(
        op_type, framework_type=_FRAMEWORK_TYPE__ONNX
    )
    assert "ParseParamMyOp" in plugin
    assert 'REGISTER_CUSTOM_OP("MyOp")' in plugin
    assert ".FrameworkType(ONNX)" in plugin
    assert '.OriginOpType("MyOp")' in plugin

    host = cpp_mod._generate_op_custom_def_cpp(
        infer_shape_samples.infer_shape_two_by_two,
        infer_shape_samples.infer_dtype_two,
        op_type=op_type,
        dtypes=["torch.float16", "torch.float16"],
    )
    assert "class MyOp : public OpDef" in host
    assert "explicit MyOp(const char *name)" in host
    assert "OP_ADD(MyOp)" in host

    exe = cpp_mod._generate_custom_executor_cpp(op_type, calc_workspace_func=_dummy_calc_workspace)
    assert "class MyOp : public SinkableExecuteOp" in exe
    assert "REG_AUTO_MAPPING_OP(MyOp)" in exe


def test_generate_op_custom_plugin_cpp_framework_type_argument():
    text = cpp_mod._generate_op_custom_plugin_cpp("MyOp", framework_type="CAFFE")
    assert ".FrameworkType(CAFFE)" in text
    assert ".FrameworkType(ONNX)" not in text


@pytest.mark.parametrize(
    "fn, expected_modes, expected_dims, expected_elem_cpp",
    [
        (
            calc_workspace_samples.calc_workspace_fixed,
            ["fixed", "fixed"],
            [2, 2],
            [None, None],
        ),
        (
            calc_workspace_samples.calc_workspace_variadic,
            ["variadic"],
            [None],
            ["int64_t"],
        ),
    ],
)
def test_parse_calc_workspace_metadata(
    fn, expected_modes, expected_dims, expected_elem_cpp
):
    meta = cpp_mod._parse_calc_workspace_for_codegen(fn)
    assert meta.input_modes == expected_modes
    assert meta.input_dims == expected_dims
    assert meta.input_elem_cpp == expected_elem_cpp
    assert meta.cpp_bind_name == "calcWorkspace"


def test_workspace_block_fixed_contains_tuple_and_dim_checks():
    meta = cpp_mod._parse_calc_workspace_for_codegen(calc_workspace_samples.calc_workspace_fixed)
    body = cpp_mod._generate_workspace_block_for_calc_workspace(meta)
    assert "GetInputTensor(0)" in body
    assert "GetInputTensor(1)" in body
    assert "GetDataType()" in body
    assert "GetShape()" in body
    assert "GetDimNum()" in body
    assert "std::make_tuple" in body
    assert "in0_shape_tuple" in body and "in1_shape_tuple" in body
    assert "auto ws = calcWorkspace(" in body
    assert "GeDataTypeElementSize" in body
    assert "GeDataTypeElementSize(in0_tensor->GetDataType())" in body
    assert "in0_dtype_size" in body and "in1_dtype_size" in body
    assert "if (ws < 0)" in body
    assert "return GRAPH_FAILED;" in body
    assert "size_t workspaceSize = static_cast<size_t>(ws);" in body


def test_workspace_block_variadic_uses_vector_and_casts():
    meta = cpp_mod._parse_calc_workspace_for_codegen(calc_workspace_samples.calc_workspace_variadic)
    body = cpp_mod._generate_workspace_block_for_calc_workspace(meta)
    assert "std::vector<int64_t> in0_shape_vec" in body
    assert "in0_shape_vec.reserve" in body
    assert "in0_shape_vec.push_back" in body
    assert "GetDataType()" in body
    assert "auto ws = calcWorkspace(in0_shape_vec, static_cast<int64_t>(in0_dtype_size));" in body


def test_invalid_op_type_raises():
    with pytest.raises(ValueError, match="op_type"):
        cpp_mod._validate_op_type_identifier("")
    with pytest.raises(ValueError, match="op_type"):
        cpp_mod._generate_custom_executor_cpp("1Bad", calc_workspace_func=_dummy_calc_workspace)
    with pytest.raises(ValueError, match="op_type"):
        cpp_mod._generate_op_custom_plugin_cpp(
            "1Bad", framework_type=_FRAMEWORK_TYPE__ONNX
        )


def test_generate_op_custom_def_cpp_builds_inputs_from_dtypes():
    text = cpp_mod._generate_op_custom_def_cpp(
        infer_shape_samples.infer_shape_three_4d,
        infer_shape_samples.infer_dtype_three,
        op_type="MyOp",
        dtypes=["torch.float16", "torch.float32", "torch.bfloat16"],
    )
    assert 'this->Input("in0")' in text
    assert 'this->Input("in1")' in text
    assert 'this->Input("in2")' in text
    assert 'this->Output("out0")' in text
    # Input dtypes come from dtype_mapping._torch_dtype_to_ge_dtype; output dtype matches infer_dtype (input0).
    assert '.DataType({ge::DT_FLOAT16})' in text  # input 0 and output
    assert '.DataType({ge::DT_FLOAT})' in text    # input 1
    assert '.DataType({ge::DT_BF16})' in text  # input 2


def test_generate_op_custom_def_cpp_rejects_empty_dtypes():
    with pytest.raises(ValueError, match="dtypes"):
        cpp_mod._generate_op_custom_def_cpp(
            infer_shape_samples.infer_shape_one_2d,
            infer_shape_samples.infer_dtype_one,
            op_type="MyOp",
            dtypes=[],
        )


def _infer_dtype_copy_input1(input0_dtype: str, input1_dtype: str) -> str:
    return input1_dtype


def _infer_dtype_const_fp32(input0_dtype: str):
    return torch.float32


def test_generate_op_custom_def_cpp_infer_dtype_copy_and_const():
    text_copy = cpp_mod._generate_op_custom_def_cpp(
        infer_shape_samples.infer_shape_two_by_two,
        _infer_dtype_copy_input1,
        op_type="MyOp",
        dtypes=["torch.float16", "torch.float32"],
    )
    # Output dtype should match input1 (float32) and InferDataType should copy from input 1.
    assert '.DataType({ge::DT_FLOAT})' in text_copy
    assert "GetInputDataType(1)" in text_copy

    text_const = cpp_mod._generate_op_custom_def_cpp(
        infer_shape_samples.infer_shape_one_2d,
        _infer_dtype_const_fp32,
        op_type="MyOpConst",
        dtypes=["torch.float16"],
    )
    # Output dtype should be DT_FLOAT regardless of input, and InferDataType sets constant DT_FLOAT.
    assert '.DataType({ge::DT_FLOAT})' in text_const
    assert "SetOutputDataType(0, ge::DT_FLOAT)" in text_const


def test_generate_op_custom_def_cpp_two_outputs_and_infer_dtype_tuple():
    text = cpp_mod._generate_op_custom_def_cpp(
        infer_shape_samples.infer_shape_two_outputs,
        infer_shape_samples.infer_dtype_two_outputs,
        op_type="DualOut",
        dtypes=["torch.float16", "torch.float32"],
    )
    assert 'this->Output("out0")' in text
    assert 'this->Output("out1")' in text
    assert "SetOutputDataType(0, context->GetInputDataType(0))" in text
    assert "SetOutputDataType(1, context->GetInputDataType(1))" in text
    assert "std::get<0>(out_shape_tuple)" in text
    assert "std::get<1>(out_shape_tuple)" in text