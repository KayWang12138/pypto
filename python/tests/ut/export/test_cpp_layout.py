# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0
"""Tests for cpp_sources ``cpp_layout.json`` manifest (2B layout)."""

from __future__ import annotations

import json

import pytest

import pypto.export
import pypto.export.cpp_layout as cpp_layout_mod


def test_build_cpp_layout_manifest_paths():
    m = cpp_layout_mod._build_cpp_layout_manifest(op_type="AddPyptoCustomOp")
    assert m["schema_version"] == cpp_layout_mod._SCHEMA_VERSION
    assert m["op_type"] == "AddPyptoCustomOp"
    assert m["op_custom_def"] == "op_host/add_pypto_custom_op_def.cpp"
    assert m["executor_tu"] == "op_host/src/add_pypto_custom_op_executor.cpp"
    assert m["onnx_plugin"] == "framework/onnx_plugin/add_pypto_custom_op_plugin.cpp"


def test_load_cpp_layout_ok(tmp_path):
    base = tmp_path / "AddPyptoCustomOp"
    base.mkdir()
    for rel in [
        "op_host/add_pypto_custom_op_def.cpp",
        "op_host/src/add_pypto_custom_op_executor.cpp",
        "framework/onnx_plugin/add_pypto_custom_op_plugin.cpp",
    ]:
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("// x\n", encoding="utf-8")
    manifest = cpp_layout_mod._build_cpp_layout_manifest(op_type="AddPyptoCustomOp")
    (base / cpp_layout_mod._CPP_LAYOUT_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    data, resolved_base = pypto.export.load_cpp_layout(tmp_path)
    assert resolved_base == base
    assert data["op_custom_def"] == manifest["op_custom_def"]


def test_load_cpp_layout_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match=cpp_layout_mod._CPP_LAYOUT_FILENAME):
        pypto.export.load_cpp_layout(tmp_path)


def test_load_cpp_layout_missing_file(tmp_path):
    base = tmp_path / "op"
    base.mkdir()
    manifest = cpp_layout_mod._build_cpp_layout_manifest(op_type="X")
    (base / cpp_layout_mod._CPP_LAYOUT_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Layout file missing"):
        pypto.export.load_cpp_layout(tmp_path)
