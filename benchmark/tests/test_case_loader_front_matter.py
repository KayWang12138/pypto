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
"""Tests for KernelBench SPEC.md YAML front matter generation."""

from __future__ import annotations

import json
import textwrap

from benchmark.case_loader import (
    CaseSpec,
    TensorSpec,
    load_case,
    render_spec_md,
    write_spec,
)


def _parse_front_matter(markdown: str) -> dict:
    assert markdown.startswith("---\n")
    block = markdown.split("---", 2)[1].strip()
    out = {}
    for line in block.splitlines():
        key, value = line.split(": ", 1)
        out[key] = value
    out["schema_version"] = int(out["schema_version"])
    out["supported_dtypes"] = json.loads(out["supported_dtypes"])
    out["p0_shapes"] = json.loads(out["p0_shapes"])
    out["tolerance"] = json.loads(out["tolerance"])
    if "dynamic_axis" in out:
        out["dynamic_axis"] = json.loads(out["dynamic_axis"])
    return out


def test_render_spec_front_matter_from_tensor_specs() -> None:
    case = CaseSpec(
        op_name="Foo",
        case_id="1_Foo",
        source_file="/tmp/1_Foo.py",
        task_desc="def get_inputs(): return []",
        inputs=[
            TensorSpec(name="x0", shape=[2, 3], dtype="torch.float16"),
            TensorSpec(name="x1", shape=[4], dtype="float16"),
            TensorSpec(name="x2", shape=None, dtype="torch.float32"),
        ],
    )

    front_matter = _parse_front_matter(render_spec_md(case))

    assert front_matter["schema_version"] == 1
    assert front_matter["op_name"] == "Foo"
    assert front_matter["supported_dtypes"] == ["float16", "float32"]
    assert front_matter["p0_shapes"] == [[2, 3], [4]]
    assert front_matter["tolerance"] == {"rtol": 0.004, "atol": 0.004}


def test_render_spec_front_matter_probe_fallback() -> None:
    case = CaseSpec(
        op_name="Fallback",
        case_id="2_Fallback",
        source_file="/tmp/2_Fallback.py",
        task_desc="def get_inputs(): return []",
    )

    front_matter = _parse_front_matter(render_spec_md(case))

    assert front_matter["supported_dtypes"] == ["float32"]
    assert front_matter["p0_shapes"] == []
    assert front_matter["tolerance"] == {"rtol": 0.001, "atol": 0.001}
    assert "dynamic_axis" not in front_matter
    assert "### 1.3 数学公式" not in render_spec_md(case)


def test_render_spec_new_interface_globals_when_present() -> None:
    case = CaseSpec(
        op_name="DynamicAxisAdd",
        case_id="101_DynamicAxisAdd",
        source_file="/tmp/101_DynamicAxisAdd.py",
        task_desc="FORMULA = 'out = x + bias'",
        inputs=[TensorSpec(name="x0", shape=[2, 4, 8], dtype="float32")],
        dynamic_axis=["B", "S"],
        formula="out[b, s, d] = x[b, s, d] + bias[d]",
    )

    markdown = render_spec_md(case)
    front_matter = _parse_front_matter(markdown)

    assert front_matter["dynamic_axis"] == ["B", "S"]
    assert "### 1.3 数学公式" in markdown
    assert "out[b, s, d] = x[b, s, d] + bias[d]" in markdown


def test_load_case_populates_front_matter_fields_and_write_spec(tmp_path) -> None:
    case_file = tmp_path / "19_Softmax.py"
    case_file.write_text(
        textwrap.dedent(
            """
            class Model:
                def forward(self, x):
                    return x

            class FakeTensor:
                shape = (16, 256, 256)
                dtype = "float32"

            def get_inputs():
                return [FakeTensor()]

            def get_init_inputs():
                return []
            """
        ).strip(),
        encoding="utf-8",
    )

    case = load_case(case_file, case_id="19_Softmax")
    spec_path = write_spec(case, tmp_path / "custom")
    front_matter = _parse_front_matter(spec_path.read_text(encoding="utf-8"))

    assert case.supported_dtypes == ["float32"]
    assert case.p0_shapes == [[16, 256, 256]]
    assert case.tolerance == {"rtol": 0.001, "atol": 0.001}
    assert front_matter["op_name"] == "Softmax"
    assert front_matter["supported_dtypes"] == ["float32"]
    assert front_matter["p0_shapes"] == [[16, 256, 256]]


def test_load_case_extracts_formula_and_dynamic_axis_globals(tmp_path) -> None:
    case_file = tmp_path / "101_DynamicAxisAdd.py"
    case_file.write_text(
        textwrap.dedent(
            """
            FORMULA = "out[b, s, d] = x[b, s, d] + bias[d]"
            DYNAMIC_AXIS = ["B", "S"]

            class Model:
                def __init__(self, hidden_size):
                    self.hidden_size = hidden_size

                def forward(self, x, bias):
                    return x + bias

            class FakeTensor:
                shape = (2, 4, 8)
                dtype = "float32"

            def get_inputs():
                return [FakeTensor(), FakeTensor()]

            def get_init_inputs():
                return [8]
            """
        ).strip(),
        encoding="utf-8",
    )

    case = load_case(case_file, case_id="101_DynamicAxisAdd")
    markdown = render_spec_md(case)
    front_matter = _parse_front_matter(markdown)

    assert case.formula == "out[b, s, d] = x[b, s, d] + bias[d]"
    assert case.dynamic_axis == ["B", "S"]
    assert front_matter["dynamic_axis"] == ["B", "S"]
    assert "### 1.3 数学公式" in markdown
