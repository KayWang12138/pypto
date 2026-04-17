#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

from .helpers import load_lint_module


def test_parse_front_matter_returns_empty_when_missing():
    mod = load_lint_module()
    _parse = getattr(mod, '_parse_front_matter')
    meta, body = _parse("# SPEC\n")
    assert meta == {}
    assert body == "# SPEC\n"


def test_parse_front_matter_extracts_values():
    mod = load_lint_module()
    _parse = getattr(mod, '_parse_front_matter')
    text = """---
schema_version: 1
op_name: demo
supported_dtypes: [bfloat16]
---
# SPEC
"""
    meta, body = _parse(text)
    assert meta["op_name"] == "demo"
    assert meta["supported_dtypes"] == ["bfloat16"]
    assert body.startswith("# SPEC")


def test_validate_doc_schema_requires_fields_by_doc_type():
    mod = load_lint_module()
    _validate = getattr(mod, '_validate_doc_schema')
    spec_errors = _validate("SPEC", {"op_name": "x"})
    design_errors = _validate("DESIGN", {"op_name": "x"})
    api_errors = _validate("API_REPORT", {})
    assert any("supported_dtypes" in err for err in spec_errors)
    assert any("dynamic_axes" in err for err in design_errors)
    assert any("op_name" in err for err in api_errors)
