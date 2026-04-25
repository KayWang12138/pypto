#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

from .helpers import load_lint_module


def test_parse_front_matter_returns_empty_when_missing():
    mod = load_lint_module()
    meta, body = mod._parse_front_matter("# SPEC\n")  # noqa: G.CLS.11
    assert not meta
    assert body == "# SPEC\n"


def test_parse_front_matter_extracts_values():
    mod = load_lint_module()
    text = """---
schema_version: 1
op_name: demo
supported_dtypes: [bfloat16]
---
# SPEC
"""
    meta, body = mod._parse_front_matter(text)  # noqa: G.CLS.11
    assert meta["op_name"] == "demo"
    assert meta["supported_dtypes"] == ["bfloat16"]
    assert body.startswith("# SPEC")


def test_validate_doc_schema_requires_fields_by_doc_type():
    mod = load_lint_module()
    spec_errors = mod._validate_doc_schema("SPEC", {"op_name": "x"})  # noqa: G.CLS.11
    design_errors = mod._validate_doc_schema("DESIGN", {"op_name": "x"})  # noqa: G.CLS.11
    api_errors = mod._validate_doc_schema("API_REPORT", {})  # noqa: G.CLS.11
    assert any("supported_dtypes" in err for err in spec_errors)
    assert any("dynamic_axes" in err for err in design_errors)
    assert any("op_name" in err for err in api_errors)


def test_validate_doc_schema_allows_empty_dynamic_axes():
    """dynamic_axes: [] 表示算子无动态轴，应当视为合法声明，不应报缺失。

    回归测试：防止 OL40 将空列表误判为缺失，导致与 OL31 产生死锁
    （OL40 要求非空 => 声明动态轴 => OL31 要求 impl 用 pypto.DYNAMIC，
    但部分算子如 Cube matmul 不支持显式 DYNAMIC）。
    """
    mod = load_lint_module()
    meta = {"schema_version": 1, "op_name": "demo", "dynamic_axes": []}
    errors = mod._validate_doc_schema("DESIGN", meta)  # noqa: G.CLS.11
    assert errors == []


def test_validate_doc_schema_still_rejects_empty_nonempty_fields():
    """SPEC 中 supported_dtypes/p0_shapes/tolerance 仍不允许为空。

    回归测试：确保放宽 dynamic_axes 的空值判定不会误伤其他必填字段。
    """
    mod = load_lint_module()
    meta = {
        "schema_version": 1,
        "op_name": "demo",
        "supported_dtypes": [],
        "p0_shapes": [],
        "tolerance": {},
    }
    errors = mod._validate_doc_schema("SPEC", meta)  # noqa: G.CLS.11
    assert any("supported_dtypes" in err for err in errors)
    assert any("p0_shapes" in err for err in errors)
    assert any("tolerance" in err for err in errors)
