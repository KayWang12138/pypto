# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0
"""Manifest for cpp_sources zip layout (cpp_layout.json).

Codegen writes this file next to generated ``.cpp`` sources. Consumers resolve paths
only from the manifest (no path heuristics).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from .helpers import _camel_case_to_snake_case

__all__ = ("load_cpp_layout",)

_CPP_LAYOUT_FILENAME = "cpp_layout.json"
_SCHEMA_VERSION = 4

# Internal: JSON keys (relative paths use forward slashes).
_KEY_SCHEMA_VERSION = "schema_version"
_KEY_OP_TYPE = "op_type"
_KEY_OP_CUSTOM_DEF = "op_custom_def"
_KEY_EXECUTOR_TU = "executor_tu"
_KEY_ONNX_PLUGIN = "onnx_plugin"

_REQUIRED_PATH_KEYS: tuple[str, ...] = (
    _KEY_OP_CUSTOM_DEF,
    _KEY_EXECUTOR_TU,
    _KEY_ONNX_PLUGIN,
)


def _build_cpp_layout_manifest(*, op_type: str) -> Dict[str, Any]:
    """Return the manifest dict for the canonical cpp_sources tree (paths relative to zip root).

    *op_type* is the C++/ONNX op identifier (e.g. ``AddPyptoCustomOp``). File path stems use
    snake_case derived from it, matching codegen in ``pypto_op``.
    """
    stem = _camel_case_to_snake_case(op_type)
    return {
        _KEY_SCHEMA_VERSION: _SCHEMA_VERSION,
        _KEY_OP_TYPE: op_type,
        _KEY_OP_CUSTOM_DEF: f"op_host/{stem}_def.cpp",
        _KEY_EXECUTOR_TU: f"op_host/src/{stem}_executor.cpp",
        _KEY_ONNX_PLUGIN: f"framework/onnx_plugin/{stem}_plugin.cpp",
    }


def load_cpp_layout(extracted_dir: Path) -> Tuple[Dict[str, Any], Path]:
    """Find ``cpp_layout.json`` under *extracted_dir*, validate schema and files, return (data, base_dir).

    *base_dir* is the directory containing the manifest; relative paths in *data* are resolved against it.
    """
    manifests = sorted(extracted_dir.rglob(_CPP_LAYOUT_FILENAME))
    if not manifests:
        raise FileNotFoundError(
            f"No {_CPP_LAYOUT_FILENAME} found under {extracted_dir}. "
            "Expected a cpp_sources extract with a layout manifest."
        )
    if len(manifests) > 1:
        raise ValueError(
            f"Expected exactly one {_CPP_LAYOUT_FILENAME}, found {len(manifests)}: "
            + ", ".join(str(p) for p in manifests)
        )
    mpath = manifests[0]
    base = mpath.parent
    data = json.loads(mpath.read_text(encoding="utf-8"))
    if data.get(_KEY_SCHEMA_VERSION) != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {_KEY_SCHEMA_VERSION}: {data.get(_KEY_SCHEMA_VERSION)!r} "
            f"(expected {_SCHEMA_VERSION})"
        )
    for k in _REQUIRED_PATH_KEYS:
        if k not in data:
            raise KeyError(f"{_CPP_LAYOUT_FILENAME} missing required key {k!r}")
        rel = data[k]
        if not rel or not isinstance(rel, str):
            raise ValueError(f"{_CPP_LAYOUT_FILENAME} key {k!r} must be a non-empty string")
        full = (base / rel).resolve()
        try:
            full.relative_to(base.resolve())
        except ValueError as e:
            raise ValueError(f"{_CPP_LAYOUT_FILENAME} path {k!r} escapes extract root: {rel!r}") from e
        if not full.is_file():
            raise FileNotFoundError(f"Layout file missing: {full} (from {_CPP_LAYOUT_FILENAME}[{k!r}])")
    return data, base
