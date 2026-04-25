from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import onnx

from .meta_schema import (
    _get_meta_attr_name,
    _get_meta_key_spec,
    _extractable_string_meta_fields,
    _extractable_zip_meta_fields,
)
from .zip import _unzip_b64_to_dir

DEFAULT_ONNX_DOMAIN = "ai.onnx.contrib"


def _try_call(fn, strict=True):
    """Run fn; if not strict, return None on exception."""
    def wrapper(*args, **kwargs):
        if strict:
            return fn(*args, **kwargs)
        else:
            try:
                return fn(*args, **kwargs)
            except Exception:
                return None
    return wrapper


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------


def _extract_node_from_onnx(
    onnx_model: onnx.ModelProto,
    domain: str,
    op_type: str,
):
    """Return the first ONNX node matching domain and op_type."""
    for node in onnx_model.graph.node:
        if node.domain == domain and node.op_type == op_type:
            return node
    raise ValueError(f"No node found for {domain}::{op_type}")


def _extract_attr_from_onnx_node(
    onnx_node: onnx.NodeProto,
    attr_name: str,
):
    """Return the ONNX node attribute with the given name."""
    for attr in onnx_node.attribute:
        if attr.name == attr_name:
            return attr
    raise KeyError(f"Could not find {attr_name} attribute in onnx node")


def _extract_string_attr_from_onnx_node(
    onnx_node: onnx.NodeProto,
    attr_name: str,
):
    """Read a string attribute from an ONNX node."""
    attr = _extract_attr_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=attr_name,
    )
    if attr.type != onnx.AttributeProto.STRING:
        raise TypeError(f"{attr.name} is not a STRING attribute")
    value = attr.s.decode("utf-8", errors="strict").strip()
    if not value:
        raise ValueError(f"{attr.name} is empty")
    return value


def _extract_zip_from_onnx_node(
    onnx_node: onnx.NodeProto,
    b64_attr_name: str,
    out_dir: Union[str, None] = None,
):
    """Extract a base64-zipped attribute from an ONNX node to out_dir."""
    b64 = _extract_string_attr_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=b64_attr_name,
    )
    op_out_dir = os.path.join(out_dir, onnx_node.op_type)
    return _unzip_b64_to_dir(b64, op_out_dir)


# ---------------------------------------------------------------------------
# Torch Air (GE)
# ---------------------------------------------------------------------------


def _extract_node_from_ge_graph(
    ge_graph: "torchair.ge._ge_graph.GeGraph",
    op_type: str,
):
    """Return the first GE graph op matching op_type."""
    for op in ge_graph._proto.op:
        if op.type == op_type:
            return op
    raise ValueError(f"No node found for {op_type}")


# ---------------------------------------------------------------------------
# Unified model loading + node extraction
# ---------------------------------------------------------------------------


def load_model(path: Union[str, Path]) -> Tuple[Any, str]:
    """Load a pypto-exported model file and return ``(model, format)``.

    *format* is ``"onnx"`` or ``"air"`` (from the file suffix). *model* is an
    ``onnx.ModelProto`` for ``.onnx`` or a ``torchair.ge._ge_graph.GeGraph``
    for ``.air``. The torchair import is lazy so .onnx-only environments do
    not need it installed.
    """
    path = Path(path)
    fmt = path.suffix.lstrip(".").lower()
    if fmt == "onnx":
        return onnx.load(str(path)), fmt
    if fmt == "air":
        try:
            from torchair.ge._ge_graph import GeGraph  # type: ignore
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "Loading .air requires torchair (not installed in this env). "
                "Use an .onnx input instead."
            ) from e
        with open(path, "rb") as f:
            return GeGraph(serialized_model_def=f.read()), fmt
    raise ValueError(f"Unsupported model format: {path.suffix!r}")


def extract_node(
    model: Any,
    op_type: str,
    domain: str = DEFAULT_ONNX_DOMAIN,
):
    """Return the first custom-op node matching *op_type* in *model*.

    Dispatches on the model's Python type:
      * ``onnx.ModelProto`` — matches ``(domain, op_type)``.
      * GE graph (from ``.air``) — matches the fully-qualified
        ``"domain::op_type"``. If *op_type* already contains ``"::"`` it is
        used verbatim; otherwise *domain* is prefixed.
    """
    if isinstance(model, onnx.ModelProto):
        return _extract_node_from_onnx(model, domain=domain, op_type=op_type)
    ge_op_type = op_type if "::" in op_type else f"{domain}::{op_type}"
    return _extract_node_from_ge_graph(model, op_type=ge_op_type)


def _extract_string_attr_from_ge_node(
    ge_node,
    attr_name: str,
):
    """Read a string attribute from a GE node."""
    attr = ge_node.attr[attr_name].s
    if attr is None:
        raise TypeError(f"{attr_name} is not a STRING attribute")
    return attr.decode()


def _extract_zip_from_ge_node(
    ge_node,
    b64_attr_name: str,
    out_dir: Union[str, None] = None,
):
    """Extract a base64-zipped attribute from a GE node to out_dir."""
    b64 = _extract_string_attr_from_ge_node(
        ge_node=ge_node,
        attr_name=b64_attr_name,
    )
    op_out_dir = os.path.join(out_dir, ge_node.type)
    return _unzip_b64_to_dir(b64, op_out_dir)


# ---------------------------------------------------------------------------
# Schema-driven extractors
# ---------------------------------------------------------------------------


def _extract_string_field_from_node(
    node: Any,
    field_name: str,
) -> str:
    """Extract a string-typed meta field from an ONNX or GE node."""
    attr_name = _get_meta_attr_name(field_name, is_zip=False)
    if isinstance(node, onnx.NodeProto):
        return _extract_string_attr_from_onnx_node(node, attr_name)
    return _extract_string_attr_from_ge_node(node, attr_name)


def _extract_zip_field_from_node(
    node: Any,
    field_name: str,
    out_dir: str,
    strict: bool = True,
):
    """Extract a zip-typed meta field from an ONNX or GE node to out_dir."""
    attr_name = _get_meta_attr_name(field_name, is_zip=True)
    if isinstance(node, onnx.NodeProto):
        fn = lambda n, d: _extract_zip_from_onnx_node(n, attr_name, d)
    else:
        fn = lambda n, d: _extract_zip_from_ge_node(n, attr_name, d)
    return _try_call(fn, strict=strict)(node, out_dir)


def _make_string_extractor(field_name: str):
    """Build an extract_<field> callable for a string meta field."""
    def extractor(node):
        return _extract_string_field_from_node(node, field_name)
    return extractor


def _make_zip_extractor(field_name: str):
    """Build an extract_<field> callable for a zip meta field."""
    def extractor(node, out_dir: str, strict: bool = True):
        return _extract_zip_field_from_node(node, field_name, out_dir, strict)
    return extractor


# Build public extract_<field> (or _extract_<field> when not public) from schema
_current_module = sys.modules[__name__]
for _field in _extractable_string_meta_fields():
    spec = _get_meta_key_spec(_field)
    attr_name = (
        f"_extract_{_field}"
        if (spec and spec.extraction and not spec.extraction.public_extractor)
        else f"extract_{_field}"
    )
    setattr(_current_module, attr_name, _make_string_extractor(_field))
for _field in _extractable_zip_meta_fields():
    spec = _get_meta_key_spec(_field + "_zip")  # registry key for zip is field_zip
    attr_name = (
        f"_extract_{_field}"
        if (spec and spec.extraction and not spec.extraction.public_extractor)
        else f"extract_{_field}"
    )
    setattr(_current_module, attr_name, _make_zip_extractor(_field))


def extract_pypto_meta(node):
    """Parse meta_json from node and return as dict."""
    return json.loads(_extract_meta_json(node))


def _export_all_extractors() -> tuple[str, ...]:
    """Names exported by this module (including schema-driven extract_* / _extract_*)."""
    names = [
        "DEFAULT_ONNX_DOMAIN",
        "load_model",
        "extract_node",
        "extract_pypto_meta",
    ]
    for _field in _extractable_string_meta_fields():
        spec = _get_meta_key_spec(_field)
        attr_name = (
            f"_extract_{_field}"
            if (spec and spec.extraction and not spec.extraction.public_extractor)
            else f"extract_{_field}"
        )
        names.append(attr_name)
    for _field in _extractable_zip_meta_fields():
        spec = _get_meta_key_spec(_field + "_zip")
        attr_name = (
            f"_extract_{_field}"
            if (spec and spec.extraction and not spec.extraction.public_extractor)
            else f"extract_{_field}"
        )
        names.append(attr_name)
    return tuple(names)


__all__ = _export_all_extractors()
