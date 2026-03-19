"""
Shared schema and registration for pypto export meta keys.

Defines which keys exist, their type (string vs zip), and attributes (hidden, user_defined).
Both dump (pypto_op) and extract (extract) use this module. Add new keys via _register_meta_key().
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

__all__: tuple[str, ...] = ()

# ---------------------------------------------------------------------------
# Types and registry
# ---------------------------------------------------------------------------


class _ExtractedType(Enum):
    """Type of value after extraction from node (string vs zip)."""
    STRING = "string"
    ZIP = "zip"


@dataclass(frozen=True)
class _ExtractSpec:
    """Extraction-related attributes for an extractable meta key."""
    extracted_type: _ExtractedType  # Type after extraction from node (string vs zip).
    public_extractor: bool = True  # If False, expose as _extract_<field> instead of extract_<field>.


@dataclass(frozen=True)
class _MetaKeySpec:
    """Spec for a meta key: attribute name, and dump/extract behavior."""
    name: str
    hidden: bool = False
    user_defined: bool = True
    extractable: bool = True
    extraction: Optional[_ExtractSpec] = None  # Set when extractable=True; None otherwise.


_REGISTRY: Dict[str, _MetaKeySpec] = {}


def _register_meta_key(
    name: str,
    *,
    hidden: bool = False,
    user_defined: bool = True,
    extractable: bool = True,
    extracted_type: Optional[_ExtractedType] = None,
    public_extractor: bool = True,
) -> None:
    """Register a meta key at module load."""
    if extractable and extracted_type is None:
        raise ValueError("extracted_type must be set when extractable=True")
    extraction = (
        _ExtractSpec(extracted_type=extracted_type, public_extractor=public_extractor)
        if extractable
        else None
    )
    _REGISTRY[name] = _MetaKeySpec(
        name=name,
        hidden=hidden,
        user_defined=user_defined,
        extractable=extractable,
        extraction=extraction,
    )


def _get_meta_key_spec(key: str) -> Optional[_MetaKeySpec]:
    """Return the spec for a key, or None if not registered."""
    return _REGISTRY.get(key)


def _is_hidden(key: str) -> bool:
    """True if the key should not be dumped to the graph."""
    spec = _get_meta_key_spec(key)
    return spec.hidden if spec else False


def _is_user_defined(key: str) -> bool:
    """True if the key should be included in the meta_json summary."""
    spec = _get_meta_key_spec(key)
    return spec.user_defined if spec else True


def _extractable_string_meta_fields() -> Tuple[str, ...]:
    """Field names for string attributes that are extractable from a node."""
    return tuple(
        k
        for k, s in _REGISTRY.items()
        if s.extraction is not None
        and s.extraction.extracted_type == _ExtractedType.STRING
        and not s.hidden
        and s.extractable
    )


def _extractable_zip_meta_fields() -> Tuple[str, ...]:
    """Field names for zip attributes (node attr name is field + '_zip')."""
    return tuple(
        k[:-4]
        for k, s in _REGISTRY.items()
        if s.extraction is not None
        and s.extraction.extracted_type == _ExtractedType.ZIP
        and not s.hidden
        and s.extractable
        and k.endswith("_zip")
    )


def _get_meta_attr_name(field_name: str, is_zip: bool = False) -> str:
    """Return the attribute name used when reading/writing this field on a node."""
    return f"{field_name}_zip" if is_zip else field_name


# ---------------------------------------------------------------------------
# Meta key name constants
# ---------------------------------------------------------------------------

_META_KEY__KERNEL_NAME = "kernel_name"
_META_KEY__KERNEL_FORMAT = "kernel_format"
_META_KEY__KERNEL_SOURCE_ZIP = "kernel_source_zip"
_META_KEY__KERNEL_BINARY_ZIP = "kernel_binary_zip"
_META_KEY__KERNEL_IR_ZIP = "kernel_ir_zip"
_META_KEY__CPP_SOURCES_ZIP = "cpp_sources_zip"
_META_KEY__TILE_SHAPES = "tile_shapes"
_META_KEY__INFER_SHAPE_SOURCE = "infer_shape_source"
_META_KEY__INFER_SHAPE_SOURCE_CPP = "infer_shape_source_cpp"
_META_KEY__CALC_WORKSPACE_SOURCE = "calc_workspace_source"
_META_KEY__CALC_WORKSPACE_SOURCE_CPP = "calc_workspace_source_cpp"
_META_KEY__META_JSON = "meta_json"


# ---------------------------------------------------------------------------
# Register all known meta keys
# ---------------------------------------------------------------------------

_register_meta_key(_META_KEY__KERNEL_NAME, hidden=False, user_defined=True, extractable=False)
_register_meta_key(_META_KEY__KERNEL_FORMAT, hidden=False, user_defined=True, extractable=True, extracted_type=_ExtractedType.STRING)
_register_meta_key(_META_KEY__KERNEL_SOURCE_ZIP, hidden=False, user_defined=False, extractable=True, extracted_type=_ExtractedType.ZIP)
_register_meta_key(_META_KEY__KERNEL_BINARY_ZIP, hidden=False, user_defined=False, extractable=True, extracted_type=_ExtractedType.ZIP)
_register_meta_key(_META_KEY__KERNEL_IR_ZIP, hidden=False, user_defined=False, extractable=True, extracted_type=_ExtractedType.ZIP)
_register_meta_key(_META_KEY__CPP_SOURCES_ZIP, hidden=False, user_defined=False, extractable=True, extracted_type=_ExtractedType.ZIP)
_register_meta_key(_META_KEY__TILE_SHAPES, hidden=False, user_defined=True, extractable=False)
_register_meta_key(_META_KEY__INFER_SHAPE_SOURCE, hidden=False, user_defined=False, extractable=True, extracted_type=_ExtractedType.STRING)
_register_meta_key(_META_KEY__INFER_SHAPE_SOURCE_CPP, hidden=True, user_defined=False, extractable=False)
_register_meta_key(_META_KEY__CALC_WORKSPACE_SOURCE, hidden=False, user_defined=False, extractable=True, extracted_type=_ExtractedType.STRING)
_register_meta_key(_META_KEY__CALC_WORKSPACE_SOURCE_CPP, hidden=True, user_defined=False, extractable=False)
_register_meta_key(_META_KEY__META_JSON, hidden=False, user_defined=True, extractable=True, extracted_type=_ExtractedType.STRING, public_extractor=False)
