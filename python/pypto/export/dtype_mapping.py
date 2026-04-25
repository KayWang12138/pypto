"""Torch dtype ↔ GE / IR conversion for export codegen."""

__all__: tuple[str, ...] = ()

# Torch dtype basenames (``str(torch.dtype)`` without ``torch.``) we map to GE.
# Aligns with ``ge::DataType`` in ``gert_ge_minimal.hpp`` where PyTorch exposes a matching
# basename. Still omitted (no stable ``torch.*`` tensor dtype / not user dtypes): STRING,
# STRING_REF, RESOURCE, DUAL, DUAL_SUB_INT8, DUAL_SUB_UINT8, UNDEFINED, HIFLOAT8.
_SUPPORTED_TORCH_BASES_GE: frozenset[str] = frozenset(
    {
        "float16", "float32", "float64",
        "bfloat16",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "bool",
        "complex32", "complex64", "complex128",
        "qint8", "qint16", "qint32",
        "quint8", "quint16",
    }
)

# Torch basenames whose ``ge::DataType`` tail is not ``<base>.upper()`` (e.g. float32 → FLOAT).
_TORCH_BASE_TO_GE_DT_NAME_EXCEPTIONS: dict[str, str] = {
    "float32": "FLOAT",
    "float64": "DOUBLE",
    "bfloat16": "BF16",
}

# Numeric ``ge::DataType`` values from ``gert_ge_minimal.hpp`` → torch dtype basename.
# Must stay aligned with that header and with ``_SUPPORTED_TORCH_BASES_GE``.
_GE_DATA_TYPE_VALUE_TO_TORCH_BASE: dict[int, str] = {
    0: "float32",  # DT_FLOAT
    1: "float16",  # DT_FLOAT16
    2: "int8",  # DT_INT8
    3: "int32",  # DT_INT32
    4: "uint8",  # DT_UINT8
    6: "int16",  # DT_INT16
    7: "uint16",  # DT_UINT16
    8: "uint32",  # DT_UINT32
    9: "int64",  # DT_INT64
    10: "uint64",  # DT_UINT64
    11: "float64",  # DT_DOUBLE
    12: "bool",  # DT_BOOL
    16: "complex64",  # DT_COMPLEX64
    17: "complex128",  # DT_COMPLEX128
    18: "qint8",  # DT_QINT8
    19: "qint16",  # DT_QINT16
    20: "qint32",  # DT_QINT32
    21: "quint8",  # DT_QUINT8
    22: "quint16",  # DT_QUINT16
    27: "bfloat16",  # DT_BF16
    33: "complex32",  # DT_COMPLEX32
}

if frozenset(_GE_DATA_TYPE_VALUE_TO_TORCH_BASE.values()) != _SUPPORTED_TORCH_BASES_GE:
    raise AssertionError(
        "_GE_DATA_TYPE_VALUE_TO_TORCH_BASE values must match _SUPPORTED_TORCH_BASES_GE exactly"
    )


def _ge_dtype_token_from_base(base: str) -> str:
    """Convert a validated torch dtype basename to a ``ge::DT_*`` token."""
    ge_dt_name = _TORCH_BASE_TO_GE_DT_NAME_EXCEPTIONS.get(base)
    if ge_dt_name is not None:
        return f"ge::DT_{ge_dt_name}"
    # int*/uint*, bool, complex*, qint*, quint* — ``DT_`` + basename uppercased.
    return f"ge::DT_{base.upper()}"


# Per-element size in bytes by ``ge::DataType`` enumerator name (``enum DataType`` in
# ``gert_ge_minimal.hpp``). Keys are the C++ identifiers ``DT_*`` (no ``ge::`` prefix).
# Used for ``calc_workspace`` C++ glue (no PyTorch on the embedded runtime path).
# Must list exactly the DT names implied by ``_GE_DATA_TYPE_VALUE_TO_TORCH_BASE``.
_GE_DATA_TYPE_VALUE_TO_ELEMENT_SIZE: dict[str, int] = {
    "DT_FLOAT": 4,
    "DT_FLOAT16": 2,
    "DT_INT8": 1,
    "DT_INT32": 4,
    "DT_UINT8": 1,
    "DT_INT16": 2,
    "DT_UINT16": 2,
    "DT_UINT32": 4,
    "DT_INT64": 8,
    "DT_UINT64": 8,
    "DT_DOUBLE": 8,
    "DT_BOOL": 1,
    "DT_COMPLEX64": 8,
    "DT_COMPLEX128": 16,
    "DT_QINT8": 1,
    "DT_QINT16": 2,
    "DT_QINT32": 4,
    "DT_QUINT8": 1,
    "DT_QUINT16": 2,
    "DT_BF16": 2,
    "DT_COMPLEX32": 4,
}

if frozenset(_GE_DATA_TYPE_VALUE_TO_ELEMENT_SIZE.keys()) != frozenset(
    _ge_dtype_token_from_base(b).removeprefix("ge::")
    for b in _GE_DATA_TYPE_VALUE_TO_TORCH_BASE.values()
):
    raise AssertionError(
        "_GE_DATA_TYPE_VALUE_TO_ELEMENT_SIZE keys must match GE DT names for "
        "_GE_DATA_TYPE_VALUE_TO_TORCH_BASE entries"
    )


def _torch_dtype_to_ge_dtype(dtype) -> str:
    """Map torch dtype-like values to GE dtype token string.

    Only dtypes in ``_SUPPORTED_TORCH_BASES_GE`` are accepted; entries in
    ``_TORCH_BASE_TO_GE_DT_NAME_EXCEPTIONS`` override the default ``ge::DT_`` +
    uppercase-basename rule.
    """
    dtype_name = str(dtype)
    if not dtype_name.startswith("torch."):
        raise ValueError(f"Unsupported dtype for GE mapping: {dtype!r}")
    base = dtype_name.replace("torch.", "", 1)
    if base not in _SUPPORTED_TORCH_BASES_GE:
        raise ValueError(f"Unsupported torch dtype for GE mapping: {dtype!r}")
    return _ge_dtype_token_from_base(base)


def _ge_data_type_enum_value_to_torch_dtype(value: int):
    """Map ``ge::DataType`` enumerator value (per ``gert_ge_minimal.hpp``) to ``torch.dtype``.

    Only values listed in ``_GE_DATA_TYPE_VALUE_TO_TORCH_BASE`` are supported.
    """
    import torch

    try:
        base = _GE_DATA_TYPE_VALUE_TO_TORCH_BASE[value]
    except KeyError as e:
        raise ValueError(f"Unsupported ge::DataType enum value for torch mapping: {value!r}") from e
    td = getattr(torch, base, None)
    if td is None:
        raise ValueError(
            f"torch has no dtype attribute {base!r} (ge::DataType enum value {value}); "
            "needs a newer PyTorch build or this GE value is not exposed as torch.{base}"
        )
    return td


def _ge_data_type_enum_value_to_element_size(value: int) -> int:
    """Map ``ge::DataType`` enumerator value to element size in bytes.

    Only values with a row in ``_GE_DATA_TYPE_VALUE_TO_TORCH_BASE`` (and thus in
    ``_GE_DATA_TYPE_VALUE_TO_ELEMENT_SIZE``) are supported.
    """
    try:
        base = _GE_DATA_TYPE_VALUE_TO_TORCH_BASE[value]
    except KeyError as e:
        raise ValueError(f"Unsupported ge::DataType enum value for element size: {value!r}") from e
    dt_member = _ge_dtype_token_from_base(base).removeprefix("ge::")
    try:
        return _GE_DATA_TYPE_VALUE_TO_ELEMENT_SIZE[dt_member]
    except KeyError as e:
        raise ValueError(
            f"No element size for ge::DataType {dt_member!r} (enum value {value})"
        ) from e


# TODO: refactor and make more explicit
def _torch_dtype_to_ir_dtype(dtype):
    """Map torch dtype-like values to ``pypto_ir.DataType`` enum."""
    import pypto_ir

    dtype_name = str(dtype)
    if not dtype_name.startswith("torch."):
        raise ValueError(f"unsupported dtype ::: {dtype}")
    base = dtype_name.replace("torch.", "")
    # Special-case bfloat16 to map to BF16 instead of BFP16
    if base == "bfloat16":
        ir_dtype_name = "BF16"
    else:
        ir_dtype_name = (
            base.replace("float", "FP")
            .replace("int", "INT")
            .replace("_", "")
            .upper()
        )
    return getattr(pypto_ir.DataType, ir_dtype_name)
