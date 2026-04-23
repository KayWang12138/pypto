"""PyPTO VF API module - direct VF instruction control.

API naming references AscendC for easy migration. Backend directly emits
VF instructions without a C++ API middle layer.
"""

from .op.vf_api_ops import (
    ALL,
    RegTensor,
    VFScope,
    vf_scope,
    CreateMask,
    UpdateMask,
    Duplicate,
    LoadAlign,
    StoreAlign,
    MemBar,
    Max,
    Add,
    Sub,
    Mul,
    Muls,
    Ln,
    FusedExpSub,
    Cast,
    DeInterleave,
    Select,
)

__all__ = [
    "ALL",
    "RegTensor",
    "VFScope",
    "vf_scope",
    "CreateMask",
    "UpdateMask",
    "Duplicate",
    "LoadAlign",
    "StoreAlign",
    "MemBar",
    "Max",
    "Add",
    "Sub",
    "Mul",
    "Muls",
    "Ln",
    "FusedExpSub",
    "Cast",
    "DeInterleave",
    "Select",
]
