# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal parser models for Python-level DSL constructs."""

from typing import Any

from pypto_block.pypto_core import ir


class _StructVar:
    """Compile-time grouping of IR expressions, accessed via attribute syntax.

    Created by ``pl.struct(field1=val1, field2=val2, ...)``.
    """

    def __init__(self, fields: dict[str, Any], name: str = "") -> None:
        self.fields = fields
        self.name = name


class _StructArrayVar:
    """List/tuple of homogeneous structs supporting dynamic index access.

    Created by assigning a list or tuple of ``_StructVar`` objects that share
    the same field names::

        ctx_arr = [ctx_0, ctx_1, ctx_2]   # all created via pl.struct(...)

    Dynamic indexing ``ctx_arr[idx]`` produces a ``_DynamicStructView`` which
    can be used for field reads (``view.field``), field writes
    (``view.field = val``), or as a function argument.
    """

    def __init__(self, structs: list[_StructVar], name: str = "") -> None:
        self.structs = structs
        self.field_names = list(structs[0].fields.keys())
        self.name = name  # C++ array variable name


class _DynamicStructView:
    """Runtime view into a ``_StructArrayVar`` at a dynamic IR index.

    Field reads lower to ``struct.get`` IR calls; field writes to ``struct.set``.
    The CCE codegen translates these to direct C++ array access: ``arr[idx].field``.
    """

    def __init__(self, array: _StructArrayVar, index_expr: ir.Expr, ref_name: str = "") -> None:
        self.array = array
        self.index_expr = index_expr
        self.ref_name = ref_name  # C++ reference variable name (empty = no ref)


__all__ = ["_DynamicStructView", "_StructArrayVar", "_StructVar"]
