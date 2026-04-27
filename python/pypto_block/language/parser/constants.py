# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Constants shared by the AST parser helpers."""

from pypto_block.pypto_core.ir import MemorySpace

_MEMORY_SPACE_MAP: dict[str, MemorySpace] = {
    "Left": MemorySpace.Left,
    "Right": MemorySpace.Right,
    "Vec": MemorySpace.Vec,
    "Mat": MemorySpace.Mat,
    "Acc": MemorySpace.Acc,
    "Scaling": MemorySpace.Scaling,
}

_BUFFER_CLASS_NAMES = frozenset({
    "NBuffer", "Buffer", "UBBuffer", "UBNBuffer",
    "L1Buffer", "L1NBuffer",
    "L0ABuffer", "L0ANBuffer",
    "L0BBuffer", "L0BNBuffer",
    "L0CBuffer", "L0CNBuffer",
})

__all__ = ["_BUFFER_CLASS_NAMES", "_MEMORY_SPACE_MAP"]
