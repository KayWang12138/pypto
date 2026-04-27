# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO language operations module.

The auto operation layer has been removed. Manual operations are exported
directly from this package so callers can use ``pypto_block.language.op`` as
the operation namespace.
"""

from . import manual as block
from . import manual
from . import mutex_ops as mutex
from . import ptr_ops as ptr
from . import system_ops as system
from . import vf_api
from .manual import *  # noqa: F401, F403
from .manual import __all__ as _manual_all
from .ptr_ops import addptr, make_tensor

__all__ = [
    *_manual_all,
    "block",
    "manual",
    "mutex",
    "ptr",
    "system",
    "vf_api",
    "make_tensor",
    "addptr",
]
