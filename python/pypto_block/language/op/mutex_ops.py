# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Mutex (buffer-id token) namespace for PyPTO Language DSL.

Exposes ``pl.mutex.lock`` / ``pl.mutex.unlock`` for the A5-only Mutex
synchronization primitive. These are thin re-exports of
``pypto_block.ir.op.system_ops.mutex_lock`` / ``mutex_unlock`` — renamed
so users write the natural form::

    pl.mutex.lock(pl.PipeType.MTE2, buf_id)
    plm.load(tile, ...)
    pl.mutex.unlock(pl.PipeType.MTE2, buf_id)

The underlying ops lower to ``pto.get_buf`` / ``pto.rls_buf`` in the PTO
backend, corresponding to Ascend C's ``Mutex::Lock<pipe>(id)`` /
``Mutex::Unlock<pipe>(id)``.
"""

from pypto_block.ir.op.system_ops import mutex_lock as lock
from pypto_block.ir.op.system_ops import mutex_unlock as unlock

__all__ = ["lock", "unlock"]
