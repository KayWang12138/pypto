#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Autograd context management.

This module re-exports from _core.py for backward compatibility.
New code should import directly from autograd or _core.
"""

# Re-export all from _core for backward compatibility
from ._core import (
    get_current_tracer,
    set_current_tracer,
    is_tracing,
    trace,
    no_trace,
)

__all__ = [
    "get_current_tracer",
    "set_current_tracer",
    "is_tracing",
    "trace",
    "no_trace",
]
