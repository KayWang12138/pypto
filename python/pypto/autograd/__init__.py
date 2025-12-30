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
PyPTO Autograd Module.

Provides automatic differentiation (backward/VJP) capabilities for PyPTO.
This module is designed to be orthogonal to pypto.jit - you can use jit
for compilation/execution and autograd for gradient computation.

Example:
    # Define forward kernel
    def fwd(x, weight, out, loss_out):
        y = x @ weight
        out[:] = y
        loss_out[:] = pypto.sum(y, dim=0)

    # Create training step
    train = pypto.autograd.train_step(
        fwd,
        wrt=("weight",),
        loss="loss_out",
    )

    # Execute training step
    grads = train(x, weight=w, out=out, loss_out=loss, grads={"weight": dw})
"""

# Import core context functions from _core.py (zero dependencies)
from ._core import (
    get_current_tracer,
    is_tracing,
    no_trace,
    set_current_tracer,
    trace,
)

from .graph import (
    Graph,
    Node,
    NodeType,
    Value,
)

from .tracer import (
    Tracer,
    _autograd_hook_op,
    _autograd_hook_move,
)

from .registry import (
    VJPContext,
    VJP_REGISTRY,
    get_vjp,
    has_vjp,
    list_registered_ops,
    register_vjp,
)

from .utils import (
    is_differentiable_dtype,
    unbroadcast,
    zeros_like,
    ones_like,
)

from .engine import (
    BackwardEngine,
    compute_gradients,
)

from .builder import (
    TrainGraphBuilder,
)

from .api import (
    TrainStep,
    train_step,
    jit,
)

# NOTE: Do NOT import rules at module load time to avoid circular imports.
# VJP rules are lazily registered when trace() is first called.
# See _core._ensure_rules_registered()

__all__ = [
    # Public API (M3)
    "train_step",
    "jit",
    "TrainStep",
    # Context management
    "get_current_tracer",
    "is_tracing",
    "no_trace",
    "set_current_tracer",
    "trace",
    # Graph IR
    "Graph",
    "Node",
    "NodeType",
    "Value",
    # Tracer
    "Tracer",
    # Engine
    "BackwardEngine",
    "compute_gradients",
    # Builder
    "TrainGraphBuilder",
    # Registry
    "VJPContext",
    "VJP_REGISTRY",
    "get_vjp",
    "has_vjp",
    "list_registered_ops",
    "register_vjp",
    # Utils
    "is_differentiable_dtype",
    "unbroadcast",
    "zeros_like",
    "ones_like",
    # Internal hooks (used by _op_wrapper and tensor)
    "_autograd_hook_op",
    "_autograd_hook_move",
]
