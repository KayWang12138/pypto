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
"""PyPTO Autograd Module."""
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

__all__ = [
    "train_step",
    "jit",
    "TrainStep",
    "get_current_tracer",
    "is_tracing",
    "no_trace",
    "set_current_tracer",
    "trace",
    "Graph",
    "Node",
    "NodeType",
    "Value",
    "Tracer",
    "BackwardEngine",
    "compute_gradients",
    "TrainGraphBuilder",
    "VJPContext",
    "VJP_REGISTRY",
    "get_vjp",
    "has_vjp",
    "list_registered_ops",
    "register_vjp",
    "is_differentiable_dtype",
    "unbroadcast",
    "zeros_like",
    "ones_like",
    "_autograd_hook_op",
    "_autograd_hook_move",
]
